import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def encode_clip_query(raw_image, clip_encoder):
    """
    Encode the whole CT scan with the (frozen) finegrain CLIP vision encoder and
    mean-pool it down to one retrieval query vector per sample.
    Args:
        raw_image: [B, S, C, H, W, D]
        clip_encoder: a ViT instance (e.g. MyEmbedding.finegrain_clip_vision_encoder)
    Returns:
        [B, vis_dim]
    """
    B = raw_image.shape[0]
    img = rearrange(raw_image, "b S c h w d -> (b S) c h w d")
    # fVLM's finegrain_clip_vision_encoder expects (D, H, W) axis order, not (H, W, D) -
    # see the same permute in MyEmbedding.forward's region-embedding loop.
    img = img.permute(0, 1, 4, 2, 3)
    feats, _ = clip_encoder(img)          # (b S), tokens, vis_dim
    feats = feats.mean(dim=1)             # (b S), vis_dim
    feats = rearrange(feats, "(b S) d -> b S d", b=B)
    return feats.mean(dim=1)              # b, vis_dim


# =====================================================================
# 1. 单器官知识增强模块 (Cross-Modal Knowledge Enhancer / KE)
# =====================================================================
class CrossModalKnowledgeEnhancer(nn.Module):
    """
    用 finegrain CLIP 编码器把当前样本的整张 CT 编码成检索 query，去 report bank 里
    检索 Top-K 最相似的报告特征，再用多头交叉注意力让视觉 token 参考这些报告特征，
    并通过一个非线性层进行特征提纯与增强。
    """
    def __init__(self, d_model, bank_dim, vis_dim, num_heads=8, dropout=0.1, top_k=16):
        super().__init__()
        self.d_model = d_model
        self.bank_dim = bank_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_model ** -0.5
        # 把 CLIP 视觉 query 投影到 report bank 的向量空间，用于检索
        self.retrieval_query_proj = nn.Linear(vis_dim, bank_dim)
        # 三个核心线性投影层
        self.visual_query = nn.Linear(d_model, d_model)
        self.top_k_report_key = nn.Linear(bank_dim, d_model)
        self.top_k_report_value = nn.Linear(bank_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)  # TODO: swap back to KANLinear once modules.efficient_kan exists

    def forward(self, raw_image, visual_tokens, clip_encoder, report_bank):
        """
        Args:
            raw_image: 当前样本整张 CT，形状 [batch_size, S, C, H, W, D]
            visual_tokens: 待增强的视觉特征，形状 [batch_size, visual_len, d_model]
            clip_encoder: 用于生成检索 query 的 finegrain CLIP 视觉编码器
            report_bank: 整个 report embedding bank，形状 [N, bank_dim]
        """
        query_vis_feat = encode_clip_query(raw_image, clip_encoder)  # [batch_size, vis_dim]
        return self._enhance(query_vis_feat, visual_tokens, report_bank)

    def _enhance(self, query_vis_feat, visual_tokens, report_bank):
        top_k_report_tokens = self._retrieve(query_vis_feat, report_bank)
        return self._cross_attend(visual_tokens, top_k_report_tokens)

    def _retrieve(self, query_vis_feat, report_bank):
        """
        Args:
            query_vis_feat: [batch_size, vis_dim]
            report_bank: [N, bank_dim]
        Returns:
            [batch_size, top_k, bank_dim]
        """
        query = self.retrieval_query_proj(query_vis_feat)  # [batch_size, bank_dim]
        bank = report_bank.to(device=query.device, dtype=query.dtype)
        query_norm = F.normalize(query, p=2, dim=-1)
        bank_norm = F.normalize(bank, p=2, dim=-1)
        sim_scores = torch.matmul(query_norm, bank_norm.T)  # [batch_size, N]
        k = min(self.top_k, bank.size(0))
        _, topk_indices = torch.topk(sim_scores, k=k, dim=-1)  # [batch_size, k]
        return bank[topk_indices]  # [batch_size, k, bank_dim]

    def _cross_attend(self, visual_tokens, top_k_report_tokens):
        """
        Args:
            visual_tokens: 当前器官/整图的视觉特征，形状 [batch_size, visual_len, d_model]
            top_k_report_tokens: 从 Report Bank 检索的文本特征，形状 [batch_size, top_k, bank_dim]
        """
        batch_size = visual_tokens.size(0)

        # 1. 投影并变换多头 Query (视觉特征)
        query = self.visual_query(visual_tokens)
        res_query = query.view(batch_size, -1, self.d_model)  # 保留残差连接用
        query = query.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        query = query.permute(0, 2, 1, 3)

        # 2. 投影并变换多头 Key (文本特征)
        key = self.top_k_report_key(top_k_report_tokens)
        key = key.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        key = key.permute(0, 2, 3, 1)

        # 3. 投影并变换多头 Value (文本特征)
        value = self.top_k_report_value(top_k_report_tokens)
        value = value.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        value = value.permute(0, 2, 1, 3)

        # 4. 计算跨模态注意力 (Cross-Attention)
        attn = torch.matmul(query, key) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 5. 加权聚合与残差/非线性增强
        out = torch.matmul(attn, value)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        out = self.norm(res_query + out)
        out = self.fc1(out)
        return out


# =====================================================================
# 2. 区域级局部知识增强模块 (RWLKE)
# =====================================================================
class RegionWiseLocalKnowledgeEnhancer(nn.Module):
    """
    管理多个器官，为每个器官独立维护一个专属的 CrossModalKnowledgeEnhancer 实例。
    """
    def __init__(self, organs_list, d_model, bank_dim, vis_dim, num_heads=8, dropout=0.1, top_k=16, region_token_len=33):
        super().__init__()
        self.organs_list = organs_list
        self.region_token_len = region_token_len
        self.enhancers = nn.ModuleDict({
            organ: CrossModalKnowledgeEnhancer(d_model, bank_dim, vis_dim, num_heads, dropout, top_k)
            for organ in organs_list
        })

    def forward(self, raw_image, vision_region_embedding, clip_encoder, report_bank, region2areas):
        """
        vision_region_embedding 里每个样本的 region_token_len (33) token 一段，
        第 i 个样本第 j 段对应的器官是 region2areas[i][j]（跟 MyEmbedding.forward 里
        把 region_embeddings 写进 vision_region_embedding 的那个 scatter 循环用的是
        同一套切分规则）。
        Args:
            raw_image: [batch_size, S, C, H, W, D]
            vision_region_embedding: [batch_size, region_token_len * max_region, d_model]
            clip_encoder: finegrain CLIP 视觉编码器
            report_bank: [N, bank_dim]
            region2areas: List[List[str]]，每个样本按 slot 顺序排列的器官名
        Returns:
            跟 vision_region_embedding 形状相同的增强后 tensor
        """
        enhanced = vision_region_embedding.clone()
        for i, organs_for_sample in enumerate(region2areas):
            query_vis_feat = encode_clip_query(raw_image[i:i + 1], clip_encoder)  # [1, vis_dim]，每个样本算一次，器官间共用
            for j, organ in enumerate(organs_for_sample):
                start, end = j * self.region_token_len, (j + 1) * self.region_token_len
                organ_tokens = vision_region_embedding[i:i + 1, start:end, :]
                enhanced[i:i + 1, start:end, :] = self.enhancers[organ]._enhance(query_vis_feat, organ_tokens, report_bank)
        return enhanced
 
 
# =====================================================================
# 3. KAN 空间自适应注意力池化模块 (K-SAP - 核心创新点)
# =====================================================================
class KANEnhancedSpatialPooling(nn.Module):
    """
    利用 KAN 的 B 样条非线性拟合能力，动态过滤背景噪音，
    将密集 Patch 序列平滑转换为 GCN 所需的器官级紧凑节点向量。
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.kan_gate = KANLinear(d_model, 1)  # 替代传统 MLP 计算空间注意力权重
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
 
    def forward(self, enhanced_local_features_dict):
        organ_names = list(enhanced_local_features_dict.keys())
        pooled_organ_list = []
        for organ in organ_names:
            feat = enhanced_local_features_dict[organ]      # 形状: [batch_size, patch_len, d_model]
            raw_scores = self.kan_gate(feat)                 # 形状: [batch_size, patch_len, 1]
            attn_weights = F.softmax(raw_scores, dim=1)      # 归一化注意力权重
            # 加权求和池化 (Attention Pooling)
            pooled_feat = torch.sum(feat * attn_weights, dim=1) # 形状: [batch_size, d_model]
            pooled_feat = self.norm(self.proj(pooled_feat))
            pooled_organ_list.append(pooled_feat)
        # 堆叠成 GCN 输入矩阵，形状: [batch_size, num_organs, d_model]
        gcn_node_inputs = torch.stack(pooled_organ_list, dim=1)
        return gcn_node_inputs, organ_names
 
 
# =====================================================================
# 4. 全局知识增强与拓扑图网络模块 (GKE)
# =====================================================================
class GlobalKnowledgeEnhancerWithKSAP(nn.Module):
    """
    结合 K-SAP 池化与 GCN 算子，学习器官间的解剖学拓扑与全局关联。
    """
    def __init__(self, organs_list, d_model):
        super().__init__()
        self.organs_list = organs_list
        self.pooler = KANEnhancedSpatialPooling(d_model)
        # GCN 拓扑学习层
        self.gcn1 = GCNConv(d_model, d_model)
        self.gcn2 = GCNConv(d_model, d_model)
        self.global_proj = nn.Linear(d_model, d_model)
 
    def forward(self, enhanced_local_features_dict, edge_index):
        """
        Args:
            enhanced_local_features_dict: RWLKE 的输出字典
            edge_index: PyG 格式的器官拓扑图邻接边索引 [2, num_edges]
        """
        batch_size = enhanced_local_features_dict[self.organs_list[0]].size(0)
        # 1. KAN 空间池化降维对齐
        node_inputs, _ = self.pooler(enhanced_local_features_dict)  # [batch_size, num_organs, d_model]
        # 2. 逐样本通过 GCN 学习器官间空间拓扑关联
        gcn_outputs = []
        for i in range(batch_size):
            x = node_inputs[i]  # [num_organs, d_model]
            x = F.relu(self.gcn1(x, edge_index))
            x = self.gcn2(x, edge_index)
            gcn_outputs.append(x)
        updated_nodes = torch.stack(gcn_outputs, dim=0)  # [batch_size, num_organs, d_model]
        # 3. 汇聚全局解剖学特征
        global_anatomical_features = self.global_proj(updated_nodes.mean(dim=1))  # [batch_size, d_model]
        return global_anatomical_features, updated_nodes
 
 
# =====================================================================
# 5. 器官专属 Bank 独立检索函数 (Cosine Similarity)
# =====================================================================
def get_organ_specific_topk_reports(organ_visual_tokens_dict, organ_report_banks, top_k=16):
    """
    针对每个器官，去其各自专属的 Report Bank 中利用余弦相似度独立检索 Top-K 文本特征。
    """
    organ_topk_tokens_dict = {}
    for organ, v_tokens in organ_visual_tokens_dict.items():
        batch_size = v_tokens.size(0)
        query = v_tokens.mean(dim=1)  # 使用器官视觉特征均值作为 Query [batch_size, d_model]
        bank = organ_report_banks[organ].to(query.device)  # 对应器官的专属文本库
        # 归一化并计算余弦相似度
        query_norm = F.normalize(query, p=2, dim=-1)
        bank_norm = F.normalize(bank, p=2, dim=-1)
        sim_scores = torch.matmul(query_norm, bank_norm.T)  # [batch_size, Total_Reports]
        # 检索 Top-K 索引并捞取文本特征
        _, topk_indices = torch.topk(sim_scores, k=top_k, dim=-1)  # [batch_size, top_k]
        topk_tokens = bank[topk_indices]  # [batch_size, top_k, d_model]
        organ_topk_tokens_dict[organ] = topk_tokens
    return organ_topk_tokens_dict