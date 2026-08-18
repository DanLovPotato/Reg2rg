import torch.nn as nn
import torch.nn.functional as F
import torch
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from .fvlm_vit.vit import ViT as FvlmViT
from einops.layers.torch import Rearrange
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel
from monai.networks.nets.swin_unetr import SwinTransformer
from .cross_attention import TwoWayTransformer
from .cross_modal_knowledge_enhancer import CrossModalKnowledgeEnhancer, RegionWiseLocalKnowledgeEnhancer
import numpy as np
 
CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]
 
SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]
 
# img_size/patch_size that the fVLM checkpoint's visual_encoder was pretrained/finetuned at
# (see fvlm/finetune.py CROP_SIZE/PATCH_SIZE) - (D, H, W) order. Only used to construct the
# module and size its position_embeddings table; PatchEmbeddingBlock.forward() interpolates
# that table to whatever (D, H, W) actually comes in at runtime, so callers aren't limited to
# this exact input shape as long as they pass axes in (D, H, W) order (see forward() below).
FVLM_NATIVE_CROP_SIZE = (112, 256, 352)
FVLM_PATCH_SIZE = (16, 16, 32)


def load_fvlm_visual_encoder(vit_module, checkpoint_path):
    """Loads the visual_encoder.* weights from a fvlm finetune.py checkpoint (which stores
    the whole BlipPretrain model's state_dict under "model") into a standalone fVLM ViT.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    prefix = 'visual_encoder.'
    sub_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    vit_module.load_state_dict(sub_state_dict, strict=True)


REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]
 
class MyEmbedding(nn.Module):
    def __init__(self, pretrained_visual_encoder=None,pretrained_finegrain_visual_encoder=None, pretrained_adapter=None,bank_npy_path=None, num_embeddings=32000, embedding_dim=4096, perceiver_num=32, vis_dim=768, patch_size=32, frame_patch_size=4, seg_channel=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.torch.randn((num_embeddings, embedding_dim)), requires_grad=True)  # NOTE: will be initialized using the weight from MedLLaMA
        self.image_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.region_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel
 
        self.vision_encoder = ViT(
            image_size=512,          # image size
            frames=512,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=frame_patch_size,      # frame patch size
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        ############ finegrain_clip 用微调好的 fVLM 视觉编码器（对比学习预训练，见
        # fvlm/finetune.py），而不是随机初始化的 vit_3d.ViT
        self.finegrain_clip_vision_encoder = FvlmViT(
            in_channels=1,
            img_size=FVLM_NATIVE_CROP_SIZE,
            patch_size=FVLM_PATCH_SIZE,
            hidden_size=vis_dim,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            qkv_bias=True,
            dropout_rate=0.1,
        )
        ##################
 
        self.mask_encoder = ViT(
            image_size=256,          # image size
            frames=64,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=16,      # frame patch size
            dim=255,
            depth=3,
            heads=8,
            mlp_dim=512,
            channels = 1,
            dropout=0.1,
            emb_dropout=0.1
        )
 
        # load pretrained vision encoder from RadFM
        if pretrained_visual_encoder is not None:
            vit3d_ckpt = torch.load(pretrained_visual_encoder, map_location='cpu')
            self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        #####
        # load fine-tuned fVLM visual_encoder weights (extracted from the full BlipPretrain
        # checkpoint saved by fvlm/finetune.py)
        if pretrained_finegrain_visual_encoder is not None:
            load_fvlm_visual_encoder(self.finegrain_clip_vision_encoder, pretrained_finegrain_visual_encoder)
            #####
 
        # frozen the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        ######    
        for param in self.finegrain_clip_vision_encoder.parameters():
            param.requires_grad = False
        ######
        self.vis_dim = vis_dim
 
        self.perceiver = PerceiverResampler(
            dim=self.vis_dim, num_latents=perceiver_num)
        # load pretrained perceiver and fc from RadFM
        if pretrained_adapter is not None:
            state_dict = torch.load(pretrained_adapter, map_location='cpu')
            self.perceiver.load_state_dict(state_dict['perceiver'])
            # self.fc.load_state_dict(state_dict['fc'])
       
        # self.cross_attn = TwoWayTransformer(
        #     depth=3,
        #     embedding_dim=self.vis_dim,
        #     num_heads=8,
        #     mlp_dim=1024,
        # )
       
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        self.mask_fc = nn.Linear(255, self.embedding_dim)

        # load the report embedding bank once (supports a flat .npy array or a
        # .npz with a 'data' array); GKE/RWLKE retrieve from it every forward pass
        assert bank_npy_path is not None, "bank_npy_path is required for CrossModalKnowledgeEnhancer/RegionWiseLocalKnowledgeEnhancer"
        bank_raw = np.load(bank_npy_path, allow_pickle=True)
        if isinstance(bank_raw, np.lib.npyio.NpzFile):
            bank_array = bank_raw['data']
        elif bank_raw.dtype == object:
            bank_array = bank_raw.item()['data']
        else:
            bank_array = bank_raw
        self.register_buffer('report_bank', torch.from_numpy(bank_array).float(), persistent=False)
        bank_dim = bank_array.shape[-1]

        self.gke = CrossModalKnowledgeEnhancer(d_model=embedding_dim, bank_dim=bank_dim, vis_dim=vis_dim)
        self.rwlke = RegionWiseLocalKnowledgeEnhancer(organs_list=REGIONS, d_model=embedding_dim, bank_dim=bank_dim, vis_dim=vis_dim)


    def forward(self, vision_x, mask_x, text_input, region2areas):
        # 获取输入张量vision_x的形状，并将各个维度的大小赋值给相应的变量
        # B: 批量大小，一次处理的样本数量
        # S: 序列长度，例如在处理文本时，这可能是句子的长度
        # C: 通道数，例如在处理图像时，这可能是颜色通道的数量（红、绿、蓝）
        # H: 高度，例如在处理图像时，这可能是图像的高度
        # W: 宽度，例如在处理图像时，这可能是图像的宽度
        # D: 深"度，例如在处理三维数据（如3D图像或点云）时，这可能是数据的深度
############# Global image embedding
        B, S, C, H, W, D = next(iter(vision_x.values())).shape
       
        raw_image = vision_x['image']
        vision_temp = raw_image
        vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
        vision_temp, pos_embedding = self.vision_encoder(vision_temp)
        vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
        vision_temp = vision_temp.unsqueeze(2)
        vision_temp = self.perceiver(vision_temp)
        n = vision_temp.shape[2]
        vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
        vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n*S)
        image_embedding = vision_temp
       
        del vision_x['image']
 ################ local
        region_embeddings = vision_x
        mask_embeddings = mask_x
               
        for key in region_embeddings.keys():
            ####extract  region texture embedding
            vision_temp = region_embeddings[key]
            vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
            # fVLM's patch/position embeddings are defined in (D, H, W) order (see
            # FVLM_NATIVE_CROP_SIZE/FVLM_PATCH_SIZE above); the tensor here is (..., h, w, d)
            vision_temp = vision_temp.permute(0, 1, 4, 2, 3)
            # the dataset triplicates the single HU channel to 3 (for the old vit_3d
            # encoder); fVLM's patch embedding was pretrained with in_channels=1
            vision_temp = vision_temp[:, :1]
            vision_temp, _ = self.finegrain_clip_vision_encoder(vision_temp)
            vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
            vision_temp = vision_temp.unsqueeze(2)
            vision_temp = self.perceiver(vision_temp)
            n = vision_temp.shape[2]
            vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
            vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n*S)
           
            region_embeddings[key] = vision_temp
            ####extract  region mask embedding
            mask_embedding, _ = self.mask_encoder(mask_x[key])
            mask_embedding = torch.mean(mask_embedding, dim=1)
            mask_embeddings[key] = mask_embedding
 
       
       
        image_embedding = self.fc(image_embedding)
       
        for key in region_embeddings.keys():
            region_embeddings[key] = self.fc(region_embeddings[key])
            mask_embeddings[key] = self.mask_fc(mask_embeddings[key])
 
        # fuse region embeddings and mask embeddings
        for key in region_embeddings.keys():
            region_embeddings[key] = torch.cat([region_embeddings[key], mask_embeddings[key].unsqueeze(1)], dim=1)
       
        max_region = len(region_embeddings)  
        # zero init the vision embedding
        vision_region_embedding = torch.zeros(
            (B, 33*max_region, self.embedding_dim), device=text_input.device) # NOTE: 2 means 1 region and 1 mask embeddings
       
        for i in range(B):
            for j in range(len(region2areas[i])):
                region = region2areas[i][j]
                vision_region_embedding[i, j*33:(j+1)*33, :] = region_embeddings[region][i, :, :]
 
        embedding_weight = torch.cat([self.weight, self.image_token_weight, self.region_token_weight], dim=0)  # num_embeddings+2, embedding_dim
        embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)  # B, num_embeddings+2, embedding_dim
        # B, num_embeddings+2+n, embedding_dim
       
       
        ###把GKE RWLKE
        enhanced_image_embedding = self.gke(raw_image, image_embedding, self.finegrain_clip_vision_encoder, self.report_bank)
        enhanced_vision_region_embedding = self.rwlke(raw_image, vision_region_embedding, self.finegrain_clip_vision_encoder, self.report_bank, region2areas)
 
        embedding_weight = torch.cat([embedding_weight, enhanced_image_embedding, enhanced_vision_region_embedding], dim=1)
       
        ######
        text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(
            vision_region_embedding.dtype).to(text_input.device)  # B, N, num_embeddings+2+n
        out_put = torch.matmul(text_input, embedding_weight)
       
        return out_put
 
 