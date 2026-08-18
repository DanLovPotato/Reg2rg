import os
import glob
import pickle
import random

import numpy as np
import pandas as pd
import torch
import nibabel as nib
import tqdm

from torch.utils.data import Dataset
import monai.transforms as transforms

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


class RadGenomeDataset_Train(Dataset):
    """
    RadGenome 训练集 Dataset 类。

    真实数据加载（load_accession_sentences / prepare_samples / mask_nii_img_to_tensor /
    text_add_image_tokens / text_add_region_tokens）移植自 ../../../Reg2RG（未修改的原始仓库）
    里已验证可跑通的同名实现，基本保持原样。

    在此基础上新增的是“双分支”结构：每个样本同时构建一份完整分支（Full）和一份挖空降级分支
    （Mask，随机挑选部分区域，将其体素置空、GT 替换为“缺失”提示），供 train_radgenome.py 里的
    CustomTrainer 分别前向传播、分别算 loss。
    """

    # 归一化后的“空气值”，和 mask_nii_img_to_tensor 里 (-1024 -> (x+400)/600) 的换算保持一致
    BLANK_VALUE = (-1024 + 400) / 600

    def __init__(self, text_tokenizer, image_padding_tokens, region_padding_tokens,
                 data_folder, mask_folder, csv_file, cache_dir=None, max_seq=2048, voc_size=32000):
        self.tokenizer = text_tokenizer
        self.image_padding_tokens = image_padding_tokens
        self.region_padding_tokens = region_padding_tokens
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.max_seq = max_seq
        self.voc_size = voc_size

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.samples = self.prepare_samples()

        self.target_size = (256, 256, 64)  # NOTE: the target input size of the image

        def threshold(x):
            return x > -1000

        self.region_transform = transforms.Compose([
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])

    def load_accession_sentences(self, csv_file):
        df = pd.read_csv(csv_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(current_file_dir, 'train_samples.pkl')

        if os.path.exists(cache_file):
            samples = pickle.load(open(cache_file, 'rb'))
        else:
            for patient_folder in tqdm.tqdm(patient_folders):
                accession_folders = glob.glob(os.path.join(patient_folder, '*'))

                for accession_folder in accession_folders:
                    nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
                    for nii_file in nii_files:
                        accession_number = nii_file.split("/")[-1]

                        if accession_number not in self.accession_to_sentences:
                            continue

                        single_sample = {}
                        volume_name = accession_number.split(".")[0]
                        mask_path = os.path.join(self.mask_folder, 'seg_' + volume_name)
                        single_sample['image'] = nii_file

                        flag = False
                        for region in REGIONS:
                            if region in self.accession_to_sentences[accession_number]:
                                mask_file = os.path.join(mask_path, region + '.nii.gz')
                                region_report = self.accession_to_sentences[accession_number][region]
                                single_sample[region] = [mask_file, region_report]
                                flag = True
                        if not flag:
                            continue

                        samples.append(single_sample)

            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)

        print('Number of training samples: ', len(samples))
        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths):
        img_data = nib.load(img_path).get_fdata()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path).get_fdata()
            masks.append(mask_data)
            mask_keys.append(key)

            if np.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data
            mask_img[mask_data == 0] = -1024
            mask_img = mask_img[np.newaxis, ...]

            tensor = self.region_transform(mask_img)

            hu_min, hu_max = -1000, 200
            tensor = torch.clamp(tensor, hu_min, hu_max)
            tensor = (((tensor + 400) / 600)).float()
            tensor = tensor.repeat(3, 1, 1, 1)
            tensor = tensor.unsqueeze(0)  # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True

        if not flag:
            print('No mask: ', img_path)

        img_data = img_data[np.newaxis, ...]
        masks_data = np.stack(masks, axis=0)
        tensors = self.image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor + 400) / 600)).float()
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0)  # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors

    def text_add_image_tokens(self, text):
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text
        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. '
        text = region_text + text
        return text

    def _build_branch(self, region_order, region_reports, mask_img_tensors, mask_tensors, dropped_regions):
        """
        构建单条分支样本：
        - dropped_regions 为空集合 -> Full 分支：完整图像 + 完整 Prompt + 完整 GT。
        - dropped_regions 非空 -> Mask 分支：被选中区域的全局体素及自身裁剪图置空，GT 替换为“缺失”提示。
        掩码本身（mask_x，标识区域解剖位置）始终保持完整，不随体素挖空而改变。
        """
        # 1. 挖空全局图：用各区域在变换后坐标系下对齐的 mask_tensors 直接在归一化后的全局图上置空
        global_image = mask_img_tensors['image'].clone()
        for region in dropped_regions:
            region_mask = mask_tensors[region][0] > 0.5  # (256, 256, 64)，和 global_image 空间对齐
            global_image[:, :, region_mask] = self.BLANK_VALUE

        vision_x = {'image': global_image}
        mask_x = {}
        region2area = {}
        for i, area in enumerate(region_order):
            region2area[i] = area
            if area in dropped_regions:
                vision_x[area] = torch.full_like(mask_img_tensors[area], self.BLANK_VALUE)
            else:
                vision_x[area] = mask_img_tensors[area]
            mask_x[area] = mask_tensors[area]

        # 2. 构造 Prompt（占位符不透露区域名字，需要模型自己识别）
        instruction = ("Given the provided global and regional information from this CT scan, please generate a "
                        "comprehensive medical report for each region. First, identify the anatomical area "
                        "corresponding to each region, then provide detailed information about these anatomical "
                        "structures and any abnormalities that are essential. You can refer to the global "
                        "information as the context and take it as a supplement when generating each region report.")
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))
        prompt = self.text_add_image_tokens(prompt)

        # 3. 构造 GT：被挖空的区域用固定的“缺失”提示替代真实报告
        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            if area in dropped_regions:
                region_report = "Not evaluable due to missing data."
            else:
                region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        self.tokenizer.padding_side = "right"
        text_tensor = self.tokenizer(
            prompt + ' ' + combined_report, max_length=self.max_seq, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]
        text_input[torch.sum(attention_mask)] = self.tokenizer.eos_token_id

        prompt_tensor = self.tokenizer(
            prompt, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        prompt_length = torch.sum(prompt_tensor["attention_mask"][0])

        label = text_input.clone()
        label[label == self.tokenizer.pad_token_id] = -100
        label[label >= self.voc_size] = -100  # 特殊占位符 token 也不计入 loss
        label[:prompt_length] = -100  # 只在答案部分计算 loss

        return {
            'lang_x': text_input,
            'vision_x': vision_x,
            'mask_x': mask_x,
            'region2area': region2area,
            'attention_mask': attention_mask,
            'label': label
        }

    def __getitem__(self, index):
        sample = self.samples[index]
        img_file = sample['image']

        region_reports = {}
        mask_files = {}
        for key in sample:
            if key == 'image':
                continue
            mask_file, region_report = sample[key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors = self.mask_nii_img_to_tensor(img_file, mask_files)

        # NOTE: 按实际生成出 tensor 的区域为准（个别 mask 可能是空的，被 mask_nii_img_to_tensor 跳过）
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                region_reports.pop(key)

        region_order = list(region_reports.keys())
        random.shuffle(region_order)  # 对应架构图里的 "Shuffle & Replace"

        # Full 分支：不挖空任何区域
        full_sample = self._build_branch(region_order, region_reports, mask_img_tensors, mask_tensors,
                                          dropped_regions=set())

        # Mask 分支：随机挑选 30%~50% 的有效区域挖空
        dropped_regions = set()
        if region_order:
            num_to_drop = max(1, int(len(region_order) * random.uniform(0.3, 0.5)))
            dropped_regions = set(random.sample(region_order, num_to_drop))
        mask_sample = self._build_branch(region_order, region_reports, mask_img_tensors, mask_tensors,
                                          dropped_regions=dropped_regions)

        return {
            "full": full_sample,
            "mask": mask_sample
        }
