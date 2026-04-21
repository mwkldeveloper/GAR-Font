import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import random

class GeneratorNFADataset(Dataset):
    def __init__(self, args, transform, split_type='train'):
        super().__init__()
        self.samples =[]
        self.transform = transform
        self.split_type = split_type
        
        self.n_ref = args.n_ref
        self.data_dir_path = args.data_dir_path
        self.data_nfa_dir_path = args.data_nfa_dir_path
        
        self.target_token_len = args.dataset_target_token_len # target token len
        self.condition_token_len = args.dataset_condition_token_len # condition token len
        
        with open(args.data_style_info_json, "r", encoding="utf-8") as f:
            data_style_info = json.load(f)
        with open(args.data_content_info_json, "r", encoding="utf-8") as f:
            data_content_info = json.load(f)

        nfa_styles = [
            d for d in os.listdir(self.data_nfa_dir_path)
            if os.path.isdir(os.path.join(self.data_nfa_dir_path, d))
        ]
        self.nfa_style_content_map = {}
        for style in nfa_styles:
            style_dir = os.path.join(self.data_nfa_dir_path, style)
            img_files = [
                f for f in os.listdir(style_dir)
                if f.endswith(".png")
            ]
            indices = [int(os.path.splitext(f)[0]) for f in img_files]
            self.nfa_style_content_map[style] = indices

        all_nfa_content_indices = set()

        for indices in self.nfa_style_content_map.values():
            all_nfa_content_indices.update(indices)

        all_nfa_content_indices = list(all_nfa_content_indices)

        self.content_font_name = data_style_info['content_ref_style'][0]

        train_styles = data_style_info.get('train_style', [])
        train_contents = data_content_info.get('train_content', [])
        test_contents = data_content_info.get('test_content', [])
        


        filtered_train_contents = [
            c for c in train_contents
            if c['index'] not in all_nfa_content_indices
        ]

        filtered_test_contents = [
            c for c in test_contents
            if c['index'] not in all_nfa_content_indices
        ]
        
        train_content_indices = [c['index'] for c in train_contents]
        filtered_train_contents_indices = [c['index'] for c in filtered_train_contents]
        filtered_test_contents_indices = [c['index'] for c in filtered_test_contents]


        all_content_indices = list(set(all_nfa_content_indices + filtered_train_contents_indices + filtered_test_contents_indices))
        self.no_nfa_ref_pool = {
            idx: [other for other in train_content_indices if other != idx]
            for idx in all_content_indices
        }


        if split_type in ['train', 'val_nfacontent_nfafont']:
            styles_to_use = nfa_styles
        elif split_type == 'val_precontent_nfafont':
            styles_to_use = nfa_styles
            contents_indices_to_use = filtered_train_contents_indices
        elif split_type == 'val_testcontent_nfafont':
            styles_to_use = nfa_styles
            contents_indices_to_use = filtered_test_contents_indices

        else:
            raise ValueError(f"Invalid 'split_type': {split_type}")

        for style_name in styles_to_use:
            if split_type in ['train', 'val_nfacontent_nfafont']:
                style_indices = self.nfa_style_content_map[style_name]

                for content_index in style_indices:
                    sample_info = {
                        'tar_image_path': os.path.join(
                            self.data_nfa_dir_path,
                            style_name,
                            f"{content_index:04d}.png"
                        ),
                        'content_image_path': os.path.join(
                            self.data_dir_path,
                            self.content_font_name,
                            f"{content_index:04d}.png"
                        ),
                        'content_index': content_index,
                        'tar_style': style_name,
                        'is_nfa':True
                    }
                    self.samples.append(sample_info)

            else:
                for content_index in contents_indices_to_use:
                    sample_info = {
                        'tar_image_path': os.path.join(
                            self.data_dir_path,
                            style_name,
                            f"{content_index:04d}.png"
                        ),
                        'content_image_path': os.path.join(
                            self.data_dir_path,
                            self.content_font_name,
                            f"{content_index:04d}.png"
                        ),
                        'content_index': content_index,
                        'tar_style': style_name,
                        'is_nfa':True
                    }
                    self.samples.append(sample_info)



        self.remaining_train_contents = train_contents.copy()
        random.shuffle(self.remaining_train_contents) 
        
        if split_type == 'train':
            for style_name in train_styles:
                sampled_contents = []

                while len(sampled_contents) < args.nfa_ori_train_num:
                    if not self.remaining_train_contents:
                        self.remaining_train_contents = train_contents.copy()
                        random.shuffle(self.remaining_train_contents)
                    sampled_contents.append(self.remaining_train_contents.pop())

                for content_item in sampled_contents:
                    content_index = content_item['index']
                    sample_info = {
                        'tar_image_path': f'{self.data_dir_path}/{style_name}/{content_index:04d}.png',
                        'content_image_path': f'{self.data_dir_path}/{self.content_font_name}/{content_index:04d}.png',
                        'content_index': content_index,
                        'tar_style': style_name,
                        'is_nfa':False
                    }
                    self.samples.append(sample_info)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tar_style = sample['tar_style']
        content_index = sample['content_index']
        try:
            tar_img = Image.open(sample['tar_image_path']).convert("RGB")
            content_img = Image.open(sample['content_image_path']).convert("RGB")

            if self.transform:
                tar_img = self.transform(tar_img)
                content_img = self.transform(content_img)


            if sample['is_nfa']:
                style_dir = self.data_nfa_dir_path
                style_indices = self.nfa_style_content_map[tar_style]
                ref_candidates = [idx for idx in style_indices if idx != content_index]
            else:
                style_dir = self.data_dir_path
                ref_candidates = self.no_nfa_ref_pool[content_index]

            if len(ref_candidates) < self.n_ref:
                if not ref_candidates:
                    raise FileNotFoundError(f"No available style reference characters for content index {content_index} in '{self.split_type}'.")
                style_ref_indices = random.choices(ref_candidates, k=self.n_ref)
            else:
                style_ref_indices = random.sample(ref_candidates, self.n_ref)

            style_img_paths = [
                f"{style_dir}/{tar_style}/{ref_index:04d}.png"
                for ref_index in style_ref_indices
            ]


            style_imgs = [Image.open(p).convert("RGB") for p in style_img_paths]
            if self.transform:
                style_imgs = [self.transform(img) for img in style_imgs]
            
            stacked_style_imgs = torch.stack(style_imgs, dim=0)  # (n_ref, C, H, W)

        except Exception as e:
            print(f"An error occurred while processing index {idx}, sample info: {sample}")
            print(f"Error details: {e}")
            exit(0)

        full_token_len = self.condition_token_len + self.target_token_len
        attn_mask = torch.tril(torch.ones(full_token_len, full_token_len)).to(torch.bool)

        return tar_img, content_img, stacked_style_imgs, attn_mask, torch.tensor(1)
        


def build_generator_NFA_datasets(args, transform):
    datasets = {
        'train': GeneratorNFADataset(args, transform=transform, split_type='train'),
        'val_nfacontent_nfafont': GeneratorNFADataset(args, transform=transform, split_type='val_nfacontent_nfafont'),
        'val_precontent_nfafont': GeneratorNFADataset(args, transform=transform, split_type='val_precontent_nfafont'),
        'val_testcontent_nfafont': GeneratorNFADataset(args, transform=transform, split_type='val_testcontent_nfafont'),
    }
    return datasets