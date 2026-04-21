import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import random

class GeneratorPreDataset(Dataset):
    def __init__(self, args, transform, split_type='train'):
        super().__init__()
        self.samples =[]
        self.transform = transform
        self.split_type = split_type
        
        self.n_ref = args.n_ref
        self.data_dir_path = args.data_dir_path
        
        
        self.target_token_len = args.dataset_target_token_len # target token len
        self.condition_token_len = args.dataset_condition_token_len # condition token len
        
        with open(args.data_style_info_json, "r", encoding="utf-8") as f:
            data_style_info = json.load(f)
        with open(args.data_content_info_json, "r", encoding="utf-8") as f:
            data_content_info = json.load(f)
            
        self.content_font_name = data_style_info['content_ref_style'][0]
        
        train_styles = data_style_info.get('train_style', [])
        val_styles = data_style_info.get('test_style', [])

        all_contents = data_content_info.get('all_content', [])
        train_contents = data_content_info.get('train_content', [])
        val_contents = data_content_info.get('test_content', [])

        all_content_indices = [c['index'] for c in all_contents]
        train_content_indices = [c['index'] for c in train_contents]
        
        self.train_ref_pool = {
            idx: [other for other in train_content_indices if other != idx]
            for idx in train_content_indices
        }
        self.val_ref_pool = {
            idx: [other for other in train_content_indices if other != idx]
            for idx in all_content_indices
        }

        if split_type == 'train' or split_type == 'seenfont_seencontent':
            styles_to_use, contents_to_use = train_styles, train_contents
        elif split_type == 'unseenfont_seencontent':
            styles_to_use, contents_to_use = val_styles, train_contents
        elif split_type == 'seenfont_unseencontent':
            styles_to_use, contents_to_use = train_styles, val_contents
        elif split_type == 'unseenfont_unseencontent':
            styles_to_use, contents_to_use = val_styles, val_contents
        else:
            raise ValueError(f"Invalid 'split_type': {split_type}")
        
        for style_name in styles_to_use:
            for content_item in contents_to_use:
                content_index = content_item['index']
                
                sample_info = {
                    'tar_image_path':f'{self.data_dir_path}/{style_name}/{content_index:04d}.png',
                    'content_image_path': f'{self.data_dir_path}/{self.content_font_name}/{content_index:04d}.png',
                    'content_index': content_index,
                    'tar_style': style_name
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
            ref_pool = self.train_ref_pool if self.split_type == 'train' else self.val_ref_pool
            
              
            ref_candidates = ref_pool[content_index]
            if len(ref_candidates) < self.n_ref:
                if not ref_candidates:
                    raise FileNotFoundError(f"No available style reference characters for content index {content_index} in '{self.split_type}'.")
                style_ref_indices = random.choices(ref_candidates, k=self.n_ref)
            else:
                style_ref_indices = random.sample(ref_candidates, self.n_ref)
                

            style_img_paths = [
                f"{self.data_dir_path}/{tar_style}/{ref_index:04d}.png"
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
        

def build_generator_PRE_datasets(args, transform):
    datasets = {
        'train': GeneratorPreDataset(args, transform=transform, split_type='train'),
        'val_seenfont_seencontent': GeneratorPreDataset(args, transform=transform, split_type='seenfont_seencontent'),
        'val_unseenfont_seencontent': GeneratorPreDataset(args, transform=transform, split_type='unseenfont_seencontent'),
        'val_seenfont_unseencontent': GeneratorPreDataset(args, transform=transform, split_type='seenfont_unseencontent'),
        'val_unseenfont_unseencontent': GeneratorPreDataset(args, transform=transform, split_type='unseenfont_unseencontent')
    }
    return datasets