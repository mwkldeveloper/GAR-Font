import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
import random

class GeneratorSEDataset(Dataset):
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
        basic_styles = data_style_info.get('basic_style',[])

        all_train_contents = data_content_info.get('all_content', [])
        pre_train_contents = data_content_info.get('train_content', [])

        pre_train_content_indices = [c['index'] for c in pre_train_contents]

        pre_train_set = set(pre_train_content_indices)

        new_train_contents = [c for c in all_train_contents if c['index'] not in pre_train_set]

        self.train_ref_pool = {
            c['index']: [other for other in pre_train_content_indices if other != c['index']]
            for c in all_train_contents
        }
        
        for content_item in pre_train_contents:
            style_names = random.sample(train_styles, k=args.se_fonts_num)
            for style_name in style_names:
                content_index = content_item['index']
                content_name = content_item['char']
                sample_info = {
                    'tar_image_path':f'{self.data_dir_path}/{style_name}/{content_index:04d}.png',
                    'content_image_path': f'{self.data_dir_path}/{self.content_font_name}/{content_index:04d}.png',
                    'content_index': content_index,
                    'content_name': content_name,
                    'tar_style': style_name
                }
                self.samples.append(sample_info)

        for content_item in new_train_contents:
            for style_name in basic_styles:
                content_index = content_item['index']
                content_name = content_item['char']
                sample_info = {
                    'tar_image_path':f'{self.data_dir_path}/{style_name}/{content_index:04d}.png',
                    'content_image_path': f'{self.data_dir_path}/{self.content_font_name}/{content_index:04d}.png',
                    'content_index': content_index,
                    'content_name': content_name,
                    'tar_style': style_name
                }
                self.samples.append(sample_info)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tar_style = sample['tar_style']
        content_index = sample['content_index']
        content_name = sample['content_name']

        try:
            tar_img = Image.open(sample['tar_image_path']).convert("RGB")
            content_img = Image.open(sample['content_image_path']).convert("RGB")

            if self.transform:
                tar_img = self.transform(tar_img)
                content_img = self.transform(content_img)
            ref_pool = self.train_ref_pool 
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

        return tar_img, content_img, stacked_style_imgs, content_name, attn_mask
        

def build_generator_SE_datasets(args, transform):
    datasets = {
        'train': GeneratorSEDataset(args, transform=transform, split_type='train') }
    return datasets