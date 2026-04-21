import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class ScorerDataset(Dataset):
    def __init__(self, args, transform, split='train'):
        self.data_dir_path = args.data_dir_path
        self.transform = transform
        self.split = split
        
        with open(args.data_style_info_json, "r", encoding="utf-8") as f:
            data_style_info = json.load(f)
        with open(args.data_content_info_json, "r", encoding="utf-8") as f:
            data_content_info = json.load(f)
            
        if split == 'train':
            styles = data_style_info.get('train_style', [])
            contents = data_content_info.get('train_content', [])
        elif split == 'val':
            styles = data_style_info.get('test_style', [])
            contents = data_content_info.get('test_content', [])
        else:
            raise ValueError(f"Invalid split type: {split}")

        self.styles = styles
        self.contents = [c['index'] for c in contents]
        
        self.style_to_contents = {style: self.contents for style in self.styles}
        
        self.dataset_len = len(self.styles) * len(self.contents)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if random.random() < 0.5:
            style = random.choice(self.styles)
            
            content1, content2 = random.sample(self.style_to_contents[style], 2)
            
            img_path1 = f"{self.data_dir_path}/{style}/{content1:04d}.png"
            img_path2 = f"{self.data_dir_path}/{style}/{content2:04d}.png"
            label = 1.0
        else:
            style1, style2 = random.sample(self.styles, 2)
            
            content1 = random.choice(self.style_to_contents[style1])
            content2 = random.choice(self.style_to_contents[style2])
            
            img_path1 = f"{self.data_dir_path}/{style1}/{content1:04d}.png"
            img_path2 = f"{self.data_dir_path}/{style2}/{content2:04d}.png"
            label = 0.0

        try:
            img1 = Image.open(img_path1).convert("RGB")
            img2 = Image.open(img_path2).convert("RGB")

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        except Exception as e:
            print(f"Error loading images: {img_path1}, {img_path2}")
            print(e)
            return self.__getitem__(0)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def build_style_reward_dataset(args, transform):
    datasets = {
        'train': ScorerDataset(args, transform=transform, split='train'),
        'val': ScorerDataset(args, transform=transform, split='val')
    }
    return datasets