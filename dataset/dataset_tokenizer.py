import os
import json
from torch.utils.data import Dataset
from PIL import Image

class TokenizerDataset(Dataset):
    def __init__(self, args, transform,split_type='train'):
        self.samples = []
        self.transform = transform
        with open(args.data_style_info_json, "r", encoding="utf-8") as f:
            data_style_info = json.load(f)
        with open(args.data_content_info_json, "r", encoding="utf-8") as f:
            data_content_info = json.load(f)
        
        if split_type == 'train':
            dataset_styles = data_style_info.get('train_style', [])
            dataset_contents = data_content_info.get('train_content', [])
        elif split_type == 'val':
            dataset_styles = data_style_info.get('test_style', [])
            dataset_contents = data_content_info.get('test_content', [])
        
        if not dataset_styles or not dataset_contents:
            print(f"Warning: The '{split_type}' split is empty. Check your JSON files.")
        
        for content_item in dataset_contents:
            char_index = content_item['index']
            for style_name in dataset_styles:
                image_path = f'{args.data_dir_path}/{style_name}/{char_index:04d}.png'
                self.samples.append(image_path)
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image



def build_tokenizer_datasets(args, transform):
    train_dataset = TokenizerDataset(args, transform=transform, split_type='train')
    val_dataset = TokenizerDataset(args, transform=transform, split_type='val')
    return train_dataset, val_dataset