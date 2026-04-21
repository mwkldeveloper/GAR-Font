import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import json

from language.t5 import T5Embedder
from tqdm import tqdm
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, json_file_path):
        """
        Expecting a JSON file like:
        {
          "FontA": "A font style that xxxx",
          "FontB": "A font style that xxxx",
          ...
        }
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        # Convert dict to list of (label, caption)
        self.label_caption_pair = [(k, v) for k, v in data_dict.items()]

    def __len__(self):
        return len(self.label_caption_pair)

    def __getitem__(self, index):
        label, caption = self.label_caption_pair[index]
        return label, caption


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):

    device = "cuda:0"
    torch.cuda.set_device(device)
    seed = args.global_seed 
    torch.manual_seed(seed)

    print(f"Starting seed: {seed}, on device: {device}.")

    # Setup data:
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.style_prompt_json_file_path)

    loader = DataLoader(
        dataset,
        batch_size=1,  # important!
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} prompts")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    
    assert os.path.exists(args.t5_model_path)
    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.max_token_len,
    )

    os.makedirs(args.t5_feat_save_path, exist_ok=True)
    
    

    for style_name, caption in tqdm(loader, desc="Processing styles"):
        caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
        valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        x = valid_caption_embs.to(torch.float32).detach().cpu().numpy()

        if x.shape[1] >= args.max_token_len:
            print(f"Warning: Style '{style_name[0]}' embedding length {x.shape[1]} may exceeds max_token_len {args.max_token_len}")

        np.save(os.path.join(args.t5_feat_save_path, f'{style_name[0]}.npy'), x)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-prompt-json-file-path", type=str, required=True)
    parser.add_argument("--t5-feat-save-path", type=str, required=True)
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-token-len", type=int, default=120)
    args = parser.parse_args()
    main(args)
