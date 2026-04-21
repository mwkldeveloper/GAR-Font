import argparse
from PIL import Image
import os
import json
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from peft import get_peft_model
from model.generator.generator import Generator
from model.model_config import VQ_models,tokenizer_kwargs, style_args, gpt_models, gpt_kwargs, ffm_args, lora_config, downsample_ratio, img_size
from model.tokenizer.tokenizer import Tokenizer



def load_style_refs(args):
    style_ref_list_path = os.path.join(args.style_ref_dir_path, "style_ref_list.json")

    if os.path.exists(style_ref_list_path):
        with open(style_ref_list_path, "r") as f:
            chosen_style_refs = json.load(f)["style_refs"]
        chosen_style_refs = [os.path.join(args.style_ref_dir_path, p) for p in chosen_style_refs]
        print(f"[INFO] style_ref_list.json exists,  {len(chosen_style_refs)} style refs in total")
    else:
        all_style_paths = [
            f for f in os.listdir(args.style_ref_dir_path)
            if f.lower().endswith(('.png'))
        ]
        assert len(all_style_paths) >= args.n_ref, f"{args.style_ref_dir_path} need {args.n_ref} references at least"

        chosen_style_refs = np.random.choice(all_style_paths, args.n_ref, replace=False).tolist()
        with open(style_ref_list_path, "w") as f:
            json.dump({"style_refs": chosen_style_refs}, f, indent=2, ensure_ascii=False)
        chosen_style_refs = [os.path.join(args.style_ref_dir_path, p) for p in chosen_style_refs]
        print(f"[INFO] randomly choose {args.n_ref} style_refs and saved info to {style_ref_list_path}")

    return chosen_style_refs

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)

def save_image(tensor, save_path):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    img.save(save_path)




class SampleFTDataset(Dataset):
    def __init__(self, args, chosen_style_refs,transform):
        super().__init__()
        self.samples = []
        self.transform = transform

        self.data_dir_path = args.style_ref_dir_path 
        self.content_ref_dir = args.content_ref_dir_path
        self.chosen_style_refs = chosen_style_refs  

        self.target_token_len = gpt_kwargs['target_token_len']

        self.condition_token_len = gpt_kwargs['img_feature_code_len']

        for fname in sorted(os.listdir(self.data_dir_path)):
            if not fname.endswith(".png"):
                continue
            tar_img_path = os.path.join(self.data_dir_path, fname)
            content_img_path = os.path.join(self.content_ref_dir, fname)

            sample_info = {
                "tar_img_path": tar_img_path,
                "content_img_path": content_img_path,
                "style_img_path_list":self.chosen_style_refs
            }
            self.samples.append(sample_info)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            tar_img = Image.open(sample["tar_img_path"]).convert("RGB")
            content_img = Image.open(sample["content_img_path"]).convert("RGB")

            if self.transform:
                tar_img = self.transform(tar_img)
                content_img = self.transform(content_img)

            style_imgs = [Image.open(p).convert("RGB") for p in sample["style_img_path_list"]]
            if self.transform:
                style_imgs = [self.transform(img) for img in style_imgs]

            stacked_style_imgs = torch.stack(style_imgs, dim=0) 

        except Exception as e:
            print(f"An error occurred while processing index {idx}, sample info: {sample}")
            print(f"Error details: {e}")
            exit(0)

        full_token_len = self.condition_token_len + self.target_token_len
        attn_mask = torch.tril(torch.ones(full_token_len, full_token_len)).to(torch.bool)

        return tar_img, content_img, stacked_style_imgs, attn_mask, torch.tensor(1)


def sample_finetune(generator, vq_model, args, chosen_style_refs):
    device = next(generator.parameters()).device
    has_lora = has_lora = any("lora_" in name for name, _ in generator.gpt.named_parameters())
    if not has_lora:
        generator.gpt = get_peft_model(generator.gpt, lora_config)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = SampleFTDataset(
        args,
        chosen_style_refs,
        transform,
    )

    loader = DataLoader(dataset, batch_size=args.ft_batch_size, shuffle=True, num_workers=4)

    total_layers = len(generator.gpt.base_model.model.layers)
    num_trainable_layers = args.ft_layer_num 

    print(f'UnFrozen {num_trainable_layers} GPT layers in total {total_layers} layers')
    

    for param in generator.parameters():
        param.requires_grad = False

    for name, param in generator.gpt.named_parameters():
        if "lora_" in name:
            layer_id = int(name.split("layers.")[1].split(".")[0])
            if layer_id < num_trainable_layers:
                param.requires_grad = True

    print(f'[Finetune] Number of trainable parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}')
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.ft_lr)

    generator.train()
    vq_model.eval()

    codebook = vq_model.quantizer.embedding.weight

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print(f"[Finetune] Start finetuning for {args.ft_epoch} epochs, {len(dataset)} samples...")
    hw_ = None
    for epoch in range(args.ft_epoch):
        for tar_img, content_img, style_imgs, attn_mask, valid in loader:
            tar_img = tar_img.to(device)
            content_img = content_img.to(device)
            style_imgs = style_imgs.to(device)
            attn_mask = attn_mask.to(device)
            valid = valid.to(device)

            B = tar_img.shape[0]
            with torch.no_grad():
                _, _, [_, _, tar_vq_indices], hw = vq_model.encode(tar_img)
                hw_ = hw
            vq_indices = tar_vq_indices.view(B, -1)
            attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1])


            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, logits_loss = generator(
                    content_img=content_img,
                    style_imgs=style_imgs,
                    vq_indices=vq_indices,
                    gpt_valid=valid,
                    gpt_attn_mask=attn_mask
                )

                probs = F.softmax(logits, dim=-1)
                soft_quantized_vectors = torch.matmul(probs, codebook)
                soft_reconstructed_img = vq_model.decode(soft_quantized_vectors,hw_)
                rec_loss = F.l1_loss(soft_reconstructed_img, tar_img)

                total_loss = logits_loss + rec_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        print(f"[Finetune] Epoch {epoch+1}/{args.ft_epoch}, Loss={total_loss.item():.4f}, Logits Loss={logits_loss.item():.4f}, Rec Loss={rec_loss.item():.4f}")

    generator.eval()
    print("[Finetune] Done, return finetuned generator")
    return generator



def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    # logits: (B, vocab)
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def gpt_sample_one_step(logits, temperature=1.0, top_k=0, top_p=1.0, sample=True):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx,probs

def generator_gpt_generate(
    generator, content_img, style_imgs, seq_len, device,
    temperature=1.0, top_k=0, top_p=1.0, sample=True
):
    # content_img: (1, 3, H, W)
    # style_imgs: (1, n_ref, 3, H, W)
    # seq_len: token sequence length
    B, n_ref, C_in, H, W = style_imgs.shape
    encoded_content = generator.content_encoder(content_img)

    style_imgs_flat = style_imgs.reshape(B * n_ref, C_in, H, W)
    style_feats = generator.style_encoder(style_imgs_flat)

    _, C_out, h, w = style_feats.shape
    style_feats = style_feats.view(B, n_ref, C_out, h, w)
    fused_feat = generator.ffm(encoded_content, style_feats)
    cat_fused = torch.cat([encoded_content, fused_feat], dim=1)

    T_cond = h*w
    T_total = T_cond + seq_len
    
    device = cat_fused.device
    with torch.device(device):
        generator.gpt.setup_caches(max_batch_size=B, max_seq_length=T_total, dtype=generator.gpt.tok_embeddings.weight.dtype,device = device)
    
    seq  = torch.empty((B, seq_len), dtype=torch.int, device=device)
    all_logits = []
    
    input_pos = torch.arange(0, T_cond, device=device)
    logits, _ = generator.gpt(idx=None, imgs_feature_map=cat_fused, input_pos=input_pos)
    all_logits.append(logits[:, -1:, :]) 
    
    next_token, _ = gpt_sample_one_step(logits,temperature,top_k,top_p,sample)
    seq[:, 0:1] = next_token
    
    

    for i in range(1,seq_len):
        cur_input_pos = torch.tensor([T_cond + i - 1], device=device, dtype=torch.long) 
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            logits, _ = generator.gpt(idx=seq[:, i-1:i], imgs_feature_map=None, input_pos=cur_input_pos)
            all_logits.append(logits[:, -1:, :])
            
            next_token, _ = gpt_sample_one_step(logits,temperature,top_k,top_p,sample)
            
            seq[:, i:i+1] = next_token

    final_logits = torch.cat(all_logits, dim=1)
    return seq, final_logits

def get_vq_content_args(vq_model):
    content_args = {
        'in_channels':3,
        'ch': vq_model.config.mid_ch,
        'ch_mult':vq_model.config.encoder_ch_mult,
        'num_res_blocks':2,
        'norm_type':'group',
        'dropout':vq_model.config.dropout_p,
        'z_channels':vq_model.config.z_channels
    }
    return content_args

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    vq_model = Tokenizer(VQ_models[args.vq_model](**tokenizer_kwargs)).to(device)
    vq_model.eval()

    content_args = get_vq_content_args(vq_model)
    generator = Generator(content_args,style_args,gpt_models[args.gpt_model](**gpt_kwargs),ffm_args).to(device)


    checkpoint = torch.load(args.generator_ckpt, map_location="cpu",weights_only=False)
    
    
    has_lora = any("lora_" in k for k in checkpoint["model"].keys())
    if has_lora:
        generator.gpt = get_peft_model(generator.gpt, lora_config)
    generator.eval()

    vq_model.load_state_dict(checkpoint["vq_model"])
    generator.load_state_dict(checkpoint["model"])

    del checkpoint
    print(f"Loaded Model checkpoint: {args.generator_ckpt}")
    print(f"Loaded VQ Model & Generator Model")
    print(f'args : {args}')
    all_char_list = args.all_char_list
    generate_char_list = args.generate_char_list

    chosen_style_refs = load_style_refs(args)

    if args.ft_epoch > 0:
        generator = sample_finetune(generator,vq_model,args,chosen_style_refs)


    batch_size = args.batch_size

    vq_codebook = vq_model.quantizer.embedding.weight
    
    start_time = time.time()
    total_images = 0
    
    hw_ = (img_size // downsample_ratio, img_size // downsample_ratio)
    total_chars = len(generate_char_list)

    for i in range(0, total_chars, batch_size):
        batch_chars = generate_char_list[i:i+batch_size]
        content_imgs, tar_names = [], []
        style_imgs = torch.cat(
            [load_image(p, img_size) for p in chosen_style_refs], dim=0
        ).unsqueeze(0).repeat(len(batch_chars), 1, 1, 1, 1).to(device)  # (B, n_ref, 3, H, W)

        for ch in batch_chars:
            idx = all_char_list.index(ch)
            content_img_path = os.path.join(args.content_ref_dir_path, f"{idx:04d}.png")
            content_img = load_image(content_img_path, img_size)
            content_imgs.append(content_img)

            style_name = os.path.basename(args.style_ref_dir_path)
            tar_names.append(f'{style_name}+{idx:04d}.png')

        content_imgs = torch.cat(content_imgs, dim=0).to(device)  # (B, 3, H, W)
        
        with torch.no_grad():
            seq_len = gpt_kwargs['target_token_len']
            vq_indices, generated_logits = generator_gpt_generate(
                generator, content_imgs, style_imgs, seq_len, device,temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, sample=args.sample
            )
           
            rec_imgs_dict = {}
            if args.sample_mode in ["soft", "all"]:
                # ========== Soft decoding ==========
                probs = F.softmax(generated_logits, dim=-1)  # (B, seq_len, n_codes)
                soft_quantized_vectors = torch.matmul(probs, vq_codebook)  # (B, seq_len, code_dim)
                rec_imgs_soft = vq_model.decode(soft_quantized_vectors, hw_)
                rec_imgs_dict["soft"] = rec_imgs_soft

            if args.sample_mode in ["hard", "all"]:
                # ========== Hard decoding ==========
                hard_indices = torch.argmax(generated_logits, dim=-1)  # (B, seq_len)
                hard_quantized_vectors = F.embedding(hard_indices, vq_codebook)  # (B, seq_len, code_dim)
                rec_imgs_hard = vq_model.decode(hard_quantized_vectors, hw_)
                rec_imgs_dict["hard"] = rec_imgs_hard

        for mode, rec_imgs in rec_imgs_dict.items():
            total_images += rec_imgs.shape[0]
            output_dir_mode = f"{args.output_dir}"
            os.makedirs(output_dir_mode, exist_ok=True)

            for j in range(rec_imgs.shape[0]):
                output_path = os.path.join(output_dir_mode, f"gen_{mode}_{tar_names[j]}")
                save_image(rec_imgs[j].unsqueeze(0), output_path)
                print(f"Saved: {output_path}")


    end_time = time.time()
    print(f"\nTotal inference time: {end_time-start_time:.2f}s, Average per image: {(end_time-start_time)/total_images:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vq-model", type=str, default="VQ-8", choices=list(VQ_models.keys()))
    parser.add_argument("--generator-ckpt", type=str, required=True)
    parser.add_argument("--gpt-model", type=str, default="GPT-314M", choices=list(gpt_models.keys()))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Use multinomial sampling, otherwise greedy (argmax)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for sampling")
    parser.add_argument("--all-char-list", type=str, required=True)
    parser.add_argument("--generate-char-list", type=str, required=True)
    parser.add_argument("--content-ref-dir-path", type=str, required=True)
    parser.add_argument("--style-ref-dir-path", type=str, required=True)
    parser.add_argument("--n-ref", type=int, default=8)
    parser.add_argument("--ft-epoch", type=int, default=0)
    parser.add_argument("--ft-layer-num", type=int, default=1, help="Number of Finetuned GPT Layers before sampling")
    parser.add_argument("--ft-batch-size", type=int, default=32)
    parser.add_argument("--ft-lr", type=float, default=2e-5)
    parser.add_argument("--sample-mode",type=str,default="soft",choices=["soft", "hard", "all"])
    args = parser.parse_args()
    main(args)