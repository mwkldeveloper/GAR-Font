import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np


from model.generator.generator import Generator
from model.model_config import VQ_models, tokenizer_kwargs, style_args, gpt_models, gpt_kwargs, ffm_args, downsample_ratio, lora_config, img_size
from model.tokenizer.tokenizer import Tokenizer

from model.generator_adapter.generator_adapter import PromptFusionAligner


import time
from language.t5 import T5Embedder
from peft import get_peft_model

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
    generator, content_img,style_imgs, style_t5_feat,aligner_model,seq_len, device,
    temperature=1.0, top_k=0, top_p=1.0, sample=True
):
    B, n_ref, C_in, H, W = style_imgs.shape

    encoded_content = generator.content_encoder(content_img)
    style_imgs_flat = style_imgs.reshape(B * n_ref, C_in, H, W)
    style_feats = generator.style_encoder(style_imgs_flat)
    _, C_out, h, w = style_feats.shape
    style_feats = style_feats.view(B, n_ref, C_out, h, w)
    pseudo_style_feats = aligner_model(style_feats, style_t5_feat)
    predicted_fused = generator.ffm(encoded_content, pseudo_style_feats)
    aligned_fused_map = torch.cat([encoded_content, predicted_fused], dim=1)

    T_cond = gpt_kwargs['img_feature_code_len']
    T_total = T_cond + seq_len

    device = aligned_fused_map.device
    with torch.device(device):
        generator.gpt.setup_caches(max_batch_size=B, max_seq_length=T_total, dtype=generator.gpt.tok_embeddings.weight.dtype, device=device)
    
    seq  = torch.empty((B, seq_len), dtype=torch.int, device=device)
    all_logits = []

    input_pos = torch.arange(0, T_cond, device=device)
    logits, _ = generator.gpt(idx=None, imgs_feature_map=aligned_fused_map, input_pos=input_pos)
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

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    vq_model = Tokenizer(VQ_models[args.vq_model](**tokenizer_kwargs))
    vq_model.to(device)
    vq_model.eval()

    content_args = {
        'in_channels':3,
        'ch': vq_model.config.mid_ch,
        'ch_mult':vq_model.config.encoder_ch_mult,
        'num_res_blocks':2,
        'norm_type':'group',
        'dropout':vq_model.config.dropout_p,
        'z_channels':vq_model.config.z_channels
    }

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

    aligner_model = PromptFusionAligner(feat_dim=vq_model.config.z_channels,text_dim= 2048).to(device)
    aligner_model.eval()
    checkpoint = torch.load(args.adapter_ckpt, map_location="cpu",weights_only=False)
    aligner_model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.max_token_len,
    )


    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    batch_size = args.batch_size

    
    start_time = time.time()
    total_images = 0

    hw_ = (img_size // downsample_ratio, img_size // downsample_ratio)
    codebook = vq_model.quantizer.embedding.weight

    for i in range(0, total, batch_size):
        batch_lines = lines[i:i+batch_size]
        content_imgs = []
        style_imgs_batch =[]
        style_prompts = []
        tar_names = []
        
        for line in batch_lines:
            data = json.loads(line.strip())
            content_img_path = data["content_image"]
            
            content_img = load_image(content_img_path, img_size)
            content_imgs.append(content_img)

            style_img_paths = data["style_images"]
            assert len(style_img_paths) == args.vision_n_ref, f"Expected {args.vision_n_ref} style images, got {len(data['style_images'])} in {line}."

            style_img_list = [load_image(p, img_size) for p in style_img_paths]
            style_img_tensor = torch.cat(style_img_list, dim=0)
            style_imgs_batch.append(style_img_tensor)


            style_prompt = data["style_prompt"]
            style_prompts.append(style_prompt)


            style_name = os.path.basename(os.path.dirname(style_img_paths[0]))
            file_name = os.path.basename(content_img_path)
            tar_name = f"{style_name}+{file_name}"
            
            tar_names.append(tar_name)

        content_imgs = torch.cat(content_imgs, dim=0).to(device)
        style_imgs = torch.stack(style_imgs_batch, dim=0).to(device)

        t5_embeddings, t5_masks = t5_model.get_text_embeddings(style_prompts)
        t5_embeddings = t5_embeddings.to(torch.float32)
        t5_masks = t5_masks.to(torch.float32)
        
        pad_t5_masks = torch.flip(t5_masks, dims=[-1])
        pad_t5_embeddings = []
        for idx, (t5_embedding, pad_t5_mask) in enumerate(zip(t5_embeddings, pad_t5_masks)):
            valid_num = int(pad_t5_masks.sum().item())
            pad_t5_embedding = torch.cat([t5_embedding[valid_num:], t5_embedding[:valid_num]])
            pad_t5_embeddings.append(pad_t5_embedding)

        pad_t5_embeddings = torch.stack(pad_t5_embeddings)


        with torch.no_grad():
            seq_len = gpt_kwargs['target_token_len']
            vq_indices, generated_logits = generator_gpt_generate(
                generator, content_imgs, style_imgs, pad_t5_embeddings, aligner_model, seq_len, device,temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, sample=args.sample
            )
            

            rec_imgs_dict = {}
            if args.sample_mode in ["soft", "all"]:
                # ========== Soft decoding ==========
                probs = F.softmax(generated_logits, dim=-1)  # (B, seq_len, n_codes)
                soft_quantized_vectors = torch.matmul(probs, codebook)  # (B, seq_len, code_dim)
                rec_imgs_soft = vq_model.decode(soft_quantized_vectors, hw_)
                rec_imgs_dict["soft"] = rec_imgs_soft

            if args.sample_mode in ["hard", "all"]:
                # ========== Hard decoding ==========
                hard_indices = torch.argmax(generated_logits, dim=-1)  # (B, seq_len)
                hard_quantized_vectors = F.embedding(hard_indices, codebook)  # (B, seq_len, code_dim)
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
    total_time = end_time - start_time
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    print(f"\nTotal inference time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to sampling jsonl file.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--max-token-len", type=int, default=120)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--vq-model", type=str, default="VQ-8", choices=list(VQ_models.keys()))
    parser.add_argument("--generator-ckpt", type=str, required=True)
    parser.add_argument('--adapter-ckpt', type=str, required=True)
    parser.add_argument("--gpt-model", type=str, default="GPT-314M", choices=list(gpt_models.keys()))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Use multinomial sampling, otherwise greedy (argmax)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for sampling")
    parser.add_argument("--vision-n-ref", type=int, default=8, help="Number of style references")
    parser.add_argument("--sample-mode",type=str,default="soft",choices=["soft", "hard", "all"])
    args = parser.parse_args()
    main(args)