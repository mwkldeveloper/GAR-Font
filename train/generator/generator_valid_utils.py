import os
import torch
from torchvision.utils import save_image
from model.model_config import gpt_kwargs
import torch.nn.functional as F
import numpy as np

def _gpt_sample_one_step(logits, temperature=1.0, sample=False):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    probs = F.softmax(logits, dim=-1)
    
    if sample:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
        
    return idx

def _generator_gpt_generate(generator, content_img, style_imgs, seq_len, device, temperature=1.0, sample=False):
    B, n_ref, C_in, H, W = style_imgs.shape
    encoded_content = generator.content_encoder(content_img)

    
    style_imgs_flat = style_imgs.view(B * n_ref,C_in,H,W)
    style_feats = generator.style_encoder(style_imgs_flat)
    
    _, C_out, h, w = style_feats.shape
    style_feats = style_feats.view(B, n_ref, C_out, h, w)
    
    fused_feat = generator.ffm(encoded_content, style_feats)
    cat_fused = torch.cat([encoded_content, fused_feat], dim=1)

    T_cond = h * w
    T_total = T_cond + seq_len

    generator.gpt.setup_caches(max_batch_size=B, max_seq_length=T_total, dtype=generator.gpt.tok_embeddings.weight.dtype, device=device)
    
    all_logits = []
    
    input_pos = torch.arange(0, T_cond, device=device)
    logits, _ = generator.gpt(idx=None, imgs_feature_map=cat_fused, input_pos=input_pos)
    all_logits.append(logits[:, -1:, :])
    
    next_token = _gpt_sample_one_step(logits, temperature=temperature, sample=sample)
    
    generated_seq = [next_token]
    for i in range(1, seq_len):
        cur_input_pos = torch.tensor([T_cond + i - 1], device=device, dtype=torch.long)
        
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            logits, _ = generator.gpt(idx=generated_seq[-1], imgs_feature_map=None, input_pos=cur_input_pos)
            all_logits.append(logits[:, -1:, :])
            
            next_token = _gpt_sample_one_step(logits, temperature=temperature, sample=sample)
            generated_seq.append(next_token)

    final_logits = torch.cat(all_logits, dim=1) # (B, seq_len, vocab_size)
    return final_logits

def validate_and_save_images(generator, vq_model, val_loader, device, train_steps, experiment_dir, hw, logger, filename_prefix):
    generator.eval()
    
    val_dir = os.path.join(experiment_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)
    codebook = vq_model.quantizer.embedding.weight

    
    with torch.no_grad():
        try:
            tar_img, content_ref, style_refs, _, _ = next(iter(val_loader))
        except StopIteration:
            logger.warning(f"Validation loader for prefix '{filename_prefix}' is empty. Skipping.")
            return
        
        tar_img = tar_img.to(device)
        content_ref = content_ref.to(device)
        style_refs = style_refs.to(device)
        
        seq_len = gpt_kwargs['target_token_len']
        generated_logits = _generator_gpt_generate(
            generator,
            content_img=content_ref,
            style_imgs=style_refs,
            seq_len=seq_len,
            device=device,
            temperature=1.0,
            sample=False
        )
        
        probs = F.softmax(generated_logits, dim=-1)
        soft_quantized_vectors = torch.matmul(probs, codebook)
        
        
        gen_img = vq_model.decode(soft_quantized_vectors,hw)
        
        rmse_values = []
        B = tar_img.shape[0]
        for i in range(B):
            gen_np = gen_img[i].permute(1, 2, 0).cpu().numpy()
            tar_np = tar_img[i].permute(1, 2, 0).cpu().numpy()
            rmse = np.sqrt(np.mean((gen_np - tar_np) ** 2))
            rmse_values.append(rmse)
        avg_rmse = np.mean(rmse_values)
        logger.info(f"Validation on {filename_prefix} - RMSE loss at step {train_steps}: {avg_rmse:.6f}")
        
        ref_img_for_display = style_refs[:, 0, ...]
        comparison_grid = torch.cat([ref_img_for_display, tar_img, gen_img])
        
        save_path = os.path.join(val_dir, f"generator_step{train_steps:07d}_{filename_prefix}_val.png")
        save_image(comparison_grid, save_path, nrow=tar_img.size(0), normalize=True)
        