import os
import time
import math
import argparse
import inspect
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from utils.logger import create_logger
from dataset.build_dataset import build_dataset


from model.generator.generator import Generator
from model.model_config import VQ_models,tokenizer_kwargs, style_args, gpt_models, gpt_kwargs, ffm_args, lora_config,downsample_ratio,img_size
from model.tokenizer.tokenizer import Tokenizer

import torch.nn.functional as F

from peft import get_peft_model

import easyocr
from reward.style_reward_model import StyleRewardModel

def _gpt_sample_one_step(logits, temperature=1.0, sample=False):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    probs = F.softmax(logits, dim=-1)
    if sample:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx

@torch.no_grad()
def gpt_autoregressive_generate(generator, content_img, style_imgs, seq_len, device,
                                temperature=1.0, sample=True):
    generator.eval()
    B, n_ref, C_in, H, W = style_imgs.shape
    encoded_content = generator.content_encoder(content_img)

    style_imgs_flat = style_imgs.reshape(B * n_ref, C_in, H, W)
    style_feats = generator.style_encoder(style_imgs_flat)

    _, C_out, h, w = style_feats.shape
    style_feats = style_feats.view(B, n_ref, C_out, h, w)
    fused_feat = generator.ffm(encoded_content, style_feats)
    cat_fused = torch.cat([encoded_content, fused_feat], dim=1)

    T_cond = h * w
    T_total = T_cond + seq_len

    generator.gpt.setup_caches(max_batch_size=B, max_seq_length=T_total,
                               dtype=generator.gpt.tok_embeddings.weight.dtype, device=device)
    all_logits = []

    input_pos = torch.arange(0, T_cond, device=device)

    logits, _ = generator.gpt(idx=None, imgs_feature_map=cat_fused,
                              input_pos=input_pos, targets=None, mask=None,
                              valid=None)
    all_logits.append(logits[:, -1:, :])

    next_token = _gpt_sample_one_step(logits, temperature=temperature, sample=sample)
    generated_seq = [next_token]

    for i in range(1, seq_len):
        cur_input_pos = torch.tensor([T_cond + i - 1], device=device, dtype=torch.long)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            logits, _ = generator.gpt(idx=generated_seq[-1], imgs_feature_map=None,
                                      input_pos=cur_input_pos, targets=None, mask=None,
                                      valid=None)
            all_logits.append(logits[:, -1:, :])
            next_token = _gpt_sample_one_step(logits, temperature=temperature, sample=sample)
            generated_seq.append(next_token)

    final_logits = torch.cat(all_logits, dim=1)
    indices = torch.cat(generated_seq, dim=1)
    return final_logits, indices

def get_logits_parallel(generator, content_ref, style_refs, indices_seq,full_attn_mask):
    B, n_ref, C_in, H, W = style_refs.shape
    encoded_content = generator.content_encoder(content_ref)

    style_imgs_flat = style_refs.reshape(B * n_ref, C_in, H, W)
    style_feats = generator.style_encoder(style_imgs_flat)

    _, C_out, h, w = style_feats.shape
    style_feats = style_feats.view(B, n_ref, C_out, h, w)
    fused_feat = generator.ffm(encoded_content, style_feats)
    cat_fused = torch.cat([encoded_content, fused_feat], dim=1)

    logits, _ = generator.gpt(
        idx=indices_seq[:, :-1],
        imgs_feature_map=cat_fused,
        targets=None, 
        mask=full_attn_mask[:,:, :-1, :-1],
        valid=None,
        eval_mode_sample=True
    )

    return logits


def gather_log_probs(logits, indices):
    logp = F.log_softmax(logits, dim=-1)  # [B,T,V]
    gather = torch.gather(logp, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)
    return gather  # [B,T]

def kl_logprobs(logits_pi, logits_ref):
    p = F.log_softmax(logits_pi, dim=-1)
    q = F.log_softmax(logits_ref, dim=-1)
    p_prob = p.exp()
    kl = (p_prob * (p - q)).sum(dim=-1)  # [B,T]
    return kl

def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed tensor: {len(decay_params)}, params: {num_decay_params:,}")
    logger.info(f"num non-decayed tensor: {len(nodecay_params)}, params: {num_nodecay_params:,}")

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = "cuda:0"
    torch.cuda.set_device(device)

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    torch.manual_seed(args.global_seed)


    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.gpt_model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-SE-{model_string_name}"
    ckpt_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f'args : {args}')

    generator_ckpt = torch.load(args.generator_ckpt, map_location="cpu",weights_only=False)

    vq_model = Tokenizer(VQ_models[args.vq_model](**tokenizer_kwargs)).to(device)
    vq_model.eval()
    vq_model.load_state_dict(generator_ckpt["vq_model"])

    logger.info("VQ Tokenizer: eval()")


    content_args = {
        'in_channels': 3,
        'ch': vq_model.config.mid_ch,
        'ch_mult': vq_model.config.encoder_ch_mult,
        'num_res_blocks': 2,
        'norm_type': 'group',
        'dropout': vq_model.config.dropout_p,
        'z_channels': vq_model.config.z_channels
    }
    generator = Generator(content_args,style_args,gpt_models[args.gpt_model](**gpt_kwargs),ffm_args).to(device)
    has_lora = any("lora_" in k for k in generator_ckpt["model"].keys())
    if has_lora:
        generator.gpt = get_peft_model(generator.gpt, lora_config)
        
    generator.load_state_dict(generator_ckpt["model"], strict=True)
    
    del generator_ckpt  
    for name, param in generator.named_parameters():
        param.requires_grad = ("lora_" in name)
    
    logger.info(f"Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    

    generator_ref = Generator(content_args,style_args,gpt_models[args.gpt_model](**gpt_kwargs),ffm_args).to(device)
    generator_ref.gpt = get_peft_model(generator_ref.gpt, lora_config)
    generator_ref.load_state_dict(generator.state_dict())
    for p in generator_ref.parameters(): p.requires_grad = False

    generator_ref.eval()


    logger.info(f"Trainable Params: {sum(p.numel() for p in generator.parameters() if p.requires_grad):,}")

    optimizer = create_optimizer(generator, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    if args.se_generator_ckpt:
        ckpt = torch.load(args.se_generator_ckpt, map_location="cpu", weights_only=False)
        generator.load_state_dict(ckpt["model"], strict=True)
        generator_ref.load_state_dict(ckpt["model"], strict=True)
        vq_model.load_state_dict(ckpt["vq_model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        del ckpt
        logger.info(f"Resumed SE checkpoint: {args.se_generator_ckpt}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], inplace=True)
    ])
    args.dataset_target_token_len = gpt_kwargs['target_token_len']
    args.dataset_condition_token_len = gpt_kwargs['img_feature_code_len']
    datasets = build_dataset(args, transform=transform)

    loader = DataLoader(
        datasets['train'],
        batch_size=int(args.global_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Train set: {len(datasets['train']):,}")

    codebook = vq_model.quantizer.embedding.weight  

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    start_time = time.time()
    global_step = 0

    log_total_loss = 0.0
    log_avg_reward = 0.0
    log_style_score = 0.0
    log_rec_score = 0.0
    log_ocr_score = 0.0
    log_kl_term = 0.0
    


    if args.ocr_reward_weight > 0:
        logger.info("Initializing EasyOCR Reader for Chinese...")
        ocr_reader = easyocr.Reader(['ch_sim'], gpu=device) 
        logger.info(f"EasyOCR Reader initialized on {device}")
    else:
        ocr_reader = None

    if args.style_reward_weight > 0:
        logger.info(f"Loading Style Consistency Scorer from: {args.style_reward_model_ckpt}")
        ckpt = torch.load(args.style_reward_model_ckpt, map_location=device, weights_only=False)
        style_scorer = StyleRewardModel().to(device)
        style_scorer.load_state_dict(ckpt['model'], strict=True)
        del ckpt
        style_scorer.eval()
        logger.info("Style Scorer loaded and set to eval mode.")
    else:
        style_scorer = None

    generator.train()
    hw_ = (img_size//downsample_ratio, img_size//downsample_ratio)
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        

        for tar_img, content_ref, style_refs, content_name, full_attn_mask in loader:
            tar_img = tar_img.to(device, non_blocking=True)
            content_ref = content_ref.to(device, non_blocking=True)
            style_refs = style_refs.to(device, non_blocking=True)
            full_attn_mask = full_attn_mask.to(device)
            full_attn_mask = full_attn_mask.unsqueeze(1)

            B = tar_img.size(0)
            seq_len = gpt_kwargs['target_token_len']

            generator.eval()
            group_logits = []
            group_indices = []
            with torch.no_grad():
                for _ in range(args.num_samples_per_group):  # K
                    logits_k, idx_k = gpt_autoregressive_generate(
                        generator, content_ref, style_refs,
                        seq_len=seq_len, device=device,
                        temperature=args.temperature, sample=True
                    )
                    group_logits.append(logits_k)   # [B, T, V]
                    group_indices.append(idx_k)     # [B, T]

            rewards_group = []  # list of [B]
            gen_images_group = [] 

            style_scores_acc = torch.tensor(0.0, device=device)
            rec_scores_acc = torch.tensor(0.0, device=device)
            ocr_scores_acc = torch.tensor(0.0, device=device)

            for k in range(args.num_samples_per_group):
                indices = group_indices[k]  # [B, T]
                emb_map = F.embedding(indices, codebook)  # [B,T,dim]
                with torch.no_grad():
                    gen_img = vq_model.decode(emb_map,hw_)
                gen_images_group.append(gen_img)

                reward = torch.zeros(B, device=device)

                # Style_Reward -> Style Fidelity
                if args.style_reward_weight > 0:
                    with torch.no_grad():
                        ref_img = style_refs[:, 0]
                        style_consistency_reward = style_scorer(gen_img, ref_img)
                        
                        reward += style_consistency_reward * args.style_reward_weight
                        style_scores_acc += style_consistency_reward.mean()

                # OCR_Reward -> Structure Correctness
                if args.ocr_reward_weight > 0:
                    ocr_rewards_batch = []
                    inv_tensor = gen_img.detach().clone()
                    inv_tensor = inv_tensor * 0.5 + 0.5 
                    np_images_batch = (inv_tensor * 255.0).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

                    for b in range(B):
                        img_to_ocr = np_images_batch[b]
                        ground_truth_text = content_name[b]
                        
                        result = ocr_reader.readtext(img_to_ocr, detail=1, paragraph=False)
                        
                        if result:
                            recognized_text = result[0][1]
                            confidence_score = result[0][2]
                            
                            if recognized_text == ground_truth_text:
                                ocr_rewards_batch.append(confidence_score)
                            else:
                                ocr_rewards_batch.append(0.0)
                        else:
                            ocr_rewards_batch.append(0.0)

                    ocr_reward_tensor = torch.tensor(ocr_rewards_batch, device=device, dtype=torch.float32)
                    reward += ocr_reward_tensor * args.ocr_reward_weight
                    ocr_scores_acc += ocr_reward_tensor.mean()

                if args.rec_reward_weight > 0:
                    with torch.no_grad():
                        rec_l1 = F.l1_loss(gen_img, tar_img, reduction='none')
                        rec_l1 = rec_l1.view(B, -1).mean(dim=1)  # [B]
                    reward = reward + args.rec_reward_weight * (-rec_l1)
                    rec_scores_acc += (-rec_l1).mean()

                

                rewards_group.append(reward) 

            rewards_stack = torch.stack(rewards_group, dim=0)
            rewards_bk = rewards_stack.permute(1, 0).contiguous()
            mean_b = rewards_bk.mean(dim=1, keepdim=True)  # [B,1]
            std_b = rewards_bk.std(dim=1, keepdim=True) + 1e-8
            adv_bk = (rewards_bk - mean_b) / std_b  # [B, K]
            advantages_group = adv_bk.permute(1, 0).contiguous() # [K,B]

            generator.gpt.clear_caches()
            generator_ref.gpt.clear_caches()
            
            ref_logits_group = []
            with torch.no_grad():
                generator_ref.eval()
                for k in range(args.num_samples_per_group):
                    logits_k = get_logits_parallel(
                        generator_ref, content_ref, style_refs, group_indices[k], full_attn_mask
                    )
                    ref_logits_group.append(logits_k)

            generator.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=ptdtype):
                pi_logits_group = []
                for k in range(args.num_samples_per_group):
                    logits_k = get_logits_parallel(
                        generator, content_ref, style_refs, group_indices[k], full_attn_mask
                    )
                    pi_logits_group.append(logits_k)

                total_loss, policy_loss_acc, kl_loss_acc, entropy_loss_acc = [torch.tensor(0.0, device=device) for _ in range(4)]
                
                for k in range(args.num_samples_per_group):
                    logits_pi = pi_logits_group[k]      # Shape: [B, T, V]
                    logits_ref = ref_logits_group[k].detach() # Shape: [B, T, V]
                    indices_k = group_indices[k]          # Shape: [B, T]
                    adv_k = advantages_group[k].unsqueeze(1)  # Shape: [B, 1]

                    indices_for_loss = indices_k # Shape: [B, T]

                    logp_pi = gather_log_probs(logits_pi, indices_for_loss)

                    if args.kl_approx_method == 'logp_diff':
                        logp_ref = gather_log_probs(logits_ref, indices_for_loss)
                        kl_term = args.kl_coef * (logp_pi - logp_ref).mean()
                    elif args.kl_approx_method == 'full_kl':
                        kl_term = args.kl_coef * kl_logprobs(logits_pi, logits_ref).mean()
                    else:
                        raise ValueError(f"Unknown kl_approx_method: {args.kl_approx_method}")
                    
                    pg = -(adv_k * logp_pi).mean()

        
                    ent_term = torch.tensor(0.0, device=device)
                    if args.entropy_coef > 0:
                        entropy = -(torch.softmax(logits_pi, dim=-1) * torch.log_softmax(logits_pi, dim=-1)).sum(-1).mean()
                        ent_term = -args.entropy_coef * entropy

                    loss_k = pg + kl_term + ent_term
                    total_loss += loss_k
                    policy_loss_acc += pg.detach()
                    kl_loss_acc += kl_term.detach()
                    entropy_loss_acc += ent_term.detach()
                
                total_loss /= args.num_samples_per_group

            scaler.scale(total_loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            log_total_loss += total_loss.item()
            log_avg_reward += rewards_stack.mean().item()
            log_style_score += (style_scores_acc / args.num_samples_per_group).item()
            log_ocr_score += (ocr_scores_acc / args.num_samples_per_group).item()
            log_rec_score += (rec_scores_acc / args.num_samples_per_group).item()
            log_kl_term += (kl_loss_acc / args.num_samples_per_group).item()

            if global_step % args.log_every == 0:
                steps_per_sec = args.log_every / (time.time() - start_time)
                
                avg_loss = log_total_loss / args.log_every
                avg_reward = log_avg_reward / args.log_every
                avg_style_score = log_style_score / args.log_every
                avg_ocr_score = log_ocr_score / args.log_every
                avg_rec_score = log_rec_score / args.log_every
                avg_kl = log_kl_term / args.log_every

                logger.info(
                    f"(step={global_step:07d}) Loss: {avg_loss:.4f} | "
                    f"Reward: {avg_reward:.4f} | Style: {avg_style_score:.4f} | "
                    f"Rec: {avg_rec_score:.4f} | "
                    f"OCR: {avg_ocr_score:.4f} | KL: {avg_kl:.4f} | "
                    f"steps/s: {steps_per_sec:.2f}"
                )
                
                log_total_loss = 0.0
                log_avg_reward = 0.0
                log_style_score = 0.0
                log_ocr_score = 0.0
                log_rec_score = 0.0
                log_kl_term = 0.0

                start_time = time.time()

            if global_step % args.ckpt_every == 0:
                generator.gpt.clear_caches()
                ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                torch.save({
                    "model": generator.state_dict(),
                    "vq_model": vq_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": global_step,
                    "train_args": args,
                    "tokenizer_kwargs": tokenizer_kwargs,
                    "content_encoder_args": content_args,
                    "style_encoder_args": style_args,
                    "feature_fusion_module_args": ffm_args,
                    "gpt_kwargs": gpt_kwargs,
                    "lora_config": lora_config,

                }, ckpt_path)
                logger.info(f"Saved ckpt to {ckpt_path}")

            if args.sample_every > 0 and global_step % args.sample_every == 0:
                vis_dir = os.path.join(experiment_dir, "samples")
                os.makedirs(vis_dir, exist_ok=True)

                all_grid_tensors = []
                for b in range(B):
                    rewards_for_sample = rewards_stack[:, b]
                    images_for_sample = torch.stack([gen_img[b] for gen_img in gen_images_group])

                    _, sorted_indices = torch.sort(rewards_for_sample, descending=True)
                    sorted_images = images_for_sample[sorted_indices]

                    style_ref_img = style_refs[b, 0].unsqueeze(0)
                    target_img = tar_img[b].unsqueeze(0)
                    placeholder = torch.ones_like(target_img) * -1

                    all_grid_tensors.append(style_ref_img)
                    all_grid_tensors.append(content_ref[b].unsqueeze(0))
                    all_grid_tensors.append(target_img)
                    all_grid_tensors.append(placeholder)
                    all_grid_tensors.extend(list(torch.chunk(sorted_images, chunks=args.num_samples_per_group, dim=0)))

                final_grid = torch.cat(all_grid_tensors, dim=0)
                
                save_path = os.path.join(vis_dir, f"step{global_step:07d}_grid.png")
                
                num_cols = args.num_samples_per_group + 4
                save_image(final_grid, save_path, nrow=num_cols, normalize=True, value_range=(-1, 1))
                rewards_for_sample_0 = rewards_stack[:, 0]
                sorted_rewards_0, _ = torch.sort(rewards_for_sample_0, descending=True)
                reward_str_0 = ", ".join([f"{r.item():.3f}" for r in sorted_rewards_0])
                logger.info(f"Saved consolidated sample grid to {save_path}")
                logger.info(f" -> e.g., Sample 0 Rewards (desc): [{reward_str_0}]")
                
        if args.update_ref_every_epoch > 0 and (epoch + 1) % args.update_ref_every_epoch == 0:
            logger.info(f"Updating Ref_Model after Epoch {epoch}")
            torch.cuda.synchronize() 
            generator_ref.gpt.clear_caches()
            generator.gpt.clear_caches()
            generator_ref.load_state_dict(generator.state_dict())
            generator_ref.eval() 
            
            logger.info(f"Done Ref_Model Updating.")
    logger.info("Done GRPO!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-path", type=str, required=True)
    parser.add_argument("--data-style-info-json", type=str, required=True)
    parser.add_argument("--data-content-info-json", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='generator_se')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--generator-ckpt", type=str, default=None)
    parser.add_argument("--se-generator-ckpt", type=str, default=None)
    parser.add_argument("--gpt-model", type=str, choices=list(gpt_models.keys()), default="GPT-314M")
    parser.add_argument("--n-ref", type=int, default=8)
    parser.add_argument("--se-fonts-num", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--style-reward-model-ckpt", type=str, default=None)
    parser.add_argument("--num-samples-per-group", type=int, default=4) 
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--kl-approx-method", type=str, default='logp_diff', choices=['logp_diff', 'full_kl'],help="KL divergence approximation method. 'logp_diff' uses the difference of log-probabilities on sampled tokens. 'full_kl' computes the full KL divergence over the distributions.")
    parser.add_argument("--update-ref-every-epoch", type=int, default=2)
    parser.add_argument("--rec-reward-weight", type=float, default=0.8, help="Weight for the reconstruction reward.")
    parser.add_argument("--ocr-reward-weight", type=float, default=0.0, help="Weight for the OCR-based reward.")
    parser.add_argument("--style-reward-weight", type=float, default=0.2, help="Weight for the style consistency reward.")
    parser.add_argument("--results-dir", type=str, default="./results/results_se")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=500)
    args = parser.parse_args()
    main(args)
