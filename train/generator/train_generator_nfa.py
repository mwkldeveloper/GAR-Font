from glob import glob
import time
import argparse
import os

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from utils.logger import create_logger
from dataset.build_dataset import build_dataset


from model.generator.generator import Generator
from model.model_config import VQ_models,tokenizer_kwargs, style_args, gpt_models, gpt_kwargs, ffm_args, lora_config
from model.tokenizer.tokenizer import Tokenizer

import torch.nn.functional as F
from train.generator.generator_valid_utils import validate_and_save_images


from peft import get_peft_model


import inspect
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
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
    
    seed = args.global_seed
    torch.manual_seed(seed)
    
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.gpt_model.replace("/", "-") 
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-NFA-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f'args : {args}')
    
    generator_ckpt = torch.load(args.generator_ckpt, map_location="cpu", weights_only=False) 

    vq_model = Tokenizer(VQ_models[args.vq_model](**tokenizer_kwargs))
    vq_model.to(device)
    vq_model.eval()
    
    vq_model.load_state_dict(generator_ckpt["vq_model"])
    logger.info("VQ Tokenizer is in eval mode")
    
    logger.info(f"{args}")
    
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
    has_lora = any("lora_" in k for k in generator_ckpt["model"].keys())
    if has_lora:
        generator.gpt = get_peft_model(generator.gpt, lora_config)

    logger.info(f"Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    generator.load_state_dict(generator_ckpt["model"], strict=True)

    generator.gpt = get_peft_model(generator.gpt, lora_config)

    del generator_ckpt

    
    for name, param in generator.named_parameters():
        param.requires_grad = ("lora_" in name)

    optimizer = creat_optimizer(generator, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    args.dataset_target_token_len = gpt_kwargs['target_token_len']
    args.dataset_condition_token_len = gpt_kwargs['img_feature_code_len']
    
    all_datasets = build_dataset(args, transform=transform)
    
    loader = DataLoader(
        all_datasets['train'],
        batch_size=int(args.global_batch_size),
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Train dataset contains {len(all_datasets['train']):,} images.")
    
    val_loaders = {}
    val_types = [
        'val_nfacontent_nfafont',
        'val_precontent_nfafont',
        'val_testcontent_nfafont',
    ]
    
    for val_type in val_types:
        if len(all_datasets[val_type]) > 0:
            val_loaders[val_type] = DataLoader(
                all_datasets[val_type],
                batch_size=args.num_val_samples,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            )
    logger.info("Created validation loaders for: %s", ", ".join(val_loaders.keys()))
    
    # resume training
    if args.nfa_generator_ckpt:
        checkpoint = torch.load(args.nfa_generator_ckpt, map_location="cpu",weights_only=False)
        # Load the model weights and optimizer state
        generator.load_state_dict(checkpoint["model"], strict=True)
        vq_model.load_state_dict(checkpoint["vq_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.ft_generator_ckpt.split('/')[-1].split('.')[0])
        train_steps = 0
        start_epoch = 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.nfa_generator_ckpt}, train_steps={resume_train_steps}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
    
    generator = generator.to(device=device)

    generator.train()
    
    loss_logits_weight = args.loss_logits_weight
    loss_rec_weight = args.loss_rec_weight

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    
    log_steps = 0
    running_total_loss = 0
    running_logits_loss = 0
    running_rec_loss = 0
    
    start_time = time.time()

    
    
    logger.info(f"Training for {args.epochs} epochs...")
    
    
    codebook = vq_model.quantizer.embedding.weight
    hw_ = None

    for epoch in range(start_epoch,args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for tar_img, content_ref, style_refs, attn_mask, valid in loader:
            tar_img = tar_img.to(device,non_blocking=True)
            content_ref = content_ref.to(device,non_blocking=True)
            style_refs = style_refs.to(device,non_blocking=True)
            attn_mask = attn_mask.to(device)
            valid = valid.to(device)
            
            B = tar_img.shape[0]
            
            with torch.no_grad():
                _, _, [_, _, tar_vq_indices], hw = vq_model.encode(tar_img)
                hw_ = hw
            vq_indices = (tar_vq_indices.reshape(B,-1)) # [b, H*W]
            attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1])
            
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                logits, logits_loss = generator(content_img=content_ref, style_imgs=style_refs,
                                                vq_indices = vq_indices, gpt_valid = valid, gpt_attn_mask = attn_mask)

                if loss_rec_weight > 0:
                    probs = F.softmax(logits, dim=-1)          
                    soft_quantized_vectors = torch.matmul(probs, codebook)
                    
                    soft_reconstructed_img = vq_model.decode(soft_quantized_vectors,hw_)
                    rec_loss = F.l1_loss(soft_reconstructed_img, tar_img)
                else:
                    rec_loss = torch.tensor(0.0, device=device) 
        
                total_loss = loss_logits_weight * logits_loss + loss_rec_weight * rec_loss
                
            scaler.scale(total_loss).backward()
            
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            running_total_loss += total_loss.item()
            running_logits_loss += logits_loss.item()
            running_rec_loss += rec_loss.item()
            
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_total_loss = torch.tensor(running_total_loss / log_steps, device=device)
                avg_logits_loss = torch.tensor(running_logits_loss / log_steps, device=device)
                avg_rec_loss = torch.tensor(running_rec_loss / log_steps, device=device)
                
                avg_total_loss = avg_total_loss.item()
                avg_logits_loss = avg_logits_loss.item()
                avg_rec_loss = avg_rec_loss.item()
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_total_loss:.4f}, Logits Loss: {avg_logits_loss:.4f}, Rec Loss: {avg_rec_loss:.4f} Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Reset monitoring variables:
                running_total_loss = 0
                running_logits_loss = 0
                running_rec_loss = 0
                log_steps = 0
                start_time = time.time()
                
            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                model_weight = generator.state_dict()  
                checkpoint = {
                    "model": model_weight,
                    "vq_model": vq_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "train_args": args,
                    "tokenizer_kwargs": tokenizer_kwargs,
                    "content_encoder_args": content_args,
                    "style_encoder_args": style_args,
                    "feature_fusion_module_args": ffm_args,
                    "gpt_kwargs": gpt_kwargs,
                    "lora_config": lora_config,
                }
                
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
            if train_steps > 0 and train_steps % args.val_every == 0:
                logger.info(f"--- Running Full Validation at Step {train_steps} ---")
                val_name_to_prefix = {
                    'val_nfacontent_nfafont':'nc_nf',
                    'val_precontent_nfafont':'pc_nf',
                    'val_testcontent_nfafont':'tc_nf',
                }
                
                val_generator = Generator(content_args,style_args,gpt_models[args.gpt_model](**gpt_kwargs),ffm_args).to(device)
                val_generator.gpt = get_peft_model(val_generator.gpt, lora_config)
                
                val_generator.load_state_dict(generator.state_dict())
                val_generator.eval()  # Set validation generator to eval mode
                
                # Loop through all validation loaders and generate an image for each
                for val_name, val_loader in val_loaders.items():
                    prefix = val_name_to_prefix.get(val_name, "val")
                    validate_and_save_images(
                        val_generator, vq_model, val_loader, device, train_steps, 
                        experiment_dir, hw_, logger, prefix
                    )
                del val_generator
                logger.info(f"--- Finished Full Validation, images saved. ---")
                

    logger.info("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-path", type=str, required=True)
    parser.add_argument("--data-style-info-json", type=str, required=True)
    parser.add_argument("--data-content-info-json", type=str, required=True)
    parser.add_argument("--data-nfa-dir-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='generator_nfa')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--generator-ckpt", type=str, default=None, help="ckpt path of GAR-Font ckpt")
    parser.add_argument("--nfa-generator-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-model", type=str, choices=list(gpt_models.keys()), default="GPT-314M")
    parser.add_argument("--n-ref", type=int, default=8)
    parser.add_argument("--nfa-ori-train-num", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--loss-logits-weight", type=float, default=1.0, help="Weight for the logits loss.")
    parser.add_argument("--loss-rec-weight", type=float, default=1.0, help="Weight for the reconstruction loss.")
    parser.add_argument("--results-dir", type=str, default="./results/results_nfa")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--val-every", type=int, default=1000, help="Run validation.")
    parser.add_argument("--num-val-samples", type=int, default=128, help="Number of validation samples.")
    
    args = parser.parse_args()
    main(args)