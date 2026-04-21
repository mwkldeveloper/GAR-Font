from glob import glob
import time
import argparse
import os

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn as nn
from utils.logger import create_logger
from dataset.build_dataset import build_dataset


from model.generator.generator import Generator
from model.model_config import VQ_models,tokenizer_kwargs, style_args, gpt_models, gpt_kwargs, ffm_args, lora_config
from model.tokenizer.tokenizer import Tokenizer

from model.generator_adapter.generator_adapter import PromptFusionAligner
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


def validate_aligner_model(
    aligner_model,
    val_loader,
    teacher_generator,
    train_steps,
    logger,
    prefix,
    device,
    n_keep
):
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        _, content_ref, style_refs, style_t5_tokens = next(iter(val_loader))
        content_ref = content_ref.to(device, non_blocking=True)
        style_refs = style_refs.to(device, non_blocking=True)
        style_t5_tokens = style_t5_tokens.to(device, non_blocking=True)
        encoded_content = teacher_generator.content_encoder(content_ref)

        B, n_ref, C_in, H, W = style_refs.shape
        style_imgs_flat = style_refs.reshape(B * n_ref, C_in, H, W)
        style_feats = teacher_generator.style_encoder(style_imgs_flat)
        _, C_out, h, w = style_feats.shape
        style_feats = style_feats.view(B, n_ref, C_out, h, w)

        feature_fused = teacher_generator.ffm(encoded_content, style_feats)
        target_fused_map = torch.cat([encoded_content, feature_fused], dim=1)

        B, n_ref, C_out, h, w = style_feats.shape

        if n_keep > 0:
            keep_idxs = torch.randperm(n_ref)[:n_keep]
            style_feats_keep = style_feats[:, keep_idxs]  # [B, n_keep, C, h, w]
        else:
            style_feats_keep = None

        pseudo_style_feats = aligner_model(style_feats_keep, style_t5_tokens)  # [B, n_keep+1, C, h, w]
        predicted_fused = teacher_generator.ffm(encoded_content, pseudo_style_feats)

        predicted_fused_map = torch.cat([encoded_content, predicted_fused], dim=1)
        val_loss = loss_fn(predicted_fused_map, target_fused_map)
        logger.info(f"Validation on {prefix} - MSE Loss at step {train_steps}: {val_loss.item():.6f}")

    aligner_model.train()


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
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-Adapter-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    for param in generator.parameters():
        param.requires_grad = False
    
    del generator_ckpt  
    


    aligner_model = PromptFusionAligner(feat_dim=vq_model.config.z_channels,text_dim= 2048).to(device)

    optimizer = creat_optimizer(aligner_model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    


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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(loader),
        eta_min=args.lr * 0.1
    )

    val_loaders = {}
    val_types = [
        'val_seenfont_seencontent', 'val_unseenfont_seencontent',
        'val_seenfont_unseencontent', 'val_unseenfont_unseencontent'
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
    if args.adapter_ckpt:
        checkpoint = torch.load(args.adapter_ckpt, map_location="cpu",weights_only=False)
        # Load the model weights and optimizer state
        aligner_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.generator_ckpt.split('/')[-1].split('.')[0])
        train_steps = 0
        start_epoch = 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.adapter_ckpt}, train_steps={resume_train_steps}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    
    loss_fn = nn.MSELoss()

    log_steps = 0
    running_rec_loss = 0
    running_total_loss = 0
    start_time = time.time()

    
    
    logger.info(f"Training for {args.epochs} epochs...")
    
    n_keep = args.adapter_n_keep
    print(f' N_Keep = {n_keep}')

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for tar_img, content_ref, style_refs, style_t5_tokens in loader:
        
            tar_img = tar_img.to(device, non_blocking=True)
            content_ref = content_ref.to(device, non_blocking=True)
            style_refs = style_refs.to(device, non_blocking=True)
            style_t5_tokens = style_t5_tokens.to(device, non_blocking=True)

            with torch.no_grad():
                B, n_ref, C_in, H, W = style_refs.shape
                encoded_content = generator.content_encoder(content_ref)

                style_imgs_flat = style_refs.reshape(B * n_ref, C_in, H, W)
                style_feats = generator.style_encoder(style_imgs_flat)
                _, C_out, h, w = style_feats.shape
                style_feats = style_feats.view(B, n_ref, C_out, h, w)

                feature_fused = generator.ffm(encoded_content, style_feats)
                target_fused_map = torch.cat([encoded_content, feature_fused], dim=1)

            with torch.cuda.amp.autocast(dtype=ptdtype):
                B, n_ref, C_out, h, w = style_feats.shape

                if n_keep > 0:
                    keep_idxs = torch.randperm(n_ref)[:n_keep]
                    style_feats_keep = style_feats[:, keep_idxs]  # [B, n_keep, C, h, w]
                else:
                    style_feats_keep = None
                pseudo_style_feats = aligner_model(style_feats_keep, style_t5_tokens)  # [B, n_keep+1, C, h, w]


                predicted_fused = generator.ffm(encoded_content, pseudo_style_feats)

                predicted_fused_map = torch.cat([encoded_content, predicted_fused], dim=1)

                rec_loss = loss_fn(predicted_fused_map, target_fused_map)

            total_loss = rec_loss
            scaler.scale(total_loss).backward()
            
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
             # Log loss values:
            running_total_loss += total_loss.item()
            running_rec_loss += rec_loss.item()
            
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_total_loss = torch.tensor(running_total_loss / log_steps, device=device)
                avg_rec_loss = torch.tensor(running_rec_loss / log_steps, device=device)
                
                avg_total_loss = avg_total_loss.item()
                avg_rec_loss = avg_rec_loss.item()
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_total_loss:.4f}, Rec Loss: {avg_rec_loss:.4f} Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Reset monitoring variables:
                running_total_loss = 0
                running_rec_loss = 0
                log_steps = 0
                start_time = time.time()
                
            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                model_weight = aligner_model.state_dict()  
                checkpoint = {
                    "model": model_weight,
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "train_args": args,
                }
                
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
            if train_steps > 0 and train_steps % args.val_every == 0:
                logger.info(f"--- Running Full Validation at Step {train_steps} ---")
                val_name_to_prefix = {
                    'val_seenfont_seencontent': 'sfsc', 'val_unseenfont_seencontent': 'ufsc',
                    'val_seenfont_unseencontent': 'sfuc', 'val_unseenfont_unseencontent': 'ufuc'
                }
                
                val_aligner_model = PromptFusionAligner(feat_dim=vq_model.config.z_channels,text_dim= 2048).to(device)
                val_aligner_model.load_state_dict(aligner_model.state_dict())
                val_aligner_model.eval()  # Set validation generator to eval mode
                
                # Loop through all validation loaders and generate an image for each
                for val_name, val_loader in val_loaders.items():
                    prefix = val_name_to_prefix.get(val_name, "val")
                    validate_aligner_model(
                        val_aligner_model, val_loader, generator,
                        train_steps, logger, prefix, device, n_keep
                    )


                del val_aligner_model
                logger.info(f"--- Finished Full Validation. ---")

    logger.info("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-path", type=str, required=True)
    parser.add_argument("--data-style-info-json", type=str, required=True)
    parser.add_argument("--data-content-info-json", type=str, required=True)
    parser.add_argument("--data-t5-feat-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='generator_adapter')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--generator-ckpt", type=str, required=True)
    parser.add_argument("--gpt-model", type=str, choices=list(gpt_models.keys()), default="GPT-314M")
    parser.add_argument("--n-ref", type=int, default=8)
    parser.add_argument("--train-num-per-font", type=int, default=20)
    parser.add_argument('--adapter-ckpt', type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--t5-feat-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--loss-rec-weight", type=float, default=1.0, help="Weight for the reconstruction loss.")
    parser.add_argument("--results-dir", type=str, default="./results/results_generator_adapter")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=1000, help="Run validation.")
    parser.add_argument("--num-val-samples", type=int, default=8, help="Number of validation samples.")
    parser.add_argument("--adapter-n-keep", type=int, default=4)
    args = parser.parse_args()
    main(args)