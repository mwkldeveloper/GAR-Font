# Modified from: llamagen/vq_train.py
import os
import time
import argparse
from glob import glob



from model.tokenizer.vq_loss import VQ_loss
from utils.logger import create_logger

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.build_dataset import build_dataset

from model.tokenizer.tokenizer import Tokenizer
from model.model_config import VQ_models,tokenizer_kwargs,img_size
from torchvision.utils import save_image

import warnings
warnings.filterwarnings('ignore')


def validate_and_save_images(vq_model, val_loader, device, train_steps, experiment_dir, logger):
    vq_model.eval()
    val_dir = os.path.join(experiment_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)

    with torch.no_grad():
        try:
            val_images = next(iter(val_loader))
        except StopIteration:
            logger.warning("Validation loader is empty. Skipping validation step.")
            vq_model.train()
            return
            
        val_images = val_images.to(device)
        recons_imgs, _ = vq_model(val_images)
        comparison_grid = torch.cat([val_images, recons_imgs])
        
        save_path = os.path.join(val_dir, f"vqvae_step{train_steps:07d}_val.png")
        save_image(
            comparison_grid, 
            save_path, 
            nrow=val_images.size(0),
            normalize=True, 
        )
        logger.info(f"Saved validation image grid to {save_path}")

    vq_model.train()
    
def main(args):
    """
    Train a Tokenizer.
    """
    device = "cuda:0"
    torch.cuda.set_device(device)
    
    seed = args.global_seed
    torch.manual_seed(seed)
    
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.vq_model
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f'args : {args}')
    
    logger.info(f"Start train for {args.vq_model} seed={args.global_seed} on {device}")
    vq_model = Tokenizer(VQ_models[args.vq_model](**tokenizer_kwargs))
    
    logger.info(f"Tokenizer Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    logger.info(f'{tokenizer_kwargs}')
    
    vq_model = vq_model.to(device)
    
    vq_loss = VQ_loss(
        image_size=img_size,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,  
        perceptual_weight=args.perceptual_weight
    ).to(device)

    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset, val_dataset = build_dataset(args, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size ),
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.num_val_samples,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Train dataset contains {len(train_dataset):,} images.")

    if len(val_dataset) > 0:
        logger.info(f"Validation dataset contains {len(val_dataset):,} images.")
    else:
        logger.warning("Validation dataset is empty. Validation will be skipped.")
    
    # check if resume training.
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        vq_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        resume_train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])

        train_steps = 0
        start_epoch = 0    
               
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}, steps={resume_train_steps}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
    
    vq_model = vq_model.to(device)
    vq_loss = vq_loss.to(device)
        
    vq_model.train()
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x in train_loader:
            imgs = x.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, codebook_loss = vq_model(imgs)
                
                loss_gen = vq_loss(codebook_loss, imgs, recons_imgs, global_step=train_steps+1, 
                                   logger=logger, log_every=args.log_every)
                
            scaler.scale(loss_gen).backward()
            
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss_gen.item() 
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                
                avg_loss = avg_loss.item()
                    
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                model_weight = vq_model.state_dict()   
                checkpoint = {
                    "model": model_weight,
                    "optimizer": optimizer.state_dict(),
                    "tokenizer_kwargs": tokenizer_kwargs,
                    "steps": train_steps,
                    "args": args
                }
                
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if train_steps % args.val_every == 0 and train_steps > 0:
                logger.info(f"Running validation at step {train_steps}...")
                validate_and_save_images(
                    vq_model,
                    val_loader,
                    device,
                    train_steps,
                    experiment_dir,
                    logger
                )
                vq_loss.train()
                

    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-path", type=str, required=True)
    parser.add_argument("--data-style-info-json", type=str, required=True)
    parser.add_argument("--data-content-info-json", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='tokenizer')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=0.001)
    parser.add_argument("--reconstruction-loss", type=str, default='l1')
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--results-dir", type=str, default="./results/results_Tokenizer")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20000)
    parser.add_argument("--val-every", type=int, default=10000, help="Run validation.")
    parser.add_argument("--num-val-samples", type=int, default=128, help="Number of validation samples.")
    
    args = parser.parse_args()
    main(args)
