import os
import time
import argparse
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from reward.style_reward_model import StyleRewardModel
from reward.style_reward_dataset import build_style_reward_dataset
from utils.logger import create_logger 

def main(args):
    assert torch.cuda.is_available()
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed)
    
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-StyleRewardModel"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Args: {args}")

    model = StyleRewardModel().to(device)
    logger.info(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    datasets = build_style_reward_dataset(args, transform)
    train_loader = DataLoader(datasets['train'], batch_size=args.global_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(datasets['val'], batch_size=args.global_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Train dataset size: {len(datasets['train']):,}")
    logger.info(f"Validation dataset size: {len(datasets['val']):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    loss_fn = nn.BCELoss()


    train_steps = 0
    start_epoch = 0

    if args.reward_model_ckpt:
        checkpoint = torch.load(args.reward_model_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])

        resume_train_steps = checkpoint["steps"] if "steps" in checkpoint else None
        
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.reward_model_ckpt}, resume_train_steps = {resume_train_steps}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")


    logger.info(f"Training for {args.epochs} epochs...")
    
    log_steps = 0
    running_loss = 0.0
    running_correct_num = 0
    data_train_num = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        model.train()

        for i, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            scores = model(img1, img2)
            loss = loss_fn(scores, labels)
            
            loss.backward()

            if args.max_grad_norm > 0:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()

            log_steps += 1
            train_steps += 1
            running_loss += loss.item()

            predicted = (scores > 0.5).float()
            running_correct_num += (predicted == labels).sum().item()
            data_train_num += labels.size(0)

            if train_steps % args.log_every == 0 and train_steps > 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                avg_loss = running_loss / log_steps
                avg_acc = running_correct_num / data_train_num
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Steps/Sec: {steps_per_sec:.2f}")

                running_loss = 0.0
                log_steps = 0
                running_correct_num = 0
                data_train_num = 0
                start_time = time.time()

            if train_steps % args.val_every == 0 and train_steps > 0:
                model.eval()
                val_loss = 0.0
                val_corrects = 0
                val_total = 0
                with torch.no_grad():
                    for val_img1, val_img2, val_labels in val_loader:
                        val_img1, val_img2, val_labels = val_img1.to(device), val_img2.to(device), val_labels.to(device)
                        
                        val_scores = model(val_img1, val_img2)
                        loss = loss_fn(val_scores, val_labels)
                        val_loss += loss.item()

                        val_predicted = (val_scores > 0.5).float()
                        val_total += val_labels.size(0)
                        val_corrects += (val_predicted == val_labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_corrects / val_total

                logger.info(f"--- (step={train_steps:07d}) Validation --- Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")
                model.train() 

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"{train_steps:07d}.pt")
                checkpoint = {
                    'steps': train_steps,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-path", type=str, required=True)
    parser.add_argument("--data-style-info-json", type=str, required=True)
    parser.add_argument("--data-content-info-json", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--reward-model-ckpt", type=str, default=None, help="Path to the checkpoint to resume training.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--results-dir", type=str, default="results_style_reward_model")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=1000, help="Run validation")

    args = parser.parse_args()
    main(args)