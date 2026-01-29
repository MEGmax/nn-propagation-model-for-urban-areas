#!/usr/bin/env python3
"""
Training script for the diffusion-based radio map prediction model.

Usage:
    python run_training.py --dataset-dir dataset_samples --epochs 50 --batch-size 8
    python run_training.py --help  (for all options)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.diffusion import TimeCondUNet, RadioMapDataset, train


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for radio map prediction."
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_samples",
        help="Directory containing .npy training files (default: dataset_samples)"
    )
    
    # Model architecture
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base number of channels in U-Net (default: 64)"
    )
    parser.add_argument(
        "--channel-mults",
        type=str,
        default="1,2,4,8",
        help="Channel multipliers for each scale (default: 1,2,4,8)"
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=2,
        help="Number of residual blocks per scale (default: 2)"
    )
    parser.add_argument(
        "--time-emb-dim",
        type=int,
        default=256,
        help="Dimension of time embedding (default: 256)"
    )
    parser.add_argument(
        "--cond-emb-dim",
        type=int,
        default=128,
        help="Dimension of conditioning embedding (default: 128)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (default: 1000)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)"
    )
    
    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use: 'auto' (default), 'cuda', or 'cpu'"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more debug info"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # ---- Setup device ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # ---- Load dataset ----
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist!")
        print(f"First run the notebook 'xml_to_radiomap_pipeline.ipynb' to generate samples.")
        sys.exit(1)
    
    # Get list of .npy files
    filenames = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".npy")])
    if len(filenames) == 0:
        print(f"ERROR: No .npy files found in '{dataset_dir}'!")
        sys.exit(1)
    
    print(f"Found {len(filenames)} samples in {dataset_dir}")
    
    # Create dataset
    ds = RadioMapDataset(str(dataset_dir), filenames)
    
    # Infer conditioning channels from first sample
    first_sample_path = dataset_dir / filenames[0]
    first_sample = np.load(first_sample_path, allow_pickle=True).item()
    cond_channels = first_sample["cond"].shape[0]
    
    print(f"Detected {cond_channels} conditioning channels")
    print(f"  (typically: elevation + TX heatmap + material channels)")
    
    if args.verbose:
        print(f"First sample keys: {list(first_sample.keys())}")
        print(f"  rss shape: {first_sample['rss'].shape}")
        print(f"  cond shape: {first_sample['cond'].shape}")
        print(f"  elevation shape: {first_sample['elevation'].shape}")
    
    # ---- Create model ----
    channel_mults = tuple(map(int, args.channel_mults.split(",")))
    
    model = TimeCondUNet(
        in_ch=1,
        cond_channels=cond_channels,
        base_ch=args.base_channels,
        channel_mults=channel_mults,
        num_res_blocks=args.num_res_blocks,
        time_emb_dim=args.time_emb_dim,
        cond_emb_dim=args.cond_emb_dim
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params / 1e6:.2f}M parameters")
    
    # ---- Train ----
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    train(
        model=model,
        dataset=ds,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        timesteps=args.timesteps,
        save_every=args.save_every,
        out_dir=args.checkpoint_dir
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("="*60)
    print("\nTo generate samples, use:")
    print("  from interm_demo import TimeCondUNet, sample_and_save")
    print(f"  model = TimeCondUNet(in_ch=1, cond_channels={cond_channels}, ...)")
    print(f"  model.load_state_dict(torch.load('{args.checkpoint_dir}/model_final.pt'))")
    print("  samples = sample_and_save(model, cond_img, device='cuda', steps=50)")


if __name__ == "__main__":
    main()
