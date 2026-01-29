#!/usr/bin/env python
# ========================================================================
# Quick-Start: Backtesting & Evaluation
# ========================================================================
"""
Run this script to evaluate your trained model against the test dataset.

Usage:
    python run_backtest.py [--checkpoint PATH] [--output-dir DIR] [--quick]

Examples:
    # Full backtesting with default settings
    python run_backtest.py
    
    # Quick evaluation (fewer samples, fewer diffusion steps)
    python run_backtest.py --quick
    
    # Use specific checkpoint
    python run_backtest.py --checkpoint models/checkpoints/model_epoch50.pt
"""

import os
import sys
import argparse
from pathlib import Path

# Add models to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))

import torch
from diffusion import TimeCondUNet, RadioMapDataset
from eval.backtest_evaluation import (
    backtest_on_dataset,
    plot_evaluation_results,
    print_evaluation_report
)


def main():
    parser = argparse.ArgumentParser(
        description='Run backtesting & evaluation on trained model'
    )
    parser.add_argument(
        '--checkpoint',
        default='models/checkpoints/model_final.pt',
        help='Path to model checkpoint (default: models/checkpoints/model_final.pt)'
    )
    parser.add_argument(
        '--input-dir',
        default='model_input/data/training/input',
        help='Path to input tensors'
    )
    parser.add_argument(
        '--target-dir',
        default='model_input/data/training/target',
        help='Path to target tensors'
    )
    parser.add_argument(
        '--output-dir',
        default='backtest_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--samples-per-scene',
        type=int,
        default=5,
        help='Number of diffusion samples per scene for ensemble'
    )
    parser.add_argument(
        '--diffusion-steps',
        type=int,
        default=50,
        help='Number of reverse diffusion steps'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick evaluation (3 samples, 20 steps, batch_size=1)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.samples_per_scene = 3
        args.diffusion_steps = 20
        args.batch_size = 1
        print("[Quick Mode] samples_per_scene=3, diffusion_steps=20, batch_size=1")
    
    # Device setup
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"Expected location: {os.path.abspath(args.checkpoint)}")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print("Initializing model...")
    model = TimeCondUNet(
        in_ch=1,
        cond_channels=3,
        base_ch=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=128,
        cond_emb_dim=64
    )
    
    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Load dataset
    print(f"Loading dataset...")
    dataset = RadioMapDataset(args.input_dir, args.target_dir)
    print(f"✓ Loaded {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        sys.exit(1)
    
    # Run backtesting
    print("\nStarting backtesting...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples per scene: {args.samples_per_scene}")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print()
    
    try:
        metrics, aggregate = backtest_on_dataset(
            model=model,
            dataset=dataset,
            device=device,
            num_samples_per_scene=args.samples_per_scene,
            diffusion_steps=args.diffusion_steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_evaluation_results(metrics, aggregate, args.output_dir)
        
        # Print report
        print_evaluation_report(metrics, aggregate)
        
        print(f"✓ Results saved to: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
