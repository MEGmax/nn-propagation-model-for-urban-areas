#!/usr/bin/env python
# ========================================================================
# RSS Map Visualization Script
# Generate PNG visualizations of model predictions vs ground truth
# ========================================================================
"""
Visualize RSS map predictions from the diffusion model.

Generates side-by-side comparisons of:
1. Predicted RSS map (from model)
2. Ground truth RSS map
3. Error map (prediction - ground truth)
4. Input conditioning (elevation + distance + frequency)

Usage:
    python visualize_rss_maps.py [--checkpoint PATH] [--output-dir DIR] [--num-scenes N]

Examples:
    # Visualize all 5 training scenes
    python visualize_rss_maps.py
    
    # Quick visualization of 2 scenes
    python visualize_rss_maps.py --num-scenes 2
    
    # Use specific checkpoint
    python visualize_rss_maps.py --checkpoint models/checkpoints/model_epoch30.pt
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Add models to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'models'))

from diffusion import TimeCondUNet, RadioMapDataset, Diffusion

# Normalization constants (from model_input.py)
RSS_DB_FLOOR = -100.0
RSS_DB_SCALE = 50.0


def denormalize_rss(rss_normalized):
    """Convert from [-1, 1] normalized to dBm."""
    return rss_normalized * RSS_DB_SCALE + RSS_DB_FLOOR


def create_rss_map_figure(predicted_rss, ground_truth_rss, conditioning, scene_name, output_dir):
    """
    Create comprehensive 2x3 figure showing:
    1. Predicted RSS map
    2. Ground truth RSS map
    3. Error map
    4. Elevation (conditioning)
    5. Distance map (conditioning)
    6. Frequency (conditioning)
    """
    
    # Denormalize if needed
    if predicted_rss.min() >= -1.5 and predicted_rss.max() <= 1.5:
        predicted_rss = denormalize_rss(predicted_rss)
    if ground_truth_rss.min() >= -1.5 and ground_truth_rss.max() <= 1.5:
        ground_truth_rss = denormalize_rss(ground_truth_rss)
    
    # Compute error
    error = predicted_rss - ground_truth_rss
    
    # Extract conditioning channels (if available)
    elevation = conditioning[0] if conditioning.shape[0] > 0 else None
    distance = conditioning[1] if conditioning.shape[0] > 1 else None
    frequency = conditioning[2] if conditioning.shape[0] > 2 else None
    
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== TOP ROW: RSS MAPS =====
    
    # 1. Predicted RSS
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(predicted_rss, cmap='viridis', origin='upper')
    ax1.set_title('Predicted RSS Map', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('dBm')
    
    # 2. Ground Truth RSS
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ground_truth_rss, cmap='viridis', origin='upper')
    ax2.set_title('Ground Truth RSS Map', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('dBm')
    
    # 3. Error Map
    ax3 = fig.add_subplot(gs[0, 2])
    # Use diverging colormap for error (red = over, blue = under)
    im3 = ax3.imshow(error, cmap='RdBu_r', origin='upper', 
                     vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
    ax3.set_title(f'Error Map (RMSE: {np.sqrt(np.mean(error**2)):.2f} dB)', 
                 fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Error (dB)')
    
    # ===== BOTTOM ROW: CONDITIONING INPUTS =====
    
    # 4. Elevation
    if elevation is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(elevation, cmap='terrain', origin='upper')
        ax4.set_title('Elevation (Input)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        cbar4 = plt.colorbar(im4, ax=ax4)
        cbar4.set_label('Height (norm)')
    
    # 5. Distance Map
    if distance is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(distance, cmap='hot', origin='upper')
        ax5.set_title('Distance from TX (Input)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X (pixels)')
        cbar5 = plt.colorbar(im5, ax=ax5)
        cbar5.set_label('Distance (norm)')
    
    # 6. Frequency (scalar as heatmap)
    if frequency is not None:
        ax6 = fig.add_subplot(gs[1, 2])
        # Frequency is uniform, so visualize it as text + single color
        ax6.imshow(np.ones_like(frequency) * frequency[0, 0], cmap='cool', origin='upper')
        freq_val = frequency[0, 0]
        ax6.set_title(f'Frequency (Input)\nlog10(GHz) = {freq_val:.2f}', 
                     fontsize=12, fontweight='bold')
        ax6.set_xlabel('X (pixels)')
        ax6.text(frequency.shape[1]//2, frequency.shape[0]//2, f'{freq_val:.2f}',
                ha='center', va='center', fontsize=20, color='white', fontweight='bold')
    
    # Add main title with scene info and statistics
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    bias = np.mean(error)
    
    fig.suptitle(
        f'{scene_name}\n'
        f'RMSE: {rmse:.2f} dB | MAE: {mae:.2f} dB | Bias: {bias:.2f} dB',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    # Save figure
    output_path = os.path.join(output_dir, f'{scene_name}_rss_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return {'rmse': rmse, 'mae': mae, 'bias': bias}


def create_simple_heatmap(rss_map, title, output_path, colorbar_label='dBm'):
    """Create a simple single heatmap visualization."""
    
    # Denormalize if needed
    if rss_map.min() >= -1.5 and rss_map.max() <= 1.5:
        rss_map = denormalize_rss(rss_map)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rss_map, cmap='viridis', origin='upper')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    
    # Add statistics to plot
    mean_val = np.mean(rss_map)
    std_val = np.std(rss_map)
    min_val = np.min(rss_map)
    max_val = np.max(rss_map)
    
    stats_text = f'Mean: {mean_val:.1f} dB\nStd: {std_val:.1f} dB\nMin: {min_val:.1f} dB\nMax: {max_val:.1f} dB'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


@torch.no_grad()
def generate_visualizations(
    model,
    dataset,
    device,
    output_dir='rss_visualizations',
    num_scenes=None,
    diffusion_steps=50,
    num_samples=5
):
    """Generate visualization PNGs for model predictions."""
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to(device)
    
    # Initialize diffusion
    diffusion = Diffusion(model, timesteps=1000, device=device)
    
    # Determine number of scenes to visualize
    if num_scenes is None:
        num_scenes = len(dataset)
    else:
        num_scenes = min(num_scenes, len(dataset))
    
    print(f"Generating visualizations for {num_scenes} scenes...")
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    metrics_list = []
    
    for scene_idx in tqdm(range(num_scenes), desc="Visualizing"):
        batch = dataset[scene_idx]
        
        # Extract data
        cond_input = batch['input'].unsqueeze(0).to(device)  # (1, 3, H, W)
        gt_target = batch['target'].squeeze(0).cpu().numpy()  # (H, W)
        
        scene_name = dataset.scene_names[scene_idx]
        
        # Generate ensemble predictions
        with torch.no_grad():
            samples = diffusion.sample(cond_input, steps=diffusion_steps)  # (1, 1, H, W)
        
        # Average samples if multiple
        pred_rss = samples[0, 0].cpu().numpy()  # (H, W)
        
        # Create comparison figure
        cond_numpy = cond_input[0].cpu().numpy()  # (3, H, W)
        metrics = create_rss_map_figure(
            pred_rss,
            gt_target,
            cond_numpy,
            scene_name,
            output_dir
        )
        metrics_list.append({**metrics, 'scene': scene_name})
        
        # Also save individual heatmaps
        create_simple_heatmap(
            pred_rss,
            f'{scene_name} - Predicted RSS',
            os.path.join(output_dir, f'{scene_name}_predicted.png')
        )
        
        create_simple_heatmap(
            gt_target,
            f'{scene_name} - Ground Truth RSS',
            os.path.join(output_dir, f'{scene_name}_groundtruth.png')
        )
    
    # Print summary statistics
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    print(f"{'Scene':<20} {'RMSE (dB)':<15} {'MAE (dB)':<15} {'Bias (dB)':<15}")
    print("-"*80)
    
    for m in metrics_list:
        print(f"{m['scene']:<20} {m['rmse']:<15.2f} {m['mae']:<15.2f} {m['bias']:<15.2f}")
    
    print("-"*80)
    rmse_mean = np.mean([m['rmse'] for m in metrics_list])
    mae_mean = np.mean([m['mae'] for m in metrics_list])
    bias_mean = np.mean([m['bias'] for m in metrics_list])
    print(f"{'AVERAGE':<20} {rmse_mean:<15.2f} {mae_mean:<15.2f} {bias_mean:<15.2f}")
    print("="*80 + "\n")
    
    print(f"✓ Generated visualizations for {num_scenes} scenes")
    print(f"✓ Saved to: {os.path.abspath(output_dir)}")
    print(f"\nFiles created:")
    print(f"  - {num_scenes} comparison figures (*_rss_comparison.png)")
    print(f"  - {num_scenes} predicted maps (*_predicted.png)")
    print(f"  - {num_scenes} ground truth maps (*_groundtruth.png)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PNG visualizations of RSS map predictions'
    )
    parser.add_argument(
        '--checkpoint',
        default='models/checkpoints/model_final.pt',
        help='Path to model checkpoint'
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
        default='rss_visualizations',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--num-scenes',
        type=int,
        default=None,
        help='Number of scenes to visualize (default: all)'
    )
    parser.add_argument(
        '--diffusion-steps',
        type=int,
        default=50,
        help='Number of reverse diffusion steps'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    args = parser.parse_args()
    
    # Device setup
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
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
    
    # Load dataset
    print("Loading dataset...")
    dataset = RadioMapDataset(args.input_dir, args.target_dir)
    print(f"✓ Loaded {len(dataset)} samples\n")
    
    # Generate visualizations
    try:
        generate_visualizations(
            model,
            dataset,
            device,
            output_dir=args.output_dir,
            num_scenes=args.num_scenes,
            diffusion_steps=args.diffusion_steps
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
