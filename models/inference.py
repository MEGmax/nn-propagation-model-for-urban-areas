# =========================================================================
# Inference Script for Conditional Diffusion Model
# Generates radio map predictions from a trained model checkpoint
# =========================================================================
import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from diffusion import TimeCondUNet, Diffusion


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load a trained model from checkpoint."""
    # Model configuration (must match training config)
    model_config = {
        'in_ch': 1,
        'cond_channels': 2,  # elevation, distance
        'base_ch': 32,
        'channel_mults': (1, 2, 4),
        'num_res_blocks': 2,
        'time_emb_dim': 128,
        'cond_emb_dim': 64
    }
    
    model = TimeCondUNet(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle both state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def run_inference(model, cond_input: np.ndarray, device: str = 'cpu', 
                  sampling_steps: int = 50, timesteps: int = 1000):
    """
    Run diffusion model inference to generate RSS prediction.
    
    Args:
        model: Trained TimeCondUNet model
        cond_input: Conditioning input array (H, W, 2) with channels [elevation, distance]
        device: Device to run inference on
        sampling_steps: Number of reverse diffusion steps
        timesteps: Total diffusion timesteps (should match training)
    
    Returns:
        RSS prediction as numpy array (H, W, 1) in dBm
    """
    # Convert input to torch tensor with batch dimension
    # Input expected: (H, W, 2) -> need (B, 2, H, W)
    cond_tensor = torch.from_numpy(cond_input).permute(2, 0, 1).unsqueeze(0).float()
    cond_tensor = cond_tensor.to(device)
    
    # Create diffusion sampler
    diffusion = Diffusion(model, timesteps=timesteps, device=device)
    
    # Sample from the model
    with torch.no_grad():
        # shape: (B, 1, H, W)
        samples = diffusion.sample(cond_tensor, steps=sampling_steps)
    
    # Convert back to numpy: (B, 1, H, W) -> (H, W, 1)
    rss_prediction = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return rss_prediction


def save_rss_numpy(rss_prediction: np.ndarray, output_path: str):
    """
    Save RSS prediction as numpy file in the specified format.
    
    Output format:
    - Shape: (H, W, 1) spatial map of signal strength
    - Values: In dBm (decibels referenced to 1 milliwatt)
    - Resolution: Matches input elevation map resolution
    """
    # Ensure shape is (H, W, 1)
    if rss_prediction.ndim == 2:
        rss_prediction = rss_prediction[:, :, np.newaxis]
    
    np.save(output_path, rss_prediction.astype(np.float32))
    print(f"✓ Saved RSS values to: {output_path}")
    print(f"  Shape: {rss_prediction.shape}")
    print(f"  Value range: {rss_prediction.min():.2f} to {rss_prediction.max():.2f} dBm")


def save_rss_png(rss_prediction: np.ndarray, output_path: str, 
                 vmin: float = None, vmax: float = None,
                 cmap: str = 'jet', title: str = 'Predicted RSS Heatmap'):
    """
    Save RSS prediction as PNG visualization.
    
    Args:
        rss_prediction: RSS values (H, W, 1) or (H, W) in dBm
        output_path: Path to save PNG
        vmin: Minimum value for colormap (weak signal). Auto if None.
        vmax: Maximum value for colormap (strong signal). Auto if None.
        cmap: Colormap to use
        title: Title for the plot
    """
    # Squeeze to 2D for visualization
    if rss_prediction.ndim == 3:
        rss_2d = rss_prediction.squeeze(-1)
    else:
        rss_2d = rss_prediction
    
    # Auto-scale if not provided
    if vmin is None:
        vmin = rss_2d.min()
    if vmax is None:
        vmax = rss_2d.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap with custom normalization
    im = ax.imshow(rss_2d, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                   origin='upper', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('RSS (dBm)', fontsize=12)
    
    # Labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (grid cells)', fontsize=11)
    ax.set_ylabel('Y (grid cells)', fontsize=11)
    
    # Add statistics annotation
    stats_text = (f"Min: {rss_2d.min():.1f} dBm\n"
                  f"Max: {rss_2d.max():.1f} dBm\n"
                  f"Mean: {rss_2d.mean():.1f} dBm")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved RSS visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with trained diffusion model for RSS prediction'
    )
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/model_final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input conditioning tensor (.npy file with shape H,W,2)')
    parser.add_argument('--output-dir', type=str, default='../rss_visualizations',
                        help='Directory to save outputs')
    parser.add_argument('--output-name', type=str, default='prediction',
                        help='Base name for output files (without extension)')
    parser.add_argument('--sampling-steps', type=int, default=50,
                        help='Number of reverse diffusion sampling steps')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum dBm value for visualization colormap (auto if not specified)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum dBm value for visualization colormap (auto if not specified)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda/cpu). Auto-detected if not specified.')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("✓ Model loaded successfully")
    
    # Load input conditioning data
    print(f"Loading input from: {args.input}")
    cond_input = np.load(args.input)
    print(f"  Input shape: {cond_input.shape}")
    
    # Validate input shape
    if cond_input.ndim != 3 or cond_input.shape[2] != 2:
        print(f"Error: Expected input shape (H, W, 2), got {cond_input.shape}")
        sys.exit(1)
    
    # Run inference
    print(f"Running inference with {args.sampling_steps} sampling steps...")
    rss_prediction = run_inference(model, cond_input, device=device, 
                                    sampling_steps=args.sampling_steps)
    print("✓ Inference complete")
    
    # Save outputs
    npy_path = os.path.join(args.output_dir, f"{args.output_name}.npy")
    png_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    
    save_rss_numpy(rss_prediction, npy_path)
    save_rss_png(rss_prediction, png_path, vmin=args.vmin, vmax=args.vmax)
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"RSS Numpy file: {npy_path}")
    print(f"  - Shape: (H, W, 1) = {rss_prediction.shape}")
    print(f"  - Values: dBm (decibels referenced to 1 milliwatt)")
    print(f"  - Range: {rss_prediction.min():.2f} to {rss_prediction.max():.2f} dBm")
    print(f"RSS Visualization: {png_path}")
    print("="*60)


if __name__ == '__main__':
    main()
