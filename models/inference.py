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
        'cond_channels': 3,  # elevation, distance, frequency
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
    Run diffusion model inference to generate pathloss prediction.
    
    Args:
        model: Trained TimeCondUNet model
        cond_input: Conditioning input array (H, W, 3) with channels [elevation, distance, frequency]
        device: Device to run inference on
        sampling_steps: Number of reverse diffusion steps
        timesteps: Total diffusion timesteps (should match training)
    
    Returns:
        Pathloss prediction as numpy array (H, W, 1) in dB
    """
    # Convert input to torch tensor with batch dimension
    # Input expected: (H, W, 3) -> need (B, 3, H, W)
    cond_tensor = torch.from_numpy(cond_input).permute(2, 0, 1).unsqueeze(0).float()
    cond_tensor = cond_tensor.to(device)
    
    # Create diffusion sampler
    diffusion = Diffusion(model, timesteps=timesteps, device=device)
    
    # Sample from the model
    with torch.no_grad():
        # shape: (B, 1, H, W)
        samples = diffusion.sample(cond_tensor, steps=sampling_steps)
    
    # Convert back to numpy: (B, 1, H, W) -> (H, W, 1)
    pathloss_prediction = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return pathloss_prediction


def save_pathloss_numpy(pathloss_prediction: np.ndarray, output_path: str):
    """
    Save pathloss prediction as numpy file in the specified format.
    
    Output format:
    - Shape: (H, W, 1) spatial map of signal strength
    - Values: In dB
    - Resolution: Matches input elevation map resolution
    """
    # Ensure shape is (H, W, 1)
    if pathloss_prediction.ndim == 2:
        pathloss_prediction = pathloss_prediction[:, :, np.newaxis]
    
    np.save(output_path, pathloss_prediction.astype(np.float32))
    print(f"✓ Saved pathloss values to: {output_path}")
    print(f"  Shape: {pathloss_prediction.shape}")
    print(f"  Value range: {pathloss_prediction.min():.2f} to {pathloss_prediction.max():.2f} dB")


def save_pathloss_png(pathloss_prediction: np.ndarray, output_path: str, 
                 vmin: float = None, vmax: float = None,
                 cmap: str = 'jet', title: str = 'Predicted Pathloss Heatmap'):
    """
    Save pathloss prediction as PNG visualization.
    
    Args:
        pathloss_prediction: Pathloss values (H, W, 1) or (H, W) in dB
        output_path: Path to save PNG
        vmin: Minimum value for colormap. Auto if None.
        vmax: Maximum value for colormap. Auto if None.
        cmap: Colormap to use
        title: Title for the plot
    """
    # Squeeze to 2D for visualization
    if pathloss_prediction.ndim == 3:
        pathloss_2d = pathloss_prediction.squeeze(-1)
    else:
        pathloss_2d = pathloss_prediction
    
    # Auto-scale if not provided
    if vmin is None:
        vmin = pathloss_2d.min()
    if vmax is None:
        vmax = pathloss_2d.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap with custom normalization
    im = ax.imshow(pathloss_2d, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                   origin='upper', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pathloss (dB)', fontsize=12)
    
    # Labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (grid cells)', fontsize=11)
    ax.set_ylabel('Y (grid cells)', fontsize=11)
    
    # Add statistics annotation
    stats_text = (f"Min: {pathloss_2d.min():.1f} dB\n"
                  f"Max: {pathloss_2d.max():.1f} dB\n"
                  f"Mean: {pathloss_2d.mean():.1f} dB")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved pathloss visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with trained diffusion model for pathloss prediction'
    )
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints_pathloss/model_final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input conditioning tensor (.npy file with shape H,W,3)')
    parser.add_argument('--output-dir', type=str, default='../pathloss_visualizations',
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
    if cond_input.ndim != 3 or cond_input.shape[2] != 3:
        print(f"Error: Expected input shape (H, W, 3), got {cond_input.shape}")
        sys.exit(1)
    
    # Run inference
    print(f"Running inference with {args.sampling_steps} sampling steps...")
    pathloss_prediction = run_inference(model, cond_input, device=device, 
                                    sampling_steps=args.sampling_steps)
    print("✓ Inference complete")
    
    # Save outputs
    npy_path = os.path.join(args.output_dir, f"{args.output_name}.npy")
    png_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    
    save_pathloss_numpy(pathloss_prediction, npy_path)
    save_pathloss_png(pathloss_prediction, png_path, vmin=args.vmin, vmax=args.vmax)
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Pathloss Numpy file: {npy_path}")
    print(f"  - Shape: (H, W, 1) = {pathloss_prediction.shape}")
    print("  - Values: dB")
    print(f"  - Range: {pathloss_prediction.min():.2f} to {pathloss_prediction.max():.2f} dB")
    print(f"Pathloss Visualization: {png_path}")
    print("="*60)


if __name__ == '__main__':
    main()
