from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pathloss import DEFAULT_STATS_FILENAME, denormalize_path_loss, load_stats, validate_stats  # noqa: E402
from diffusion import Diffusion, TimeCondUNet  # noqa: E402


def load_checkpoint_payload(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {
        "model_state_dict": checkpoint,
        "model_config": {
            "noisy_channels": 1,
            "cond_channels": 2,
            "base_ch": 32,
            "channel_mults": (1, 2, 4),
            "num_res_blocks": 2,
            "time_emb_dim": 128,
            "out_ch": 1,
        },
        "normalization_stats": None,
    }


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[TimeCondUNet, dict]:
    payload = load_checkpoint_payload(checkpoint_path, device)
    model_config = payload["model_config"]
    model = TimeCondUNet(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def run_inference(
    model: TimeCondUNet,
    cond_input: np.ndarray,
    stats,
    device: str = "cpu",
    sampling_steps: int = 50,
    timesteps: int = 1000,
) -> np.ndarray:
    if cond_input.ndim != 3 or cond_input.shape[2] != 2:
        raise ValueError(f"Expected conditioning tensor shape (H, W, 2), got {cond_input.shape}")

    cond_tensor = torch.from_numpy(cond_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
    diffusion = Diffusion(model, timesteps=timesteps, device=device)

    with torch.no_grad():
        samples = diffusion.sample(cond_tensor, steps=sampling_steps)

    normalized_path_loss = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return denormalize_path_loss(normalized_path_loss, stats)


def save_pathloss_numpy(path_loss_prediction: np.ndarray, output_path: str) -> None:
    if path_loss_prediction.ndim == 2:
        path_loss_prediction = path_loss_prediction[:, :, np.newaxis]
    np.save(output_path, path_loss_prediction.astype(np.float32))
    print(f"Saved path-loss tensor to: {output_path}")
    print(f"Shape: {path_loss_prediction.shape}")
    print(f"Value range: {path_loss_prediction.min():.2f} to {path_loss_prediction.max():.2f} dB")


def save_pathloss_png(
    path_loss_prediction: np.ndarray,
    output_path: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    title: str = "Predicted Path Loss",
) -> None:
    path_loss_2d = path_loss_prediction.squeeze(-1) if path_loss_prediction.ndim == 3 else path_loss_prediction
    if vmin is None:
        vmin = float(path_loss_2d.min())
    if vmax is None:
        vmax = float(path_loss_2d.max())

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(path_loss_2d, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax), origin="upper")
    colorbar = plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Path Loss (dB)")
    axis.set_title(title)
    axis.set_xlabel("X (grid cells)")
    axis.set_ylabel("Y (grid cells)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved path-loss visualization to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the path-loss diffusion model")
    parser.add_argument("--checkpoint", default="./checkpoints/model_final.pt", help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Path to conditioning tensor (.npy, shape H,W,2)")
    parser.add_argument(
        "--stats-file",
        default=None,
        help=f"Path to normalization stats JSON; defaults to checkpoint stats or {DEFAULT_STATS_FILENAME}",
    )
    parser.add_argument("--output-dir", default="../pathloss_visualizations", help="Directory for outputs")
    parser.add_argument("--output-name", default="prediction", help="Base name for output files")
    parser.add_argument("--sampling-steps", type=int, default=50, help="Reverse diffusion steps")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum plot value")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum plot value")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    model, payload = load_model(args.checkpoint, device=device)
    stats_payload = payload.get("normalization_stats")

    if args.stats_file is not None:
        stats = load_stats(Path(args.stats_file))
    elif stats_payload is not None:
        from common.pathloss import PathLossStats  # noqa: E402

        stats = PathLossStats.from_dict(stats_payload)
    else:
        stats = load_stats(Path(DEFAULT_STATS_FILENAME))
    validate_stats(stats)

    cond_input = np.load(args.input).astype(np.float32)
    expected_channels = model.cond_channels
    if cond_input.ndim != 3 or cond_input.shape[2] != expected_channels:
        raise ValueError(f"Expected input shape (H, W, {expected_channels}), got {cond_input.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    path_loss_prediction = run_inference(
        model=model,
        cond_input=cond_input,
        stats=stats,
        device=device,
        sampling_steps=args.sampling_steps,
    )

    npy_path = os.path.join(args.output_dir, f"{args.output_name}.npy")
    png_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    save_pathloss_numpy(path_loss_prediction, npy_path)
    save_pathloss_png(path_loss_prediction, png_path, vmin=args.vmin, vmax=args.vmax)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Path-loss tensor: {npy_path}")
    print(f"Path-loss visualization: {png_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
