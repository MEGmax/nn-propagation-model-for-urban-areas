#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.backtest_evaluation import load_trained_model  # noqa: E402
from common.pathloss import DEFAULT_STATS_FILENAME, PathLossStats, denormalize_path_loss, load_stats, validate_stats  # noqa: E402
from models.diffusion import Diffusion, RadioMapDataset  # noqa: E402


def create_pathloss_figure(
    predicted_db: np.ndarray,
    ground_truth_db: np.ndarray,
    conditioning: np.ndarray,
    scene_name: str,
    output_dir: str,
) -> dict[str, float]:
    error = predicted_db - ground_truth_db
    elevation = conditioning[0]
    electrical_distance = conditioning[1]

    figure = plt.figure(figsize=(15, 10))
    grid = GridSpec(2, 2, figure=figure, hspace=0.3, wspace=0.3)

    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]

    pred_im = axes[0].imshow(predicted_db, cmap="viridis", origin="upper")
    axes[0].set_title("Predicted Path Loss")
    plt.colorbar(pred_im, ax=axes[0], label="dB")

    gt_im = axes[1].imshow(ground_truth_db, cmap="viridis", origin="upper")
    axes[1].set_title("Ground Truth Path Loss")
    plt.colorbar(gt_im, ax=axes[1], label="dB")

    error_im = axes[2].imshow(
        error,
        cmap="RdBu_r",
        origin="upper",
        vmin=-np.max(np.abs(error)),
        vmax=np.max(np.abs(error)),
    )
    axes[2].set_title("Error")
    plt.colorbar(error_im, ax=axes[2], label="dB")

    cond_im = axes[3].imshow(electrical_distance, cmap="magma", origin="upper")
    axes[3].contour(elevation, levels=5, colors="white", linewidths=0.5)
    axes[3].set_title("Conditioning: electrical distance with elevation contours")
    plt.colorbar(cond_im, ax=axes[3], label="normalized")

    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    figure.suptitle(f"{scene_name} | RMSE {rmse:.2f} dB | MAE {mae:.2f} dB | Bias {bias:.2f} dB")
    output_path = os.path.join(output_dir, f"{scene_name}_pathloss_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return {"rmse": rmse, "mae": mae, "bias": bias}


@torch.no_grad()
def generate_visualizations(
    model,
    dataset,
    stats,
    device,
    output_dir="pathloss_visualizations",
    num_scenes=None,
    diffusion_steps=50,
    timesteps=1000,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to(device)
    diffusion = Diffusion(model, timesteps=timesteps, device=device)

    total_scenes = len(dataset) if num_scenes is None else min(num_scenes, len(dataset))
    metrics_list = []

    for scene_idx in tqdm(range(total_scenes), desc="Visualizing"):
        batch = dataset[scene_idx]
        cond_input = batch["input"].unsqueeze(0).to(device)
        gt_target = batch["target"].squeeze(0).cpu().numpy()
        scene_name = dataset.scene_names[scene_idx]

        samples = diffusion.sample(cond_input, steps=diffusion_steps)
        pred_norm = samples[0, 0].cpu().numpy()

        pred_db = denormalize_path_loss(pred_norm, stats)
        gt_db = denormalize_path_loss(gt_target, stats)
        cond_np = cond_input[0].cpu().numpy()
        metrics = create_pathloss_figure(pred_db, gt_db, cond_np, scene_name, output_dir)
        metrics_list.append({**metrics, "scene": scene_name})

    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"{'Scene':<20} {'RMSE (dB)':<15} {'MAE (dB)':<15} {'Bias (dB)':<15}")
    print("-" * 80)
    for metric in metrics_list:
        print(f"{metric['scene']:<20} {metric['rmse']:<15.2f} {metric['mae']:<15.2f} {metric['bias']:<15.2f}")
    print("-" * 80)
    print(f"Saved figures to: {os.path.abspath(output_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize path-loss diffusion predictions")
    parser.add_argument("--checkpoint", default="models/checkpoints/model_final.pt", help="Checkpoint path")
    parser.add_argument("--input-dir", default="model_input/data/training/input", help="Input tensor directory")
    parser.add_argument("--target-dir", default="model_input/data/training/target", help="Target tensor directory")
    parser.add_argument(
        "--stats-file",
        default=f"model_input/data/training/{DEFAULT_STATS_FILENAME}",
        help="Normalization statistics JSON file",
    )
    parser.add_argument("--output-dir", default="pathloss_visualizations", help="Output directory")
    parser.add_argument("--num-scenes", type=int, default=None, help="Number of scenes to visualize")
    parser.add_argument("--diffusion-steps", type=int, default=50, help="Reverse diffusion steps")
    parser.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps from checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, payload = load_trained_model(args.checkpoint, device=device)
    training_config = payload.get("training_config") or {}
    if payload.get("normalization_stats") is not None:
        stats = PathLossStats.from_dict(payload["normalization_stats"])
    else:
        stats = load_stats(Path(args.stats_file))
    validate_stats(stats)

    dataset = RadioMapDataset(args.input_dir, args.target_dir)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    generate_visualizations(
        model=model,
        dataset=dataset,
        stats=stats,
        device=device,
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        diffusion_steps=args.diffusion_steps,
        timesteps=args.timesteps or int(training_config.get("timesteps", 1000)),
    )


if __name__ == "__main__":
    main()
