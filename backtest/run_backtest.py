#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.backtest_evaluation import (  # noqa: E402
    backtest_on_dataset,
    load_trained_model,
    plot_evaluation_results,
    print_evaluation_report,
)
from common.pathloss import (
    DEFAULT_STATS_FILENAME,
    PathLossStats,
    load_stats,
    validate_stats,
)  # noqa: E402
from models.diffusion import RadioMapDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backtesting for the path-loss diffusion model"
    )
    parser.add_argument(
        "--checkpoint",
        default="../models/checkpoints/model_final.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--input-dir",
        default="../model_input/data/training/input",
        help="Input tensor directory",
    )
    parser.add_argument(
        "--target-dir",
        default="../model_input/data/training/target",
        help="Target tensor directory",
    )
    parser.add_argument(
        "--stats-file",
        default=f"../model_input/data/training/{DEFAULT_STATS_FILENAME}",
        help="Normalization statistics JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_results",
        help="Directory for metrics and plots",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Inference batch size"
    )
    parser.add_argument(
        "--samples-per-scene", type=int, default=5, help="Ensemble samples per scene"
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=50, help="Reverse diffusion steps"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer samples and fewer diffusion steps",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.samples_per_scene = 3
        args.diffusion_steps = 20
        args.batch_size = 1
        print("[Quick Mode] samples_per_scene=3, diffusion_steps=20, batch_size=1")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, payload = load_trained_model(args.checkpoint, device=device)
    if payload.get("normalization_stats") is not None:
        stats = PathLossStats.from_dict(payload["normalization_stats"])
    else:
        stats = load_stats(Path(args.stats_file))
    validate_stats(stats)

    dataset = RadioMapDataset(args.input_dir, args.target_dir)
    print(f"Loaded {len(dataset)} samples")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    metrics, aggregate = backtest_on_dataset(
        model=model,
        dataset=dataset,
        stats=stats,
        device=device,
        num_samples_per_scene=args.samples_per_scene,
        diffusion_steps=args.diffusion_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    plot_evaluation_results(metrics, aggregate, args.output_dir)
    print_evaluation_report(metrics, aggregate)
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
