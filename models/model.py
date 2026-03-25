from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pathloss import DEFAULT_STATS_FILENAME, load_stats, validate_stats  # noqa: E402
from diffusion import RadioMapDataset, TimeCondUNet, train  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the path-loss diffusion model")
    parser.add_argument(
        "--input-dir",
        default="../model_input/data/training/input",
        help="Path to normalized input tensors",
    )
    parser.add_argument(
        "--target-dir",
        default="../model_input/data/training/target",
        help="Path to normalized target tensors",
    )
    parser.add_argument(
        "--stats-file",
        default=f"../model_input/data/training/{DEFAULT_STATS_FILENAME}",
        help="Path to the shared normalization statistics JSON file",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint cadence")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_70scenes", help="Checkpoint directory")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not os.path.isdir(args.target_dir):
        raise FileNotFoundError(f"Target directory not found: {args.target_dir}")

    stats = load_stats(Path(args.stats_file))
    validate_stats(stats)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU name: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    dataset = RadioMapDataset(args.input_dir, args.target_dir)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    cond_channels, target_channels = dataset.sample_spec()
    if cond_channels != 3 or target_channels != 1:
        raise ValueError(
            f"Expected dataset channels cond=2,target=1 but found cond={cond_channels},target={target_channels}"
        )

    model = TimeCondUNet(
        noisy_channels=1,
        cond_channels=cond_channels,
        base_ch=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        out_ch=1,
    )
    model.to(device)

    num_params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    logger.info("Loaded %d samples", len(dataset))
    logger.info("Model parameters: %.2fM", num_params)
    logger.info("Normalization stats: %s", stats.to_dict())

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "timesteps": args.timesteps,
        "save_every": args.save_every,
        "num_workers": args.num_workers,
        "device": device,
        "input_dir": args.input_dir,
        "target_dir": args.target_dir,
        "stats_file": args.stats_file,
    }

    train(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        timesteps=args.timesteps,
        save_every=args.save_every,
        out_dir=args.checkpoint_dir,
        normalization_stats=stats.to_dict(),
        num_workers=args.num_workers,
        training_config=training_config,
    )

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Samples: {len(dataset)}")
    print(f"Conditioning channels: {cond_channels}")
    print(f"Target channels: {target_channels}")
    print(f"Parameters: {num_params:.2f}M")
    print(f"Device: {device}")
    print(f"Stats file: {args.stats_file}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
