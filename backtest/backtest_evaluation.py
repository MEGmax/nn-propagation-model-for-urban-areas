from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pathloss import denormalize_path_loss  # noqa: E402
from models.diffusion import Diffusion, RadioMapDataset, TimeCondUNet  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PathLossEvaluator:
    def compute_metrics(
        self, predicted_db: np.ndarray, ground_truth_db: np.ndarray, scenario_name: str
    ) -> Dict:
        pred = predicted_db.flatten()
        gt = ground_truth_db.flatten()
        error = pred - gt
        abs_error = np.abs(error)

        metrics = {
            "scenario": scenario_name,
            "num_pixels": int(gt.size),
            "rmse_db": float(np.sqrt(np.mean(error**2))),
            "mae_db": float(np.mean(abs_error)),
            "median_error_db": float(np.median(abs_error)),
            "std_error_db": float(np.std(error)),
            "bias_db": float(np.mean(error)),
            "percentile_90_db": float(np.percentile(abs_error, 90)),
            "percentile_75_db": float(np.percentile(abs_error, 75)),
            "percentile_50_db": float(np.percentile(abs_error, 50)),
            "pred_mean_db": float(np.mean(pred)),
            "gt_mean_db": float(np.mean(gt)),
        }

        for threshold in (3.0, 5.0, 10.0):
            metrics[f"within_{int(threshold)}_db"] = float(
                (abs_error <= threshold).mean()
            )

        correlation = np.corrcoef(pred, gt)[0, 1] if gt.size > 1 else np.nan
        metrics["pearson_correlation"] = float(correlation)
        return metrics

    def summarize_batch(self, batch_metrics: List[Dict]) -> Dict:
        summary: Dict[str, float] = {"num_scenarios": len(batch_metrics)}
        if not batch_metrics:
            return summary

        keys = [
            "rmse_db",
            "mae_db",
            "median_error_db",
            "bias_db",
            "within_3_db",
            "within_5_db",
            "within_10_db",
            "pearson_correlation",
        ]
        for key in keys:
            values = [metric[key] for metric in batch_metrics if key in metric]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
        return summary


class DiffusionSampler:
    def __init__(
        self,
        model: TimeCondUNet,
        device: str = "cuda",
        timesteps: int = 1000,
    ):
        self.model = model
        self.device = device
        self.diffusion = Diffusion(model, timesteps=timesteps, device=device)

    @torch.no_grad()
    def sample(
        self, cond_img: torch.Tensor, num_samples: int = 1, steps: int = 50
    ) -> np.ndarray:
        self.model.eval()
        outputs = []
        for _ in range(num_samples):
            sample = self.diffusion.sample(cond_img, steps=steps)
            outputs.append(sample.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)


def load_trained_model(
    checkpoint_path: str, device: str = "cuda"
) -> tuple[TimeCondUNet, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        payload = {
            "model_state_dict": payload,
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

    model = TimeCondUNet(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def backtest_on_dataset(
    model: TimeCondUNet,
    dataset: RadioMapDataset,
    stats,
    device: str = "cuda",
    num_samples_per_scene: int = 5,
    diffusion_steps: int = 50,
    batch_size: int = 2,
    output_dir: str = "./backtest_results",
    timesteps: int = 1000,
) -> Tuple[List[Dict], Dict]:
    os.makedirs(output_dir, exist_ok=True)
    evaluator = PathLossEvaluator()
    sampler = DiffusionSampler(model, device=device, timesteps=timesteps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_metrics: List[Dict] = []

    logger.info("Starting backtesting on %d scenes", len(dataset))
    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        inputs = batch["input"].to(device)
        targets = batch["target"].cpu().numpy()
        predictions = sampler.sample(
            inputs, num_samples=num_samples_per_scene, steps=diffusion_steps
        )

        for scene_idx in range(inputs.shape[0]):
            sample_preds = predictions[
                scene_idx * num_samples_per_scene : (scene_idx + 1)
                * num_samples_per_scene,
                0,
            ]
            ensemble_pred = np.mean(sample_preds, axis=0)
            pred_db = denormalize_path_loss(ensemble_pred, stats)
            target_db = denormalize_path_loss(targets[scene_idx, 0], stats)
            scenario_name = f"batch{batch_idx}_scene{scene_idx}"
            all_metrics.append(
                evaluator.compute_metrics(pred_db, target_db, scenario_name)
            )

    aggregate = evaluator.summarize_batch(all_metrics)
    results_file = Path(output_dir) / "backtest_results.json"
    with results_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "per_scene": all_metrics,
                "aggregate": aggregate,
                "config": {
                    "num_scenes": len(all_metrics),
                    "num_samples_per_scene": num_samples_per_scene,
                    "diffusion_steps": diffusion_steps,
                    "timesteps": timesteps,
                },
            },
            handle,
            indent=2,
        )
    logger.info("Results saved to %s", results_file)
    return all_metrics, aggregate


def plot_evaluation_results(
    metrics: List[Dict], aggregate: Dict, output_dir: str = "./backtest_results"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rmses = [metric["rmse_db"] for metric in metrics]
    maes = [metric["mae_db"] for metric in metrics]
    biases = [metric["bias_db"] for metric in metrics]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(rmses, bins=20, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("RMSE Distribution")
    axes[0, 0].set_xlabel("RMSE (dB)")

    axes[0, 1].hist(maes, bins=20, alpha=0.7, color="green", edgecolor="black")
    axes[0, 1].set_title("MAE Distribution")
    axes[0, 1].set_xlabel("MAE (dB)")

    sorted_maes = np.sort(maes)
    cdf = np.arange(1, len(sorted_maes) + 1) / len(sorted_maes)
    axes[1, 0].plot(sorted_maes, cdf, linewidth=2)
    axes[1, 0].set_title("MAE CDF")
    axes[1, 0].set_xlabel("Absolute Error (dB)")
    axes[1, 0].set_ylabel("CDF")

    axes[1, 1].hist(biases, bins=20, alpha=0.7, color="orange", edgecolor="black")
    axes[1, 1].set_title("Bias Distribution")
    axes[1, 1].set_xlabel("Bias (dB)")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "backtest_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_evaluation_report(metrics: List[Dict], aggregate: Dict) -> None:
    print("\n" + "=" * 80)
    print("BACKTEST REPORT")
    print("=" * 80)
    print(f"Scenes: {aggregate.get('num_scenarios', 0)}")
    print(
        f"RMSE: {aggregate.get('rmse_db_mean', float('nan')):.2f} ± {aggregate.get('rmse_db_std', float('nan')):.2f} dB"
    )
    print(
        f"MAE: {aggregate.get('mae_db_mean', float('nan')):.2f} ± {aggregate.get('mae_db_std', float('nan')):.2f} dB"
    )
    print(f"Median error: {aggregate.get('median_error_db_mean', float('nan')):.2f} dB")
    print(f"Bias: {aggregate.get('bias_db_mean', float('nan')):.2f} dB")
    print(f"Coverage @3 dB: {aggregate.get('within_3_db_mean', 0.0) * 100:.1f}%")
    print(f"Coverage @5 dB: {aggregate.get('within_5_db_mean', 0.0) * 100:.1f}%")
    print(f"Coverage @10 dB: {aggregate.get('within_10_db_mean', 0.0) * 100:.1f}%")
    print(f"Pearson r: {aggregate.get('pearson_correlation_mean', float('nan')):.3f}")
    print("=" * 80 + "\n")
