#!/usr/bin/env python
"""
compare_prediction.py
---------------------
Compare a model prediction (.npy) against a ground truth target (.npy)
and produce a publication-quality figure with side-by-side maps and error analysis.

Usage
-----
    python compare_prediction.py \
        --prediction  pathloss_visualizations_inference/prediction.npy \
        --ground-truth model_input/data/training/target/scene0_target.npy \
        --stats-file   model_input/data/training/normalization_stats.json \
        --output-path  report_figures/comparison_scene0.png

Notes
-----
- prediction.npy  : raw dB values output by inference.py  (H, W) or (H, W, 1)
- ground-truth    : normalized target tensor (H, W, 1) — will be denormalized
  OR raw dB values if --gt-is-raw is passed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pathloss import denormalize_path_loss, load_stats  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colorbar(ax, im, label: str, fontsize: int = 9):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=8)
    return cb


def _imshow(ax, data, cmap, title, vmin=None, vmax=None, label="", norm=None):
    if norm is not None:
        im = ax.imshow(data, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    else:
        im = ax.imshow(data, cmap=cmap, origin="upper", interpolation="nearest",
                       norm=Normalize(vmin=vmin, vmax=vmax))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("x (grid cells)", fontsize=8)
    ax.set_ylabel("y (grid cells)", fontsize=8)
    ax.tick_params(labelsize=7)
    _colorbar(ax, im, label)
    return im


def load_prediction(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr.squeeze(0)
    return arr  # (H, W) in dB


def load_ground_truth(path: Path, stats, gt_is_raw: bool) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr[0]
    if gt_is_raw:
        return arr  # already in dB
    return denormalize_path_loss(arr, stats)  # normalized → dB


def print_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    error = pred - gt
    abs_error = np.abs(error)
    metrics = {
        "RMSE (dB)":         float(np.sqrt(np.mean(error**2))),
        "MAE (dB)":          float(np.mean(abs_error)),
        "Median error (dB)": float(np.median(abs_error)),
        "Bias (dB)":         float(np.mean(error)),
        "Std error (dB)":    float(np.std(error)),
        "Max error (dB)":    float(abs_error.max()),
        "Within 3 dB (%)":   float((abs_error <= 3).mean() * 100),
        "Within 5 dB (%)":   float((abs_error <= 5).mean() * 100),
        "Within 10 dB (%)":  float((abs_error <= 10).mean() * 100),
        "Pearson r":         float(np.corrcoef(pred.flatten(), gt.flatten())[0, 1]),
    }

    print("\n" + "=" * 50)
    print("PREDICTION vs GROUND TRUTH METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<22} {v:>8.3f}")
    print("=" * 50 + "\n")
    return metrics


def make_figure(
    pred: np.ndarray,
    gt: np.ndarray,
    metrics: dict,
    output_path: Path,
    scene_name: str,
    dpi: int = 200,
):
    error = pred - gt
    abs_error = np.abs(error)

    # Shared colour limits for pred/gt panels
    vmin = float(np.percentile(np.concatenate([pred.flatten(), gt.flatten()]), 1))
    vmax = float(np.percentile(np.concatenate([pred.flatten(), gt.flatten()]), 99))
    err_abs_max = float(np.percentile(abs_error, 99))

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#f7f7f7")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    ax_pred  = fig.add_subplot(gs[0, 0])
    ax_gt    = fig.add_subplot(gs[0, 1])
    ax_err   = fig.add_subplot(gs[0, 2])
    ax_aerr  = fig.add_subplot(gs[0, 3])
    ax_hist  = fig.add_subplot(gs[1, 0:2])
    ax_scat  = fig.add_subplot(gs[1, 2:4])

    # ------------------------------------------------------------------
    # Row 0: spatial maps
    # ------------------------------------------------------------------
    _imshow(ax_pred, pred, "plasma", "Prediction", vmin=vmin, vmax=vmax, label="Path Loss (dB)")
    _imshow(ax_gt,   gt,   "plasma", "Ground Truth", vmin=vmin, vmax=vmax, label="Path Loss (dB)")

    # Signed error — diverging colormap centred at 0
    err_norm = TwoSlopeNorm(vmin=-err_abs_max, vcenter=0, vmax=err_abs_max)
    _imshow(ax_err, error, "RdBu_r", "Signed Error (Pred − GT)",
            norm=err_norm, label="Error (dB)")

    _imshow(ax_aerr, abs_error, "hot_r", "Absolute Error",
            vmin=0, vmax=err_abs_max, label="|Error| (dB)")

    # ------------------------------------------------------------------
    # Row 1 left: error histogram
    # ------------------------------------------------------------------
    ax_hist.hist(error.flatten(), bins=80, color="#2E86AB", edgecolor="none", alpha=0.85)
    ax_hist.axvline(0, color="black", lw=1.2, linestyle="--", label="Zero error")
    ax_hist.axvline(metrics["Bias (dB)"], color="#E84855", lw=1.5, linestyle="-",
                    label=f"Bias = {metrics['Bias (dB)']:.2f} dB")
    for sign, pct in [(1, 90), (-1, 90)]:
        val = sign * np.percentile(abs_error, pct)
        ax_hist.axvline(val, color="#F4A261", lw=1.0, linestyle=":",
                        label=f"p{pct} |err| = ±{abs(val):.1f} dB" if sign == 1 else None)
    ax_hist.set_title("Error Distribution (Pred − GT)", fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("Error (dB)", fontsize=9)
    ax_hist.set_ylabel("Pixel Count", fontsize=9)
    ax_hist.tick_params(labelsize=8)
    ax_hist.legend(fontsize=8)
    ax_hist.set_facecolor("#efefef")

    # ------------------------------------------------------------------
    # Row 1 right: scatter plot pred vs gt
    # ------------------------------------------------------------------
    # Subsample for speed if large
    n_pts = pred.size
    if n_pts > 5000:
        idx = np.random.choice(n_pts, 5000, replace=False)
        px, gx = pred.flatten()[idx], gt.flatten()[idx]
    else:
        px, gx = pred.flatten(), gt.flatten()

    ax_scat.scatter(gx, px, s=2, alpha=0.3, color="#2E86AB", rasterized=True)
    lims = [min(vmin, gx.min()), max(vmax, gx.max())]
    ax_scat.plot(lims, lims, "r--", lw=1.2, label="Perfect prediction")
    ax_scat.set_xlim(lims)
    ax_scat.set_ylim(lims)
    ax_scat.set_title(f"Scatter: Pred vs GT  (r = {metrics['Pearson r']:.3f})",
                      fontsize=10, fontweight="bold")
    ax_scat.set_xlabel("Ground Truth (dB)", fontsize=9)
    ax_scat.set_ylabel("Prediction (dB)", fontsize=9)
    ax_scat.tick_params(labelsize=8)
    ax_scat.legend(fontsize=8)
    ax_scat.set_facecolor("#efefef")
    ax_scat.set_aspect("equal")

    # ------------------------------------------------------------------
    # Metrics annotation
    # ------------------------------------------------------------------
    summary = (
        f"Scene: {scene_name}\n"
        f"RMSE:     {metrics['RMSE (dB)']:.2f} dB\n"
        f"MAE:      {metrics['MAE (dB)']:.2f} dB\n"
        f"Bias:     {metrics['Bias (dB)']:.2f} dB\n"
        f"Pearson r:{metrics['Pearson r']:.3f}\n"
        f"@3 dB:    {metrics['Within 3 dB (%)']:.1f}%\n"
        f"@5 dB:    {metrics['Within 5 dB (%)']:.1f}%\n"
        f"@10 dB:   {metrics['Within 10 dB (%)']:.1f}%"
    )
    fig.text(0.01, 0.01, summary, fontsize=8, va="bottom", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.92))

    fig.suptitle(f"Path Loss Prediction Comparison — {scene_name}",
                 fontsize=13, fontweight="bold", y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model prediction against ground truth and plot errors."
    )
    parser.add_argument(
        "--prediction", required=True,
        help="Path to prediction .npy file (raw dB, shape H×W or H×W×1)",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth .npy file",
    )
    parser.add_argument(
        "--stats-file",
        default="model_input/data/training/normalization_stats.json",
        help="Normalization stats JSON (needed to denormalize ground truth)",
    )
    parser.add_argument(
        "--gt-is-raw", action="store_true",
        help="Pass this flag if ground-truth is already in dB (not normalized)",
    )
    parser.add_argument(
        "--output-path",
        default="report_figures/comparison.png",
        help="Output path for the figure",
    )
    parser.add_argument(
        "--scene-name", default="",
        help="Scene label shown in the figure title",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Figure DPI (200 for reports, 300 for publication)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = load_stats(Path(args.stats_file))
    pred = load_prediction(Path(args.prediction))
    gt   = load_ground_truth(Path(args.ground_truth), stats, args.gt_is_raw)

    # Resize if shapes don't match (e.g. prediction was upsampled)
    if pred.shape != gt.shape:
        from skimage.transform import resize as sk_resize
        print(f"Warning: pred shape {pred.shape} != gt shape {gt.shape}, resizing prediction.")
        pred = sk_resize(pred, gt.shape, order=1, preserve_range=True,
                         anti_aliasing=False).astype(np.float32)

    scene_name = args.scene_name or Path(args.prediction).stem
    metrics = print_metrics(pred, gt)
    make_figure(pred, gt, metrics, Path(args.output_path), scene_name, dpi=args.dpi)


if __name__ == "__main__":
    main()

    """Example usage:
    python visualize/compare_prediction.py \
  --prediction pathloss_visualizations_inference_70scene/prediction.npy \
  --ground-truth model_input/data/testing/target/scene69_target.npy \
  --stats-file model_input/data/training/normalization_stats.json \
  --scene-name scene69\
  --output-path report_figures/comparison_scene69.png
  """