#!/usr/bin/env python
"""
visualize_scene.py
------------------
Publication-quality visualization of a single scene showing:
  - Raw elevation map
  - Raw Sionna path-loss simulation (dB)
  - Normalized elevation channel (model input ch 0)
  - Normalized electrical-distance channel (model input ch 1)
  - Normalized path-loss target (model target)

Usage
-----
From the project root:

    python visualize/visualize_scene.py \
        --scene-name scene0 \
        --input-dir  model_input/data_50/testing\
        --target-dir model_input/data_50/training/target \
        --stats-file model_input/data_50/training/normalization_stats.json \
        --raw-scenes-root scene_generation/automated_scenes \
        --output-path report_figures/scene0_overview.png

All --raw-scenes-root scenes are optional; if omitted, the raw panels are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pathloss import denormalize_path_loss, load_stats  # noqa: E402


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_colorbar(ax, im, label: str, fontsize: int = 9):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=8)
    return cb


def _imshow(ax, data: np.ndarray, cmap: str, title: str,
            vmin=None, vmax=None, label: str = "", fontsize: int = 10):
    im = ax.imshow(data, cmap=cmap, origin="upper",
                   norm=Normalize(vmin=vmin, vmax=vmax),
                   interpolation="nearest")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=6)
    ax.set_xlabel("x (grid cells)", fontsize=8)
    ax.set_ylabel("y (grid cells)", fontsize=8)
    ax.tick_params(labelsize=7)
    _add_colorbar(ax, im, label, fontsize=9)
    return im


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed(scene_name: str, input_dir: Path, target_dir: Path):
    inp = np.load(input_dir / f"{scene_name}_input.npy").astype(np.float32)
    tgt = np.load(target_dir / f"{scene_name}_target.npy").astype(np.float32)
    # inp: (H, W, 2)  tgt: (H, W, 1)
    return inp, tgt


def load_raw(scene_name: str, raw_scenes_root: Path):
    """Load raw elevation and pathloss from the scene generation folder."""
    scene_dir = raw_scenes_root / scene_name

    # Elevation
    elev_files = sorted(scene_dir.glob("elevation*.npy"))
    elevation = np.load(elev_files[0]).astype(np.float32) if elev_files else None

    # Pathloss
    pl_files = sorted(scene_dir.glob("pathloss_values*.npy"))
    if pl_files:
        pathloss = np.load(pl_files[0]).astype(np.float32)
        if pathloss.ndim == 3:
            pathloss = np.squeeze(pathloss)
    else:
        pathloss = None

    return elevation, pathloss


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(
    scene_name: str,
    input_dir: Path,
    target_dir: Path,
    stats_path: Path,
    raw_scenes_root: Path | None,
    output_path: Path,
    dpi: int = 200,
):
    stats = load_stats(stats_path)
    inp, tgt = load_processed(scene_name, input_dir, target_dir)

    elev_norm = inp[:, :, 0]        # normalized elevation
    elec_norm = inp[:, :, 1]        # normalized electrical distance
    pl_norm   = tgt[:, :, 0]        # normalized path loss
    pl_db     = denormalize_path_loss(pl_norm, stats)  # dB

    has_raw = raw_scenes_root is not None
    elevation_raw, pathloss_raw = (None, None)
    if has_raw:
        elevation_raw, pathloss_raw = load_raw(scene_name, raw_scenes_root)

    # Layout: 2 rows × 3 cols if raw available, else 1 row × 3 cols
    if has_raw:
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        axes_raw  = [fig.add_subplot(gs[0, i]) for i in range(3)]
        axes_proc = [fig.add_subplot(gs[1, i]) for i in range(3)]
    else:
        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.35)
        axes_raw  = []
        axes_proc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    fig.patch.set_facecolor("#f8f8f8")
    title_kw = dict(fontsize=11, fontweight="bold", pad=6)

    # ------------------------------------------------------------------
    # Row 0: Raw data
    # ------------------------------------------------------------------
    if has_raw:
        if elevation_raw is not None:
            _imshow(axes_raw[0], elevation_raw, "terrain",
                    "Raw Elevation", label="Height (m)", **{"vmin": 0})
        else:
            axes_raw[0].text(0.5, 0.5, "No raw elevation", ha="center",
                             va="center", transform=axes_raw[0].transAxes)
            axes_raw[0].set_title("Raw Elevation", **title_kw)

        if pathloss_raw is not None:
            _imshow(axes_raw[1], pathloss_raw, "plasma",
                    "Sionna Path-Loss Simulation",
                    vmin=np.percentile(pathloss_raw, 1),
                    vmax=np.percentile(pathloss_raw, 99),
                    label="Path Loss (dB)")
        else:
            axes_raw[1].text(0.5, 0.5, "No raw pathloss", ha="center",
                             va="center", transform=axes_raw[1].transAxes)
            axes_raw[1].set_title("Sionna Path-Loss Simulation", **title_kw)

        # Histogram of raw path loss values
        if pathloss_raw is not None:
            axes_raw[2].hist(pathloss_raw.flatten(), bins=60,
                             color="#2E86AB", edgecolor="none", alpha=0.85)
            axes_raw[2].axvline(float(np.mean(pathloss_raw)),
                                color="#E84855", lw=1.5, linestyle="--",
                                label=f"Mean = {np.mean(pathloss_raw):.1f} dB")
            axes_raw[2].axvline(float(np.percentile(pathloss_raw, 5)),
                                color="#F4A261", lw=1.2, linestyle=":",
                                label=f"p5 = {np.percentile(pathloss_raw, 5):.1f} dB")
            axes_raw[2].axvline(float(np.percentile(pathloss_raw, 95)),
                                color="#F4A261", lw=1.2, linestyle=":",
                                label=f"p95 = {np.percentile(pathloss_raw, 95):.1f} dB")
            axes_raw[2].set_title("Raw Path-Loss Distribution", **title_kw)
            axes_raw[2].set_xlabel("Path Loss (dB)", fontsize=9)
            axes_raw[2].set_ylabel("Pixel Count", fontsize=9)
            axes_raw[2].tick_params(labelsize=8)
            axes_raw[2].legend(fontsize=8)
            axes_raw[2].set_facecolor("#f0f0f0")
        else:
            axes_raw[2].set_visible(False)

    # ------------------------------------------------------------------
    # Row 1 (or only row): Processed / normalized tensors
    # ------------------------------------------------------------------
    _imshow(axes_proc[0], elev_norm, "terrain",
            "Input Ch 0 — Normalized Elevation",
            vmin=0, vmax=1, label="Normalized [0, 1]")

    _imshow(axes_proc[1], elec_norm, "viridis",
            "Input Ch 1 — Normalized Electrical Distance",
            vmin=0, vmax=1, label="Normalized [0, 1]")

    _imshow(axes_proc[2], pl_db, "plasma",
            "Target — Path Loss (dB, denormalized)",
            vmin=np.percentile(pl_db, 1),
            vmax=np.percentile(pl_db, 99),
            label="Path Loss (dB)")

    # ------------------------------------------------------------------
    # Stats annotation box
    # ------------------------------------------------------------------
    stats_text = (
        f"Scene: {scene_name}\n"
        f"Shape: {inp.shape[0]}×{inp.shape[1]} px\n"
        f"PL mean: {stats.path_loss_mean_db:.1f} dB\n"
        f"PL std:  {stats.path_loss_std_db:.1f} dB\n"
        f"Freq:    {stats.frequency_hz/1e9:.2f} GHz"
    )
    fig.text(0.01, 0.01, stats_text, fontsize=8, va="bottom",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    row_labels = []
    if has_raw:
        row_labels.append((0.5, 0.94, "RAW SCENE DATA"))
    row_labels.append((0.5, 0.47 if has_raw else 0.97, "PREPROCESSED MODEL TENSORS"))

    for x, y, label in row_labels:
        fig.text(x, y, label, ha="center", fontsize=12,
                 fontweight="bold", color="#444444",
                 transform=fig.transFigure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-quality single-scene visualization for reporting."
    )
    parser.add_argument(
        "--scene-name", required=True,
        help="Scene name, e.g. scene0",
    )
    parser.add_argument(
        "--input-dir",
        default="model_input/data/training/input",
        help="Directory of preprocessed input tensors",
    )
    parser.add_argument(
        "--target-dir",
        default="model_input/data/training/target",
        help="Directory of preprocessed target tensors",
    )
    parser.add_argument(
        "--stats-file",
        default="model_input/data/training/normalization_stats.json",
        help="Normalization stats JSON file",
    )
    parser.add_argument(
        "--raw-scenes-root",
        default=None,
        help="Root directory of raw scene folders (optional). If provided, raw elevation and pathloss panels are included.",
    )
    parser.add_argument(
        "--output-path",
        default="report_figures/scene_overview.png",
        help="Output path for the figure",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Figure DPI (200 for reports, 300 for publication)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(
        scene_name=args.scene_name,
        input_dir=Path(args.input_dir),
        target_dir=Path(args.target_dir),
        stats_path=Path(args.stats_file),
        raw_scenes_root=Path(args.raw_scenes_root) if args.raw_scenes_root else None,
        output_path=Path(args.output_path),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
