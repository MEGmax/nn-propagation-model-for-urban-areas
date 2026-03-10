# Build multi-channel input tensors and training targets from generated scenes.
import argparse
import json
import os
from pathlib import Path

import numpy as np


TARGET_RSS = "rss"
TARGET_PATHLOSS = "pathloss"


def _load_first(scene_path: Path, pattern: str, error_label: str) -> np.ndarray:
    files = list(scene_path.glob(pattern))
    if not files:
        raise RuntimeError(f"No {error_label} found in {scene_path}")
    return np.load(files[0])


def _build_input_tensor(scene_path: Path, h: int, w: int, tx_meta: dict) -> np.ndarray:
    elevation = _load_first(scene_path, "elevation*.npy", "elevation*.npy")
    eh, ew = elevation.shape
    yi = np.clip((np.linspace(0, eh - 1, h)).round().astype(int), 0, eh - 1)
    xi = np.clip((np.linspace(0, ew - 1, w)).round().astype(int), 0, ew - 1)
    elevation_rs = elevation[np.ix_(yi, xi)].astype(np.float32)

    tx_pos = np.array(tx_meta.get("tx_position", [0.0, 0.0, 0.0]), dtype=np.float32)
    tx_x, tx_y = float(tx_pos[0]), float(tx_pos[1])

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx_centered = xx - w / 2.0
    yy_centered = yy - h / 2.0
    distance_map = np.sqrt((xx_centered - tx_x) ** 2 + (yy_centered - tx_y) ** 2).astype(np.float32)

    frequency_hz = float(tx_meta.get("frequency", 2.4e9))
    wavelength = 3e8 / max(frequency_hz, 1.0)
    distance_in_wavelengths = distance_map / wavelength

    return np.stack(
        [
            elevation_rs.astype(np.float32),
            distance_in_wavelengths.astype(np.float32),
        ],
        axis=-1,
    )


def _build_rss_target(scene_path: Path) -> np.ndarray:
    rss_linear_w = _load_first(scene_path, "rss_values*.npy", "rss_values*.npy")
    if rss_linear_w.ndim == 2:
        rss_linear_w = rss_linear_w[None, ...]

    rss_safe = np.clip(rss_linear_w, 1e-30, None)
    rss_dbm = 10.0 * np.log10(rss_safe) + 30.0
    return np.transpose(rss_dbm.astype(np.float32), (1, 2, 0))


def _build_pathloss_target(scene_path: Path, tx_meta: dict) -> np.ndarray:
    pathloss_files = list(scene_path.glob("pathloss_values*.npy"))
    if pathloss_files:
        pathloss_db = np.load(pathloss_files[0])
    else:
        # Compatibility fallback for old scenes that only contain RSS.
        # RSS[dBm] = P_tx[dBm] - PathLoss[dB] => PathLoss[dB] = P_tx[dBm] - RSS[dBm].
        rss_linear_w = _load_first(scene_path, "rss_values*.npy", "rss_values*.npy")
        if rss_linear_w.ndim == 2:
            rss_linear_w = rss_linear_w[None, ...]
        rss_dbm = 10.0 * np.log10(np.clip(rss_linear_w, 1e-30, None)) + 30.0
        tx_power_dbm = float(tx_meta.get("tx_power_dbm", 44.0))
        pathloss_db = tx_power_dbm - rss_dbm

    if pathloss_db.ndim == 2:
        pathloss_db = pathloss_db[None, ...]

    return np.transpose(pathloss_db.astype(np.float32), (1, 2, 0))


def scene_to_tensors(scene_dir: Path, target_kind: str) -> tuple[np.ndarray, np.ndarray]:
    scene_path = Path(scene_dir)
    metadata_file = scene_path / "tx_metadata.json"
    if not metadata_file.exists():
        raise RuntimeError(f"No tx_metadata.json found in {scene_path}")

    with open(metadata_file, "r") as f:
        tx_meta = json.load(f)

    if target_kind == TARGET_RSS:
        target_tensor = _build_rss_target(scene_path)
    elif target_kind == TARGET_PATHLOSS:
        target_tensor = _build_pathloss_target(scene_path, tx_meta)
    else:
        raise ValueError(f"Unknown target kind: {target_kind}")

    h, w = target_tensor.shape[0], target_tensor.shape[1]
    input_tensor = _build_input_tensor(scene_path, h, w, tx_meta)
    return input_tensor, target_tensor


def create_data_tensors_from_scenes(
    scenes_root: str,
    output_dir_input: str,
    output_dir_target: str,
    target_kind: str = TARGET_RSS,
) -> int:
    scene_dirs = sorted([d for d in Path(scenes_root).iterdir() if d.is_dir()])
    os.makedirs(output_dir_input, exist_ok=True)
    os.makedirs(output_dir_target, exist_ok=True)

    target_suffix = "_target.npy" if target_kind == TARGET_RSS else "_pathloss_target.npy"

    generated = 0
    for scene_dir in scene_dirs:
        input_tensor, target_tensor = scene_to_tensors(scene_dir, target_kind=target_kind)
        scene_name = scene_dir.name

        np.save(os.path.join(output_dir_input, f"{scene_name}_input.npy"), input_tensor)
        np.save(os.path.join(output_dir_target, f"{scene_name}{target_suffix}"), target_tensor)
        generated += 1

    print(
        f"Saved {target_kind} tensors for {generated} scenes to "
        f"{output_dir_input} and {output_dir_target}"
    )
    return generated


def _default_output_dirs(dataset_split: str, target_kind: str) -> tuple[str, str]:
    if dataset_split not in {"training", "testing", "validation"}:
        raise ValueError("DATASET_SPLIT must be one of: training, testing, validation")

    output_dir_input = f"data/{dataset_split}/input"
    if target_kind == TARGET_RSS:
        output_dir_target = f"data/{dataset_split}/target"
    else:
        output_dir_target = f"data/{dataset_split}/target_pathloss"

    return output_dir_input, output_dir_target


def main():
    default_scenes_root = Path(__file__).resolve().parent.parent / "scene_generation" / "automated_scenes"

    parser = argparse.ArgumentParser(description="Build input/target tensors for model training")
    parser.add_argument(
        "--scenes-root",
        type=str,
        default=str(default_scenes_root),
        help="Directory containing scene folders",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="training",
        choices=["training", "testing", "validation"],
        help="Dataset split name used for default output folders",
    )
    parser.add_argument(
        "--target-kind",
        type=str,
        default=TARGET_RSS,
        choices=[TARGET_RSS, TARGET_PATHLOSS],
        help="Target semantic to generate",
    )
    parser.add_argument("--output-dir-input", type=str, default=None)
    parser.add_argument("--output-dir-target", type=str, default=None)

    args = parser.parse_args()

    default_input, default_target = _default_output_dirs(args.dataset_split, args.target_kind)
    output_dir_input = args.output_dir_input or default_input
    output_dir_target = args.output_dir_target or default_target

    count = create_data_tensors_from_scenes(
        scenes_root=args.scenes_root,
        output_dir_input=output_dir_input,
        output_dir_target=output_dir_target,
        target_kind=args.target_kind,
    )

    print(f"Created {args.target_kind} data tensors in {output_dir_input} and {output_dir_target}")
    print(f"Scene count: {count}")


if __name__ == "__main__":
    main()
