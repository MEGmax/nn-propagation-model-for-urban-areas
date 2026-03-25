from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from skimage.transform import resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pathloss import (  # noqa: E402
    DEFAULT_STATS_FILENAME,
    coerce_scalar,
    denormalize_path_loss,
    load_stats,
    make_stats,
    normalize_electrical_distance,
    normalize_elevation,
    normalize_obstruction,
    normalize_path_loss,
    save_stats,
    validate_stats,
)


DEFAULT_CELL_SIZE_M = (0.15, 0.15)
DEFAULT_MAP_CENTER_M = (0.0, 0.0)


def is_scene_dir(scene_dir: Path) -> bool:
    if not scene_dir.is_dir():
        return False
    has_elevation = any(scene_dir.glob("elevation*.npy"))
    has_pathloss = any(scene_dir.glob("pathloss_values*.npy"))
    return has_elevation and has_pathloss and (scene_dir / "tx_metadata.json").exists()


def load_scene_payload(scene_dir: Path) -> dict:
    elevation_files = sorted(scene_dir.glob("elevation*.npy"))
    pathloss_files = sorted(scene_dir.glob("pathloss_values*.npy"))
    metadata_file = scene_dir / "tx_metadata.json"

    if not elevation_files:
        raise FileNotFoundError(f"No elevation map found in {scene_dir}")
    if not pathloss_files:
        raise FileNotFoundError(f"No pathloss map found in {scene_dir}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"No tx_metadata.json found in {scene_dir}")

    elevation = np.load(elevation_files[0]).astype(np.float32)
    pathloss_db = np.load(pathloss_files[0]).astype(np.float32)
    if pathloss_db.ndim == 3:
        pathloss_db = np.squeeze(pathloss_db)

    # Load obstruction map if present
    obstruction_files = sorted(scene_dir.glob("obstruction_map_scene*.npy"))
    if obstruction_files:
        obstruction = np.load(obstruction_files[0]).astype(np.float32)
        if obstruction.ndim == 3:
            obstruction = np.squeeze(obstruction)
    else:
        obstruction = None

    with metadata_file.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    tx_position = np.asarray(metadata["tx_position"], dtype=np.float32).reshape(-1)
    if tx_position.size < 2:
        raise ValueError(f"tx_position must contain x and y coordinates in {metadata_file}")

    radio_map = metadata.get("radio_map", {})
    cell_size = tuple(radio_map.get("cell_size", DEFAULT_CELL_SIZE_M))
    map_center = tuple(radio_map.get("center", DEFAULT_MAP_CENTER_M))

    return {
        "scene_name": scene_dir.name,
        "elevation_m": elevation,
        "pathloss_db": pathloss_db,
        "obstruction": obstruction,
        "tx_position_m": tx_position[:2],
        "frequency_hz": coerce_scalar(metadata["frequency"]),
        "cell_size_m": (float(cell_size[0]), float(cell_size[1])),
        "map_center_m": (float(map_center[0]), float(map_center[1])),
    }


def resize_elevation(elevation_m: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return resize(
        elevation_m,
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)


def compute_electrical_distance_map(
    height: int,
    width: int,
    tx_position_m: np.ndarray,
    cell_size_m: tuple[float, float],
    map_center_m: tuple[float, float],
    frequency_hz: float,
) -> np.ndarray:
    dx_m, dy_m = cell_size_m
    center_x_m, center_y_m = map_center_m

    x_coords = center_x_m + (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) * dx_m
    y_coords = center_y_m + (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) * dy_m
    xx_m, yy_m = np.meshgrid(x_coords, y_coords)

    metric_distance = np.sqrt((xx_m - tx_position_m[0]) ** 2 + (yy_m - tx_position_m[1]) ** 2)
    wavelength_m = 3e8 / frequency_hz
    return (metric_distance / wavelength_m).astype(np.float32)


def collect_stats(scene_payloads: list[dict]) -> object:
    if not scene_payloads:
        raise ValueError("No scenes found to preprocess")

    frequency_hz = scene_payloads[0]["frequency_hz"]
    total_pixels = 0
    pathloss_sum = 0.0
    pathloss_sq_sum = 0.0
    elevation_sum = 0.0
    elevation_sq_sum = 0.0
    electrical_distance_sum = 0.0
    electrical_distance_sq_sum = 0.0
    obstruction_sum = 0.0
    obstruction_sq_sum = 0.0
    max_elevation_m = 0.0
    max_electrical_distance = 0.0
    max_obstruction = 0.0
    has_obstruction = any(p.get("obstruction") is not None for p in scene_payloads)

    for payload in scene_payloads:
        if not np.isclose(payload["frequency_hz"], frequency_hz):
            raise ValueError(
                "Frequency varies across scenes. Fixed frequency is required when frequency is not a model input."
            )

        pathloss_db = payload["pathloss_db"]
        elevation_rs = resize_elevation(payload["elevation_m"], pathloss_db.shape)
        electrical_distance = compute_electrical_distance_map(
            height=pathloss_db.shape[0],
            width=pathloss_db.shape[1],
            tx_position_m=payload["tx_position_m"],
            cell_size_m=payload["cell_size_m"],
            map_center_m=payload["map_center_m"],
            frequency_hz=payload["frequency_hz"],
        )

        total_pixels += pathloss_db.size
        pathloss_sum += float(pathloss_db.sum())
        pathloss_sq_sum += float(np.square(pathloss_db).sum())
        elevation_sum += float(elevation_rs.sum())
        elevation_sq_sum += float(np.square(elevation_rs).sum())
        electrical_distance_sum += float(electrical_distance.sum())
        electrical_distance_sq_sum += float(np.square(electrical_distance).sum())
        max_elevation_m = max(max_elevation_m, float(elevation_rs.max()))
        max_electrical_distance = max(max_electrical_distance, float(electrical_distance.max()))

        if payload.get("obstruction") is not None:
            obs_rs = resize_elevation(payload["obstruction"], pathloss_db.shape)
            obstruction_sum += float(obs_rs.sum())
            obstruction_sq_sum += float(np.square(obs_rs).sum())
            max_obstruction = max(max_obstruction, float(obs_rs.max()))

    mean_db = pathloss_sum / total_pixels
    variance = max(pathloss_sq_sum / total_pixels - mean_db**2, 1e-8)
    std_db = variance**0.5
    elevation_mean_m = elevation_sum / total_pixels
    elevation_variance = max(elevation_sq_sum / total_pixels - elevation_mean_m**2, 1e-8)
    elevation_std_m = elevation_variance**0.5
    electrical_distance_mean = electrical_distance_sum / total_pixels
    electrical_distance_variance = max(
        electrical_distance_sq_sum / total_pixels - electrical_distance_mean**2,
        1e-8,
    )
    electrical_distance_std = electrical_distance_variance**0.5

    obstruction_mean = obstruction_sum / total_pixels if has_obstruction else None
    obstruction_std = (
        max(obstruction_sq_sum / total_pixels - obstruction_mean**2, 1e-8) ** 0.5
        if has_obstruction else None
    )
    max_obstruction_val = max_obstruction if has_obstruction else None

    return make_stats(
        frequency_hz=frequency_hz,
        path_loss_mean_db=mean_db,
        path_loss_std_db=std_db,
        max_elevation_m=max_elevation_m,
        max_electrical_distance=max_electrical_distance,
        elevation_mean_m=elevation_mean_m,
        elevation_std_m=elevation_std_m,
        electrical_distance_mean=electrical_distance_mean,
        electrical_distance_std=electrical_distance_std,
        obstruction_mean=obstruction_mean,
        obstruction_std=obstruction_std,
        max_obstruction=max_obstruction_val,
    )


def build_scene_tensors(payload: dict, stats) -> tuple[np.ndarray, np.ndarray]:
    pathloss_db = payload["pathloss_db"]
    height, width = pathloss_db.shape

    elevation_rs = resize_elevation(payload["elevation_m"], (height, width))
    electrical_distance = compute_electrical_distance_map(
        height=height,
        width=width,
        tx_position_m=payload["tx_position_m"],
        cell_size_m=payload["cell_size_m"],
        map_center_m=payload["map_center_m"],
        frequency_hz=payload["frequency_hz"],
    )

    input_channels = [
        normalize_elevation(elevation_rs, stats),
        normalize_electrical_distance(electrical_distance, stats),
    ]

    if payload.get("obstruction") is not None:
        obs_rs = resize_elevation(payload["obstruction"], (height, width))
        print("obstriction: ", obs_rs.max())
        print(normalize_obstruction(obs_rs, stats).max())
        input_channels.append(normalize_obstruction(obs_rs, stats))
    elif stats.obstruction_mean is not None:
        print(f"  Warning: scene {payload['scene_name']} missing obstruction map, filling with zeros")
        input_channels.append(np.zeros((height, width), dtype=np.float32))

    input_tensor = np.stack(input_channels, axis=-1).astype(np.float32)
    target_tensor = normalize_path_loss(pathloss_db, stats)[..., np.newaxis].astype(np.float32)
    return input_tensor, target_tensor


def preprocess_scenes(
    scenes_root: Path,
    output_input_dir: Path,
    output_target_dir: Path,
    stats_path: Path,
) -> int:
    if is_scene_dir(scenes_root):
        scene_dirs = [scenes_root]
    else:
        scene_dirs = sorted([path for path in scenes_root.iterdir() if is_scene_dir(path)])
        skipped_dirs = sorted([path.name for path in scenes_root.iterdir() if path.is_dir() and not is_scene_dir(path)])
        if skipped_dirs:
            preview = ", ".join(skipped_dirs[:5])
            suffix = "..." if len(skipped_dirs) > 5 else ""
            print(f"Skipping {len(skipped_dirs)} non-scene directories under {scenes_root}: {preview}{suffix}")
    scene_payloads = [load_scene_payload(scene_dir) for scene_dir in scene_dirs]
    stats = collect_stats(scene_payloads)
    save_stats(stats, stats_path)

    output_input_dir.mkdir(parents=True, exist_ok=True)
    output_target_dir.mkdir(parents=True, exist_ok=True)

    for payload in scene_payloads:
        print("for scene $ ", payload)
        input_tensor, target_tensor = build_scene_tensors(payload, stats)
        scene_name = payload["scene_name"]
        np.save(output_input_dir / f"{scene_name}_input.npy", input_tensor)
        np.save(output_target_dir / f"{scene_name}_target.npy", target_tensor)

    return len(scene_payloads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized path-loss tensors with elevation and electrical-distance conditioning."
    )
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=PROJECT_ROOT / "scene_generation" / "automated_scenes",
        help="Directory containing generated scene folders.",
    )
    parser.add_argument(
        "--output-input",
        type=Path,
        default=PROJECT_ROOT / "model_input" / "data" / "training" / "input",
        help="Output directory for input tensors.",
    )
    parser.add_argument(
        "--output-target",
        type=Path,
        default=PROJECT_ROOT / "model_input" / "data" / "training" / "target",
        help="Output directory for target tensors.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=PROJECT_ROOT / "model_input" / "data" / "training" / DEFAULT_STATS_FILENAME,
        help="Output JSON file for shared normalization statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.scenes_root.exists():
        raise FileNotFoundError(f"Scenes root not found: {args.scenes_root}")

    count = preprocess_scenes(
        scenes_root=args.scenes_root,
        output_input_dir=args.output_input,
        output_target_dir=args.output_target,
        stats_path=args.stats_file,
    )
    stats = load_stats(args.stats_file)
    validate_stats(stats)

    print(f"created_pairs={count}")
    print(f"input_dir={args.output_input}")
    print(f"target_dir={args.output_target}")
    print(f"stats_file={args.stats_file}")
    print(
        "normalization="
        f"path_loss(mean={stats.path_loss_mean_db:.6f},std={stats.path_loss_std_db:.6f}),"
        f" elevation(mean={stats.elevation_mean_m:.6f},std={stats.elevation_std_m:.6f},max={stats.max_elevation_m:.6f}),"
        f" electrical_distance(mean={stats.electrical_distance_mean:.6f},std={stats.electrical_distance_std:.6f},max={stats.max_electrical_distance:.6f})"
        + (f",\n obstruction(mean={stats.obstruction_mean:.6f},std={stats.obstruction_std:.6f},max={stats.max_obstruction:.6f})"
           if stats.obstruction_mean is not None else "")
    )


if __name__ == "__main__":
    main()