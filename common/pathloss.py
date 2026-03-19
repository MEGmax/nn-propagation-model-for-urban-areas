from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_STATS_FILENAME = "normalization_stats.json"


@dataclass(frozen=True)
class PathLossStats:
    frequency_hz: float
    path_loss_mean_db: float
    path_loss_std_db: float
    max_elevation_m: float
    max_electrical_distance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "frequency_hz": float(self.frequency_hz),
            "path_loss_mean_db": float(self.path_loss_mean_db),
            "path_loss_std_db": float(self.path_loss_std_db),
            "max_elevation_m": float(self.max_elevation_m),
            "max_electrical_distance": float(self.max_electrical_distance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PathLossStats":
        required_keys = {
            "frequency_hz",
            "path_loss_mean_db",
            "path_loss_std_db",
            "max_elevation_m",
            "max_electrical_distance",
        }
        missing = sorted(required_keys.difference(payload))
        if missing:
            raise KeyError(f"Missing normalization stats keys: {', '.join(missing)}")

        return cls(
            frequency_hz=float(payload["frequency_hz"]),
            path_loss_mean_db=float(payload["path_loss_mean_db"]),
            path_loss_std_db=float(payload["path_loss_std_db"]),
            max_elevation_m=float(payload["max_elevation_m"]),
            max_electrical_distance=float(payload["max_electrical_distance"]),
        )


def coerce_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Expected a scalar-like value")
    return float(array[0])


def make_stats(
    *,
    frequency_hz: float,
    path_loss_mean_db: float,
    path_loss_std_db: float,
    max_elevation_m: float,
    max_electrical_distance: float,
) -> PathLossStats:
    stats = PathLossStats(
        frequency_hz=float(frequency_hz),
        path_loss_mean_db=float(path_loss_mean_db),
        path_loss_std_db=float(path_loss_std_db),
        max_elevation_m=float(max_elevation_m),
        max_electrical_distance=float(max_electrical_distance),
    )
    validate_stats(stats)
    return stats


def save_stats(stats: PathLossStats, path: str | Path) -> None:
    validate_stats(stats)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats.to_dict(), handle, indent=2)


def load_stats(path: str | Path) -> PathLossStats:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stats = PathLossStats.from_dict(payload)
    validate_stats(stats)
    return stats


def validate_stats(stats: PathLossStats) -> None:
    values = stats.to_dict()
    non_finite = [key for key, value in values.items() if not np.isfinite(value)]
    if non_finite:
        raise ValueError(f"Normalization stats must be finite: {', '.join(non_finite)}")
    if stats.frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if stats.path_loss_std_db <= 0.0:
        raise ValueError("path_loss_std_db must be positive")
    if stats.max_elevation_m <= 0.0:
        raise ValueError("max_elevation_m must be positive")
    if stats.max_electrical_distance <= 0.0:
        raise ValueError("max_electrical_distance must be positive")


def normalize_path_loss(path_loss_db: Any, stats: PathLossStats) -> np.ndarray:
    validate_stats(stats)
    path_loss_db = np.asarray(path_loss_db, dtype=np.float32)
    return ((path_loss_db - stats.path_loss_mean_db) / stats.path_loss_std_db).astype(np.float32)


def denormalize_path_loss(normalized_path_loss: Any, stats: PathLossStats) -> np.ndarray:
    validate_stats(stats)
    normalized_path_loss = np.asarray(normalized_path_loss, dtype=np.float32)
    return (normalized_path_loss * stats.path_loss_std_db + stats.path_loss_mean_db).astype(np.float32)


def normalize_elevation(elevation_m: Any, stats: PathLossStats) -> np.ndarray:
    validate_stats(stats)
    elevation_m = np.asarray(elevation_m, dtype=np.float32)
    return (elevation_m / stats.max_elevation_m).astype(np.float32)


def normalize_electrical_distance(electrical_distance: Any, stats: PathLossStats) -> np.ndarray:
    validate_stats(stats)
    electrical_distance = np.asarray(electrical_distance, dtype=np.float32)
    return (electrical_distance / stats.max_electrical_distance).astype(np.float32)
