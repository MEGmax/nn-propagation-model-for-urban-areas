from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sionna.rt import Camera, PlanarArray, RadioMapSolver, Transmitter, load_scene

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SCENE_DIR = BASE_DIR / "automated_scenes"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.toml"

DEFAULT_FREQUENCY_HZ = 3_500_000_000
RADIO_MAP_CELL_SIZE_M = (0.15, 0.15)
RADIO_MAP_CENTER_M = (0.0, 0.0, 1.5)
RADIO_MAP_SIZE_M = (16.0, 16.0)
RADIO_MAP_MAX_DEPTH = 32
RADIO_MAP_SAMPLES_PER_TX = 10**8


def load_frequency_hz() -> float:
    if not CONFIG_PATH.exists():
        return float(DEFAULT_FREQUENCY_HZ)
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    return float(config.get("Frequency", {}).get("frequency", DEFAULT_FREQUENCY_HZ))


def iter_scene_dirs() -> list[Path]:
    return sorted([path for path in SCENE_DIR.iterdir() if path.is_dir() and (path / f"{path.name}.xml").exists()])


def _to_numpy(array_like):
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.asarray(array_like)


def _to_scalar(value) -> float:
    array = _to_numpy(value).reshape(-1)
    if array.size == 0:
        raise ValueError("Expected scalar-like value")
    return float(array[0])


def render_scene(scene_dir: Path, frequency_hz: float) -> None:
    scene_name = scene_dir.name
    scene_xml = scene_dir / f"{scene_name}.xml"
    print(f"Rendering {scene_name} from {scene_xml}...")

    scene = load_scene(scene_xml)
    scene.frequency = int(frequency_hz)
    print(f"Set frequency to {_to_scalar(scene.frequency)} Hz")

    scene.tx_array = PlanarArray(
        num_rows=4,
        num_cols=4,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    tx = Transmitter("tx", [0, 0, 1.5], [0.0, 0.0, 0.0])
    scene.add(tx)

    camera = Camera(position=[0, 0, 30], look_at=tx.position)
    solver = RadioMapSolver()
    radio_map = solver(
        scene,
        max_depth=RADIO_MAP_MAX_DEPTH,
        samples_per_tx=RADIO_MAP_SAMPLES_PER_TX,
        cell_size=RADIO_MAP_CELL_SIZE_M,
        center=RADIO_MAP_CENTER_M,
        size=RADIO_MAP_SIZE_M,
        orientation=[0, 0, 0],
    )

    scene_metadata = {
        "frequency": _to_scalar(scene.frequency),
        "tx_position": [
            _to_scalar(tx.position.x),
            _to_scalar(tx.position.y),
            _to_scalar(tx.position.z),
        ],
        "tx_orientation": [
            _to_scalar(tx.orientation.x),
            _to_scalar(tx.orientation.y),
            _to_scalar(tx.orientation.z),
        ],
        "radio_map": {
            "cell_size": list(RADIO_MAP_CELL_SIZE_M),
            "center": list(RADIO_MAP_CENTER_M),
            "size": list(RADIO_MAP_SIZE_M),
            "orientation": [0.0, 0.0, 0.0],
        },
    }
    with (scene_dir / "tx_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(scene_metadata, handle, indent=2)

    path_gain_linear = _to_numpy(radio_map.path_gain)
    path_gain_safe = np.clip(path_gain_linear, 1e-30, None)
    path_loss_db = -10.0 * np.log10(path_gain_safe)
    np.save(scene_dir / f"pathloss_values_{scene_name}.npy", path_loss_db.astype(np.float32))

    image = scene.render(camera=camera, radio_map=radio_map)
    image.savefig(str(scene_dir / f"pathloss_render_{scene_name}.png"))
    plt.close(image)
    print(f"Finished {scene_name}")


def main() -> None:
    frequency_hz = load_frequency_hz()
    scene_dirs = iter_scene_dirs()
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories with XML files found in {SCENE_DIR}")

    for scene_dir in scene_dirs:
        render_scene(scene_dir, frequency_hz)


if __name__ == "__main__":
    main()
