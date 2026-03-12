import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

if "DRJIT_LIBLLVM_PATH" not in os.environ:
    for llvm_path in (
        "/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
        "/usr/local/opt/llvm/lib/libLLVM.dylib",
    ):
        if Path(llvm_path).exists():
            os.environ["DRJIT_LIBLLVM_PATH"] = llvm_path
            break

import sionna.rt
import json
import argparse

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, RadioMapSolver

import random


def _to_numpy(array_like):
    """Convert tensors/array-like objects from Sionna/Mitsuba to NumPy arrays."""
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.array(array_like)

# pip install sionna
# pip install tensorflow
# pip install matplotlib
# Need python 3.12 and sionna version 1.*

# If you need to have llvm first in your PATH, run:
# echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc

# For compilers to find llvm you may need to set:
#   export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
#   export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"

BASE_DIR = Path(__file__).resolve().parent
SCENE_DIR = BASE_DIR / "automated_scenes"
TRUTH_DIR = BASE_DIR / "ground_truth"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Sionna radio maps and save pathloss artifacts")
    parser.add_argument("--scenes-root", type=str, default=str(SCENE_DIR), help="Directory containing scene*/ folders")
    parser.add_argument("--scene-limit", type=int, default=None, help="Optional limit on number of scenes to process")
    parser.add_argument("--samples-per-tx", type=int, default=10**8, help="Sionna RadioMapSolver samples_per_tx")
    parser.add_argument("--max-depth", type=int, default=32, help="Sionna RadioMapSolver max_depth")
    parser.add_argument("--cell-size", type=float, default=0.15, help="Radio map cell size for both axes")
    parser.add_argument("--save-rss", action="store_true", help="Also save RSS map if available from Sionna")
    parser.add_argument("--seed", type=int, default=None, help="Optional Python random seed")
    return parser.parse_args()


def _scene_dirs(scenes_root: Path, scene_limit: int | None) -> list[Path]:
    scene_dirs = sorted([p for p in scenes_root.iterdir() if p.is_dir()])
    if scene_limit is not None:
        return scene_dirs[: max(scene_limit, 0)]
    return scene_dirs


def main() -> None:
    args = _parse_args()
    scenes_root = Path(args.scenes_root)
    if not scenes_root.exists():
        raise RuntimeError(f"Scenes root does not exist: {scenes_root}")

    if args.seed is not None:
        random.seed(args.seed)

    scene_dirs = _scene_dirs(scenes_root, args.scene_limit)
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {scenes_root}")

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        scene_id = scene_name.replace("scene", "")
        scene_xml = scene_dir / f"{scene_name}.xml"
        if not scene_xml.exists():
            raise RuntimeError(f"Expected scene XML not found: {scene_xml}")

        print(f"Rendering {scene_name}...")
        scene = load_scene(scene_xml)
        # scene = load_scene(TRUTH_DIR / "ground_truth.xml")

        # set frequency in between 1 GHz and 5.3 GHz
        scene.frequency = int(random.uniform(1e9, 5.3e9))
        print(f"Set frequency to {scene.frequency} GHz")

        scene.tx_array = PlanarArray(
            num_rows=4,
            num_cols=4,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        )

        # place transmitter at origin
        tx = Transmitter("tx", [0, 0, 1.5], [0.0, 0.0, 0.0])
        scene.add(tx)
        # Prof offered to change camera location to 2 meters?
        my_cam = Camera(position=[0, 0, 30], look_at=tx.position)

        # Instantiate the radio map solver
        rm_solver = RadioMapSolver()
        # Compute radio map using the mesh example
        rm = rm_solver(
            scene,
            max_depth=args.max_depth,  # Maximum number of ray scene interactions
            samples_per_tx=args.samples_per_tx,  # Increase for less Monte Carlo noise
            cell_size=(args.cell_size, args.cell_size),  # Resolution of the radio map
            center=[0, 0, 1.5],  # Center of the radio map
            size=[16, 16],  # Total size of the radio map
            orientation=[0, 0, 0],
        )  # Orientation of the radio map, e.g., could be also vertical

        # save configuration of scene
        scene_config = {
            "frequency": np.array(scene.frequency).item(),
            "tx_position": [np.array(tx.position.x).item(), np.array(tx.position.y).item(), np.array(tx.position.z).item()],
            "tx_orientation": [np.array(tx.orientation.x).item(), np.array(tx.orientation.y).item(), np.array(tx.orientation.z).item()],
            "tx_power_dbm": float(tx.power_dbm),
            "tx_power_w": float(tx.power),
        }
        metadata_path = scene_dir / "tx_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(scene_config, f, indent=4)

        path_gain_linear = _to_numpy(rm.path_gain)

        # Path loss in dB uses path gain convention from Sionna coverage map:
        # path_loss_db = -10 * log10(path_gain_linear).
        path_gain_safe = np.clip(path_gain_linear, 1e-30, None)
        path_loss_db = -10.0 * np.log10(path_gain_safe)

        np.save(scene_dir / f"pathloss_values{scene_id}.npy", path_loss_db.astype(np.float32))

        if args.save_rss and hasattr(rm, "rss"):
            rss_dbm = _to_numpy(rm.rss).astype(np.float32)
            np.save(scene_dir / f"rss_values{scene_id}.npy", rss_dbm)

        print(f"Done rendering {scene_name}")


if __name__ == "__main__":
    main()
