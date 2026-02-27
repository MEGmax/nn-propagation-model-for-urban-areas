import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sionna.rt
import json

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, RadioMapSolver
from pathlib import Path

import random

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

for i in range(len(list(Path(SCENE_DIR).iterdir()))):
    print(f"Rendering scene {i}...")
    scene = load_scene(SCENE_DIR / f"scene{i}" / f"scene{i}.xml")
    # scene = load_scene(TRUTH_DIR / "ground_truth.xml")

    #set frequency in between 1 GHz and 5.3 GHz
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
        max_depth=32,  # Maximum number of ray scene interactions
        samples_per_tx=10**8,  # If you increase: less noise, but more memory required
        cell_size=(0.15, 0.15),  # Resolution of the radio map
        center=[0, 0, 1.5],  # Center of the radio map
        size=[16, 16],  # Total size of the radio map
        orientation=[0, 0, 0],
    )  # Orientation of the radio map, e.g., could be also vertical

    # save configuration of scene
    scene_config = {
        "frequency": np.array(scene.frequency).item(),
        "tx_position": [np.array(tx.position.x).item(), np.array(tx.position.y).item(), np.array(tx.position.z).item()],
        "tx_orientation": [np.array(tx.orientation.x).item(), np.array(tx.orientation.y).item(), np.array(tx.orientation.z).item()],
    }
    metadata_path = SCENE_DIR / f"scene{i}" / "tx_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(scene_config, f, indent=4)

    # Handle -inf values before saving
    # rss_map = rm.rss
    # rss_map[rss_map == -np.inf] = 99999

    # set -inf values back to minimum for difference calculation
    # rss_map[rss_map == 99999] = rss_map.min()

    np.save(SCENE_DIR / f"scene{i}" / f"rss_values{i}.npy", rm.rss)

    print(f"Done rendering scene {i}")
