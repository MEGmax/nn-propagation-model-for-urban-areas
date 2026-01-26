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

    #set frequency in between 900 MHz and 5.3 GHz
    scene.frequency = int(random.uniform(900e6, 5.3e9))
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
        samples_per_tx=10**7,  # If you increase: less noise, but more memory required
        cell_size=(0.4, 0.4),  # Resolution of the radio map
        center=[0, 0, 1.5],  # Center of the radio map
        size=[16, 16],  # Total size of the radio map
        orientation=[0, 0, 0],
    )  # Orientation of the radio map, e.g., could be also vertical

    # Save the rendered image to your project folder
    scene.render_to_file(camera=my_cam, radio_map=rm, filename=f"automated_scenes/scene{i}/radio_map{i}.png")
    # scene.render_to_file(camera=my_cam, radio_map=rm, filename="ground_truth/radio_map.png")

    #scene.render(camera=my_cam, radio_map=rm)
    # 1. Generate the visualization (this creates a Matplotlib figure)
    # We set 'show_plot=False' to prevent windows from popping up in a loop
    #rm.show(metric="rss")

    # 2. Save the figure manually using Matplotlib
    #plt.savefig(f"automated_scenes/scene{i}/rss_map{i}.png", dpi=300)

    # Save .npy file of the rss values
    np.save(f"automated_scenes/scene{i}/rss_values{i}.npy", rm.rss)

    # 3. Close the figure to free up memory (important for loops!)
    # plt.close()

    # Save the scene configuration to a JSON file

    # Ensure folder exists
    scene_path = SCENE_DIR / f"scene{i}"
    scene_path.mkdir(parents=True, exist_ok=True)

    scene_config = {
        "frequency": float(np.array(scene.frequency)),
        "tx_position": [float(np.array(tx.position.x)), float(np.array(tx.position.y)), float(np.array(tx.position.z))],
        "tx_orientation": [float(np.array(tx.orientation.x)), float(np.array(tx.orientation.y)), float(np.array(tx.orientation.z))],
    }
    metadata_path = scene_path / "tx_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(scene_config, f, indent=4)

    print(f"Done rendering scene {i}")
