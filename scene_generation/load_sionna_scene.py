import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sionna.rt

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, RadioMapSolver
from pathlib import Path

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

for i in range(100):
    scene = load_scene(SCENE_DIR / f"scene{i}" / f"scene{i}.xml")
    # scene = load_scene(TRUTH_DIR / "ground_truth.xml")

    for obj in scene.objects.values():
        print(f"{obj.name}: {obj.radio_material.name}")

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
    my_cam = Camera(position=[0, 0, 30], look_at=tx.position)

    # Instantiate the radio map solver
    rm_solver = RadioMapSolver()
    # Compute radio map using the mesh example
    rm = rm_solver(
        scene,
        max_depth=32,  # Maximum number of ray scene interactions
        samples_per_tx=10**7,  # If you increase: less noise, but more memory required
        cell_size=(0.4, 0.4),  # Resolution of the radio map
        center=[0, 0, 0],  # Center of the radio map
        size=[16, 16],  # Total size of the radio map
        orientation=[0, 0, 0],
    )  # Orientation of the radio map, e.g., could be also vertical

    # Save the rendered image to your project folder
    # scene.render_to_file(camera=my_cam, radio_map=rm, filename=f"automated_scenes/scene{i}/radio_map{i}.png")
    # scene.render_to_file(camera=my_cam, radio_map=rm, filename="ground_truth/radio_map.png")

    scene.render(camera=my_cam, radio_map=rm)
    # 1. Generate the visualization (this creates a Matplotlib figure)
    # We set 'show_plot=False' to prevent windows from popping up in a loop
    rm.show(metric="rss")

    # 2. Save the figure manually using Matplotlib
    plt.savefig(f"automated_scenes/scene{i}/rss_map{i}.png", dpi=300)

    # 3. Close the figure to free up memory (important for loops!)
    plt.close()

    print(f"Done rendering scene {i}"),
