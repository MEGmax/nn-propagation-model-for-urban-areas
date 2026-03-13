import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sionna.rt
import json
import drjit as dr

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, RadioMapSolver
from pathlib import Path
import random
import os
if "DRJIT_LIBLLVM_PATH" not in os.environ:
    for llvm_path in (
        "/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
        "/usr/local/opt/llvm/lib/libLLVM.dylib",
    ):
        if Path(llvm_path).exists():
            os.environ["DRJIT_LIBLLVM_PATH"] = llvm_path
            break

# pip install sionna
# pip install tensorflow
# pip install matplotlib
# Need python 3.12, sionna version 1.*, tensorflow version 2.20.0

# If you need to have llvm first in your PATH, run:
# echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc

# For compilers to find llvm you may need to set:
#   export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
#   export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"



BASE_DIR = Path(__file__).resolve().parent
SCENE_DIR = BASE_DIR / "automated_scenes"
TRUTH_DIR = BASE_DIR / "ground_truth"

def _to_numpy(array_like):
    """Convert tensors/array-like objects from Sionna/Mitsuba to NumPy arrays."""
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.array(array_like)

for i in range(len(list(Path(SCENE_DIR).iterdir()))):
    print(f"Rendering scene {i}...")
    scene = load_scene(SCENE_DIR / f"scene{i}" / f"scene{i}.xml")
    # scene = load_scene(TRUTH_DIR / "ground_truth.xml")

    #set frequency in between 1 GHz and 5.3 GHz
    scene.frequency = int(random.uniform(1e9, 5.3e9))
    #scene.frequency = 5.3e9
    print(f"Set frequency to {scene.frequency} Hz")

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )

    # place transmitter at origin
    tx = Transmitter("tx", [0, 0, 1.5], [0.0, 0.0, 0.0])
    scene.add(tx)

    # place camera 30 meters above center of scene
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

    # additionally calculate pathloss
    path_gain_linear = _to_numpy(rm.path_gain)

    # Path loss in dB uses path gain convention from Sionna coverage map:
    # path_loss_db = -10 * log10(path_gain_linear).
    path_gain_safe = np.clip(path_gain_linear, 1e-30, None)
    path_loss_db = -10.0 * np.log10(path_gain_safe)
    pathloss_path = SCENE_DIR / f"scene{i}" 
    path_loss_db = path_loss_db.squeeze()
    print("shape of path loss map:", path_loss_db.shape)

    np.save(pathloss_path / f"pathloss_values{i}.npy", path_loss_db.astype(np.float32))

    #np.save(SCENE_DIR / f"scene{i}" / f"rss_values{i}.npy", rm.rss)

    # debug statement to save rendered radio map from Sionna
    # img = scene.render(camera=my_cam, radio_map=rm)
    # img.savefig(str(SCENE_DIR / f"scene{i}" / f"rss_render{i}.png"))
    # plt.close(img)

    # image rendering of pathloss map
    save_file = os.path.join(pathloss_path, "pathloss_map.png")
    plt.figure(figsize=(6,6))
    plt.imshow(path_loss_db, origin="lower", cmap="viridis")
    plt.colorbar(label="Path Loss (dB)")
    plt.title("Path Loss Map")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Done rendering scene {i}")
