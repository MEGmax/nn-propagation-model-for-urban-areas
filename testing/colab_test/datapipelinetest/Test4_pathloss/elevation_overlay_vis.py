# visualize elevation map overlayed on pathloss map for a scene

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.transform import resize

root_dir = Path("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes")
pathloss = np.load(root_dir / "scene0" / "pathloss_values_scene0.npy").squeeze()
elevation = np.load(root_dir / "scene0" / "elevation_map_scene0.npy")

plt.figure(figsize=(6,6))
print(f"Pathloss shape: {pathloss.shape}, Elevation shape: {elevation.shape}")

# Base layer (e.g., path loss)
plt.imshow(pathloss, cmap="viridis", origin="lower")

# Overlay (elevation/buildings)
plt.imshow(elevation, cmap="gray", alpha=0.4, origin="lower")

plt.colorbar(label="Path Loss (dB)")
plt.title("Path Loss with Elevation Overlay")
plt.show()