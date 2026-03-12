# multi channel tensor input for nn model
import os
import sys
from tracemalloc import start
from matplotlib import pyplot as plt
import torch
# we want to have a function that takes in the path to the scene directory and returns the multi channel tensor input for the nn model
import numpy as np
from pathlib import Path
import json
from skimage.transform import resize
import sionna.rt as rt
from math import tanh

#compare boolean image vs elevation image

boolean = np.load("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes_masks/scene0_mask.npy")
elevation = np.load("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/elevation.npy")
#compare with normalized elevation
elevation_norm = np.tanh(elevation / 8.0 )  # simple normalization, can be tuned
# compare with rss map 
rss_map = np.load("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/rss_values0.npy").squeeze()
rss_map = 10 * np.log10(rss_map) + 30

#print sizes of all
print(f"Boolean mask shape: {boolean.shape}, min: {boolean.min()}, max: {boolean.max()}")
print(f"Elevation shape: {elevation.shape}, min: {elevation.min()}, max: {elevation.max()}")
print(f"Normalized Elevation shape: {elevation_norm.shape}, min: {elevation_norm.min()}, max: {elevation_norm.max()}")
print(f"RSS map shape: {rss_map.shape}, min: {rss_map.min()}, max: {rss_map.max()}")

elevation = resize(
    elevation,
    (40, 40),
    order=0,                # nearest neighbor
    preserve_range=True,
    anti_aliasing=False
).astype(np.float32)


"""
#plot the two images side by side
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(boolean, cmap='gray')
axs[0].set_title("Boolean Mask")
axs[0].axis('off')

axs[1].imshow(elevation, cmap='gray')
axs[1].set_title("Elevation")
axs[1].axis('off')

axs[2].imshow(rss_map, cmap='coolwarm')
axs[2].set_title("Normalized Elevation")
axs[2].axis('off')


plt.show()
"""

# Plot overlay
plt.figure(figsize=(6, 6))
plt.imshow(rss_map, cmap='viridis', origin='lower')  # RSS as base
plt.imshow(elevation, cmap='gray', alpha=0.4, origin='lower')  # Boolean on top with transparency
plt.colorbar(label="RSS (dB)")
plt.title("RSS Map Overlayed with Boolean Mask")
plt.axis('off')
plt.show()
