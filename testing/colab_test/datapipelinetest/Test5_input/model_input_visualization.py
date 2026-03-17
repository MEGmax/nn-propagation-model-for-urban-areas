# This script is to visualize the model input and target tensors for a given scene, to verify that they are being generated correctly.

# multi channel tensor input for nn model
import os
import sys
from tracemalloc import start
from matplotlib import pyplot as plt
import torch
import numpy as np
from pathlib import Path
import json
from skimage.transform import resize
import sionna.rt as rt
from math import tanh

# Load input and target tensors
input_tensor = np.load("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/model_input/data/training/input/scene0_input.npy")
target_tensor = np.load("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/model_input/data/training/target/scene0_target.npy")   

# Visualize input channels
elevation = input_tensor[:, :, 0]
distance  = input_tensor[:, :, 1]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(elevation, cmap='terrain')
plt.title("Elevation Channel")
plt.colorbar(label="Elevation (m)")
plt.subplot(1, 2, 2)
plt.imshow(distance, cmap='viridis')
plt.title("Distance Channel")
plt.colorbar(label="Distance (normalized)")
plt.show()

# Visualize target tensor (RSS map)
pathloss = target_tensor[:, :, 0]  # Assuming target is H x W x
print(f"Target tensor shape: {target_tensor.shape}, min: {target_tensor.min()}, max: {target_tensor.max()}")
plt.figure(figsize=(6, 5))
plt.imshow(pathloss, cmap='inferno')
plt.title("Target Pathloss Map")
plt.colorbar(label="Pathloss (dB)")
plt.show()
