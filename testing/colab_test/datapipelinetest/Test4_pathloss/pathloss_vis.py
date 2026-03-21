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

pathloss = np.load("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/pathloss_values0.npy")

pathloss = pathloss.squeeze()
# Plot overlay
plt.figure(figsize=(6, 6))
plt.imshow(pathloss, cmap='viridis', origin='lower')  # RSS as base
plt.colorbar(label="RSS (dB)")
plt.title("RSS Map Overlayed with Boolean Mask")
plt.axis('off')
plt.show()