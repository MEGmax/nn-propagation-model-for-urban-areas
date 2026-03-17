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

pathloss = np.load("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/pathloss_values0.npy")
plt.figure(figsize=(6, 5))
plt.imshow(pathloss, cmap='inferno')
plt.title("Pathloss Map")
plt.colorbar(label="Pathloss (dB)")
plt.show()
