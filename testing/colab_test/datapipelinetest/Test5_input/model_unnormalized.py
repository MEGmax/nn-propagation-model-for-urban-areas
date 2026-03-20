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

def main():

    BASE_DIR = Path(__file__).resolve().parent
    ROOT_DIR = BASE_DIR.parent.parent.parent.parent
    TARGET_DIR = ROOT_DIR / "model_input" / "data" / "training" / "target"

    for file_path in TARGET_DIR.iterdir():
        if file_path.is_file():
            pathloss = np.load(file_path)
            plt.figure(figsize=(6, 5))
            plt.imshow(pathloss, cmap='inferno')
            plt.title("Pathloss Map")
            plt.colorbar(label="Pathloss (dB)")
            plt.show()

if __name__ == "__main__":
    main()
