import torch
import numpy as np
from PIL import Image

def preprocess_elevation_map(path, max_elevation=250.0):
    """
    Loads a grayscale elevation-map image and returns a tensor of shape [1, H, W]
    where values represent elevation in meters.
    """
    # Load in grayscale
    img = Image.open(path).convert("L")

    # Convert image to array
    arr = np.array(img).astype("float32")  # range 0-255

    # Convert pixel → meters
    elevation = (arr / 255.0) * max_elevation  # scale to actual elevation
    
    # Convert to tensor and add channel dimension
    tensor = torch.from_numpy(elevation).unsqueeze(0)  # shape: [1, H, W]

    return tensor
