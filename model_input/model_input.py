# multi channel tensor input for nn model
import os
from matplotlib import pyplot as plt

# we want to have a function that takes in the path to the scene directory and returns the multi channel tensor input for the nn model
import numpy as np
from pathlib import Path
import json
from skimage.transform import resize

def scene_to_tensor_simple(scene_dir: str, distance_normalize=True, freq_log_scale=True):
    """
    Converts a scene folder into an ML-ready input tensor and RSS target tensor.

    Parameters
    ----------
    scene_dir : str or Path
        Path to a scene folder containing:
        - elevation.npy
        - rss_values*.npy
        - tx_metadata.json
    distance_normalize : bool
        Whether to normalize the distance map to 0-1 (optional)
    freq_log_scale : bool
        Whether to use log10(frequency in GHz) for the frequency channel

    Returns
    -------
    input_tensor : np.ndarray
        H x W x 3 tensor:
        [elevation, distance, frequency]
    target_tensor : np.ndarray
        H x W RSS map
    """
    scene_path = Path(scene_dir)

    # Load elevation
    elevation_files = list(scene_path.glob("elevation*.npy"))
    if not elevation_files:
        raise RuntimeError(f"No elevation.npy found in {scene_dir}")
    elevation = np.load(elevation_files[0])  # H x W 


    rss_files = list(scene_path.glob("rss_values*.npy"))
    if not rss_files:
        raise RuntimeError(f"No rss_values*.npy found in {scene_dir}")
    rss = np.load(rss_files[0])  # H x W

    #resize elevation map to shape of RSS map if needed
    H, W = rss.shape[1], rss.shape[2]
    elevation_rs = resize(
    elevation,
    (H, W),
    order=0,                # nearest neighbor
    preserve_range=True,
    anti_aliasing=False
    ).astype(np.float32)
    

    # Load TX metadata
    metadata_file = scene_path / "tx_metadata.json"
    if not metadata_file.exists():
        raise RuntimeError(f"No tx_metadata.json found in {scene_dir}")
    with open(metadata_file, "r") as f:
        tx_meta = json.load(f)

    tx_pos = np.array(tx_meta["tx_position"])  # [x, y, z]
    frequency_hz = tx_meta["frequency"]

   # H, W = elevation_rs.shape

    # Distance map (XY plane)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    distance_map = np.sqrt((xx - tx_pos[0])**2 + (yy - tx_pos[1])**2)
    if distance_normalize:
        distance_map = distance_map / distance_map.max()  # scale 0-1

    # Frequency channel
    if freq_log_scale:
        freq_val = np.log10(frequency_hz / 1e9)  # log10(GHz)
    else:
        freq_val = float(frequency_hz)
    freq_map = np.full_like(elevation_rs, freq_val, dtype=np.float32)

    # Stack input channels: elevation, distance, frequency
    input_tensor = np.stack([
        elevation_rs.astype(np.float32),
        distance_map.astype(np.float32),
        freq_map.astype(np.float32)
    ], axis=-1)


    # Target tensor: RSS

    # Convert rss from dB to dBm
    rss_dbm = 10 * np.log10(rss) + 30
    target_tensor = rss_dbm.astype(np.float32)

    
    # Permute target tensor to H x W X C
    target_tensor = np.transpose(target_tensor, (1, 2, 0))  # H x W x C


    #model input visualization: elevation and rss side by side
    elev = elevation_rs   # (40,40)
    rss  = rss_dbm.squeeze()   # (40,40)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Elevation")
    plt.imshow(elev)
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title("RSS")
    plt.imshow(rss)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title("Elevation edges over RSS")
    plt.imshow(rss, alpha=0.8)
    plt.imshow(elev, alpha=0.35)
    plt.colorbar()

    plt.show()

    return input_tensor, target_tensor



    
# Example usage:

scene_folder = Path("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0").resolve()
print(scene_folder.exists())  # Should print True
input_tensor, target_tensor = scene_to_tensor_simple(scene_folder)
print("Input tensor shape:", input_tensor.shape)
print("Target tensor shape:", target_tensor.shape)    

# Create automation here to convert all scenes in a root directory