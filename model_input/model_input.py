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

#normalization constant for elevation, use config file after
H_MAX = 8.0
Frequency_max = 5.3e9

def data_repo_setup(PARENT_DIR, OUTPUT_DIR_INPUT, OUTPUT_DIR_TARGET):
    if str(PARENT_DIR / "scene_generation") not in sys.path:
        sys.path.append(str(PARENT_DIR / "scene_generation"))
    from studio_setup import repository_setup

    repository_setup(OUTPUT_DIR_INPUT)
    repository_setup(OUTPUT_DIR_TARGET)

#creates a directory called data split into training, validation and testing that stores input and target tensors used as input for the nn model
def createDataTensorsFromScenes(scenes_root: Path, output_dir_input: Path, output_dir_target: Path):
    scene_dirs = [d for d in scenes_root.iterdir() if d.is_dir()]
    # os.makedirs(output_dir_input, exist_ok=True)
    # os.makedirs(output_dir_target, exist_ok=True)
    for scene_dir in scene_dirs:
        input_tensor, target_tensor = scene_to_tensor_simple(scene_dir)
        scene_name = scene_dir.name
        np.save(output_dir_input / f"{scene_name}_input.npy", input_tensor)
        np.save(output_dir_target / f"{scene_name}_target.npy", target_tensor)
    print(f"Saved tensors for {len(scene_dirs)} scenes to {output_dir_input} and {output_dir_target}")

def scene_to_tensor_simple(scene_dir: str, freq_log_scale=True):
    """
    Converts a scene folder into an ML-ready input tensor and RSS target tensor.

    Parameters
    ----------
    scene_dir : str or Path
        Path to a scene folder containing:
        - elevation.npy
        - pathloss_values*.npy
        - tx_metadata.json
    distance_normalize : bool
        Whether to normalize the distance map to 0-1 (optional)
    freq_log_scale : bool
        Whether to use log10(frequency in GHz) for the frequency channel

    Returns
    -------
    input_tensor : np.ndarray
        elevation_rs.astype(np.float32),
        distance_map.astype(np.float32),
        H x W x 2 tensor:
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

    """
    rss_files = list(scene_path.glob("rss_values*.npy"))
    if not rss_files:
        raise RuntimeError(f"No rss_values*.npy found in {scene_dir}")
    rss = np.load(rss_files[0])  # H x W
    """
    pathloss_files = list(scene_path.glob("pathloss_values*.npy"))
    if not pathloss_files:
        raise RuntimeError(f"No pathloss_values*.npy found in {scene_dir}")
    pathloss = np.load(pathloss_files[0])  # H x W

    # Resize elevation map to shape of RSS map if needed
    H, W = pathloss.shape[0], pathloss.shape[1]
    print(f"Original elevation shape: {elevation.shape}, RSS shape: {pathloss.shape}")
    elevation_rs = resize(
        elevation,
        (H, W),
        order=0,                # nearest neighbor
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.float32)

    # Normalize elevation to 0-1
    #elevation_norm = np.tanh(elevation_rs / H_MAX)  # simple normalization, can be tuned
    elevation_norm = elevation_rs / H_MAX
    elevation_norm = elevation_norm * 2 - 1
    # Print range of normalized elevation for debugging
    #print(f"Elevation range after normalization: min={elevation_norm.min()}, max={elevation_norm.max()}")

    # Load TX metadata
    metadata_file = scene_path / "tx_metadata.json"
    if not metadata_file.exists():
        raise RuntimeError(f"No tx_metadata.json found in {scene_dir}")
    with open(metadata_file, "r") as f:
        tx_meta = json.load(f)

    # Retrieve TX position and frequency
    tx_pos = np.array(tx_meta["tx_position"])  # [x, y, z]
    frequency_hz = tx_meta["frequency"]

    # Distance map (XY plane)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    # Shift origin to center: tx position is written relative to center, so we need to shift the grid as well
    xx_centered = xx - W / 2
    yy_centered = yy - H / 2

    # Assuming tx_pos[0], tx_pos[1] are also relative to center
    distance_map = np.sqrt((xx_centered - tx_pos[0])**2 + (yy_centered - tx_pos[1])**2)

    # Convert frequency → wavelength and encode distance in wavelengths
    c = 3e8
    wavelength = c / frequency_hz
    distance_wavelengths = distance_map / wavelength
    MAX_DISTANCE_WAVELENGTHS = 353.0
    distance_norm = distance_wavelengths / MAX_DISTANCE_WAVELENGTHS
    distance_norm = np.clip(distance_norm, 0, 1)
    distance_norm = distance_norm * 2 - 1

    # Stack input channels: elevation, distance, frequency
    input_tensor = np.stack([
        elevation_norm.astype(np.float32),
        distance_norm.astype(np.float32)
    ], axis=-1)

    # Target tensor: RSS
    # Convert rss from dB to dBm
    # pathloss = 10 * np.log10(pathloss) + 30
    mean_global =  95.39616 # computed across all scenes
    std_global = 300.0    # computed across all scenes

    # simple mean-zero normalization
    #normalized_pathloss = (pathloss - mean_global) / std_global
    #normalized_pathloss = np.tanh(normalized_pathloss)
    # for denormalization: pathloss_recovered = normalized_map_tanh * std_global + mean_global

    #linar normalization 
    PATHLOSS_MIN = 0.0
    PATHLOSS_MAX = 200.0

    normalized_pathloss = (pathloss - PATHLOSS_MIN) / (PATHLOSS_MAX - PATHLOSS_MIN)
    normalized_pathloss = normalized_pathloss * 2.0 - 1.0   # scale to [-1,1]

    target_tensor = normalized_pathloss.astype(np.float32)

    # Permute target tensor to H x W X C
    if target_tensor.ndim == 2:
        target_tensor = np.expand_dims(target_tensor, axis=0)  # C x H x W
    target_tensor = np.transpose(target_tensor, (1, 2, 0))  # H x W x C
    print(f"Input tensor shape: {input_tensor.shape}, Target tensor shape: {target_tensor.shape}")
    return input_tensor, target_tensor


# Create automation here to convert all scenes in a root directory
PARENT_DIR = Path(__file__).resolve().parent.parent
SCENES_ROOT = Path(PARENT_DIR / "scene_generation/automated_scenes").resolve()
print(SCENES_ROOT.exists())  # Should print True

# Choose dataset split: "training", "testing", or "validation"
DATASET_SPLIT = "training"

if DATASET_SPLIT == "training":
    OUTPUT_DIR_INPUT = PARENT_DIR / "model_input/data/training/input"
    OUTPUT_DIR_TARGET = PARENT_DIR / "model_input/data/training/target"
elif DATASET_SPLIT == "testing":
    OUTPUT_DIR_INPUT = PARENT_DIR / "model_input/data/testing/input"
    OUTPUT_DIR_TARGET = PARENT_DIR / "model_input/data/testing/target"
elif DATASET_SPLIT == "validation":
    OUTPUT_DIR_INPUT = PARENT_DIR / "model_input/data/validation/input"
    OUTPUT_DIR_TARGET = PARENT_DIR / "model_input/data/validation/target"
else:
    raise ValueError(f"Invalid DATASET_SPLIT: {DATASET_SPLIT}. Must be 'training', 'testing', or 'validation'")

data_repo_setup(PARENT_DIR, OUTPUT_DIR_INPUT, OUTPUT_DIR_TARGET)
createDataTensorsFromScenes(SCENES_ROOT, OUTPUT_DIR_INPUT, OUTPUT_DIR_TARGET)
print(f"Created data tensors in {OUTPUT_DIR_INPUT} and {OUTPUT_DIR_TARGET}")    
