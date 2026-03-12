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


    rss_files = list(scene_path.glob("rss_values*.npy"))
    if not rss_files:
        raise RuntimeError(f"No rss_values*.npy found in {scene_dir}")
    rss = np.load(rss_files[0])  # H x W

    # Resize elevation map to shape of RSS map if needed
    H, W = rss.shape[1], rss.shape[2]
    elevation_rs = resize(
        elevation,
        (H, W),
        order=0,                # nearest neighbor
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.float32)

    # Normalize elevation to 0-1
    elevation_norm = np.tanh(elevation_rs / H_MAX)  # simple normalization, can be tuned
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
    distance_norm = distance_wavelengths / 353.0  # simple normalization, can be tuned
    distance_norm = np.tanh(distance_norm * 2 - 1)

    # Create frequency map (constant channel)
    # freq_map = np.full((H, W), np.log10(frequency_hz), dtype=np.float32)
    # if not freq_log_scale:
    #      freq_map = np.full((H, W), frequency_hz / 1e9, dtype=np.float32) # normalize by GHz if not log

    #print(f"Distance range after normalization: min={distance_norm.min()}, max={distance_norm.max()}")

     # Stack input channels: elevation, distance, frequency
    input_tensor = np.stack([
        elevation_norm.astype(np.float32),
        distance_norm.astype(np.float32)
    ], axis=-1)

    # Target tensor: RSS
    # Convert rss from dB to dBm
    rss_dbm = 10 * np.log10(rss) + 30
    target_tensor = rss_dbm.astype(np.float32)

    # Permute target tensor to H x W X C
    target_tensor = np.transpose(target_tensor, (1, 2, 0))  # H x W x C
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
