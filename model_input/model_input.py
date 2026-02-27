# multi channel tensor input for nn model
import os
from matplotlib import pyplot as plt

# we want to have a function that takes in the path to the scene directory and returns the multi channel tensor input for the nn model
import numpy as np
from pathlib import Path
import json
from skimage.transform import resize

# pip install scikit-image

#creates a directory called data split into training, validation and testing that stores input and target tensors used as input for the nn model
def createDataTensorsFromScenes(scenes_root: str, output_dir_input: str, output_dir_target: str, output_dir_masks: str):
    scene_dirs = [d for d in Path(scenes_root).iterdir() if d.is_dir()]
    os.makedirs(output_dir_input, exist_ok=True)
    os.makedirs(output_dir_target, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)
    for scene_dir in scene_dirs:
        input_tensor, target_tensor, boolean_mask, rss_null_mask = scene_to_tensor_simple(scene_dir)
        scene_name = scene_dir.name
        np.save(os.path.join(output_dir_input, f"{scene_name}_input.npy"), input_tensor)
        np.save(os.path.join(output_dir_target, f"{scene_name}_target.npy"), target_tensor)
        scene_mask_dir = os.path.join(output_dir_masks, scene_name)
        os.makedirs(scene_mask_dir, exist_ok=True)
        np.save(os.path.join(scene_mask_dir, "boolean_mask.npy"), boolean_mask)
        np.save(os.path.join(scene_mask_dir, "rss_null_mask.npy"), rss_null_mask)
    print(f"Saved tensors for {len(scene_dirs)} scenes to {output_dir_input}, {output_dir_target}, and {output_dir_masks}")

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
        H x W x 2 tensor:
        [elevation, distance]
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

    # Mask indicating 0 for ground and 1 for buildings
    boolean_mask = np.where(elevation_rs > 0, 1, 0).astype(np.float32)  # H x W

    # Boolean mask to indicate invalid RSS values (will be excluded from loss)
    # Mark non-finite entries and non-positive power values (log10 undefined for <= 0)
    rss_null_mask = (~np.isfinite(rss) | (rss <= 0)).astype(np.float32)  # C x H x W

    # Load TX metadata
    metadata_file = scene_path / "tx_metadata.json"
    if not metadata_file.exists():
        raise RuntimeError(f"No tx_metadata.json found in {scene_dir}")
    with open(metadata_file, "r") as f:
        tx_meta = json.load(f)

    tx_pos = np.array(tx_meta["tx_position"])  # [x, y, z]
    frequency_hz = tx_meta["frequency"]

   # H, W = elevation_rs.shape

   # 
   

    # Distance map (XY plane)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    distance_map = np.sqrt((xx - tx_pos[0])**2 + (yy - tx_pos[1])**2)
    #if distance_normalize:
    #   distance_map = distance_map / distance_map.max()  # scale 0-1

    # Convert frequency → wavelength and encode distance in wavelengths
    c = 3e8
    wavelength = c / frequency_hz
    distance_map = distance_map / wavelength
    
    # Stack input channels: elevation, distance
    input_tensor = np.stack([
        elevation_rs.astype(np.float32),
        distance_map.astype(np.float32),
    ], axis=-1)

    # Target tensor: RSS

    # Convert rss from dB to dBm
    rss_dbm = 10 * np.log10(rss) + 30
    target_tensor = rss_dbm.astype(np.float32)

    # Permute target tensor to H x W X C
    target_tensor = np.transpose(target_tensor, (1, 2, 0))  # H x W x C

    # Store masks as H x W x 1 (same spatial shape as prediction/target)
    boolean_mask = np.expand_dims(boolean_mask.astype(np.float32), axis=-1)
    rss_null_mask = np.transpose(rss_null_mask, (1, 2, 0)).astype(np.float32)

    return input_tensor, target_tensor, boolean_mask, rss_null_mask


# Create automation here to convert all scenes in a root directory
scene_folder = Path("../scene_generation/automated_scenes").resolve()
print(scene_folder.exists())  # Should print True
SCENES_ROOT = str(scene_folder)

# Choose dataset split: "training", "testing", or "validation"
DATASET_SPLIT = "training"

if DATASET_SPLIT == "training":
    OUTPUT_DIR_INPUT = "data/training/input"
    OUTPUT_DIR_TARGET = "data/training/target"
    OUTPUT_DIR_MASKS = "data/training/masks"
elif DATASET_SPLIT == "testing":
    OUTPUT_DIR_INPUT = "data/testing/input"
    OUTPUT_DIR_TARGET = "data/testing/target"
    OUTPUT_DIR_MASKS = "data/testing/masks"
elif DATASET_SPLIT == "validation":
    OUTPUT_DIR_INPUT = "data/validation/input"
    OUTPUT_DIR_TARGET = "data/validation/target"
    OUTPUT_DIR_MASKS = "data/validation/masks"
else:
    raise ValueError(f"Invalid DATASET_SPLIT: {DATASET_SPLIT}. Must be 'training', 'testing', or 'validation'")

createDataTensorsFromScenes(SCENES_ROOT, OUTPUT_DIR_INPUT, OUTPUT_DIR_TARGET, OUTPUT_DIR_MASKS)
print(f"Created data tensors in {OUTPUT_DIR_INPUT} and {OUTPUT_DIR_TARGET}")    