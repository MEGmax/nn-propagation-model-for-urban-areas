import os
import numpy as np
import json
from pathlib import Path
from skimage.transform import resize

# Paths
base_dir = Path("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes")
out_dir = base_dir.parent / "automated_scenes_masks"
out_dir.mkdir(exist_ok=True)

def create_mask(scene_path: Path):
    # Load elevation
    elevation_files = list(scene_path.glob("elevation*.npy"))
    if not elevation_files:
        print(f"Skipping {scene_path.name}: no elevation file")
        return
    elevation = np.load(elevation_files[0])

    # Load RSS
    rss_files = list(scene_path.glob("rss_values*.npy"))
    if not rss_files:
        print(f"Skipping {scene_path.name}: no rss file")
        return
    rss = np.load(rss_files[0])  # (1, H, W)

    # Match shapes
    H, W = rss.shape[1], rss.shape[2]
    elevation_rs = resize(
        elevation,
        (H, W),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.float32)

    # Building mask
    boolean_mask = elevation_rs > 0

    # RSS null mask
    rss_null_mask = np.isinf(rss)
    rss_null_mask = np.squeeze(rss_null_mask, axis=0)

    # Merge masks
    final_mask = np.logical_or(boolean_mask, rss_null_mask).astype(np.uint8)

    # Save
    out_path = out_dir / f"{scene_path.name}_mask.npy"
    np.save(out_path, final_mask)

    print(f"Saved mask for {scene_path.name}")

# Loop over all scene folders
for scene_folder in base_dir.iterdir():
    if scene_folder.is_dir():
        create_mask(scene_folder)

print("Done.")