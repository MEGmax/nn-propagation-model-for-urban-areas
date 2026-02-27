import os
import glob
import trimesh
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from pathlib import Path as PathlibPath

# =========================
# GLOBAL PARAMETERS
# =========================

BASE_DIR = PathlibPath(__file__).resolve().parent
SCENES_ROOT = BASE_DIR / "automated_scenes"
METERS_PER_PIXEL = 0.5
H_MAX = 20   # meters (you may want to auto-compute this later)

# =========================
# LOAD & MERGE BUILDINGS
# =========================

def load_all_buildings(meshes_dir):
    """
    Load and merge all building_*.ply meshes in a directory.
    Ignores plane.py and any non-building files.
    """
    ply_files = sorted(glob.glob(os.path.join(meshes_dir, "Building_*.ply")))

    if not ply_files:
        raise RuntimeError(f"No building_*.ply files found in {meshes_dir}")

    meshes = []
    for ply in ply_files:
        mesh = trimesh.load(ply, force="mesh")
        if not mesh.is_empty:
            meshes.append(mesh)

    print(f"Loaded {len(meshes)} building meshes")
    return trimesh.util.concatenate(meshes)

# =========================
# ELEVATION MAP FUNCTION
# =========================

def generate_elevation_map(meshes_dir, output_prefix):
    print(f"Processing buildings in: {meshes_dir}")

    mesh = load_all_buildings(meshes_dir)
    vertices = mesh.vertices

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2] - vertices[:, 2].min()

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    width  = int((xmax - xmin) / METERS_PER_PIXEL) + 1
    height = int((ymax - ymin) / METERS_PER_PIXEL) + 1

    height_map = np.zeros((height, width))

    for face in mesh.faces:
        v0, v1, v2 = vertices[face]

        tri_xy = np.array([
            [(v0[0] - xmin) / METERS_PER_PIXEL, (v0[1] - ymin) / METERS_PER_PIXEL],
            [(v1[0] - xmin) / METERS_PER_PIXEL, (v1[1] - ymin) / METERS_PER_PIXEL],
            [(v2[0] - xmin) / METERS_PER_PIXEL, (v2[1] - ymin) / METERS_PER_PIXEL],
        ])

        tri_z = max(v0[2], v1[2], v2[2])

        minx, miny = np.floor(tri_xy.min(axis=0)).astype(int)
        maxx, maxy = np.ceil(tri_xy.max(axis=0)).astype(int)

        path = Path(tri_xy)

        for iy in range(miny, maxy + 1):
            for ix in range(minx, maxx + 1):
                if 0 <= ix < width and 0 <= iy < height:
                    if path.contains_point((ix + 0.5, iy + 0.5)):
                        height_map[iy, ix] = max(height_map[iy, ix], tri_z)

    normalized = np.clip(height_map / H_MAX, 0.0, 1.0)
    height_uint8 = (normalized * 255).astype(np.uint8)
    print(f"Elevation map size: {height_uint8.shape}, min: {height_uint8.min()}, max: {height_uint8.max()}")
    print(height_map.astype(np.float32))

    # Save raw grayscale
    #plt.imsave(f"{output_prefix}_raw.png", height_uint8, cmap="gray")

    # Save .npy file of the height values normalized
    np.save(f"{output_prefix}.npy", normalized.astype(np.float32))

    #np.save(f"{output_prefix}.npy", height_map.astype(np.float32))


    # Save colored visualization
    plt.figure(figsize=(width , height), dpi=300)
    img = plt.imshow(normalized, origin="lower", cmap="gray", vmin=0, vmax=1)
    cbar = plt.colorbar(img)
    cbar.set_label("Height (m)")
    plt.axis("off")
    plt.title("City Elevation Map")

    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved elevation maps → {output_prefix}.png")

# =========================
# MAIN AUTOMATION LOOP
# =========================

for scene_name in sorted(os.listdir(SCENES_ROOT)):
    scene_path = os.path.join(SCENES_ROOT, scene_name)

    if not os.path.isdir(scene_path):
        continue

    meshes_dir = os.path.join(scene_path, "meshes")

    if not os.path.isdir(meshes_dir):
        print(f"⚠️  Skipping {scene_name}: no meshes directory")
        continue

    output_prefix = os.path.join(scene_path, "elevation")

    try:
        generate_elevation_map(meshes_dir, output_prefix)
    except RuntimeError as e:
        print(f"⚠️  Skipping {scene_name}: {e}")

print("All scenes processed")
