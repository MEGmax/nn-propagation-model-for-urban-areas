import trimesh
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER PARAMETERS
# =========================

PLY_PATH = "meshes/None_buildings.ply"   # building mesh
OUTPUT_PREFIX = "elevation"
METERS_PER_PIXEL = 0.5                  # spatial resolution
H_MAX = 27.6307                          # tallest building height (meters)


#load mesh
print("Loading mesh...")
mesh = trimesh.load(PLY_PATH)

vertices = mesh.vertices
x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

# Shift so ground starts at 0
z -= z.min()


#create grid

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

width  = int((xmax - xmin) / METERS_PER_PIXEL) + 1
height = int((ymax - ymin) / METERS_PER_PIXEL) + 1

print(f"Grid size: {width} x {height}")

height_map = np.zeros((height, width))



#rasterize (make each face contribute to the height map)

from matplotlib.path import Path

print("Rasterizing elevation (triangle-based)...")

for face in mesh.faces:
    v0, v1, v2 = vertices[face]

    # Project triangle to XY grid
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


#normalize the heights for visualization

normalized = np.clip(height_map / H_MAX, 0.0, 1.0)


#save data for further processing if needed

#np.save(f"{OUTPUT_PREFIX}_heightmap.npy", height_map)

height_uint8 = (normalized * 255).astype(np.uint8)
plt.imsave(
    f"{OUTPUT_PREFIX}_raw.png",
    height_uint8,
    cmap="gray"
)

print("Saved raw height map")

# =========================
# SAVE COLORED MAP WITH LEGEND
# =========================

cmap = plt.cm.viridis
cmap.set_under("white")

plt.figure(figsize=(width / 100, height / 100), dpi=100)

img = plt.imshow(
    normalized,
    origin="lower",
    cmap="gray",
    vmin=0.0,
    vmax=1.0
)
cbar = plt.colorbar(img)
cbar.set_label("Height (m)")
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels([
    "0",
    f"{0.25 * H_MAX:.0f}",
    f"{0.50 * H_MAX:.0f}",
    f"{0.75 * H_MAX:.0f}",
    f"{H_MAX:.0f}"
])

plt.title("City Elevation Map (Orthographic Top View)")
plt.axis("off")

plt.savefig(
    f"{OUTPUT_PREFIX}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved colored elevation map")
print("Done.")
