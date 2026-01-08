import mitsuba as mi
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 1. Mitsuba Setup & Scene Load
# --------------------------
mi.set_variant("scalar_rgb")

scene_path = "/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/testing/colab_test/datapipelinetest/test1.xml"
scene = mi.load_file(scene_path)

# --------------------------
# 2. Output Resolution
# --------------------------
H = 1024
W = 1024

# --------------------------
# 3. Scene Bounds
# --------------------------
bounds = scene.bbox()
min_x, max_x = bounds.min.x, bounds.max.x
min_y, max_y = bounds.min.y, bounds.max.y

# Create sampling grid
x_coords = np.linspace(min_x, max_x, W)
y_coords = np.linspace(min_y, max_y, H)

heightmap = np.zeros((H, W), dtype=np.float32)

# --------------------------
# 4. Raycast Height Sampling
# --------------------------
print("Computing heightmap...")

for i, y in enumerate(y_coords):
    for j, x in enumerate(x_coords):

        # Ray origin: directly above building space
        o = mi.Point3f(x, y, bounds.max.z + 10.0)
        d = mi.Vector3f(0, 0, -1)

        ray = mi.Ray3f(o, d)
        si = scene.ray_intersect(ray)

        if si.is_valid():
            heightmap[i, j] = si.p.z     # hit height
        else:
            heightmap[i, j] = bounds.min.z   # ground height fallback

    if i % 50 == 0:
        print(f"Row {i}/{H} completed")

print("Heightmap extraction complete.")

# 5. Normalize + Invert (very important)
h_min = heightmap.min()
h_max = heightmap.max()

print(f"Height range: {h_min:.2f} to {h_max:.2f}")

# Avoid division by zero 
norm = (heightmap - h_min) / (h_max - h_min + 1e-8)

# Invert so tall buildings → dark 
final = 1 - norm

# --------------------------
# 6. Convert to Image
# --------------------------
elev_img = (final * 255).astype(np.uint8)

# --------------------------
# 7. Save Image
# --------------------------
output_path = "elevation_map.png"
cv2.imwrite(output_path, elev_img)

print(f"Saved elevation map to: {output_path}")
# elev_img = your generated elevation map (2D array)

plt.figure(figsize=(5, 5))
plt.imshow(elev_img, cmap='gray')
plt.colorbar(label="Elevation (m)")   # <-- This adds the scale bar!
plt.title("Elevation Map")
plt.axis('off')
plt.savefig("elevation_map_friday", dpi=300, bbox_inches='tight')
plt.show()
