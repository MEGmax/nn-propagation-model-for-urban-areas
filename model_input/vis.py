import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load obstruction layer
# ----------------------------
file_path = "/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/obstruction_layer.npy"
obstruction_map = np.load(file_path)


# ----------------------------
# Plot heatmap
# ----------------------------
plt.figure(figsize=(6,6))
plt.imshow(obstruction_map, origin='lower', cmap='viridis')
plt.colorbar(label='Number of Obstructions')
plt.title('Obstruction Layer Heatmap - Scene 0')
plt.xlabel('X grid index')
plt.ylabel('Y grid index')
plt.show()