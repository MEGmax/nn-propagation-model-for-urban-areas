#compare material scenes
import numpy as np
import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies, transform_mesh
import matplotlib.pyplot as plt

#rss_concrete = np.load("nn-propagation-model-for-urban-areas/testing/colab_test/material_test/material_comparison/concrete.npy", allow_pickle=True)
rss_glass = np.load("nn-propagation-model-for-urban-areas/testing/colab_test/material_test/material_comparison/glass.npy", allow_pickle=True)
rss_concrete = np.load("nn-propagation-model-for-urban-areas/testing/colab_test/material_test/material_comparison/concrete.npy", allow_pickle=True)


# Convert to dBm
with np.errstate(divide="ignore"):
    rss_concrete_dbm = 10 * np.log10(rss_concrete) + 30
    rss_glass_dbm = 10 * np.log10(rss_glass) + 30

# Plot RSS with Sionna defaults
plt.figure()
plt.imshow(
    rss_concrete_dbm[0],
    origin="lower",   # IMPORTANT: matches Sionna
    cmap="viridis"    # Sionna default
)
plt.colorbar(label="Received signal strength (RSS) [dBm]")
plt.xlabel("Cell index (X-axis)")
plt.ylabel("Cell index (Y-axis)")
plt.title("RSS")
#plt.show()
plt.close()

# Calculate Difference 

print(rss_concrete_dbm.min(), rss_concrete_dbm.max())

# Calculate Mean Absolute Error (MAE)
mae_db = np.mean(np.abs(rss_concrete - rss_glass))
print(f"Mean Absolute Error (MAE) between concrete and glass: {mae_db:.2f} (linear scale)")
mae_dbm = np.mean(np.abs(rss_concrete_dbm - rss_glass_dbm))
print((rss_concrete_dbm - rss_glass_dbm)[0])
print(f"Mean Absolute Error (MAE) between concrete and glass: {mae_dbm:.2f} dBm")

# Calculate dB difference with rss_concrete - rss_glass
diff = rss_concrete_dbm - rss_glass_dbm  # in dBm
print("Difference (concrete - glass) stats:")

plt.figure()
plt.imshow(diff[0], origin="lower", cmap="coolwarm")
plt.colorbar(label="RSS difference [dBm]")
plt.title("RSS Difference Map (Map1 − Map2)")
plt.xlabel("Cell index (X)")
plt.ylabel("Cell index (Y)")
plt.show()

