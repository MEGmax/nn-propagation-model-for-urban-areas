# this script is to go through all the pathloss maps and find the max pathloss to eventually normalize
# additionally we find the mean

import numpy as np
from pathlib import Path
#Global mean path loss (dB): 95.39616
#Global max pathloss: 300.0

# Root directory containing all scenes
root_dir = Path("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes")


# Get a list of all pathloss files
files = list(root_dir.glob("scene*/pathloss_values*.npy"))

# 1. Find global max
max_pathloss = -np.inf
max_file = None

for file in files:
    data = np.load(file)
    local_max = np.max(data)
    if local_max > max_pathloss:
        max_pathloss = local_max
        max_file = file

# 2. Compute global mean
total_sum = 0.0
total_count = 0

for file in files:
    data = np.load(file)
    total_sum += np.sum(data)
    total_count += data.size

global_mean = total_sum / total_count

print("Global mean path loss (dB):", global_mean)
print("Global max pathloss:", max_pathloss)
print("Found in file:", max_file)