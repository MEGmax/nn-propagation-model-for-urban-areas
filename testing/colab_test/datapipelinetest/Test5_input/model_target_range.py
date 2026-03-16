import numpy as np
from pathlib import Path
#Global min across all scenes: -0.3532237
#Global max across all scenes: 0.5928264
# Directory containing all target .npy files
target_dir = Path("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/model_input/data/training/target")

# Find all scene target files
target_files = sorted(target_dir.glob("scene*_target.npy"))

global_min = np.inf
global_max = -np.inf

print("Per-scene min and max:")

for file in target_files:
    data = np.load(file)
    local_min = np.min(data)
    local_max = np.max(data)
    
    print(f"{file.name}: min={local_min:.4f}, max={local_max:.4f}")
    
    if local_min < global_min:
        global_min = local_min
    if local_max > global_max:
        global_max = local_max

print("\nGlobal min across all scenes:", global_min)
print("Global max across all scenes:", global_max)