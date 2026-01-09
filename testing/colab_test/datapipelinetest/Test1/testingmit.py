import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

mi.set_variant("scalar_rgb")  # or "cuda_rgb" if GPU is available

# Load your scene
scene = mi.load_file("/path/to/your_scene.xml")

# Create a render
film = scene.sensors()[0].film
image = mi.render(scene, spp=64)  # spp = samples per pixel

# Convert to numpy array
img = mi.Tensor(image).numpy()

# Visualize
plt.imshow(img)
plt.axis('off')
plt.show()
