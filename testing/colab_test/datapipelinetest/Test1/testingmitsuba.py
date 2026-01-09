# Import the library using the alias "mi"
import mitsuba as mi
import matplotlib.pyplot as plt

# Set the variant of the renderer
mi.set_variant('scalar_rgb')
# Load a scene
scene = mi.load_file("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/testing/colab_test/datapipelinetest/test1.xml")



# Render the scene
img = mi.render(scene, spp=256)
# Display the rendered image using matplotlib
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(img ** (1.0 / 2.2));

# Write the rendered image to an EXR file

mi.util.write_bitmap("my_first_render.png", img)
mi.Bitmap(img).write('myfirstrender.exr')