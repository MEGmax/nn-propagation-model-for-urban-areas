import cv2
import numpy as np

depth = cv2.imread("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/testing/colab_test/datapipelinetest/depth_map.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# convert distance to height:
# height = camera_Z - depth_distance
camera_Z = 200   # whatever you set
elev = camera_Z - depth

# normalize
norm = (elev - elev.min()) / (elev.max() - elev.min())

# invert (tall buildings → dark)
final = 1 - norm

cv2.imwrite("elev_map.png", (final * 255).astype(np.uint8))
