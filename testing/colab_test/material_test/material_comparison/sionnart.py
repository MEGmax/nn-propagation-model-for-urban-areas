import numpy as np
import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies, transform_mesh


scene = load_scene("/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/testing/colab_test/material_test/glass_test/glass_scene.xml")

scene.tx_array = PlanarArray(num_rows = 4,
                             num_cols = 4,
                             vertical_spacing = 0.5,
                             horizontal_spacing = 0.5,
                             pattern = "tr38901",
                             polarization = "V")

scene.rx_array = PlanarArray(num_rows = 1,
                             num_cols = 1,
                             vertical_spacing = 0.5,
                             horizontal_spacing = 0.5,
                             pattern = "iso",
                             polarization = "V")

tx = Transmitter("tx", [-50,31,0.3],[0.0, 0.0, 0.0])
scene.add(tx)

rx = Receiver("rx",[33,-1,6],[0.0, 0.0, 0.0])
scene.add(rx)

# Instantiate the radio map solver
rm_solver = RadioMapSolver()

# Compute radio map using the mesh example

rm = rm_solver(scene,
              max_depth=5,
              samples_per_tx=10**7,
              cell_size=(5, 5))

m_maxdepth = rm_solver(scene,
               max_depth = 15,           # Maximum number of ray scene interactions
               samples_per_tx=10**9 , # If you increase: less noise, but more memory required
               cell_size=(1, 1),      # Resolution of the radio map
               center=[0, 0, 0],      # Center of the radio map
               size=[300, 300],       # Total size of the radio map
               orientation=[0, 0, 0]) # Orientation of the radio map, e.g., could be also vertical

fig = m_maxdepth.show(metric="rss")

np.save("rss_map_db.npy", fig)
