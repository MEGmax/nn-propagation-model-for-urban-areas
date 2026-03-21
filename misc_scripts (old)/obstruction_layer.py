
# multi channel tensor input for nn model
import os
from tracemalloc import start
from matplotlib import pyplot as plt
import torch
# we want to have a function that takes in the path to the scene directory and returns the multi channel tensor input for the nn model
import numpy as np
from pathlib import Path
import json
from skimage.transform import resize
import sionna.rt as rt
# pip install scikit-image
import mitsuba as mi
import drjit as dr

from model_input import scene_to_tensor_simple
import torch
import numpy as np
#note to self to check sizes and note to self to check if we need to normalize the distance map and if we need to log scale the frequency channel
# also check size of elevation map because obstruction map is size of elecation map for some reason 


def load_inputs(rss_path, tx_metadata_path):
    elevation = np.load(rss_path)

    with open(tx_metadata_path, "r") as f:
        tx_metadata = json.load(f)

    return elevation, tx_metadata

def build_rx_grid(rss, cell_size):
    H, W = rss.shape
    print("elevation shape", rss.shape)
    print(cell_size)
    rx_positions = []

    for i in range(H):
        for j in range(W):
            x = j * cell_size
            y = i * cell_size
            #z = rss[i, j]
            z = 1.5  # Set a fixed height for RX points (e.g., 1.5 meters)
            rx_positions.append([x, y, z])

    print("completed rx(pos)", rx_positions)  # Print the first RX position for verification
    return np.array(rx_positions), H, W  # (H*W, 3)
#returns list 1600


def build_rays(tx_pos, rx_positions):
    """
    tx_pos: array-like [3], TX position (x,y,z)
    rx_positions: array-like [N,3], RX positions

    Returns:
    - rays: mitsuba Rays object (with origins, directions)
    - dsts: distances from TX to each RX
    """
    tx_pos = np.array(tx_pos, dtype=np.float32)
    rx_positions = np.array(rx_positions, dtype=np.float32)

    # Directions from TX -> RX
    directions = rx_positions - tx_pos  # shape (N,3)
    dsts = np.linalg.norm(directions, axis=1)  # distance for each ray
    directions_unit = directions / dsts[:, None]  # normalize

    # Build Mitsuba rays
    N = rx_positions.shape[0]
    rays = mi.Ray3f(
        o=mi.Point3f(tx_pos),
        d=mi.Vector3f(directions_unit.T),  # transpose if Mitsuba expects (3,N)
        maxt=dr.scalar.Float(dsts),
        mint=1e-4
    )

    return rays, dr.scalar.Float(dsts)

def count_obstructions_vectorized(tx_pos, rx_positions, scene):

    rays, dsts = build_rays(tx_pos, rx_positions)

    is_ray_active = dr.ones(dr.auto.ad.Bool, rays.o.shape[1])
    num_obstructions = dr.zeros(dr.auto.ad.Int, rays.o.shape[1])
    obstruction_coeffs = dr.ones(dr.auto.ad.Int, rays.o.shape[1])

    # limit each ray to RX distance
    rays.maxt = dsts

    while dr.any(is_ray_active):
        intersections = scene.ray_intersect_preliminary(rays, is_ray_active)
        hit = is_ray_active & intersections.is_valid()

        if not dr.any(hit):
            break

        num_obstructions += dr.gather(
            dtype=dr.auto.ad.Int,
            source=obstruction_coeffs,
            index=intersections.shape_index,
            active=hit,
        )

        # optionally deactivate rays that have hit max distance
        is_ray_active &= dr.any(intersections.t < dsts)  # mask rays still valid

    return num_obstructions

def count_obstructions(scene, tx_pos, rx_positions):
    mi_scene = scene._scene
    obstruction_counts = []
    zeros_array = np.zeros((40, 40))
    positions_of_intersections = []

    print("TX position:", tx_pos)
    print("Rx positions shape:", rx_positions.shape)
    print("Rx positions sample:", rx_positions)  # Print first 5 RX positions for verification
    for  rx in rx_positions:
        direction = rx - tx_pos
        
        
        #direction vector is location of reciever minus transmitter
        distance = np.linalg.norm(direction)
        direction = direction / distance

        tx_point = mi.Point3f(
            float(tx_pos[0]),
            float(tx_pos[1]),
            float(tx_pos[2])
        )

        dir_vec = mi.Vector3f(
            float(direction[0]),
            float(direction[1]),
            float(direction[2])
        )

        "Create a ray from transmitter to receiver"
        ray = mi.Ray3f(tx_point, dir_vec)
        #ray.mint = dr.scalar.Float(1e-4)
        ray.maxt = dr.scalar.Float(distance - 1e-4)

        #si = mi_scene.ray_intersect(ray)
        count = 0
        si = mi_scene.ray_intersect_preliminary(ray)

        while si.is_valid() and si.t < distance:  # make sure we stop at RX
            count += 1
            ray = si.spawn_ray(direction)
            ray.maxt = distance - si.t  # limit ray length
            si = mi_scene.ray_intersect_preliminary(ray)
       
        #print where the ray intersects the scene
        if np.allclose(rx, rx_positions[439]):
            print("is there obstruction for this rx?", si.is_valid())
            print("Sample RX position:", rx)
            if si.is_valid():
                print("Initial intersection point:", si.p)
                positions_of_intersections.append(si.p)

        #if there is obstruction, we want to spawn a new ray from the intersection point in the same direction and check for another intersection until we reach the receiver
        """while si.is_valid():
            count += 1
            ray = si.spawn_ray(direction)
            si = mi_scene.ray_intersect(ray)
            if np.allclose(rx, rx_positions[439]):
                print("Sample RX position:", rx)
                if si.is_valid():
                    print("Next intersection point:", si.p)
                    positions_of_intersections.append(si.p)"""


        obstruction_counts.append(count)
        if np.allclose(rx, rx_positions[439]):
            print("Sample RX position:", rx)
            print("Obstruction count for this RX:", count)
            
            plt.figure(figsize=(8,6))
            W = 40
            H = 40
            
            # Example: zeros array just for demonstration
            zeros_array = np.zeros((H, W))
            
            plt.imshow(zeros_array, origin='upper', cmap='viridis', extent=[0, W-1, 0, H-1])
            plt.colorbar(label='Number of Obstructions')

            # Flip y-coordinate to match origin='upper'
            sample_y = H - 1 - int(rx_positions[439][1] / 1.0)
            sample_x = int(rx_positions[439][0] / 1.0)
            plt.scatter(sample_x, sample_y, color='red', s=100, label='Sample RX')
            """positions_of_intersections = np.array(positions_of_intersections)
            print("Positions of intersections shape:", positions_of_intersections.shape)
            for(x, y) in positions_of_intersections[:, :2]:
                plt.scatter(x, H - 1 - y, color='blue', s=50, label='Intersection Point')
            """
            plt.scatter(tx_point[0], H - 1 - tx_point[1], color='green', s=100, label='Transmitter')
            plt.xlabel('X (columns)')
            plt.ylabel('Y (rows)')
            plt.title('Obstruction Counts Grid')
            plt.legend()
            plt.show()


    return np.array(obstruction_counts)



def generate_obstruction_map(scene,
                             rss_path,
                             tx_metadata_path,
                             cell_size=1.0,
                             origin=(0, 0)):

    rss, tx_metadata = load_inputs(rss_path=rss_path, tx_metadata_path=
                                         tx_metadata_path)
    
    print("rss shape",rss.shape)

    tx_pos = np.array(tx_metadata["tx_position"])
    #change tx pos to be relative to origin
    H = rss.shape[1]
    W = rss.shape[2]
    center_i = H // 2  # row index
    center_j = W // 2  # column index
    tx_x = center_j * cell_size
    tx_y = center_i * cell_size
    tx_z = 1.5  # height of transmitter
    tx_pos = np.array([tx_x, tx_y, tx_z], dtype=np.float32)

    

    rx_positions, H, W = build_rx_grid(
        rss.squeeze(),  # remove channel dimension if present
        cell_size=cell_size
    )

    print("After building RX grid,  RX position:", rx_positions.shape)  # Print the first RX position for verification
    # origin=origin

    counts = count_obstructions_vectorized(scene,
                                 tx_pos,
                                 rx_positions)

    obstruction_map = counts.reshape(H, W)

    return obstruction_map



scene_xml = "/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/scene0.xml"
scene_dir = "/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0"
input_tensor, target_tensor = scene_to_tensor_simple(scene_dir)[0], scene_to_tensor_simple(scene_dir)[1]
# Distance map is the second channel (index 1)
distance_map = input_tensor[:, :, 0]
rss_grid = target_tensor[:, :, 0]
tx_pos = torch.tensor([0.0, 0.0, 1.5]).float()  # single Tx
#obstruction = compute_obstruction_tensor(scene_dir, rss_map=rss_grid)
scene = rt.load_scene(scene_xml)
obstruction_map = generate_obstruction_map(
    scene=scene,
    rss_path="/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/rss_values0.npy",
    tx_metadata_path="/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/tx_metadata.json",
    cell_size=1.0
)
print("obstruction shape:",obstruction_map.shape)
# Visualize
"""plt.figure(figsize=(8, 6))
plt.imshow(obstruction_map, cmap="gray")
plt.colorbar(label="Building Height")
plt.title("Elevation Channel")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
