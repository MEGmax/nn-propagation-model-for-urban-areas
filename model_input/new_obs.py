import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def load_scene_mesh(scene_path):
    meshes_dir = os.path.join(scene_path, "meshes")
    mesh_list = []

    for file in os.listdir(meshes_dir):
        if file.endswith((".obj", ".ply", ".stl")):
            mesh = trimesh.load(os.path.join(meshes_dir, file), force="mesh")
            mesh_list.append(mesh)

    if not mesh_list:
        raise ValueError(f"No mesh files found in {meshes_dir}")
    
    return trimesh.util.concatenate(mesh_list)

# -------------------------
# Compute obstruction layer
# -------------------------
def compute_obstruction_for_scene(scene_path, grid_shape=(40, 40), rx_height=0.0):
    mesh = load_scene_mesh(scene_path)

    # TX above tallest building
    tx_position = np.array([
        0.5*(mesh.bounds[0][0] + mesh.bounds[1][0]),
        0.5*(mesh.bounds[0][1] + mesh.bounds[1][1]),
        mesh.bounds[1][2] + 0.5
    ])

    # RX grid aligned to mesh bounds
    H, W = grid_shape
    x_min, y_min, z_min = mesh.bounds[0]
    x_max, y_max, z_max = mesh.bounds[1]

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, W),
        np.linspace(y_min, y_max, H)
    )
    zz = np.ones_like(xx) * rx_height
    rx_positions = np.stack([xx.flatten(order='C'), yy.flatten(order='C'), zz.flatten(order='C')], axis=1)

    # Rays
    ray_origins = np.repeat(tx_position[None, :], rx_positions.shape[0], axis=0)
    ray_directions = rx_positions - ray_origins
    ray_lengths = np.linalg.norm(ray_directions, axis=1)
    ray_directions = ray_directions / ray_lengths[:, None]

    # Ray intersector
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True
    )

    # Count obstructions per ray
    obstruction_counts = np.zeros(rx_positions.shape[0], dtype=int)
    for i, ray_idx in enumerate(index_ray):
        obstruction_counts[ray_idx] += 1

    obstruction_map = obstruction_counts.reshape(H, W, order='C')
    np.save(os.path.join(scene_path, "obstruction_layer.npy"), obstruction_map)
    
    print(f"[✓] Scene '{os.path.basename(scene_path)}': obstruction_layer.npy saved")
    return obstruction_map

def process_all_scenes(base_path, rx_height=1.5):
    """
    Generate obstruction layers for all scenes in 'automated_scenes'.
    
    base_path: path to 'automated_scenes'
    """

    # loading rss for h W
    file_path = "/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0/rss_values0.npy"
    rss = np.load(file_path)
    # rss is typically 2D: (H, W)
    C, H, W = rss.shape
    grid_shape = (H, W)


    for scene_folder in sorted(os.listdir(base_path)):
        scene_path = os.path.join(base_path, scene_folder)
        if os.path.isdir(scene_path) and scene_folder.startswith("scene"):
            try:
                compute_obstruction_for_scene(scene_path, grid_shape=grid_shape, rx_height=rx_height)
            except Exception as e:
                print(f"[✗] Scene '{scene_folder}' failed: {e}")
base_path = "/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes"
process_all_scenes(base_path)


def visualize_single_ray(
    mesh,
    tx_position,
    rx_position
):
    
    #Visualize a single TX → RX ray and its intersections.
    
    print("Mesh bounds:")
    print(mesh.bounds)
    intersector = mesh.ray

    # ----------------------------
    # Build single ray
    # ----------------------------
    ray_origin = tx_position.reshape(1, 3)

    ray_vector = rx_position - tx_position
    ray_length = np.linalg.norm(ray_vector)
    ray_direction = (ray_vector / ray_length).reshape(1, 3)

    # ----------------------------
    # Intersections
    # ----------------------------
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origin,
        ray_directions=ray_direction,
        multiple_hits=True
    )

    # Filter intersections beyond RX
    hit_vectors = locations - ray_origin[0]
    hit_distances = np.linalg.norm(hit_vectors, axis=1)

    valid_mask = hit_distances <= ray_length
    valid_hits = locations[valid_mask]

    obstruction_count = len(valid_hits) // 2

    print("Number of triangle hits:", len(valid_hits))
    print("Obstruction count (buildings crossed):", obstruction_count)

    # ----------------------------
    # Visualization
    # ----------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh (wireframe style)
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        alpha=0.1
    )

    # Plot TX
    ax.scatter(*tx_position, color='red', s=100, label='TX')

    # Plot RX
    ax.scatter(*rx_position, color='blue', s=100, label='RX')

    # Plot ray segment
    ax.plot(
        [tx_position[0], rx_position[0]],
        [tx_position[1], rx_position[1]],
        [tx_position[2], rx_position[2]],
        linestyle='--',
        label='Ray'
    )

    # Plot intersection points
    if len(valid_hits) > 0:
        ax.scatter(
            valid_hits[:, 0],
            valid_hits[:, 1],
            valid_hits[:, 2],
            s=60,
            label='Intersections'
        )

    ax.set_title(f"Obstructions: {obstruction_count}")
    ax.legend()
    plt.show()
def visualize_rx_selection(mesh, tx_position, obstruction_map, grid_shape=(40,40), rx_index=None):
    """
    Visualize top-down obstruction heatmap and 3D ray for selected RX.
    
    mesh: Trimesh object
    tx_position: np.array([x, y, z])
    obstruction_map: (H, W) numpy array
    grid_shape: (H, W)
    rx_index: optional, flat index of RX to visualize ray to
    """
    H, W = grid_shape
    x_min, y_min, z_min = mesh.bounds[0]
    x_max, y_max, z_max = mesh.bounds[1]
    
    # Build RX grid (center of each cell)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, W),
        np.linspace(y_min, y_max, H)
    )
    zz = np.zeros_like(xx)
    rx_positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    
    # -------------------
    # Top-down heatmap
    # -------------------
    plt.figure(figsize=(8,8))
    plt.imshow(
        obstruction_map,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(label='Obstructions')
    plt.scatter(mesh.vertices[:,0], mesh.vertices[:,1], s=1, color='black', alpha=0.5)
    plt.scatter(tx_position[0], tx_position[1], color='red', s=100, label='TX')
    plt.scatter(xx.flatten(), yy.flatten(), color='blue', s=5, alpha=0.5, label='RX')
    
    if rx_index is not None:
        selected_rx = rx_positions[rx_index]
        plt.scatter(selected_rx[0], selected_rx[1], color='yellow', s=60, label='Selected RX')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Top-Down Obstruction Map')
    plt.legend()
    plt.show()
    
    # -------------------
    # 3D ray visualization
    # -------------------
    if rx_index is not None:
        visualize_single_ray(mesh, tx_position, selected_rx)

rx_index = 0  # example index to visualize
scene_path = "/Users/khushipatel/Desktop/capstone/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes/scene0"

mesh = load_scene_mesh(scene_path)
tx_position = np.array([0, 0, 1.5])
obstruction_map = np.load(scene_path + "/obstruction_layer.npy")

visualize_rx_selection(mesh, tx_position, obstruction_map, grid_shape=(40,40), rx_index=rx_index)