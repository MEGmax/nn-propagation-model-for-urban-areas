import random
import os
import shutil
import math
import sys
import argparse
import numpy as np

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Command to run in terminal (depends on your Blender installation path):
# /Applications/Blender.app/Contents/MacOS/Blender --background --python scene_generation/studio_setup.py --
# Parameters example:
# --num-scenes 5 --buildings-in-center False

# Use seeds for debugging
# np.random.seed(42)
# random.seed(42)

"""
PARAMETERS FOR A 300 x 300 SCENE

NUM_BUILDINGS_MIN = 500
NUM_BUILDINGS_MAX = 600

MIN_SIZE = 4
MAX_SIZE = 17
MIN_HEIGHT = 3
MAX_HEIGHT = 30
AREA_SIZE = 150
"""

EXCLUSION_RADIUS = 2.0

BASE_DIR = Path(__file__).resolve().parent
SCENE_DIR = BASE_DIR / "automated_scenes"
CONFIG_PATH = BASE_DIR.parent / "configs" / "config.toml"

DEFAULT_CONFIG = {
    "Scene Parameters": {
        "num_buildings_min": 10,
        "num_buildings_max": 20,
        "min_size": 0.5,
        "max_size": 4.0,
        "min_height": 1.0,
        "max_height": 8.0,
        "area_size": 8.0,
        "exclusion_radius": 2.0,
    },
}


def load_config():

    config = {
        "Scene Parameters": DEFAULT_CONFIG["Scene Parameters"].copy(),
    }

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as f:
            user_config = tomllib.load(f)

        for section in ["Scene Parameters"]:
            config[section].update(user_config.get(section, {}))

    return config


def parse_arguments():
    # all args after `--` are in sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--buildings-in-center", type=bool, default=False)

    # Blender adds its own args before the '--', so skip them
    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    else:
        args = parser.parse_args()

    return args


def repository_setup(SCENE_DIR) -> None:
    # if repo does not exist, create it
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)
        print(f"Created directory at {SCENE_DIR}")
    # if repo exists, clear it
    else:
        for item in os.listdir(SCENE_DIR):
            item_path = os.path.join(SCENE_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            else:
                shutil.rmtree(item_path)
        print(f"Cleared directory at {SCENE_DIR}")

def scene_generation(buildings_in_center: bool, scene_config: dict) -> None:

    num_buildings_min = scene_config["num_buildings_min"]
    num_buildings_max = scene_config["num_buildings_max"]
    min_size = scene_config["min_size"]
    max_size = scene_config["max_size"]
    min_height = scene_config["min_height"]
    max_height = scene_config["max_height"]
    area_size = scene_config["area_size"]
    exclusion_radius = scene_config["exclusion_radius"]

    # select and delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # clean orphan data
    print(f"meshes count before deletion: {len(bpy.data.meshes)}")
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    print(f"meshes count after deletion: {len(bpy.data.meshes)}")

    # delete old materials
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)

    # declare your materials
    concrete = bpy.data.materials.new(name='itu_concrete')
    soil = bpy.data.materials.new(name='itu_medium_dry_ground')

    # add plane
    bpy.ops.mesh.primitive_plane_add(size=(area_size * 2), enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    ground = bpy.context.active_object
    ground.data.materials.append(soil)

    # add cubes to be like buildings
    num_buildings = random.randint(num_buildings_min, num_buildings_max)

    for i in range(num_buildings):
        while True:
            width = random.uniform(min_size, max_size)
            depth = random.uniform(min_size, max_size)
            height = random.uniform(min_height, max_height)

            x = random.uniform(-area_size + width/2, area_size - width/2)
            y = random.uniform(-area_size + depth/2, area_size - depth/2)

            # place cube at half height so it is not buried
            z = height / 2

            if not buildings_in_center:
                # make sure the building is not too close to the center (0, 0)
                half_diagonal = math.sqrt((width/2)**2 + (depth/2)**2)
                min_distance = exclusion_radius + half_diagonal
                distance_center = math.hypot(x, y)

                if distance_center >= min_distance:
                    break
            else:
                break

        # adding the cube at location x, y, z
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z), scale=(width / 2, depth / 2, height / 2))
        # building = bpy.context.selected_objects[0]
        building = bpy.context.active_object
        building.name = f"Building_{i}"
        building.data.materials.append(concrete)


def export_scene(scene_ID: int) -> None:

    # create directory to save xml and mesh directory
    os.makedirs(SCENE_DIR / f"scene{scene_ID}")

    # export to Mitsuba XML
    xml_filepath = os.path.abspath(SCENE_DIR / f"scene{scene_ID}" / f"scene{scene_ID}.xml")

    try:
        bpy.context.view_layer.update()
        depgraph = bpy.context.evaluated_depsgraph_get()
        depgraph.update()
        bpy.ops.export_scene.mitsuba(filepath=xml_filepath, check_existing=False, axis_forward='Y', axis_up='Z')
        print(f"Successfully exported scene to {xml_filepath}")
    except Exception as e:
        print(f"Error during Mitsuba export: {e}")

def main():
    config = load_config()
    args = parse_arguments()
    repository_setup(SCENE_DIR)
    for i in range(args.num_scenes):
        print("GENERATION SCENE ", i)
        scene_generation(args.buildings_in_center, config["Scene Parameters"])
        export_scene(i)


if __name__ == "__main__":
    # These are blender specfic libraries that we don't want to import when 
    # importing this file in another script.
    import mathutils
    import bpy
    main()
