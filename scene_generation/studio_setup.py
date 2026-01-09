import random
import os
import bpy

# Command to run in terminal (depends on your Blender installation path):
# /Applications/Blender.app/Contents/MacOS/Blender --background --python ./studio_setup.py

NUM_BUILDINGS_MIN = 10
NUM_BUILDINGS_MAX = 20

MIN_SIZE = 0.5
MAX_SIZE = 4
MIN_HEIGHT = 1.0
MAX_HEIGHT = 8.0

AREA_SIZE = 8


# select and delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# delete old materials
for mat in bpy.data.materials:
    bpy.data.materials.remove(mat, do_unlink=True)

# declare your materials
concrete = bpy.data.materials.new(name='itu_concrete')
soil = bpy.data.materials.new(name='medium_dry_ground')

# add plane
bpy.ops.mesh.primitive_plane_add(size=(AREA_SIZE * 2), enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
ground = bpy.context.active_object
ground.data.materials.append(soil)

# add cubes to be like buildings
num_buildings = random.randint(NUM_BUILDINGS_MIN, NUM_BUILDINGS_MAX)

for i in range(num_buildings):
    width = random.uniform(MIN_SIZE, MAX_SIZE)
    depth = random.uniform(MIN_SIZE, MAX_SIZE)
    height = random.uniform(MIN_HEIGHT, MAX_HEIGHT)

    x = random.uniform(-AREA_SIZE + width/2, AREA_SIZE - width/2)
    y = random.uniform(-AREA_SIZE + depth/2, AREA_SIZE - depth/2)

    # place cube at half height so it is not buried
    z = height / 2

    # adding the cube at location x, y, z
    bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
    building = bpy.context.active_object

    # scale cube (devide by 2 becasue default cube is size (2, 2, 2)
    building.scale = (width / 2, depth / 2, height / 2)

    building.name = f"Building_{i}"
    building.data.materials.append(concrete)

# export to Mitsuba XML
xml_filepath = os.path.abspath("studio_setup.xml")

try:
    bpy.ops.export_scene.mitsuba(filepath=xml_filepath, check_existing=False)
    print(f"Successfully exported scene to {xml_filepath}")
except Exception as e:
    print(f"Error during Mitsuba export: {e}")

print("Mitsuba XML export complete.")
