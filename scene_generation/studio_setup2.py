import random
import os
import bpy
import shutil
import math
import sys
import argparse
import bmesh

from pathlib import Path

# Command to run in terminal (depends on your Blender installation path):
# /Applications/Blender.app/Contents/MacOS/Blender --background --python ./studio_setup.py --

NUM_BUILDINGS_MIN = 10
NUM_BUILDINGS_MAX = 20

MIN_SIZE = 0.5
MAX_SIZE = 4
MIN_HEIGHT = 1.0
MAX_HEIGHT = 8.0

AREA_SIZE = 8
EXCLUSION_RADIUS = 2.0

# ---- NEW: hollow-building controls ----
WALL_THICKNESS = 0.30   # meters (typical 0.2–0.4)
OPEN_ROOF = True        # True => no roof (good for debugging / top-down maps)
OPEN_FLOOR = True       # True => no bottom face

BASE_DIR = Path(__file__).resolve().parent
SCENE_DIR = BASE_DIR / "automated_scenes"


def parse_arguments():
    # all args after `--` are in sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes", type=int, default=5)

    # IMPORTANT: type=bool is unreliable in argparse (e.g., "--flag False" still truthy).
    # Use a proper boolean flag:
    parser.add_argument("--buildings-in-center", action="store_true", help="Allow buildings near (0,0).")

    # Blender adds its own args before the '--', so skip them
    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    else:
        args = parser.parse_args()

    return args


def repository_setup() -> None:
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


def reset_scene():
    # Delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Purge old materials (safe unlink)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)


def make_hollow_building(building_obj, wall_thickness=0.3, open_roof=True, open_floor=True):
    """
    Convert a scaled cube into a hollow wall-shell:
    - Applies rotation/scale
    - Deletes top and/or bottom faces (optional)
    - Adds Solidify for wall thickness
    - Applies modifier + fixes normals
    """

    # Activate object
    bpy.ops.object.select_all(action="DESELECT")
    building_obj.select_set(True)
    bpy.context.view_layer.objects.active = building_obj

    # Apply transforms so "top/bottom" detection via normals is stable
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Enter edit mode and get bmesh
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(building_obj.data)
    bm.faces.ensure_lookup_table()

    # Delete roof/floor by face normal Z (cube faces are axis-aligned after transform_apply)
    to_delete = []
    for f in bm.faces:
        nz = f.normal.z
        if open_roof and nz > 0.999:     # top face
            to_delete.append(f)
        if open_floor and nz < -0.999:   # bottom face
            to_delete.append(f)

    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context="FACES")
        bmesh.update_edit_mesh(building_obj.data, destructive=True)

    # Recalculate normals outward
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.update_edit_mesh(building_obj.data, destructive=True)

    bpy.ops.object.mode_set(mode="OBJECT")

    # Add thickness (walls only)
    solid = building_obj.modifiers.new(name="SolidifyWalls", type="SOLIDIFY")
    solid.thickness = wall_thickness
    solid.offset = 0.0  # split thickness in/out around the wall surface
    solid.use_rim = True
    solid.use_rim_only = False

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier=solid.name)

    # Final normals cleanup
    bpy.ops.object.mode_set(mode="EDIT")
    bm2 = bmesh.from_edit_mesh(building_obj.data)
    bm2.faces.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
    bmesh.update_edit_mesh(building_obj.data, destructive=True)
    bpy.ops.object.mode_set(mode="OBJECT")


def scene_generation(buildings_in_center: bool) -> None:
    reset_scene()

    # declare your materials
    concrete = bpy.data.materials.new(name="itu_concrete")
    soil = bpy.data.materials.new(name="itu_medium_dry_ground")

    # add plane (ground)
    bpy.ops.mesh.primitive_plane_add(
        size=(AREA_SIZE * 2),
        enter_editmode=False,
        align="WORLD",
        location=(0, 0, 0),
        scale=(1, 1, 1),
    )
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(soil)

    # add buildings
    num_buildings = random.randint(NUM_BUILDINGS_MIN, NUM_BUILDINGS_MAX)

    for i in range(num_buildings):
        while True:
            width = random.uniform(MIN_SIZE, MAX_SIZE)
            depth = random.uniform(MIN_SIZE, MAX_SIZE)
            height = random.uniform(MIN_HEIGHT, MAX_HEIGHT)

            x = random.uniform(-AREA_SIZE + width / 2, AREA_SIZE - width / 2)
            y = random.uniform(-AREA_SIZE + depth / 2, AREA_SIZE - depth / 2)

            # place cube at half height so it is not buried
            z = height / 2

            if not buildings_in_center:
                # keep away from center (0,0)
                half_diagonal = math.sqrt((width / 2) ** 2 + (depth / 2) ** 2)
                min_distance = EXCLUSION_RADIUS + half_diagonal
                distance_center = math.hypot(x, y)
                if distance_center >= min_distance:
                    break
            else:
                break

        # Create a cube, scale to footprint+height
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
        building = bpy.context.active_object

        # scale cube (divide by 2 because default cube is 2x2x2)
        building.scale = (width / 2, depth / 2, height / 2)

        building.name = f"Building_{i}"
        building.data.materials.append(concrete)

        # ✅ convert solid cube into hollow wall shell
        make_hollow_building(
            building,
            wall_thickness=WALL_THICKNESS,
            open_roof=OPEN_ROOF,
            open_floor=OPEN_FLOOR,
        )

    # apply transforms for export (good practice)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def export_scene(scene_ID: int) -> None:
    # create directory to save xml and mesh directory
    out_dir = SCENE_DIR / f"scene{scene_ID}"
    os.makedirs(out_dir, exist_ok=True)

    # export to Mitsuba XML
    xml_filepath = os.path.abspath(out_dir / f"scene{scene_ID}.xml")

    try:
        bpy.ops.export_scene.mitsuba(
            filepath=xml_filepath,
            check_existing=False,
            axis_forward="Y",
            axis_up="Z",
        )
        print(f"Successfully exported scene to {xml_filepath}")
    except Exception as e:
        print(f"Error during Mitsuba export: {e}")


def main():
    args = parse_arguments()
    repository_setup()
    for i in range(args.num_scenes):
        scene_generation(args.buildings_in_center)
        export_scene(i)


if __name__ == "__main__":
    main()