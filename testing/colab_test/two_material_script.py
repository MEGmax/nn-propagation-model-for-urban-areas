# This script assigns two different materials to objects in a Blender scene:

import bpy


# Clean up old materials
for mat in bpy.data.materials:
    bpy.data.materials.remove(mat, do_unlink=True)

# Create two materials
mat_glass = bpy.data.materials.new(name="itu_glass")
mat_concrete = bpy.data.materials.new(name="itu_concrete")

# Assign materials
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.data.materials.clear()

        # Check if object name contains "Plane"
        # (You can adjust this to match your actual plane object name)
        if "Plane" in obj.name:
            obj.data.materials.append(mat_concrete)
        else:
            obj.data.materials.append(mat_glass)