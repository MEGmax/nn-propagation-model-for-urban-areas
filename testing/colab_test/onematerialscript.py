import bpy

#this script is used within blender to make all the materials of all the objects the same

# Clean up old materials
for mat in bpy.data.materials:
    bpy.data.materials.remove(mat, do_unlink=True)

# Create one material for the scene
mat = bpy.data.materials.new(name="MyMaterial")

# Assign to all mesh objects
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.data.materials.clear()
        obj.data.materials.append(mat)
