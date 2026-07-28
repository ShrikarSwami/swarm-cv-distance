#!/usr/bin/env python3
"""Test compositor with different use_nodes settings"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "compositor_use_nodes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Factory reset
try:
    bpy.ops.wm.read_factory_use_empty(use_empty=True)
except AttributeError:
    bpy.ops.wm.read_homefile(use_empty=True)

# Set render engine to Cycles
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
scene.render.resolution_x = 640
scene.render.resolution_y = 480

# Create a simple cube
bpy.ops.mesh.primitive_cube_add(size=5, location=(0, 0, 0))
cube = bpy.context.object
cube.name = "TestCube"

# Create camera
bpy.ops.object.camera_add(location=(0, -10, 5))
cam = bpy.context.object
cam.name = "TestCamera"
cam.rotation_euler = (1.0, 0, 0)
scene.camera = cam

# Enable Object Index pass
bpy.context.view_layer.use_pass_object_index = True
cube.pass_index = 1
print(f"use_pass_object_index: {bpy.context.view_layer.use_pass_object_index}")

# Enable compositor
scene.use_nodes = True
print(f"scene.use_nodes: {scene.use_nodes}")

# Check compositor node group
tree = scene.compositing_node_group
print(f"compositor node group: {tree}")

if tree is None:
    tree = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
    scene.compositing_node_group = tree
    print(f"Created new compositor node group: {tree.name}")

# Clear and create nodes
tree.nodes.clear()

# Render Layers node
rl = tree.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)
print(f"Render Layers outputs: {[out.name for out in rl.outputs]}")

# File Output node
out = tree.nodes.new("CompositorNodeOutputFile")
out.location = (200, 0)
out.directory = str(OUTPUT_DIR) + "/"
out.file_name = "test_output"
out.format.file_format = "OPEN_EXR_MULTILAYER"
print(f"File Output directory: {out.directory}")
print(f"File Output file_name: {out.file_name}")

# Connect Object Index to File Output
if "Object Index" in [out.name for out in rl.outputs]:
    tree.links.new(rl.outputs["Object Index"], out.inputs[0])
    print("LINKED: Object Index -> File Output")
else:
    print("ERROR: Object Index not found")

# Check all links
print(f"\nAll links in compositor:")
for link in tree.links:
    print(f"  {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

# Render
scene.render.filepath = str(OUTPUT_DIR / "render")
scene.render.image_settings.file_format = "OPEN_EXR"
print(f"\nRendering...")
bpy.ops.render.render(write_still=True)
print("Render complete!")

# Check output files
print(f"\nFiles in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
