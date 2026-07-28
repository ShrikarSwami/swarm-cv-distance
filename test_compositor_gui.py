#!/usr/bin/env python3
"""Test compositor in GUI mode (without --background)"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "compositor_gui"
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
# Get the last created object
cube = [obj for obj in bpy.data.objects if obj.type == 'MESH'][-1]
cube.name = "TestCube"
cube.pass_index = 1

# Create camera
bpy.ops.object.camera_add(location=(0, -10, 5))
cam = [obj for obj in bpy.data.objects if obj.type == 'CAMERA'][-1]
cam.name = "TestCamera"
cam.rotation_euler = (1.0, 0, 0)
scene.camera = cam

# Enable Object Index pass
bpy.context.view_layer.use_pass_object_index = True
print(f"use_pass_object_index: {bpy.context.view_layer.use_pass_object_index}")

# Enable compositor
scene.use_nodes = True
scene.render.use_compositing = True
print(f"scene.use_nodes: {scene.use_nodes}")
print(f"scene.render.use_compositing: {scene.render.use_compositing}")

# Set compositor to CPU
if hasattr(scene.render, 'compositor_device'):
    scene.render.compositor_device = 'CPU'
    print(f"scene.render.compositor_device: {scene.render.compositor_device}")

# Create compositor node group
new_group = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
scene.compositing_node_group = new_group

# Clear existing nodes
scene.compositing_node_group.nodes.clear()

# Render Layers node
rl = scene.compositing_node_group.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)

# File Output node
out = scene.compositing_node_group.nodes.new("CompositorNodeOutputFile")
out.location = (200, 0)
out.directory = str(OUTPUT_DIR) + "/"
out.file_name = "test_output"
out.format.file_format = "OPEN_EXR_MULTILAYER"

# Connect Object Index to File Output
if "Object Index" in [out.name for out in rl.outputs]:
    scene.compositing_node_group.links.new(rl.outputs["Object Index"], out.inputs[0])
    print(f"LINKED: Object Index -> File Output")

# Add GroupOutput node
try:
    group_output = scene.compositing_node_group.nodes.new("NodeGroupOutput")
    group_output.location = (400, 0)

    # Add output socket to interface
    new_socket = scene.compositing_node_group.interface.new_socket(
        "Image",
        in_out='OUTPUT',
        socket_type='NodeSocketColor'
    )

    # Link Render Layers Image to NodeGroupOutput
    scene.compositing_node_group.links.new(rl.outputs["Image"], group_output.inputs[0])
    print(f"LINKED: Render Layers.Image -> NodeGroupOutput.Image")
except Exception as e:
    print(f"ERROR creating NodeGroupOutput: {e}")

# Render
scene.render.filepath = str(OUTPUT_DIR / "render")
scene.render.image_settings.file_format = "OPEN_EXR"
print(f"Rendering...")
bpy.ops.render.render(write_still=True)
print(f"Render complete!")

# Check output files
print(f"\nFiles in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# Check for File Output with frame number suffix
import glob
pattern = str(OUTPUT_DIR / "test_output*.exr")
files = glob.glob(pattern)
print(f"\nFile Output files: {files}")
