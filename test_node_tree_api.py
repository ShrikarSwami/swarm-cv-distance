#!/usr/bin/env python3
"""Test compositor node tree API - scene.node_tree vs scene.compositing_node_group"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "node_tree_api"
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
cube.pass_index = 1

# Create camera
bpy.ops.object.camera_add(location=(0, -10, 5))
cam = bpy.context.object
cam.name = "TestCamera"
cam.rotation_euler = (1.0, 0, 0)
scene.camera = cam

# Enable Object Index pass
bpy.context.view_layer.use_pass_object_index = True
print(f"use_pass_object_index: {bpy.context.view_layer.use_pass_object_index}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 1: Print type(scene.node_tree) and whether it's None
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] type(scene.node_tree)")
try:
    node_tree = scene.node_tree
    print(f"  type(scene.node_tree): {type(node_tree)}")
    print(f"  scene.node_tree: {node_tree}")
    print(f"  Is None: {node_tree is None}")
except AttributeError as e:
    print(f"  AttributeError: {e}")
    print(f"  scene.node_tree does not exist in this Blender version")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2: Print type(scene.compositing_node_group)
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] type(scene.compositing_node_group)")
try:
    comp_group = scene.compositing_node_group
    print(f"  type(scene.compositing_node_group): {type(comp_group)}")
    print(f"  scene.compositing_node_group: {comp_group}")
    print(f"  Is None: {comp_group is None}")
except AttributeError as e:
    print(f"  AttributeError: {e}")
    print(f"  scene.compositing_node_group does not exist")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3: Enable compositor and check both APIs
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] Enable compositor and check both APIs")
scene.use_nodes = True
print(f"  scene.use_nodes: {scene.use_nodes}")

# Check scene.node_tree again
try:
    node_tree = scene.node_tree
    print(f"  After use_nodes=True:")
    print(f"    scene.node_tree: {node_tree}")
    print(f"    type(scene.node_tree): {type(node_tree)}")
    if node_tree is not None:
        print(f"    scene.node_tree.name: {node_tree.name}")
        print(f"    scene.node_tree.nodes: {node_tree.nodes}")
        print(f"    scene.node_tree.links: {node_tree.links}")
except AttributeError as e:
    print(f"  scene.node_tree: AttributeError - {e}")

# Check scene.compositing_node_group again
try:
    comp_group = scene.compositing_node_group
    print(f"  scene.compositing_node_group: {comp_group}")
    print(f"  type(scene.compositing_node_group): {type(comp_group)}")
    if comp_group is not None:
        print(f"    scene.compositing_node_group.name: {comp_group.name}")
        print(f"    scene.compositing_node_group.nodes: {comp_group.nodes}")
        print(f"    scene.compositing_node_group.links: {comp_group.links}")
except AttributeError as e:
    print(f"  scene.compositing_node_group: AttributeError - {e}")

# Try to find the actual compositor node tree
print(f"\n  Looking for actual compositor node tree:")
found_compositor = False
for ng in bpy.data.node_groups:
    print(f"    {ng.name}: type={ng.type}")
    if ng.type == 'CompositorNodeTree':
        print(f"      -> This is a CompositorNodeTree!")
        print(f"      -> nodes: {len(ng.nodes)}")
        print(f"      -> links: {len(ng.links)}")
        found_compositor = True

if not found_compositor:
    print(f"    No CompositorNodeTree found in bpy.data.node_groups")

# Check if there's a different way to access the compositor
print(f"\n  Checking other compositor access methods:")
print(f"    scene.use_nodes: {scene.use_nodes}")
print(f"    scene.render.use_compositing: {scene.render.use_compositing}")

# Try to use bpy.context.scene.node_tree (old API)
print(f"\n  Trying old API: bpy.context.scene.node_tree")
try:
    old_tree = bpy.context.scene.node_tree
    print(f"    bpy.context.scene.node_tree: {old_tree}")
except AttributeError as e:
    print(f"    AttributeError: {e}")

# Try to use bpy.context.scene.compositing_node_group
print(f"\n  Trying new API: bpy.context.scene.compositing_node_group")
try:
    new_tree = bpy.context.scene.compositing_node_group
    print(f"    bpy.context.scene.compositing_node_group: {new_tree}")
except AttributeError as e:
    print(f"    AttributeError: {e}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4: Confirm scene.render.use_compositing = True
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] scene.render.use_compositing")
try:
    print(f"  scene.render.use_compositing: {scene.render.use_compositing}")
except AttributeError as e:
    print(f"  AttributeError: {e}")
    print(f"  scene.render.use_compositing does not exist")

# Try to set it
try:
    scene.render.use_compositing = True
    print(f"  After setting True: {scene.render.use_compositing}")
except AttributeError as e:
    print(f"  Cannot set use_compositing: {e}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5: Build node graph via scene.compositing_node_group
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] Build node graph via scene.compositing_node_group")

# Get the compositor node group
comp_group = scene.compositing_node_group
if comp_group is None:
    comp_group = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
    scene.compositing_node_group = comp_group
    print(f"  Created new compositor node group: {comp_group.name}")

# Clear existing nodes
comp_group.nodes.clear()

# Render Layers node
rl = comp_group.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)
print(f"  Render Layers outputs: {[out.name for out in rl.outputs]}")

# File Output node
out = comp_group.nodes.new("CompositorNodeOutputFile")
out.location = (200, 0)
out.directory = str(OUTPUT_DIR) + "/"
out.file_name = "test_output"
out.format.file_format = "OPEN_EXR_MULTILAYER"
print(f"  File Output directory: {out.directory}")
print(f"  File Output file_name: {out.file_name}")

# Connect Object Index to File Output
if "Object Index" in [out.name for out in rl.outputs]:
    comp_group.links.new(rl.outputs["Object Index"], out.inputs[0])
    print(f"  LINKED: Object Index -> File Output")
else:
    print(f"  ERROR: Object Index not found")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 6: Render and check output
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 6] Render and check output")

scene.render.filepath = str(OUTPUT_DIR / "render")
scene.render.image_settings.file_format = "OPEN_EXR"
print(f"  Rendering...")
bpy.ops.render.render(write_still=True)
print(f"  Render complete!")

# Check output files
print(f"\nFiles in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 7: Sample pixel values using Blender's native path
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 7] Sample pixel values using Blender's native path")

import numpy as np

# Load the render EXR using Blender's native image loading
render_path = str(OUTPUT_DIR / "render.exr")
if Path(render_path).exists():
    img = bpy.data.images.load(render_path)
    print(f"  Loaded image: {img.name}")
    print(f"  Size: {img.size[0]}x{img.size[1]}")
    print(f"  Channels: {img.channels}")

    # Get pixels as numpy array
    pixels = np.array(img.pixels[:])
    print(f"  Pixels shape: {pixels.shape}")
    print(f"  Pixels min: {pixels.min()}")
    print(f"  Pixels max: {pixels.max()}")
    print(f"  Pixels mean: {pixels.mean()}")

    # Check if there's an Object Index channel
    # In Blender, Object Index is stored as a separate image or pass
    # Let's check if there are multiple images
    print(f"\n  All images in blend file:")
    for image in bpy.data.images:
        print(f"    {image.name}: {image.size[0]}x{image.size[1]}, {image.channels} channels")
else:
    print(f"  Render file not found: {render_path}")

print("\n" + "=" * 70)
print("NODE TREE API DEBUG COMPLETE")
print("=" * 70)
