#!/usr/bin/env python3
"""Test compositor node group assignment to scene"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "compositor_assignment"
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

# Enable compositor
scene.use_nodes = True
scene.render.use_compositing = True
print(f"scene.use_nodes: {scene.use_nodes}")
print(f"scene.render.use_compositing: {scene.render.use_compositing}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 1: Print scene.compositing_node_group BEFORE assignment
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] scene.compositing_node_group BEFORE assignment")
print(f"  value: {scene.compositing_node_group}")
print(f"  type: {type(scene.compositing_node_group)}")
print(f"  is None: {scene.compositing_node_group is None}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2: Create node group and ASSIGN to scene
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] Create node group and ASSIGN to scene")

# Create new compositor node group
new_group = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
print(f"  Created node group: {new_group.name}")
print(f"  type: {type(new_group)}")

# ASSIGN to scene
scene.compositing_node_group = new_group
print(f"  Assigned to scene.compositing_node_group")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3: Print scene.compositing_node_group AFTER assignment
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] scene.compositing_node_group AFTER assignment")
print(f"  value: {scene.compositing_node_group}")
print(f"  type: {type(scene.compositing_node_group)}")
print(f"  is None: {scene.compositing_node_group is None}")
if scene.compositing_node_group is not None:
    print(f"  name: {scene.compositing_node_group.name}")
    print(f"  nodes: {scene.compositing_node_group.nodes}")
    print(f"  links: {scene.compositing_node_group.links}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4: Build node graph in the assigned group
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] Build node graph in the assigned group")

# Clear existing nodes
scene.compositing_node_group.nodes.clear()

# Render Layers node
rl = scene.compositing_node_group.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)
print(f"  Render Layers outputs: {[out.name for out in rl.outputs]}")

# File Output node
out = scene.compositing_node_group.nodes.new("CompositorNodeOutputFile")
out.location = (200, 0)
out.directory = str(OUTPUT_DIR) + "/"
out.file_name = "test_output"
out.format.file_format = "OPEN_EXR_MULTILAYER"
print(f"  File Output directory: {out.directory}")
print(f"  File Output file_name: {out.file_name}")

# Connect Object Index to File Output
if "Object Index" in [out.name for out in rl.outputs]:
    scene.compositing_node_group.links.new(rl.outputs["Object Index"], out.inputs[0])
    print(f"  LINKED: Object Index -> File Output")
else:
    print(f"  ERROR: Object Index not found")

# Verify the link
print(f"\n  All links in compositor:")
for link in scene.compositing_node_group.links:
    print(f"    {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5: Render and check output
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] Render and check output")

scene.render.filepath = str(OUTPUT_DIR / "render")
scene.render.image_settings.file_format = "OPEN_EXR"
print(f"  Rendering...")
bpy.ops.render.render(write_still=True)
print(f"  Render complete!")

# Check output files
print(f"\n  Files in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 6: Check for File Output with frame number suffix
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 6] Check for File Output with frame number suffix")

import glob
# Check for test_output*.exr pattern
pattern = str(OUTPUT_DIR / "test_output*.exr")
files = glob.glob(pattern)
print(f"  Pattern: {pattern}")
print(f"  Files found: {files}")

if not files:
    # Also check for any .exr files
    all_exr = list(OUTPUT_DIR.glob("*.exr"))
    print(f"  All .exr files: {[f.name for f in all_exr]}")

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
else:
    print(f"  Render file not found: {render_path}")

print("\n" + "=" * 70)
print("COMPOSITOR ASSIGNMENT TEST COMPLETE")
print("=" * 70)
