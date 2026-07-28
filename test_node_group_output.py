#!/usr/bin/env python3
"""Test NodeGroupOutput as replacement for CompositorNodeComposite"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "node_group_output"
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
# CHECK 1: Create compositor node group and assign to scene
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] Create compositor node group and assign to scene")

# Create new compositor node group
new_group = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
scene.compositing_node_group = new_group
print(f"  scene.compositing_node_group: {scene.compositing_node_group}")

# Clear existing nodes
scene.compositing_node_group.nodes.clear()

# Render Layers node
rl = scene.compositing_node_group.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)
print(f"  Render Layers outputs: {[out.name for out in rl.outputs]}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2: Try creating NodeGroupOutput
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] Try creating NodeGroupOutput")

try:
    group_output = scene.compositing_node_group.nodes.new("NodeGroupOutput")
    group_output.location = (400, 0)
    print(f"  Created NodeGroupOutput: {group_output.name}")
    print(f"  type: {group_output.type}")
    print(f"  inputs: {[inp.name for inp in group_output.inputs]}")
    print(f"  outputs: {[out.name for out in group_output.outputs]}")
except RuntimeError as e:
    print(f"  ERROR: {e}")
    group_output = None

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3: Check node group's interface and add output socket
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] Check node group's interface and add output socket")

if hasattr(scene.compositing_node_group, 'interface'):
    print(f"  group.interface: {scene.compositing_node_group.interface}")
    print(f"  group.interface.items_tree: {scene.compositing_node_group.interface.items_tree}")
    print(f"  Number of interface items: {len(scene.compositing_node_group.interface.items_tree)}")

    # Try to add an output socket to the interface
    try:
        # Add an Image output socket
        new_socket = scene.compositing_node_group.interface.new_socket(
            "Image",
            in_out='OUTPUT',
            socket_type='NodeSocketColor'
        )
        print(f"  Added output socket: {new_socket.name}")
        print(f"  Socket type: {new_socket.socket_type}")
    except Exception as e:
        print(f"  ERROR adding socket: {e}")
else:
    print(f"  No interface attribute found")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4: Create File Output node
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] Create File Output node")

out = scene.compositing_node_group.nodes.new("CompositorNodeOutputFile")
out.location = (200, 0)
out.directory = str(OUTPUT_DIR) + "/"
out.file_name = "test_output"
out.format.file_format = "OPEN_EXR_MULTILAYER"
print(f"  File Output directory: {out.directory}")
print(f"  File Output file_name: {out.file_name}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5: Link Render Layers Image to NodeGroupOutput
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] Link Render Layers Image to NodeGroupOutput")

if group_output:
    # Check NodeGroupOutput inputs
    print(f"  NodeGroupOutput inputs: {[inp.name for inp in group_output.inputs]}")

    # Try to link Render Layers Image to NodeGroupOutput
    if len(group_output.inputs) > 0:
        scene.compositing_node_group.links.new(rl.outputs["Image"], group_output.inputs[0])
        print(f"  LINKED: Render Layers.Image -> NodeGroupOutput[0]")
    else:
        print(f"  WARNING: NodeGroupOutput has no inputs")

    # Also link Object Index to File Output
    if "Object Index" in [out.name for out in rl.outputs]:
        scene.compositing_node_group.links.new(rl.outputs["Object Index"], out.inputs[0])
        print(f"  LINKED: Object Index -> File Output")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 6: Print all nodes and links
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 6] Print all nodes and links")

print(f"\n  All nodes in compositor:")
for node in scene.compositing_node_group.nodes:
    print(f"    {node.name} ({node.type}) at {node.location}")

print(f"\n  All links in compositor:")
for link in scene.compositing_node_group.links:
    print(f"    {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 7: Render and check output
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 7] Render and check output")

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
# CHECK 8: Check for File Output with frame number suffix
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 8] Check for File Output with frame number suffix")

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
# CHECK 9: Sample pixel values using Blender's native path
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 9] Sample pixel values using Blender's native path")

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
print("NODE GROUP OUTPUT TEST COMPLETE")
print("=" * 70)
