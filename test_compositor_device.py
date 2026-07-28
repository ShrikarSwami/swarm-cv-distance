#!/usr/bin/env python3
"""Test compositor device setting - GPU vs CPU"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "compositor_device"
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
# CHECK 1: Search for compositor device/performance property
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] Search for compositor device/performance property")

# Check scene.render properties for compositor-related settings
print(f"  scene.render properties with 'compositor':")
for attr in dir(scene.render):
    if 'compositor' in attr.lower():
        try:
            val = getattr(scene.render, attr)
            if not callable(val):
                print(f"    scene.render.{attr}: {val}")
        except:
            pass

# Check the specific property
if hasattr(scene.render, 'compositor_device'):
    print(f"\n  scene.render.compositor_device: {scene.render.compositor_device}")
else:
    print(f"\n  scene.render.compositor_device: NOT FOUND")

# Try to find it in bl_rna.properties
print(f"\n  Searching bl_rna.properties for 'compositor':")
for prop_name in dir(scene.render.bl_rna.properties):
    if 'compositor' in prop_name.lower():
        prop = getattr(scene.render.bl_rna.properties, prop_name)
        print(f"    {prop_name}: {prop}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2: Set compositor_device to CPU if found
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] Set compositor_device to CPU if found")

if hasattr(scene.render, 'compositor_device'):
    original_value = scene.render.compositor_device
    print(f"  Original value: {original_value}")

    try:
        scene.render.compositor_device = 'CPU'
        print(f"  Set to CPU: {scene.render.compositor_device}")
    except Exception as e:
        print(f"  ERROR setting to CPU: {e}")
else:
    print(f"  compositor_device not found, skipping")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3: Create compositor node group and assign to scene
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] Create compositor node group and assign to scene")

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

# Add GroupOutput node
try:
    group_output = scene.compositing_node_group.nodes.new("NodeGroupOutput")
    group_output.location = (400, 0)
    print(f"  Created NodeGroupOutput: {group_output.name}")

    # Add output socket to interface
    new_socket = scene.compositing_node_group.interface.new_socket(
        "Image",
        in_out='OUTPUT',
        socket_type='NodeSocketColor'
    )
    print(f"  Added output socket: {new_socket.name}")

    # Link Render Layers Image to NodeGroupOutput
    scene.compositing_node_group.links.new(rl.outputs["Image"], group_output.inputs[0])
    print(f"  LINKED: Render Layers.Image -> NodeGroupOutput.Image")
except Exception as e:
    print(f"  ERROR creating NodeGroupOutput: {e}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4: Print all nodes and links
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] Print all nodes and links")

print(f"\n  All nodes in compositor:")
for node in scene.compositing_node_group.nodes:
    print(f"    {node.name} ({node.type}) at {node.location}")

print(f"\n  All links in compositor:")
for link in scene.compositing_node_group.links:
    print(f"    {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5: Render and check output
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] Render and check output")

scene.render.filepath = str(OUTPUT_DIR / "render")
scene.render.image_settings.file_format = "OPEN_EXR"
print(f"  Rendering with compositor_device: {scene.render.compositor_device if hasattr(scene.render, 'compositor_device') else 'N/A'}")
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

print("\n" + "=" * 70)
print("COMPOSITOR DEVICE TEST COMPLETE")
print("=" * 70)
