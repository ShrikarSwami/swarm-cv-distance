#!/usr/bin/env python3
"""Final compositor debug checks before concluding Blender bug"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "compositor_final"
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
# CHECK 1: Print id(bpy.context.scene) and id(scene) before render
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] Print id(bpy.context.scene) and id(scene) before render")
print(f"  id(bpy.context.scene): {id(bpy.context.scene)}")
print(f"  id(scene): {id(scene)}")
print(f"  Same object: {id(bpy.context.scene) == id(scene)}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2: Create compositor node group and assign to scene
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] Create compositor node group and assign to scene")

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
else:
    print(f"  ERROR: Object Index not found")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3: Print file_output_node.mute
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] Print file_output_node.mute")
print(f"  out.mute: {out.mute}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4: Print exact directory and file_name
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] Print exact directory and file_name")
print(f"  out.directory: {out.directory}")
print(f"  bpy.path.abspath(out.directory): {bpy.path.abspath(out.directory)}")
print(f"  out.file_name: {out.file_name}")

# Check file_output_items
if hasattr(out, 'file_output_items'):
    print(f"  out.file_output_items: {out.file_output_items}")
    if len(out.file_output_items) > 0:
        print(f"  out.file_output_items[0]: {out.file_output_items[0]}")
        print(f"  out.file_output_items[0].path: {out.file_output_items[0].path}")
else:
    print(f"  out.file_output_items: NOT AVAILABLE")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5: Add Composite output node (required for Blender to run compositor)
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] Add Composite output node")

# Check available compositor node types
print(f"  Available compositor node types:")
for node_type in dir(bpy.types):
    if 'Compositor' in node_type and 'Node' in node_type:
        print(f"    {node_type}")

# Try different Composite node types
composite = None
for node_type in ['CompositorNodeComposite', 'CompositorNodeOutputComposite', 'CompositeNode']:
    try:
        composite = scene.compositing_node_group.nodes.new(node_type)
        print(f"  Created Composite node: {composite.name} (type: {node_type})")
        break
    except RuntimeError as e:
        print(f"  {node_type}: {e}")

if composite:
    composite.location = (200, 100)
    # Link Render Layers Image to Composite
    scene.compositing_node_group.links.new(rl.outputs["Image"], composite.inputs["Image"])
    print(f"  LINKED: Render Layers.Image -> Composite.Image")
else:
    print(f"  ERROR: Could not create Composite node")

# Print all nodes and links
print(f"\n  All nodes in compositor:")
for node in scene.compositing_node_group.nodes:
    print(f"    {node.name} ({node.type}) at {node.location}")

print(f"\n  All links in compositor:")
for link in scene.compositing_node_group.links:
    print(f"    {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 6: Print render settings toggles
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 6] Print render settings toggles")
print(f"  scene.render.use_compositing: {scene.render.use_compositing}")

# Check for other compositing-related settings
for attr in dir(scene.render):
    if 'composit' in attr.lower() or 'file' in attr.lower():
        try:
            val = getattr(scene.render, attr)
            if not callable(val):
                print(f"  scene.render.{attr}: {val}")
        except:
            pass

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
print("FINAL COMPOSITOR DEBUG COMPLETE")
print("=" * 70)
