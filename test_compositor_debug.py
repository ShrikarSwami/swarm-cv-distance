#!/usr/bin/env python3
"""Debug compositor Object Index pass step by step"""

import sys
import os
import bpy
import math
from pathlib import Path

# Ensure blender_addon is importable
_project_root = str(Path(__file__).resolve().parent)
_addon_dir = os.path.join(_project_root, "blender_addon")
for p in [_project_root, _addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "compositor_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("COMPOSITOR OBJECT INDEX PASS DEBUG")
print("=" * 70)

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
scene.render.resolution_percentage = 100

# Apply environment preset
from blender_addon.environments import get_environment
env_preset = get_environment("desert")
env_preset.apply(scene)

# Apply weather preset
from blender_addon.weather import get_weather
weather_preset = get_weather("clear")
weather_preset.apply(scene, hdri_active=True)

# Apply HDRI preset
from blender_addon.hdri import apply as apply_hdri
apply_hdri(scene, "clear")

# Create camera
bpy.ops.object.camera_add(location=(0, -100, 50))
cam = bpy.context.object
cam.name = "TestCamera"
cam.rotation_euler = (math.radians(75), 0, 0)
scene.camera = cam

# Create cube drones
for i in range(5):
    x = (i - 2) * 10
    bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
    cube = bpy.context.object
    cube.name = f"Drone_{i:02d}"

    mat = bpy.data.materials.new(f"DroneMat_{i:02d}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)

    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 0.5
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (200, 0)
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    cube.data.materials.append(mat)
    cube.pass_index = i + 1

# ══════════════════════════════════════════════════════════════════════════
# CHECK 1: Confirm scene.use_nodes = True
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] Confirm scene.use_nodes = True")
print(f"  Before: scene.use_nodes = {scene.use_nodes}")
scene.use_nodes = True
print(f"  After: scene.use_nodes = {scene.use_nodes}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2: Enable Object Index pass and refresh node tree
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] Enable Object Index pass and refresh node tree")

# Enable Object Index pass
bpy.context.view_layer.use_pass_object_index = True
print(f"  use_pass_object_index: {bpy.context.view_layer.use_pass_object_index}")

# Force view layer update
bpy.context.view_layer.update()
print(f"  view_layer.update() called")

# Get or create compositor node group
tree = scene.compositing_node_group
if tree is None:
    tree = bpy.data.node_groups.new("CompositorNodeTree", 'CompositorNodeTree')
    scene.compositing_node_group = tree
print(f"  compositor node group: {tree.name}")

# Clear existing nodes
tree.nodes.clear()

# Create Render Layers node AFTER enabling pass
rl = tree.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)
print(f"  Render Layers node created: {rl.name}")

# Check available outputs
print(f"  Render Layers outputs: {[out.name for out in rl.outputs]}")

# Check specifically for Object Index output
has_object_index = "Object Index" in [out.name for out in rl.outputs]
print(f"  Has 'Object Index' output: {has_object_index}")

if not has_object_index:
    print("  ERROR: Object Index output not found!")
    print("  Available outputs:")
    for out in rl.outputs:
        print(f"    {out.name}: {out.type}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3: Create File Output node and LINK Object Index to it
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] Create File Output node and LINK Object Index to it")

# Create File Output node
out = tree.nodes.new("CompositorNodeOutputFile")
out.location = (200, 0)
out.directory = str(OUTPUT_DIR) + "/"
out.file_name = "obj_index"
out.format.file_format = "OPEN_EXR_MULTILAYER"
out.format.color_depth = "32"
print(f"  File Output node created: {out.name}")
print(f"  Output directory: {out.directory}")
print(f"  File name: {out.file_name}")
print(f"  Format: {out.format.file_format}")

# Check File Output node inputs
print(f"  File Output inputs: {[inp.name for inp in out.inputs]}")

# LINK Object Index to File Output
if has_object_index:
    # Get the Object Index output socket
    obj_index_output = None
    for out_socket in rl.outputs:
        if out_socket.name == "Object Index":
            obj_index_output = out_socket
            break

    if obj_index_output:
        # Create link
        link = tree.links.new(obj_index_output, out.inputs[0])
        print(f"  LINK CREATED: {out_socket.name} -> {out.inputs[0].name}")
        print(f"  Link: {link}")

        # Verify link exists
        links_count = len([l for l in tree.links if l.from_node == rl and l.to_node == out])
        print(f"  Links from Render Layers to File Output: {links_count}")
    else:
        print("  ERROR: Could not find Object Index output socket")
else:
    print("  SKIPPED: Object Index output not available")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4: Confirm render is triggered via bpy.ops.render.render
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] Render with bpy.ops.render.render(write_still=True)")

# Set main render output
scene.render.filepath = str(OUTPUT_DIR / "render")
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

print(f"  Main render filepath: {scene.render.filepath}")
print(f"  Main render format: {scene.render.image_settings.file_format}")

# Render
print("  Calling bpy.ops.render.render(write_still=True)...")
bpy.ops.render.render(write_still=True)
print("  Render complete!")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5: Check File Output node's actual output path
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] Check File Output node's actual output path")

# List all files in output directory
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Files in directory:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# Check for specific patterns
import glob
pattern1 = str(OUTPUT_DIR / "obj_index*")
pattern2 = str(OUTPUT_DIR / "render_obj_index*")
pattern3 = str(OUTPUT_DIR / "*obj*")

files_matching = glob.glob(pattern1) + glob.glob(pattern2) + glob.glob(pattern3)
if files_matching:
    print(f"  Files matching obj patterns: {files_matching}")
else:
    print(f"  No files matching obj patterns found")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 6: Sample real pixel values from Object Index channel
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 6] Sample real pixel values from Object Index channel")

# Try to find and read any EXR files
import numpy as np
try:
    import OpenEXR
    import Imath

    for exr_file in OUTPUT_DIR.glob("*.exr"):
        print(f"\n  Checking {exr_file.name}...")
        try:
            exr = OpenEXR.InputFile(str(exr_file))
            header = exr.header()
            channels = list(header['channels'].keys())
            print(f"    Channels: {channels}")

            if 'Object Index' in channels:
                print(f"    Object Index channel: FOUND ✓")

                # Read Object Index channel
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1

                obj_index = np.frombuffer(
                    exr.channel('Object Index', Imath.PixelType(Imath.PixelType.FLOAT)),
                    dtype=np.float32
                )
                obj_index = obj_index.reshape(height, width)

                print(f"    Shape: {obj_index.shape}")
                print(f"    Min: {obj_index.min()}")
                print(f"    Max: {obj_index.max()}")
                print(f"    Mean: {obj_index.mean()}")

                # Find unique values
                unique_values = np.unique(obj_index)
                print(f"    Unique values: {unique_values}")
                print(f"    Number of unique values: {len(unique_values)}")

                if len(unique_values) > 1:
                    print(f"    ✓ Object Index contains distinct values")
                else:
                    print(f"    ✗ Object Index does NOT contain distinct values")
            else:
                print(f"    Object Index channel: NOT FOUND")
                print(f"    Available channels: {channels}")

        except Exception as e:
            print(f"    Error reading EXR: {e}")

except ImportError:
    print("  OpenEXR not available")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
