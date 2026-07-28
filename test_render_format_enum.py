#!/usr/bin/env python3
"""Test actual enum options for main render output format"""

import bpy
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "render_format_enum"
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
# CHECK 1: Print FULL raw enum list for file_format
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] Print FULL raw enum list for file_format")
try:
    enum_items = scene.render.image_settings.bl_rna.properties['file_format'].enum_items.keys()
    print(f"  Full enum list: {list(enum_items)}")
    print(f"  Number of options: {len(list(enum_items))}")

    # Check specifically for EXR formats
    exr_formats = [fmt for fmt in enum_items if 'EXR' in fmt.upper()]
    print(f"  EXR formats: {exr_formats}")
except Exception as e:
    print(f"  ERROR: {e}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2: Try setting OPEN_EXR_MULTILAYER if available
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] Try setting OPEN_EXR_MULTILAYER if available")

try:
    # Remove/disable compositor node group entirely
    if scene.compositing_node_group:
        scene.compositing_node_group = None
        print(f"  Removed compositor node group")
    scene.use_nodes = False
    print(f"  scene.use_nodes: {scene.use_nodes}")

    # Try to set OPEN_EXR_MULTILAYER
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    print(f"  Set file_format to OPEN_EXR_MULTILAYER: SUCCESS")
    print(f"  Current file_format: {scene.render.image_settings.file_format}")
except Exception as e:
    print(f"  ERROR setting OPEN_EXR_MULTILAYER: {e}")
    print(f"  Current file_format: {scene.render.image_settings.file_format}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3: Render and check output
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] Render and check output")

scene.render.filepath = str(OUTPUT_DIR / "render")
scene.render.image_settings.color_depth = "32"
print(f"  Rendering with format: {scene.render.image_settings.file_format}")
bpy.ops.render.render(write_still=True)
print(f"  Render complete!")

# Check output files
print(f"\n  Files in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4: Load EXR and check channels
# ══════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] Load EXR and check channels")

import numpy as np

# Find the rendered EXR file
exr_files = list(OUTPUT_DIR.glob("*.exr"))
if exr_files:
    exr_path = str(exr_files[0])
    print(f"  Loading EXR: {exr_path}")

    img = bpy.data.images.load(exr_path)
    print(f"  Loaded image: {img.name}")
    print(f"  Size: {img.size[0]}x{img.size[1]}")
    print(f"  Channels: {img.channels}")

    # Get pixels as numpy array
    pixels = np.array(img.pixels[:])
    print(f"  Pixels shape: {pixels.shape}")
    print(f"  Pixels min: {pixels.min()}")
    print(f"  Pixels max: {pixels.max()}")
    print(f"  Pixels mean: {pixels.mean()}")

    # Check if there are multiple images (Object Index might be separate)
    print(f"\n  All images in blend file:")
    for image in bpy.data.images:
        print(f"    {image.name}: {image.size[0]}x{image.size[1]}, {image.channels} channels")
else:
    print(f"  No EXR files found in {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("RENDER FORMAT ENUM TEST COMPLETE")
print("=" * 70)
