#!/usr/bin/env python3
"""Test shadow direction empirically in Blender 5.2"""

import bpy
import math
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "shadow_test"
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
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# Create ground plane
bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
ground = bpy.context.object
ground.name = "Ground"

# Create cube
bpy.ops.mesh.primitive_cube_add(size=5, location=(0, 0, 5))
cube = bpy.context.object
cube.name = "TestCube"

# Create camera
bpy.ops.object.camera_add(location=(0, -50, 30))
cam = bpy.context.object
cam.name = "TestCamera"
cam.rotation_euler = (math.radians(60), 0, 0)
scene.camera = cam

# Create sun light
light_data = bpy.data.lights.new("Sun", "SUN")
sun = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(sun)
sun.location = (0, 0, 100)
light_data.energy = 5.0

# Test different sun rotations
test_rotations = [
    ("straight_down", (0, 0, 0)),  # Sun directly above
    ("rotated_90_z", (0, 0, math.radians(90))),  # Rotated 90° around Z
    ("rotated_minus_90_z", (0, 0, math.radians(-90))),  # Rotated -90° around Z
    ("azimuth_45", (math.radians(90 - 45), 0, math.radians(45))),  # Azimuth 45° (our preset)
]

for name, rotation in test_rotations:
    print(f"\n=== Testing rotation: {name} ===")
    print(f"  rotation_euler: {[math.degrees(r) for r in rotation]}")

    # Set sun rotation
    sun.rotation_euler = rotation

    # Render
    scene.render.filepath = str(OUTPUT_DIR / f"{name}.png")
    bpy.ops.render.render(write_still=True)

    # Analyze shadow direction
    import numpy as np
    from PIL import Image

    img = np.array(Image.open(OUTPUT_DIR / f"{name}.png"))
    h, w = img.shape[:2]

    # Find shadow region (below cube)
    shadow_region = img[h//2:h//2+50, :, :3]

    # Find left vs right brightness
    left_half = shadow_region[:, :w//2]
    right_half = shadow_region[:, w//2:]
    left_mean = left_half.mean()
    right_mean = right_half.mean()

    print(f"  Left half mean: {left_mean:.2f}")
    print(f"  Right half mean: {right_mean:.2f}")

    if left_mean < right_mean:
        print(f"  Shadow cast toward LEFT (west)")
    else:
        print(f"  Shadow cast toward RIGHT (east)")

print("\n=== Summary ===")
print("For azimuth=45° (northeast sun), shadows should point southwest (west)")
print("If shadow points east, the sign convention is inverted")
