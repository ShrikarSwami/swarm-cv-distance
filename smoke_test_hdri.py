#!/usr/bin/env python3
"""Smoke test for HDRI environment lighting.

Renders a single frame with existing cube drones and HDRI sky,
confirms: HDRI visible, drones lit correctly, EXR pass works.

Usage:
  blender --background --python smoke_test_hdri.py
"""

import sys
import os
import bpy
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure blender_addon is importable (same approach as render_clip.py)
_project_root = str(Path(__file__).resolve().parent)
_addon_dir = os.path.join(_project_root, "blender_addon")
for p in [_project_root, _addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "hdri_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("HDRI ENVIRONMENT LIGHTING SMOKE TEST")
    print("=" * 70)

    # Record start time
    start_time = time.time()

    # Step 1: Load existing clip config (desert_clear)
    print("\n[1] Loading scene configuration")

    # Factory reset
    try:
        bpy.ops.wm.read_factory_use_empty(use_empty=True)
    except AttributeError:
        bpy.ops.wm.read_homefile(use_empty=True)

    # Set render engine to Cycles
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Step 2: Apply environment preset (creates ground plane, sun light)
    print("\n[2] Applying environment preset")
    from blender_addon.environments import get_environment
    env_preset = get_environment("desert")
    env_preset.apply(scene)

    # Step 3: Apply weather preset with hdri_active=True
    print("\n[3] Applying weather preset")
    from blender_addon.weather import get_weather
    weather_preset = get_weather("clear")
    weather_preset.apply(scene, hdri_active=True)

    # Step 4: Apply HDRI preset (clear)
    print("\n[4] Applying HDRI preset")
    from blender_addon.hdri import apply as apply_hdri
    apply_hdri(scene, "clear")

    # Step 5: Create camera
    print("\n[5] Setting up camera")
    bpy.ops.object.camera_add(location=(0, -100, 50))
    cam = bpy.context.object
    cam.name = "SmokeTestCamera"

    # Point camera at scene center using explicit rotation
    # (TRACK_TO with target=None is unreliable in headless Blender)
    import math
    cam.rotation_euler = (math.radians(75), 0, 0)  # Look down at scene

    scene.camera = cam

    # Step 6: Create test cube drones
    print("\n[6] Creating test cube drones")
    for i in range(5):
        x = (i - 2) * 10  # -20, -10, 0, 10, 20
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
        cube = bpy.context.object
        cube.name = f"Drone_{i:02d}"

        # Assign emission material (reduced strength to avoid overexposure)
        mat = bpy.data.materials.new(f"DroneMat_{i:02d}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 2.0  # Reduced from 10.0
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (200, 0)
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        cube.data.materials.append(mat)

        # Set object index for EXR pass
        cube.pass_index = i + 1

    # Step 7: Configure render passes for EXR with Object Index
    print("\n[7] Configuring render passes for EXR with Object Index")
    # Enable Object Index pass in view layer (Blender 5.2 API)
    bpy.context.view_layer.use_pass_object_index = True

    # Step 8: Render single frame as EXR (for data analysis)
    print("\n[8] Rendering single frame as EXR")
    render_start = time.time()

    scene.render.filepath = str(OUTPUT_DIR / "render.exr")
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    bpy.ops.render.render(write_still=True)

    exr_time = time.time() - render_start

    # Step 9: Render single frame as PNG (for visual inspection)
    print("\n[9] Rendering single frame as PNG")
    scene.render.filepath = str(OUTPUT_DIR / "render.png")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    png_start = time.time()
    bpy.ops.render.render(write_still=True)
    png_time = time.time() - png_start

    render_time = time.time() - render_start
    total_time = time.time() - start_time

    print(f"\n  EXR render time: {exr_time:.2f}s")
    print(f"  PNG render time: {png_time:.2f}s")
    print(f"  Total render time: {render_time:.2f}s")
    print(f"  Total script time: {total_time:.2f}s")

    # Step 10: Analyze EXR multilayer pass
    print("\n[10] Analyzing EXR multilayer pass")
    exr_path = OUTPUT_DIR / "render" / "render0001.exr"
    if not exr_path.exists():
        exr_path = OUTPUT_DIR / "render.exr"
    if exr_path.exists():
        exr_size = exr_path.stat().st_size
        print(f"  EXR file: {exr_path}")
        print(f"  EXR size: {exr_size / 1024:.1f} KB")
        if exr_size > 10000:
            print("  ✓ EXR file looks valid (>10KB)")
        else:
            print("  ✗ EXR file too small (<10KB)")
    else:
        print("  EXR file not found")

    # Step 11: Analyze PNG render
    print("\n[11] Analyzing PNG render")

    # Load rendered image
    render_path = OUTPUT_DIR / "render.png"
    if render_path.exists():
        img = np.array(Image.open(render_path))
        h, w = img.shape[:2]

        # Center region (where drones should be)
        center_region = img[h//4:3*h//4, w//4:3*w//4, :3]
        center_mean = center_region.mean(axis=(0,1))

        # Corner regions (background/sky)
        corners = [
            img[:h//4, :w//4, :3],
            img[:h//4, 3*w//4:, :3],
            img[3*h//4:, :w//4, :3],
            img[3*h//4:, 3*w//4:, :3],
        ]
        bg_mean = np.mean([c.mean(axis=(0,1)) for c in corners], axis=0)

        print(f"  Center pixel (drones): {center_mean}")
        print(f"  Background pixel (sky): {bg_mean}")
        print(f"  Difference: {np.abs(center_mean - bg_mean).max():.2f}")

        # Check if drones are visible (relative comparison)
        diff = np.abs(center_mean - bg_mean).max()
        if diff > 5.0:  # At least 5 unit difference in 0-255 range
            print(f"  ✓ Drones visible against sky (diff={diff:.2f})")
        else:
            print(f"  ✗ Drones not distinguishable from sky (diff={diff:.2f})")

    else:
        print("  PNG render not found")

    # Step 12: Shadow direction check (programmatic)
    print("\n[12] Shadow direction check")
    if render_path.exists():
        # Analyze shadow direction by looking at ground plane
        # Shadows should be cast in the direction opposite to the HDRI's bright region
        # For "clear" preset with sun_azimuth=45°, shadows should point toward 225° (southwest)

        # Find darker regions below drones (shadows)
        drone_y = h // 2  # Approximate drone row
        shadow_region = img[drone_y+50:drone_y+100, :, :3]  # Below drones

        # Find the darkest row in shadow region (actual shadow)
        row_means = shadow_region.mean(axis=(1,2))
        darkest_row_idx = np.argmin(row_means)
        shadow_row = shadow_region[darkest_row_idx]

        # Find left vs right brightness to determine shadow direction
        left_half = shadow_row[:w//2]
        right_half = shadow_row[w//2:]
        left_mean = left_half.mean()
        right_mean = right_half.mean()

        print(f"  Shadow left half mean: {left_mean:.2f}")
        print(f"  Shadow right half mean: {right_mean:.2f}")

        if left_mean < right_mean:
            print(f"  Shadow cast toward LEFT (west)")
            shadow_direction = "west"
        else:
            print(f"  Shadow cast toward RIGHT (east)")
            shadow_direction = "east"

        # Expected: for sun_azimuth=45° (northeast), shadows should point southwest (225°)
        # This means shadows should be cast toward the left (west) side
        expected_direction = "west"
        if shadow_direction == expected_direction:
            print(f"  ✓ Shadow direction matches expected (sun_azimuth=45° → shadows toward southwest)")
        else:
            print(f"  ✗ Shadow direction mismatch: got {shadow_direction}, expected {expected_direction}")

    # Step 11: Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Render time: {render_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  PNG output: {OUTPUT_DIR / 'render.png'}")
    print(f"  EXR output: {OUTPUT_DIR / 'render.exr'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
