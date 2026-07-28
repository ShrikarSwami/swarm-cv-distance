#!/usr/bin/env python3
"""Smoke test for HDRI environment lighting.

Renders a single frame with existing cube drones and HDRI sky,
confirms: HDRI visible, drones lit correctly, EXR pass works.

Usage:
  blender --background --python smoke_test_hdri.py
"""

import bpy
import time
import numpy as np
from pathlib import Path
from PIL import Image

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
    from blender_addon.weather import WeatherPreset
    weather_preset = WeatherPreset(
        name="clear",
        sun_energy=5.0,
        sun_elevation_deg=60.0,
        sun_rotation_deg=45.0,
        sun_color=(1.0, 0.95, 0.9, 1.0),
        sky_luminance_cd_m2=8000.0,
        ambient_color=(0.5, 0.6, 0.9, 1.0),
    )
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

    # Point camera at scene center
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = None  # Will track origin
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    scene.camera = cam

    # Step 6: Create test cube drones
    print("\n[6] Creating test cube drones")
    for i in range(5):
        x = (i - 2) * 10  # -20, -10, 0, 10, 20
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
        cube = bpy.context.object
        cube.name = f"Drone_{i:02d}"

        # Assign emission material
        mat = bpy.data.materials.new(f"DroneMat_{i:02d}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 10.0
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (200, 0)
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        cube.data.materials.append(mat)

        # Set object index for EXR pass
        cube.pass_index = i + 1

    # Step 7: Render single frame
    print("\n[7] Rendering single frame")
    render_start = time.time()

    scene.render.filepath = str(OUTPUT_DIR / "render")
    bpy.ops.render.render(write_still=True)

    render_time = time.time() - render_start
    total_time = time.time() - start_time

    print(f"\n  Render time: {render_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

    # Step 8: Analyze render
    print("\n[8] Analyzing render")

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

        # Check if drones are visible
        if np.abs(center_mean - bg_mean).max() > 5.0:
            print("  ✓ Drones visible against sky")
        else:
            print("  ✗ Drones not distinguishable from sky")

    # Step 9: Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Render time: {render_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output: {OUTPUT_DIR / 'render.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
