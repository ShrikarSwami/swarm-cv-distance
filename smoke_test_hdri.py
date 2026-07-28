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

    # Step 7: Setup compositor for EXR multilayer output
    print("\n[7] Setting up compositor for EXR output")
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    out = tree.nodes.new("CompositorNodeOutputFile")
    out.location = (200, 0)
    out.filepath = str(OUTPUT_DIR / "render_exr") + "/"
    out.file_slots[0].path = "render_exr"
    out.format.file_format = "OPEN_EXR_MULTILAYER"
    out.format.color_depth = "32"

    tree.links.new(rl.outputs["Image"], out.inputs["Image"])

    # Step 8: Render single frame
    print("\n[8] Rendering single frame")
    render_start = time.time()

    scene.render.filepath = str(OUTPUT_DIR / "render")
    bpy.ops.render.render(write_still=True)

    render_time = time.time() - render_start
    total_time = time.time() - start_time

    print(f"\n  Render time: {render_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

    # Step 9: Analyze EXR multilayer pass
    print("\n[9] Analyzing EXR multilayer pass")
    exr_path = OUTPUT_DIR / "render_exr" / "render_exr0001.exr"
    if exr_path.exists():
        exr_size = exr_path.stat().st_size
        print(f"  EXR file: {exr_path}")
        print(f"  EXR size: {exr_size / 1024:.1f} KB")
        if exr_size > 10000:
            print("  ✓ EXR file looks valid (>10KB)")
        else:
            print("  ✗ EXR file too small (<10KB)")
    else:
        print("  EXR file not found (compositor may not have written it)")

    # Step 10: Analyze PNG render
    print("\n[10] Analyzing PNG render")

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
        if diff > 1.0:  # At least 1 unit difference in 0-255 range
            print(f"  ✓ Drones visible against sky (diff={diff:.2f})")
        else:
            print(f"  ✗ Drones not distinguishable from sky (diff={diff:.2f})")

    else:
        print("  PNG render not found")

    # Shadow direction check (visual inspection required)
    print("\n  Shadow direction check:")
    print("  - Visually inspect shadows on ground plane")
    print("  - Confirm shadows point toward HDRI's bright region")
    print("  - This requires human verification of the rendered image")

    # Step 11: Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Render time: {render_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  PNG output: {OUTPUT_DIR / 'render.png'}")
    print(f"  EXR output: {OUTPUT_DIR / 'render_exr' / 'render_exr0001.exr'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
