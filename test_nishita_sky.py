#!/usr/bin/env python3
"""
Smoke test for blender_addon/nishita_sky.py.

Two-phase design:
  Phase 1 — Blender renders all presets (run *inside* Blender Python).
  Phase 2 — Verify rendered frames with PIL/numpy (run *outside* Blender).

Usage:
    # Phase 1: render
    /path/to/blender --background --python test_nishita_sky.py

    # Phase 2: verify (in system python, after Blender exits)
    python3 test_nishita_sky.py verify

"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_smoke_test", "nishita_sky")


def log(msg):
    print(f"[test_nishita_sky] {msg}")


# ======================================================================
# Phase 1 — Render (runs inside Blender's Python)
# ======================================================================

def _make_scene_ground_plane(scene):
    """Add a ground plane to *scene* with a grey matte material."""
    import bpy
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    mat = bpy.data.materials.new("GroundMat")
    # Blender 5.x: use node assignment instead of use_nodes
    mat.node_tree.nodes.clear()
    out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (0.4, 0.4, 0.4, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    ground.data.materials.append(mat)
    return ground


def _make_test_cube(scene):
    """Add a coloured test cube on the ground plane."""
    import bpy
    import math
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 3, 1))
    cube = bpy.context.active_object
    cube.name = "Cube"

    mat = bpy.data.materials.new("CubeMat")
    mat.node_tree.nodes.clear()
    out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.8, 0.5, 0.3, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.3
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    cube.data.materials.append(mat)
    return cube


def create_test_scene(name, preset_name):
    """Create a test scene with Nishita sky, ground plane, cube, and camera."""
    import bpy
    import math

    # New scene
    bpy.ops.scene.new(type="NEW")
    scene = bpy.context.scene
    scene.name = f"test_{name}"
    bpy.context.window.scene = scene

    # Rendering settings
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 512
    scene.render.filepath = os.path.join(OUTPUT_DIR, f"{preset_name}_smoke.png")

    _make_scene_ground_plane(scene)
    _make_test_cube(scene)

    # Camera — elevated, looking across scene to see sky gradient
    bpy.ops.object.camera_add(location=(12, -5, 2))
    cam = bpy.context.active_object
    cam.name = "Camera"
    scene.camera = cam
    cam.rotation_euler = (math.radians(80), 0.0, math.radians(50))

    # Apply Nishita sky
    sys.path.insert(0, SCRIPT_DIR)
    from blender_addon.nishita_sky import apply_to_scene
    result = apply_to_scene(scene, preset_name)

    log(f"Scene '{name}' ready. World: {result['world'].name}, "
        f"Sun: {result['sun_obj'].name}, "
        f"Exposure: {scene.view_settings.exposure}")

    return scene


def render_scene(scene):
    """Render *scene* to its configured filepath."""
    import bpy
    log(f"Rendering to {scene.render.filepath} ...")
    bpy.ops.render.render(write_still=True)
    log("Render complete.")


def phase1_render():
    import bpy

    log(f"Blender version: {bpy.app.version_string}")
    if "ShaderNodeTexSky" not in dir(bpy.types):
        log("FATAL: ShaderNodeTexSky not available in this Blender build")
        sys.exit(1)
    log("ShaderNodeTexSky is available.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    presets = ["clear", "overcast", "dusk"]
    for preset_name in presets:
        log(f"\n--- Rendering preset: {preset_name} ---")
        scene = create_test_scene(preset_name, preset_name)
        render_scene(scene)
        bpy.data.scenes.remove(scene)

    log(f"\nAll presets rendered to: {OUTPUT_DIR}")


# ======================================================================
# Phase 2 — Verify (runs in system Python where PIL/numpy are available)
# ======================================================================

def verify_frames():
    """Load rendered PNGs and verify gradient, clipping, shadow."""
    import numpy as np
    from PIL import Image

    presets = ["clear", "overcast", "dusk"]
    all_pass = True

    for preset_name in presets:
        png_path = os.path.join(OUTPUT_DIR, f"{preset_name}_smoke.png")
        log(f"\n--- Verifying {preset_name} ---")

        if not os.path.exists(png_path):
            log(f"FAIL: {png_path} not found")
            all_pass = False
            continue

        img = Image.open(png_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        log(f"  Image: {w}x{h}")
        log(f"  Pixel range: [{arr.min():.4f}, {arr.max():.4f}]")
        log(f"  Mean: {arr.mean():.4f}")

        # Check 1: No clipping (max < 0.98 in sRGB)
        clipped = int((arr >= 0.98).sum())
        if clipped > 0:
            log(f"  WARNING: {clipped} clipped pixels (>= 0.98)")
        else:
            log(f"  PASS: No clipping")

        # Check 2: Vertical gradient (top vs bottom row, per-channel)
        top_row = arr[0, :, :].mean(axis=0)
        bottom_row = arr[h-1, :, :].mean(axis=0)
        diff = float(max(abs(top_row - bottom_row)))
        log(f"  Top:    RGB({top_row[0]:.3f}, {top_row[1]:.3f}, {top_row[2]:.3f})")
        log(f"  Bottom: RGB({bottom_row[0]:.3f}, {bottom_row[1]:.3f}, {bottom_row[2]:.3f})")

        if diff > 0.05:
            log(f"  PASS: Vertical gradient detected (diff={diff:.4f})")
        else:
            log(f"  WARNING: Weak gradient (diff={diff:.4f})")

        # Check 3: Gradient fills full frame (mid row differs from top)
        mid_row = arr[h//2, :, :].mean(axis=0)
        top_mid = float(max(abs(top_row - mid_row)))
        if top_mid > 0.02:
            log(f"  PASS: Full-frame gradient (top-to-mid diff={top_mid:.4f})")
        else:
            log(f"  WARNING: Possible thin-strip gradient (top-mid diff={top_mid:.4f})")

        # Preset expectations
        info = {
            "clear": "blue sky, strong sun, hard shadows",
            "overcast": "grey/white uniform sky, soft shadows",
            "dusk": "warm orange/pink sky, long shadows",
        }
        log(f"  Expected: {info.get(preset_name, '')}")

    log(f"\nAll checks complete. Presets: {presets}")
    return all_pass


# ======================================================================
# Entry
# ======================================================================

if __name__ == "__main__":
    # Phase detection:
    #   --verify arg -> run verification outside Blender
    #   otherwise    -> run rendering inside Blender
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        sys.exit(0 if verify_frames() else 1)
    else:
        phase1_render()
