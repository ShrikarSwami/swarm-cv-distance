#!/usr/bin/env python3
"""
Smoke test for ``blender_addon.terrain``.

Launched via::

    blender --background --python test_terrain.py

Creates two terrain presets (desert and forest), renders each at
1280x720 with a low-angle camera, and verifies that the rendered
images show measurable terrain relief (non-uniform pixels in the
ground region).
"""

import sys
import math
from pathlib import Path

# -- Ensure the repo root is on sys.path so we can import blender_addon ---
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bpy
import mathutils
import numpy as np

from blender_addon.terrain import (
    ProceduralTerrain,
    configure_desert,
    configure_forest_floor,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = _REPO_ROOT / "dataset_smoke_test" / "terrain_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
RENDER_SAMPLES = 64
SUN_ENERGY = 5.0
SUN_ELEVATION_DEG = 55.0
SUN_AZIMUTH_DEG = 30.0

# Threshold for pixel standard deviation in the ground region.
# A flat plane with uniform material has std < 0.01 at 64 samples
# (just Cycles firefly noise).  Terrain with visible relief should
# yield std > 0.03.
VARIANCE_THRESHOLD = 0.03


# ---------------------------------------------------------------------------
# Scene setup helpers
# ---------------------------------------------------------------------------


def reset_scene():
    """Factory-reset the scene and choose Cycles as the render engine."""
    try:
        bpy.ops.wm.read_factory_use_empty(use_empty=True)
    except AttributeError:
        bpy.ops.wm.read_homefile(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True

    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"

    # Ensure Filmic view transform is used (gives nicer contrast)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"

    return scene


def add_camera(scene):
    """Add a camera with a low angle looking across the terrain."""
    cam_data = bpy.data.cameras.new("TestCamera")
    cam_obj = bpy.data.objects.new("TestCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Position: south edge, slightly elevated
    cam_obj.location = (0.0, -1800.0, 60.0)

    # Look toward the centre of the terrain at a modest height
    target = mathutils.Vector((0.0, 0.0, 12.0))
    direction = target - cam_obj.location
    quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = quat.to_euler()

    return cam_obj


def add_sun(scene):
    """Create a bright sun light."""
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = SUN_ENERGY
    light_data.color = (1.0, 0.95, 0.90)
    sun_obj = bpy.data.objects.new("Sun", light_data)
    scene.collection.objects.link(sun_obj)
    sun_obj.location = (0.0, 0.0, 10000.0)
    sun_obj.rotation_euler = (
        math.radians(90.0 - SUN_ELEVATION_DEG),
        0.0,
        math.radians(SUN_AZIMUTH_DEG),
    )
    return sun_obj


def setup_world_background(scene):
    """Set a simple sky-blue world background."""
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True

    tree = world.node_tree
    tree.nodes.clear()

    bg = tree.nodes.new("ShaderNodeBackground")
    bg.location = (0, 0)
    bg.inputs["Color"].default_value = (0.5, 0.6, 0.9, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.location = (300, 0)

    tree.links.new(bg.outputs["Background"], output.inputs["Surface"])


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_terrain_relief(image_path: Path, label: str) -> bool:
    """Check that the rendered image has measurable pixel variance.

    If PIL is available we use it (more standard); otherwise fall back
    to Blender's bundled numpy and image data.
    """
    if image_path.exists():
        use_pil = False
        try:
            from PIL import Image as PILImage
            use_pil = True
        except ImportError:
            pass

        if use_pil:
            img = np.array(PILImage.open(str(image_path)))
            h, w = img.shape[:2]
            # Ground region: bottom 60 %
            ground = img[int(h * 0.4):, :, :3].astype(np.float32) / 255.0
        else:
            # Fall back to Blender image load
            img = bpy.data.images.load(str(image_path))
            w_img, h_img = img.size
            pixels = np.array(img.pixels[:]).reshape(h_img, w_img, 4)
            bpy.data.images.remove(img)
            h, w = h_img, w_img
            ground = pixels[int(h * 0.4):, :, :3]

        gray = np.mean(ground, axis=2)
        std = float(np.std(gray))

        print(f"  Ground region std = {std:.5f}")

        if std < VARIANCE_THRESHOLD:
            print(f"  FAIL [{label}]: terrain appears flat (std={std:.5f} < "
                  f"{VARIANCE_THRESHOLD})")
            return False
        else:
            print(f"  PASS [{label}]: visible relief (std={std:.5f})")
            return True
    else:
        print(f"  SKIP [{label}]: output file not found")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    tests_passed = 0
    tests_total = 0

    # ---- Desert terrain --------------------------------------------------
    print("\n=== Desert Terrain ===")

    scene = reset_scene()
    setup_world_background(scene)
    add_sun(scene)
    cam = add_camera(scene)

    terrain_desert = ProceduralTerrain(seed=42)
    configure_desert(terrain_desert)
    terrain_desert.build(scene)

    desert_path = OUTPUT_DIR / "desert.png"
    scene.render.filepath = str(desert_path)
    bpy.ops.render.render(write_still=True)

    tests_total += 1
    if verify_terrain_relief(desert_path, "desert"):
        tests_passed += 1

    # ---- Forest terrain --------------------------------------------------
    print("\n=== Forest Floor Terrain ===")

    scene = reset_scene()
    setup_world_background(scene)
    add_sun(scene)
    cam = add_camera(scene)

    terrain_forest = ProceduralTerrain(seed=137)
    configure_forest_floor(terrain_forest)
    terrain_forest.build(scene)

    forest_path = OUTPUT_DIR / "forest.png"
    scene.render.filepath = str(forest_path)
    bpy.ops.render.render(write_still=True)

    tests_total += 1
    if verify_terrain_relief(forest_path, "forest"):
        tests_passed += 1

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"Results: {tests_passed}/{tests_total} passed")
    print(f"Output: {OUTPUT_DIR}")

    if tests_passed < tests_total:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
