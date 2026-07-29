#!/usr/bin/env python3
"""Sweep HDRI Background.Strength for optimal exposure.

Usage:
    blender --background --python test_sweep_hdri.py
"""

import sys, os, math, time as time_module
import bpy
import numpy as np
from pathlib import Path
from PIL import Image

_project_root = str(Path(__file__).resolve().parent)
_addon_dir = os.path.join(_project_root, "blender_addon")
for p in [_project_root, _addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from blender_addon.hdri import PRESETS, download_hdri


OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "hdri_sweep_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_single(preset_name: str, hdri_strength: float):
    """Render one frame with given preset+strength, return pixel analysis."""
    # Factory reset
    try:
        bpy.ops.wm.read_factory_use_empty(use_empty=True)
    except AttributeError:
        bpy.ops.wm.read_homefile(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"

    # Environment (desert) + weather
    from blender_addon.environments import get_environment
    from blender_addon.weather import get_weather
    env_preset = get_environment("desert")
    env_preset.apply(scene)
    weather_preset = get_weather("clear")
    weather_preset.apply(scene, hdri_active=True)

    # HDRI — manual node setup with custom strength
    preset = PRESETS[preset_name]
    hdri_path = download_hdri(preset.asset_id)

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (600, 0)

    bg = nodes.new("ShaderNodeBackground")
    bg.location = (300, 0)
    bg.inputs["Strength"].default_value = hdri_strength
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (0, 0)
    env_tex.image = bpy.data.images.load(str(hdri_path))
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-200, 0)
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(preset.sun_azimuth))
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-400, 0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    # Sun lamp rotation
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            obj.rotation_euler = (
                math.radians(90.0 - preset.sun_elevation),
                0.0,
                math.radians(preset.sun_azimuth),
            )
            break

    # Camera
    bpy.ops.object.camera_add(location=(0, -100, 50))
    cam = [o for o in bpy.data.objects if o.type == "CAMERA"][-1]
    cam.name = "SweepCam"
    cam.rotation_euler = (math.radians(75), 0, 0)
    scene.camera = cam

    # Drone cubes (same as original clear preset sweep)
    mat = bpy.data.materials.new("DroneEmission")
    mat.use_nodes = True
    mn = mat.node_tree.nodes
    for n in mn: mn.remove(n)
    emission = mn.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 2.0
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    mout = mn.new("ShaderNodeOutputMaterial")
    mout.location = (200, 0)
    mat.node_tree.links.new(emission.outputs["Emission"], mout.inputs["Surface"])

    for i in range(5):
        x = (i - 2) * 10
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
        cube = [o for o in bpy.data.objects if o.type == "MESH"][-1]
        cube.name = f"Drone_{i:02d}"
        cube.data.materials.clear()
        cube.data.materials.append(mat)
        cube.pass_index = i + 1

    # Render
    out_path = OUTPUT_DIR / f"{preset_name}_{hdri_strength}.png"
    scene.render.filepath = str(out_path)
    t0 = time_module.time()
    bpy.ops.render.render(write_still=True)
    render_time = time_module.time() - t0

    # Analyze
    img = np.array(Image.open(out_path))
    h, w = img.shape[:2]

    # Center region (where drones are)
    center_region = img[h//4:3*h//4, w//4:3*w//4, :3]
    center_mean = center_region.mean(axis=(0, 1))

    # Corner regions (background/sky)
    corners = [
        img[:h//4, :w//4, :3], img[:h//4, 3*w//4:, :3],
        img[3*h//4:, :w//4, :3], img[3*h//4:, 3*w//4:, :3],
    ]
    bg_mean = np.mean([c.mean(axis=(0, 1)) for c in corners], axis=0)

    diff = np.abs(center_mean - bg_mean).max()

    # Shadow direction
    drone_y = h // 2
    shadow_region = img[drone_y + 50:drone_y + 100, :, :3]
    left_half = shadow_region[:, :w//2]
    right_half = shadow_region[:, w//2:]
    left_mean = left_half.mean()
    right_mean = right_half.mean()
    shadow_dir = "west" if left_mean < right_mean else "east"

    return center_mean, bg_mean, diff, shadow_dir, render_time


def sweep_preset(name: str, strengths: list):
    """Sweep a preset at the given strengths."""
    print(f"\n{'='*70}")
    print(f"SWEEP: {name.upper()}  —  asset={PRESETS[name].asset_id}")
    print(f"  sun_azimuth={PRESETS[name].sun_azimuth}°, "
          f"sun_elevation={PRESETS[name].sun_elevation}°")
    print(f"{'='*70}")
    print(f"{'Strength':>10} | {'Center (R,G,B)':>30} | {'BG (R,G,B)':>30} | {'Diff':>8} | {'Shadow':>8} | {'Time':>6}")
    print("-" * 100)

    results = []
    for s in strengths:
        center, bg, diff, shadow_dir, t = render_single(name, s)
        c_str = f"({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
        b_str = f"({bg[0]:.1f},{bg[1]:.1f},{bg[2]:.1f})"
        print(f"{s:>10.2f} | {c_str:>30} | {b_str:>30} | {diff:>8.2f} | {shadow_dir:>8} | {t:>5.2f}s")
        results.append((s, center, bg, diff, shadow_dir, t))

    # Find best: no clipping, best diff
    best_s, best_d = None, 0
    for s, center, bg, diff, shadow_dir, t in results:
        clipped = center.max() > 250 or center.min() < 5 or bg.max() > 250 or bg.min() < 5
        if not clipped and diff > best_d:
            best_d = diff
            best_s = s

    print(f"\n  Optimal: strength={best_s}, diff={best_d:.2f}")
    return best_s, results


# ===== MAIN =====

# 1. OVERCAST sweep
best_oc, oc_results = sweep_preset("overcast", [0.8, 0.5, 0.3, 0.2, 0.15, 0.1])

# 2. DUSK sweep
best_dk, dk_results = sweep_preset("dusk", [1.2, 0.8, 0.5, 0.3, 0.2, 0.15])

# 3. Final renders at all 3 optimal values
print("\n" + "=" * 70)
print("FINAL CONFIRMATION — ALL 3 PRESETS")
print("=" * 70)
final_values = {"clear": 0.15, "overcast": best_oc, "dusk": best_dk}

print(f"{'Preset':>12} | {'Strength':>10} | {'Center (R,G,B)':>30} | {'BG (R,G,B)':>30} | {'Diff':>8} | {'Shadow':>8} | {'Time':>6}")
print("-" * 110)
for pname, pstr in final_values.items():
    center, bg, diff, shadow_dir, t = render_single(pname, pstr)
    c_str = f"({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
    b_str = f"({bg[0]:.1f},{bg[1]:.1f},{bg[2]:.1f})"
    print(f"{pname:>12} | {pstr:>10.2f} | {c_str:>30} | {b_str:>30} | {diff:>8.2f} | {shadow_dir:>8} | {t:>5.2f}s")
