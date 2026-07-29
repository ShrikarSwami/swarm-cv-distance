#!/usr/bin/env python3
"""Sweep HDRI Background.Strength for optimal exposure"""

import sys
import os
import bpy
import math
import time as time_module
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure blender_addon is importable
_project_root = str(Path(__file__).resolve().parent)
_addon_dir = os.path.join(_project_root, "blender_addon")
for p in [_project_root, _addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "hdri_strength_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def render_with_strength(hdri_strength, preset_name="clear"):
    """Render scene with given HDRI strength and analyze."""
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

    # Apply environment preset
    from blender_addon.environments import get_environment
    env_preset = get_environment("desert")
    env_preset.apply(scene)

    # Apply weather preset
    from blender_addon.weather import get_weather
    weather_preset = get_weather("clear")
    weather_preset.apply(scene, hdri_active=True)

    # Apply HDRI preset with custom strength
    from blender_addon.hdri import PRESETS, download_hdri
    preset = PRESETS[preset_name]

    # Download HDRI
    hdri_path = download_hdri(preset.asset_id)

    # Set up world with custom HDRI strength
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

    # Set sun lamp rotation
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            obj.rotation_euler = (
                math.radians(90.0 - preset.sun_elevation),
                0.0,
                math.radians(preset.sun_azimuth),
            )

    # Create camera
    bpy.ops.object.camera_add(location=(0, -100, 50))
    cam = [obj for obj in bpy.data.objects if obj.type == 'CAMERA'][-1]
    cam.name = "TestCamera"
    cam.rotation_euler = (math.radians(75), 0, 0)
    scene.camera = cam

    # Create cube drones
    for i in range(5):
        x = (i - 2) * 10
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
        cube = [obj for obj in bpy.data.objects if obj.type == 'MESH'][-1]
        cube.name = f"Drone_{i:02d}"

        mat = bpy.data.materials.new(f"DroneMat_{i:02d}")
        mat.use_nodes = True
        mnodes = mat.node_tree.nodes
        mlinks = mat.node_tree.links
        for node in mnodes:
            mnodes.remove(node)

        emission = mnodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 2.0
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        moutput = mnodes.new("ShaderNodeOutputMaterial")
        moutput.location = (200, 0)
        mlinks.new(emission.outputs["Emission"], moutput.inputs["Surface"])

        cube.data.materials.append(mat)
        cube.pass_index = i + 1

    # Render
    scene.render.filepath = str(OUTPUT_DIR / f"{preset_name}_{hdri_strength}")
    render_start = time_module.time()
    bpy.ops.render.render(write_still=True)
    render_time = time_module.time() - render_start

    # Analyze
    img = np.array(Image.open(OUTPUT_DIR / f"{preset_name}_{hdri_strength}.png"))
    h, w = img.shape[:2]

    center_region = img[h//4:3*h//4, w//4:3*w//4, :3]
    center_mean = center_region.mean(axis=(0,1))

    corners = [
        img[:h//4, :w//4, :3],
        img[:h//4, 3*w//4:, :3],
        img[3*h//4:, :w//4, :3],
        img[3*h//4:, 3*w//4:, :3],
    ]
    bg_mean = np.mean([c.mean(axis=(0,1)) for c in corners], axis=0)

    diff = np.abs(center_mean - bg_mean).max()

    # Check shadow direction
    drone_y = h // 2
    shadow_region = img[drone_y+50:drone_y+100, :, :3]
    left_half = shadow_region[:, :w//2]
    right_half = shadow_region[:, w//2:]
    left_mean = left_half.mean()
    right_mean = right_half.mean()
    shadow_direction = "west" if left_mean < right_mean else "east"

    return center_mean, bg_mean, diff, shadow_direction, render_time

# Sweep HDRI strengths for clear preset
print("=" * 70)
print("HDRI STRENGTH SWEEP - CLEAR PRESET")
print("=" * 70)

results = []
for strength in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]:
    center, bg, diff, shadow, time = render_with_strength(strength, "clear")
    results.append((strength, center, bg, diff, shadow, time))
    print(f"\nStrength {strength}:")
    print(f"  Center: {center}")
    print(f"  Background: {bg}")
    print(f"  Diff: {diff:.4f}")
    print(f"  Shadow: {shadow}")
    print(f"  Time: {time:.2f}s")

def find_optimal(results):
    """Find strength with best contrast (diff) while avoiding clipping."""
    best_strength = None
    best_diff = 0
    for strength, center, bg, diff, shadow, time in results:
        center_clipped = center.max() > 250 or center.min() < 5
        bg_clipped = bg.max() > 250 or bg.min() < 5
        if not center_clipped and not bg_clipped and diff > best_diff:
            best_diff = diff
            best_strength = strength
    return best_strength, best_diff

# Find optimal for clear
bs, bd = find_optimal(results)
print(f"\nOptimal clear strength: {bs} (diff={bd:.4f})")
if bs:
    center, bg, diff, shadow, time = render_with_strength(bs, "clear")
    print(f"  Final: Center={center}, BG={bg}, Diff={diff:.4f}, Shadow={shadow}, Time={time:.2f}s")

# ---------------------------------------------------------------------------
# OVERCAST PRESET SWEEP
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HDRI STRENGTH SWEEP - OVERCAST PRESET")
print("=" * 70)

oc_results = []
for strength in [0.8, 0.5, 0.3, 0.2, 0.15, 0.1]:
    center, bg, diff, shadow, time = render_with_strength(strength, "overcast")
    oc_results.append((strength, center, bg, diff, shadow, time))
    print(f"\nStrength {strength}:")
    print(f"  Center: {center}")
    print(f"  Background: {bg}")
    print(f"  Diff: {diff:.4f}")
    print(f"  Shadow: {shadow}")
    print(f"  Time: {time:.2f}s")

bs_oc, bd_oc = find_optimal(oc_results)
print(f"\nOptimal overcast strength: {bs_oc} (diff={bd_oc:.4f})")
if bs_oc:
    center, bg, diff, shadow, time = render_with_strength(bs_oc, "overcast")
    print(f"  Final: Center={center}, BG={bg}, Diff={diff:.4f}, Shadow={shadow}, Time={time:.2f}s")

# ---------------------------------------------------------------------------
# DUSK PRESET SWEEP
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HDRI STRENGTH SWEEP - DUSK PRESET")
print("=" * 70)

dk_results = []
for strength in [1.2, 0.8, 0.5, 0.3, 0.2]:
    center, bg, diff, shadow, time = render_with_strength(strength, "dusk")
    dk_results.append((strength, center, bg, diff, shadow, time))
    print(f"\nStrength {strength}:")
    print(f"  Center: {center}")
    print(f"  Background: {bg}")
    print(f"  Diff: {diff:.4f}")
    print(f"  Shadow: {shadow}")
    print(f"  Time: {time:.2f}s")

bs_dk, bd_dk = find_optimal(dk_results)
print(f"\nOptimal dusk strength: {bs_dk} (diff={bd_dk:.4f})")
if bs_dk:
    center, bg, diff, shadow, time = render_with_strength(bs_dk, "dusk")
    print(f"  Final: Center={center}, BG={bg}, Diff={diff:.4f}, Shadow={shadow}, Time={time:.2f}s")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Clear:    optimal strength={bs},   diff={bd:.4f}")
print(f"  Overcast: optimal strength={bs_oc}, diff={bd_oc:.4f}")
print(f"  Dusk:     optimal strength={bs_dk}, diff={bd_dk:.4f}")
