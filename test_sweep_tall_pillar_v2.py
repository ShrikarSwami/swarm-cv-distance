#!/usr/bin/env python3
"""Sweep HDRI strength for clear and dusk using tall-pillar shadow verification.
V2: Broader sweep range since the HDRI assets were just swapped to the correct ones.
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

OUTPUT_DIR = Path(_project_root) / "dataset_smoke_test" / "sweep_tall_pillar_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_with_strength(preset_name, hdri_strength, render_name):
    """Render a scene with given preset+strength, return pixel analysis."""
    try:
        bpy.ops.wm.read_factory_use_empty(use_empty=True)
    except AttributeError:
        bpy.ops.wm.read_homefile(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"

    preset = PRESETS[preset_name]
    hdri_path = download_hdri(preset.asset_id)

    # Environment + weather
    from blender_addon.environments import get_environment
    from blender_addon.weather import get_weather
    get_environment("desert").apply(scene)
    get_weather("clear").apply(scene, hdri_active=True)

    # Manual HDRI with custom strength
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = hdri_strength
    links.new(bg.outputs["Background"], output.inputs["Surface"])
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(str(hdri_path))
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(preset.sun_azimuth))
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    tex_coord = nodes.new("ShaderNodeTexCoord")
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    # Sun lamp rotation
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            obj.rotation_euler = (math.radians(90 - preset.sun_elevation), 0, math.radians(preset.sun_azimuth))
            break

    # Camera
    bpy.ops.object.camera_add(location=(0, -100, 50))
    cam = [o for o in bpy.data.objects if o.type == "CAMERA"][-1]
    cam.rotation_euler = (math.radians(75), 0, 0)
    scene.camera = cam

    # 5 drone cubes
    m = bpy.data.materials.new("DroneEm")
    m.use_nodes = True
    ns = m.node_tree.nodes
    for n in ns: ns.remove(n)
    e = ns.new("ShaderNodeEmission")
    e.inputs["Strength"].default_value = 2.0
    e.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    o = ns.new("ShaderNodeOutputMaterial")
    o.location = (200, 0)
    m.node_tree.links.new(e.outputs["Emission"], o.inputs["Surface"])

    for i in range(5):
        x = (i - 2) * 10
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
        cube = [o for o in bpy.data.objects if o.type == "MESH"][-1]
        cube.name = f"Drone_{i:02d}"
        cube.data.materials.clear()
        cube.data.materials.append(m)
        cube.pass_index = i + 1

    # Render
    out_path = OUTPUT_DIR / f"{render_name}.png"
    scene.render.filepath = str(out_path)
    t0 = time_module.time()
    bpy.ops.render.render(write_still=True)
    render_time = time_module.time() - t0

    # Analyze
    img = np.array(Image.open(out_path))
    h, w = img.shape[:2]
    center_region = img[h//4:3*h//4, w//4:3*w//4, :3]
    center_mean = center_region.mean(axis=(0, 1))
    corners = [img[:h//4, :w//4, :3], img[:h//4, 3*w//4:, :3],
               img[3*h//4:, :w//4, :3], img[3*h//4:, 3*w//4:, :3]]
    bg_mean = np.mean([c.mean(axis=(0, 1)) for c in corners], axis=0)
    diff = np.abs(center_mean - bg_mean).max()

    return center_mean, bg_mean, diff, render_time


def render_tall_pillar(preset_name, hdri_strength, render_name):
    """Render tall pillar for shadow direction verification."""
    try:
        bpy.ops.wm.read_factory_use_empty(use_empty=True)
    except AttributeError:
        bpy.ops.wm.read_homefile(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x = 400
    scene.render.resolution_y = 400
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"

    preset = PRESETS[preset_name]
    hdri_path = download_hdri(preset.asset_id)

    from blender_addon.environments import get_environment
    from blender_addon.weather import get_weather
    get_environment("desert").apply(scene)
    get_weather("clear").apply(scene, hdri_active=True)

    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = hdri_strength
    links.new(bg.outputs["Background"], output.inputs["Surface"])
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(str(hdri_path))
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(preset.sun_azimuth))
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    tex_coord = nodes.new("ShaderNodeTexCoord")
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            obj.rotation_euler = (math.radians(90 - preset.sun_elevation), 0, math.radians(preset.sun_azimuth))
            break

    bpy.ops.object.camera_add(location=(0, -20, 8))
    cam = [o for o in bpy.data.objects if o.type == "CAMERA"][-1]
    cam.rotation_euler = (math.radians(72), 0, 0)
    scene.camera = cam

    m = bpy.data.materials.new("Em")
    m.use_nodes = True
    ns = m.node_tree.nodes
    for n in ns: ns.remove(n)
    e = ns.new("ShaderNodeEmission")
    e.inputs["Strength"].default_value = 2.0
    e.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    o = ns.new("ShaderNodeOutputMaterial")
    o.location = (200, 0)
    m.node_tree.links.new(e.outputs["Emission"], o.inputs["Surface"])

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 10))
    pillar = bpy.context.object
    pillar.scale = (2, 2, 20)
    bpy.ops.object.transform_apply()
    pillar.data.materials.clear()
    pillar.data.materials.append(m)

    out_path = OUTPUT_DIR / f"{render_name}.png"
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


# ===== MAIN =====
print("=" * 70)
print("EXPOSURE SWEEP V2 — TALL PILLAR METHOD")
print("=" * 70)

# CLEAR sweep (kloofendal_43d_clear_puresky, azimuth=45°, elev=60°)
print("\n--- CLEAR (kloofendal_43d_clear_puresky, azimuth=45°, elev=60°) ---")
print("Expected shadow: FROM NE (sun) → TOWARD SW (left from camera looking N)")
print(f"{'Strength':>10} | {'Center(RGB)':>30} | {'BG(RGB)':>30} | {'MaxDiff':>8} | {'Time':>6}")
print("-" * 95)

clear_results = []
for s in [0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02]:
    center, bg, diff, t = render_with_strength("clear", s, f"clear_{s}")
    c_str = f"({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
    b_str = f"({bg[0]:.1f},{bg[1]:.1f},{bg[2]:.1f})"
    print(f"{s:>10.3f} | {c_str:>30} | {b_str:>30} | {diff:>8.2f} | {t:>6.2f}s")
    clear_results.append((s, center, bg, diff, t))

# Find best clear: max diff without center clipping (>240 = near-white)
best_clear_s = max((r for r in clear_results if r[1].max() < 240), key=lambda x: x[3], default=None)
if best_clear_s is None:
    best_clear_s = max(clear_results, key=lambda x: x[3])[0]
else:
    best_clear_s = best_clear_s[0]
print(f"\nBest clear strength: {best_clear_s}")
print(f"  (max diff, center < 240 to avoid clipping)")

# Render tall pillar for clear shadow verification
print(f"\nRendering tall pillar for clear at {best_clear_s}...")
render_tall_pillar("clear", best_clear_s, "pillar_clear")
print("  Saved: pillar_clear.png")

# DUSK sweep (belfast_sunset_puresky, azimuth=90°, elev=15°)
print("\n--- DUSK (belfast_sunset_puresky, azimuth=90°, elev=15°) ---")
print("Expected shadow: FROM E (sun) → TOWARD W (left from camera looking N)")
print(f"{'Strength':>10} | {'Center(RGB)':>30} | {'BG(RGB)':>30} | {'MaxDiff':>8} | {'Time':>6}")
print("-" * 95)

dusk_results = []
for s in [1.0, 0.8, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05]:
    center, bg, diff, t = render_with_strength("dusk", s, f"dusk_{s}")
    c_str = f"({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
    b_str = f"({bg[0]:.1f},{bg[1]:.1f},{bg[2]:.1f})"
    print(f"{s:>10.3f} | {c_str:>30} | {b_str:>30} | {diff:>8.2f} | {t:>6.2f}s")
    dusk_results.append((s, center, bg, diff, t))

# Find best dusk: max diff without center clipping
best_dusk_s = max((r for r in dusk_results if r[1].max() < 240), key=lambda x: x[3], default=None)
if best_dusk_s is None:
    best_dusk_s = max(dusk_results, key=lambda x: x[3])[0]
else:
    best_dusk_s = best_dusk_s[0]
print(f"\nBest dusk strength: {best_dusk_s}")

# Render tall pillar for dusk shadow verification
print(f"\nRendering tall pillar for dusk at {best_dusk_s}...")
render_tall_pillar("dusk", best_dusk_s, "pillar_dusk")
print("  Saved: pillar_dusk.png")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Clear:  strength={best_clear_s}")
print(f"Dusk:   strength={best_dusk_s}")
print("\nReview pillar_clear.png and pillar_dusk.png for shadow direction.")
print(f"\nOutput directory: {OUTPUT_DIR}")
