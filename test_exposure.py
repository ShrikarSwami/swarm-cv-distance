#!/usr/bin/env python3
"""Test emission and HDRI strength independently for exposure"""

import sys
import os
import bpy
import math
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure blender_addon is importable
_project_root = str(Path(__file__).resolve().parent)
_addon_dir = os.path.join(_project_root, "blender_addon")
for p in [_project_root, _addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "exposure_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def render_and_analyze(name, emission_strength, hdri_strength):
    """Render a scene with given strengths and analyze exposure."""
    print(f"\n=== Testing: {name} ===")
    print(f"  Emission strength: {emission_strength}")
    print(f"  HDRI strength: {hdri_strength}")

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
    preset = PRESETS["clear"]

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
    bg.inputs["Strength"].default_value = hdri_strength  # Custom HDRI strength
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (0, 0)
    env_tex.image = bpy.data.images.load(str(hdri_path))
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-200, 0)
    # Use the new API for mapping rotation
    mapping.inputs["Rotation"].default_value = (0, 0, preset.sun_azimuth)
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
    cam = bpy.context.object
    cam.name = "TestCamera"
    cam.rotation_euler = (math.radians(75), 0, 0)
    scene.camera = cam

    # Create cube drones with custom emission strength
    for i in range(5):
        x = (i - 2) * 10
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
        cube = bpy.context.object
        cube.name = f"Drone_{i:02d}"

        mat = bpy.data.materials.new(f"DroneMat_{i:02d}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = emission_strength  # Custom emission strength
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (200, 0)
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        cube.data.materials.append(mat)
        cube.pass_index = i + 1

    # Render
    scene.render.filepath = str(OUTPUT_DIR / f"{name}.png")
    render_start = time.time()
    bpy.ops.render.render(write_still=True)
    render_time = time.time() - render_start

    # Analyze
    img = np.array(Image.open(OUTPUT_DIR / f"{name}.png"))
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

    print(f"  Center pixel: {center_mean}")
    print(f"  Background pixel: {bg_mean}")
    print(f"  Difference: {diff:.4f}")
    print(f"  Render time: {render_time:.2f}s")

    return center_mean, bg_mean, diff

# Test different emission strengths with fixed HDRI strength
print("\n" + "="*70)
print("TEST 1: Varying emission strength (HDRI strength fixed at 1.0)")
print("="*70)

for emission_strength in [10.0, 2.0, 0.5, 0.2]:
    render_and_analyze(f"emission_{emission_strength}", emission_strength, 1.0)

# Test different HDRI strengths with fixed emission strength
print("\n" + "="*70)
print("TEST 2: Varying HDRI strength (emission strength fixed at 2.0)")
print("="*70)

for hdri_strength in [1.0, 0.5, 0.2, 0.1]:
    render_and_analyze(f"hdri_{hdri_strength}", 2.0, hdri_strength)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
