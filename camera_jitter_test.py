"""
Camera Jitter Test (Layer 4b).

Tests subtle handheld-style camera jitter using noise modifiers on camera
animation curves. Renders a 30-frame sequence and compares stabilized vs.
jittered versions.

Two approaches:
  A) Noise modifier on camera location f-curves (built-in Blender)
  B) Python-generated random offsets per frame (more controllable)

Usage:
    blender --background --python camera_jitter_test.py
"""

import bpy
import math
import os
import random
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_smoke_test" / "camera_realism"
N_FRAMES = 30


def factory_reset():
    bpy.ops.wm.read_factory_settings()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = N_FRAMES
    return scene


def clear_scene(scene):
    for name in ["Cube", "Light", "Camera"]:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)


def setup_world(scene):
    """Sky and sun."""
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.5, 0.6, 0.8, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(bg.outputs["Background"], output.inputs["Surface"])

    # Sun
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 100))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.3, 0.0)


def create_target_scene(scene):
    """Create a simple scene with objects at varying distances."""
    # Ground
    bpy.ops.mesh.primitive_plane_add(size=500, location=(0, 0, -1))
    ground = bpy.context.object
    gmat = bpy.data.materials.new("ground")
    gmat.use_nodes = True
    bsdf = gmat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.25, 0.25, 0.25, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(gmat)

    # Central structure (tall pillar)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=8, radius=5, depth=40, location=(0, -50, 20)
    )
    pillar = bpy.context.object
    pmat = bpy.data.materials.new("pillar")
    pmat.use_nodes = True
    bsdf = pmat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.5, 0.3, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    pillar.data.materials.append(pmat)

    # Distant cubes at various depths
    for i, (x, y) in enumerate([(-30, -80), (0, -120), (30, -90), (-20, -150), (20, -60)]):
        bpy.ops.mesh.primitive_cube_add(size=8, location=(x, y, 4))
        cube = bpy.context.object
        cube.name = f"TargetCube_{i}"
        c = bpy.data.materials.new(f"cube_mat_{i}")
        c.use_nodes = True
        bsdf = c.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            color_val = 0.3 + 0.1 * i
            bsdf.inputs["Base Color"].default_value = (color_val, 0.2 + 0.1*i, 0.8 - 0.1*i, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.6
        cube.data.materials.append(c)


def create_stable_camera(scene):
    """Create a camera with no animation (stable)."""
    bpy.ops.object.camera_add(location=(0, 30, 15))
    cam = bpy.context.object
    cam.name = "StableCamera"
    scene.camera = cam
    # Look at center of scene (0, -50, 20)
    cam.rotation_euler = (0.25, 0, 0)
    return cam


def animate_camera_noise_modifier(scene, cam, pos_amp=0.3, rot_amp_deg=0.4):
    """Animate camera with noise modifiers on f-curves (Approach A).

    pos_amp: amplitude of position jitter in meters
    rot_amp_deg: amplitude of rotation jitter in degrees
    """
    cam.name = "JitteredCameraNoise"
    scene.camera = cam

    # Base position and rotation
    base_loc = cam.location.copy()
    base_rot = cam.rotation_euler.copy()

    # Insert keyframes for the whole range
    for frame in range(1, N_FRAMES + 1):
        scene.frame_current = frame
        cam.location = base_loc
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.rotation_euler = base_rot
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Add noise modifiers to each location f-curve
    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            if fc.data_path == "location":
                mod = fc.modifiers.new("NOISE")
                mod.scale = 6.0   # Frequency of noise (frames)
                mod.strength = pos_amp
                mod.depth = 0
            elif fc.data_path == "rotation_euler":
                mod = fc.modifiers.new("NOISE")
                mod.scale = 8.0
                # Convert degrees to radians
                mod.strength = math.radians(rot_amp_deg)
                mod.depth = 0

    print(f"  Noise modifier: pos_amp={pos_amp}m, rot_amp={rot_amp_deg}deg")


def animate_camera_python_jitter(scene, cam, pos_amp=0.5, rot_amp_deg=0.5, seed=42):
    """Animate camera with per-frame random jitter (Approach B).

    pos_amp: meters of random position offset per frame
    rot_amp_deg: degrees of random rotation offset per frame
    """
    import numpy as np
    cam.name = "JitteredCameraPython"
    scene.camera = cam

    base_loc = cam.location.copy()
    base_rot = cam.rotation_euler.copy()

    rng = np.random.default_rng(seed)

    # Generate jitter offsets with smoothing (gaussian per frame)
    # Low-frequency drift + high-frequency jitter
    for frame in range(1, N_FRAMES + 1):
        # Smooth drift: use previous offset + noise
        t = frame / N_FRAMES
        drift_x = 0.3 * math.sin(t * 4 * math.pi) * pos_amp
        drift_y = 0.2 * math.cos(t * 3 * math.pi) * pos_amp
        drift_z = 0.1 * math.sin(t * 5 * math.pi) * pos_amp

        # High-frequency jitter (gaussian)
        jitter = rng.normal(0, pos_amp * 0.3, size=3)

        cam.location = (
            base_loc[0] + drift_x + jitter[0],
            base_loc[1] + drift_y + jitter[1],
            base_loc[2] + drift_z + jitter[2],
        )
        cam.keyframe_insert(data_path="location", frame=frame)

        # Rotation jitter (small)
        rot_jitter = rng.normal(0, math.radians(rot_amp_deg * 0.3), size=3)
        cam.rotation_euler = (
            base_rot[0] + rot_jitter[0],
            base_rot[1] + rot_jitter[1],
            base_rot[2] + rot_jitter[2],
        )
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    print(f"  Python jitter: pos_amp={pos_amp}m, rot_amp={rot_amp_deg}deg")


def render_sequence(scene, cam, output_subdir, prefix):
    """Render all frames in the scene's frame range."""
    out_dir = OUTPUT_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set output format
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_current = frame
        scene.camera = cam

        fp = out_dir / f"{prefix}_frame_{frame:04d}.png"
        scene.render.filepath = str(fp)
        bpy.ops.render.render(write_still=True)

    print(f"  Rendered {N_FRAMES} frames to {out_dir}")
    return out_dir


def test_stable_sequence():
    """Render 30 frames with NO jitter (baseline)."""
    print("\n" + "=" * 60)
    print("Baseline: Stable camera (no jitter)")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_target_scene(scene)
    cam = create_stable_camera(scene)

    t0 = time.time()
    out = render_sequence(scene, cam, "jitter_stable", "stable")
    print(f"  Time: {time.time() - t0:.1f}s")
    return out


def test_noise_modifier_jitter():
    """Render 30 frames with Blender's noise modifier on camera."""
    print("\n" + "=" * 60)
    print("Test A: Noise modifier on camera f-curves")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_target_scene(scene)
    cam = create_stable_camera(scene)
    animate_camera_noise_modifier(scene, cam, pos_amp=0.3, rot_amp_deg=0.4)

    t0 = time.time()
    out = render_sequence(scene, cam, "jitter_noise_modifier", "jitter_nm")
    print(f"  Time: {time.time() - t0:.1f}s")
    return out


def test_python_jitter():
    """Render 30 frames with Python-generated per-frame jitter."""
    print("\n" + "=" * 60)
    print("Test B: Python-generated per-frame jitter")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_target_scene(scene)
    cam = create_stable_camera(scene)
    animate_camera_python_jitter(scene, cam, pos_amp=0.5, rot_amp_deg=0.5, seed=42)

    t0 = time.time()
    out = render_sequence(scene, cam, "jitter_python", "jitter_py")
    print(f"  Time: {time.time() - t0:.1f}s")
    return out


def test_heavy_python_jitter():
    """Render 30 frames with more pronounced jitter to see the effect."""
    print("\n" + "=" * 60)
    print("Test C: Heavy Python jitter (for visibility)")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_target_scene(scene)
    cam = create_stable_camera(scene)
    animate_camera_python_jitter(scene, cam, pos_amp=1.0, rot_amp_deg=1.0, seed=99)

    t0 = time.time()
    out = render_sequence(scene, cam, "jitter_heavy", "jitter_hvy")
    print(f"  Time: {time.time() - t0:.1f}s")
    return out


def verify_output(out_dirs):
    """Check that we got expected number of frames in each output dir."""
    print("\n  Output verification:")
    for label, path in out_dirs:
        files = sorted(path.glob("*.png"))
        print(f"    {label}: {len(files)} frames in {path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("CAMERA JITTER TEST")
    print("=================")
    t_total = time.time()

    test_stable_sequence()
    test_noise_modifier_jitter()
    test_python_jitter()
    test_heavy_python_jitter()

    print(f"\n{'=' * 60}")
    print(f"Total time: {time.time() - t_total:.1f}s")
    print(f"Images in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # Show subdirectory summaries
    for sub in sorted(OUTPUT_DIR.glob("jitter_*")):
        if sub.is_dir():
            files = list(sub.glob("*.png"))
            print(f"  {sub.name}/: {len(files)} frames ({files[0].stat().st_size // 1024} KB each)" if files else f"  {sub.name}/: (empty)")


if __name__ == "__main__":
    main()
