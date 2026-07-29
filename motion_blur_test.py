"""
Motion Blur Test (Layer 4b).

Renders a fast-moving sphere (50 m/s) with Cycles motion blur at various
shutter speeds to find the sweet spot between visible motion trails and
overwhelming blur.

Usage:
    blender --background --python motion_blur_test.py
"""

import bpy
import time
import math
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_smoke_test" / "camera_realism"


def factory_reset():
    bpy.ops.wm.read_factory_settings()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.render.fps = 30
    return scene


def clear_scene(scene):
    for name in ["Cube", "Light", "Camera"]:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)


def setup_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    bg = tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.3, 0.4, 0.6, 1.0)
    bg.inputs["Strength"].default_value = 0.8
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(bg.outputs["Background"], output.inputs["Surface"])


def create_scene_with_moving_object(scene, speed_mps=50.0, track_length=200.0):
    """Create a scene with a fast-moving sphere.

    The sphere travels along X axis at `speed_mps` m/s. The camera is
    positioned to capture the motion as a horizontal streak.

    We animate the sphere position so Cycles can compute velocity vectors
    for motion blur.
    """
    # Sun light
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 100))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.2, 0.3)

    # Ground plane with checker for motion reference
    bpy.ops.mesh.primitive_plane_add(size=500, location=(0, 0, -1))
    ground = bpy.context.object
    ground_mat = bpy.data.materials.new("ground_checker")
    ground_mat.use_nodes = True
    bsdf = ground_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(ground_mat)

    # Animated sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=5, location=(-track_length/2, 0, 20))
    sphere = bpy.context.object
    sphere.name = "FastSphere"

    # Emission material for visibility against dark background
    mat = bpy.data.materials.new("sphere_emissive")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.8, 0.2, 1.0)
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.8, 0.2, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 20.0
    sphere.data.materials.append(mat)

    # Keyframes: start position at frame 1, end position at frame 2
    duration_frames = track_length / speed_mps  # seconds given speed
    # At 30 fps, how many frames to traverse track_length?
    n_frames = max(2, int(duration_frames * 30))  # ensure at least 2 frames

    # Set scene frame range
    scene.frame_start = 1
    scene.frame_end = n_frames + 1

    # Position at start
    sphere.location = (-track_length / 2, 0, 20)
    sphere.keyframe_insert(data_path="location", frame=1)

    # Position at end
    sphere.location = (track_length / 2, 0, 20)
    sphere.keyframe_insert(data_path="location", frame=n_frames + 1)

    # Set linear interpolation
    for fc in sphere.animation_data.action.fcurves:
        for kf in fc.keyframe_points:
            kf.interpolation = "LINEAR"

    # Camera: side view, perpendicular to motion
    bpy.ops.object.camera_add(location=(0, -80, 30))
    cam = bpy.context.object
    cam.name = "MotionCam"
    scene.camera = cam
    # Look at center
    cam.rotation_euler = (0.45, 0, 0)

    # Also add static reference spheres
    for x in [-80, 0, 80]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=3, location=(x, 0, 5))
        static = bpy.context.object
        static.name = f"StaticSphere_{x}"
        static_mat = bpy.data.materials.new(f"static_mat_{x}")
        static_mat.use_nodes = True
        sbsdf = static_mat.node_tree.nodes.get("Principled BSDF")
        if sbsdf:
            sbsdf.inputs["Base Color"].default_value = (0.3, 1.0, 0.3, 1.0) if x == 0 else (0.3, 0.3, 1.0, 1.0)
        static.data.materials.append(static_mat)

    # Set current frame to the middle of the animation
    mid_frame = (n_frames // 2) + 1
    scene.frame_current = mid_frame

    print(f"  Sphere speed: {speed_mps} m/s across {track_length}m")
    print(f"  Animation: {n_frames + 1} frames, rendering at frame {mid_frame}")
    return sphere, cam


def configure_motion_blur(scene, shutter_fraction):
    """Enable Cycles motion blur with given shutter fraction.

    shutter_fraction is the fraction of the frame interval the shutter is open.
    At 30 fps, frame interval = 1/30 s.
    shutter_fraction=1.0 means open for the whole interval.
    """
    scene.render.use_motion_blur = True
    scene.render.motion_blur_position = "CENTER"  # Shutter opens centered on frame
    scene.render.motion_blur_shutter = shutter_fraction

    # Cycles-specific motion blur settings
    scene.cycles.motion_blur_position = "CENTER"
    scene.cycles.use_motion_blur = True
    scene.render.motion_blur_max = 32  # Max number of motion blur samples

    print(f"  Motion blur shutter: {shutter_fraction:.4f} "
          f"(= 1/{1/(shutter_fraction * 1/30):.0f}s at 30fps)")


def render_shutter_test(scene, shutter_label, shutter_fraction, filepath):
    """Configure and render with a given shutter fraction."""
    configure_motion_blur(scene, shutter_fraction)
    scene.render.filepath = str(filepath)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    elapsed = time.time() - t0
    print(f"  Shutter {shutter_label}: {elapsed:.1f}s -> {filepath.name}")


def test_no_motion_blur():
    """Render the scene without motion blur (baseline)."""
    print("\n" + "=" * 60)
    print("Baseline: No motion blur")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_scene_with_moving_object(scene)
    scene.render.use_motion_blur = False

    fp = OUTPUT_DIR / "motion_blur_off.png"
    t0 = time.time()
    render_and_save(scene, fp)
    print(f"  Time: {time.time() - t0:.1f}s")


def render_and_save(scene, filepath):
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)
    print(f"  Saved: {filepath.name}")


def test_motion_blur():
    """Render with various shutter speeds."""
    print("\n" + "=" * 60)
    print("Motion Blur Shutter Sweep")
    print("=" * 60)

    # Shutter fractions at 30fps:
    #  0.25 = 1/120s (fast, minimal blur)
    #  0.50 = 1/60s  (moderate)
    #  1.00 = 1/30s  (full frame interval, heavy blur)
    #  1.50 = 1/20s  (overlapping, very heavy)
    shutter_settings = [
        ("1_120", 0.25),
        ("1_60",  0.50),
        ("1_30",  1.00),
        ("1_20",  1.50),
    ]

    for label, fraction in shutter_settings:
        print(f"\n  Shutter: 1/{label.replace('_','/')}s")
        scene = factory_reset()
        clear_scene(scene)
        setup_world(scene)
        create_scene_with_moving_object(scene)
        configure_motion_blur(scene, fraction)

        fp = OUTPUT_DIR / f"motion_blur_{label}s.png"
        t0 = time.time()
        render_and_save(scene, fp)
        print(f"    Time: {time.time() - t0:.1f}s")

    # Also test with higher quality (more samples)
    print("\n  High quality shutter 1/60s (128 samples):")
    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_scene_with_moving_object(scene)
    scene.cycles.samples = 128
    configure_motion_blur(scene, 0.5)
    fp = OUTPUT_DIR / "motion_blur_1_60s_hq.png"
    t0 = time.time()
    render_and_save(scene, fp)
    print(f"    Time: {time.time() - t0:.1f}s")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("MOTION BLUR TEST")
    print("===============")
    t_total = time.time()

    test_no_motion_blur()
    test_motion_blur()

    print(f"\n{'=' * 60}")
    print(f"Total time: {time.time() - t_total:.1f}s")
    print(f"Images in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    for f in sorted(OUTPUT_DIR.glob("motion_blur_*")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
