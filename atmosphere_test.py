"""
Atmospheric Perspective Test (Layer 4b).

Renders a scene with objects at varying distances to test two approaches:
  A) Cycles Volumetric Scattering -- world volume shader with Volume Scatter
  B) Compositor Mist Pass -- distance-based fog using view layer mist pass

Both produce visible "bluing" / fading of distant objects. Output images
are saved to dataset_smoke_test/camera_realism/.

Usage:
    blender --background --python atmosphere_test.py
"""

import bpy
import os
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_smoke_test" / "camera_realism"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def factory_reset():
    """Reset scene to factory defaults (without use_empty)."""
    bpy.ops.wm.read_factory_settings()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    return scene


def clear_scene(scene):
    """Remove default Cube, Light, and Camera objects."""
    for name in ["Cube", "Light", "Camera"]:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)


def create_test_scene(scene, distances):
    """Create colored spheres at given Y-distances. Returns number of objects."""
    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=5000, location=(0, -1500, -1))
    ground = bpy.context.object
    ground_mat = bpy.data.materials.new("ground_mat")
    ground_mat.use_nodes = True
    bsdf = ground_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(ground_mat)

    # Sun light to illuminate the scene
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5000))
    sun = bpy.context.object
    sun.data.energy = 5.0
    sun.data.color = (1.0, 1.0, 1.0)
    sun.rotation_euler = (1.0, 0.0, 0.0)

    colors = [
        (1.0, 0.2, 0.2, 1.0),
        (0.2, 1.0, 0.2, 1.0),
        (0.2, 0.2, 1.0, 1.0),
        (1.0, 1.0, 0.2, 1.0),
        (1.0, 0.5, 0.0, 1.0),
    ]

    count = 0
    for i, d in enumerate(distances):
        col = colors[i % len(colors)]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=10, location=(0, -d, 30))
        sphere = bpy.context.object
        sphere.name = f"Sphere_{d}m"
        mat = bpy.data.materials.new(f"mat_{d}m")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = col
            bsdf.inputs["Roughness"].default_value = 0.6
        sphere.data.materials.append(mat)
        count += 1

        # Reference cubes on ground
        for x_ofs in [-25, 25]:
            bpy.ops.mesh.primitive_cube_add(size=5, location=(x_ofs, -d, 3))
            cube = bpy.context.object
            cube.name = f"Cube_{d}m_x{x_ofs}"
            mat = bpy.data.materials.new(f"cube_mat_{d}m")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.8
            cube.data.materials.append(mat)
            count += 1
    return count


def setup_camera(scene, location=(0, 50, 30), target=(0, -1500, 20)):
    """Create a camera pointing at target."""
    import math
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = "TestCamera"
    scene.camera = cam

    dx = target[0] - location[0]
    dy = target[1] - location[1]
    dz = target[2] - location[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    yaw = math.atan2(dx, dy)
    pitch = -math.asin(dz / max(dist, 0.001))
    cam.rotation_euler = (pitch, 0, yaw)
    return cam


def setup_world_sky(scene):
    """Set up a simple sky-colored background (no volume)."""
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


def add_volume_to_world(scene, density=0.002, color=(0.7, 0.8, 1.0, 1.0)):
    """Add Volume Scatter to world for atmospheric haze."""
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    if not world.use_nodes:
        world.use_nodes = True

    tree = world.node_tree
    nodes = tree.nodes
    links = tree.links

    output = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputWorld")

    vol = nodes.new("ShaderNodeVolumeScatter")
    vol.inputs["Color"].default_value = color
    vol.inputs["Density"].default_value = density
    vol.location = (0, -200)
    links.new(vol.outputs["Volume"], output.inputs["Volume"])

    # Configure Cycles volume settings
    scene.cycles.volume_bounces = 4
    scene.cycles.volume_max_steps = 128
    scene.cycles.volume_step_rate = 10.0


def setup_mist_compositor(scene, depth=1500, start=50, intensity=1.0):
    """Set up compositor-based mist pass fog overlay."""
    vl = bpy.context.view_layer
    vl.use_pass_mist = True

    # World mist settings
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    try:
        world.mist_settings.intensity = intensity
        world.mist_settings.depth = depth
        world.mist_settings.start = start
        world.mist_settings.falloff = "QUADRATIC"
    except AttributeError as e:
        print(f"  Warning: world.mist_settings not available: {e}")

    # Compositor
    scene.use_nodes = True
    ng = bpy.data.node_groups.new("MistCompositor", "CompositorNodeTree")
    scene.compositing_node_group = ng
    ng.nodes.clear()

    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 200)

    # Create a solid haze color image
    import numpy as np
    haze_color = (0.5, 0.6, 1.0, 1.0)  # bluish haze
    img = bpy.data.images.new("HazeColor", 960, 540, alpha=True, float_buffer=True)
    rgba = np.zeros((540, 960, 4), dtype=np.float32)
    rgba[:,:,0] = haze_color[0]
    rgba[:,:,1] = haze_color[1]
    rgba[:,:,2] = haze_color[2]
    rgba[:,:,3] = haze_color[3]
    img.pixels.foreach_set(rgba.ravel())
    img.update()

    haze_img = ng.nodes.new("CompositorNodeImage")
    haze_img.image = img
    haze_img.location = (0, -200)

    # AlphaOver: mist as factor, haze as foreground
    ao = ng.nodes.new("CompositorNodeAlphaOver")
    ao.location = (200, 0)
    ao.inputs["Type"].default_value = "Over"

    # Output to viewer
    viewer = ng.nodes.new("CompositorNodeViewer")
    viewer.location = (400, 0)

    # Links
    ng.links.new(rl.outputs["Image"], ao.inputs["Background"])
    ng.links.new(haze_img.outputs["Image"], ao.inputs["Foreground"])
    ng.links.new(rl.outputs["Mist"], ao.inputs["Factor"])
    ng.links.new(ao.outputs["Image"], viewer.inputs["Image"])

    print(f"  Mist: depth={depth}, start={start}, intensity={intensity}")
    print(f"  Mist pass available: {'Mist' in [o.name for o in rl.outputs]}")


def render_and_save(scene, filepath):
    """Render still to filepath."""
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)
    print(f"  Saved: {filepath.name}")


# ---------------------------------------------------------------------------
# Test A: Volumetric Scattering
# ---------------------------------------------------------------------------

def test_volumetric():
    """Render scene with Cycles volumetric scattering at multiple densities."""
    print("\n" + "=" * 60)
    print("Test A: Cycles Volumetric Scattering")
    print("=" * 60)

    distances = [100, 500, 1000, 2000]
    densities = {
        "subtle": 0.001,
        "medium": 0.003,
        "heavy": 0.006,
    }

    for label, density in densities.items():
        print(f"\n  Density: {label} ({density})")
        scene = factory_reset()
        clear_scene(scene)
        setup_world_sky(scene)
        create_test_scene(scene, distances)
        setup_camera(scene)
        add_volume_to_world(scene, density=density)

        fp = OUTPUT_DIR / f"atmosphere_volume_{label}.png"
        t0 = time.time()
        render_and_save(scene, fp)
        print(f"    Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Test B: Mist Pass Compositor
# ---------------------------------------------------------------------------

def test_mist_pass():
    """Render scene using compositor-based mist pass (no volumes)."""
    print("\n" + "=" * 60)
    print("Test B: Compositor Mist Pass")
    print("=" * 60)

    distances = [100, 500, 1000, 2000]

    configs = [
        ("light",  2000, 50,  0.5),
        ("medium", 1200, 50,  1.0),
        ("strong",  600, 50,  1.5),
    ]

    for label, depth, start, intensity in configs:
        print(f"\n  Config: {label} (depth={depth})")
        scene = factory_reset()
        clear_scene(scene)
        setup_world_sky(scene)
        create_test_scene(scene, distances)
        setup_camera(scene)
        setup_mist_compositor(scene, depth=depth, start=start, intensity=intensity)

        fp = OUTPUT_DIR / f"atmosphere_mist_{label}.png"
        t0 = time.time()
        render_and_save(scene, fp)
        print(f"    Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def test_baseline():
    """Render with no fog/atmosphere at all."""
    print("\n" + "=" * 60)
    print("Baseline: No atmosphere")
    print("=" * 60)

    distances = [100, 500, 1000, 2000]
    scene = factory_reset()
    clear_scene(scene)
    setup_world_sky(scene)
    create_test_scene(scene, distances)
    setup_camera(scene)
    scene.use_nodes = False

    fp = OUTPUT_DIR / "atmosphere_baseline.png"
    t0 = time.time()
    render_and_save(scene, fp)
    print(f"  Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("ATMOSPHERIC PERSPECTIVE TEST")
    print("============================")
    t_total = time.time()

    test_baseline()
    test_volumetric()
    test_mist_pass()

    print(f"\n{'=' * 60}")
    print(f"Total time: {time.time() - t_total:.1f}s")
    print(f"Images in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    for f in sorted(OUTPUT_DIR.glob("atmosphere_*")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
