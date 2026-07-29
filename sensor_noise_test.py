"""
Sensor Noise and Film Grain Test (Layer 4b).

Renders a controlled scene and tests three noise approaches:
  A) Compositor-based film grain overlay using a noise image
  B) Low sample count native Cycles noise (no denoising)
  C) ISO noise via the physical camera model (f-stop, aperture, ISO)
  D) Compositor glare node for lens effects

Each test produces a rendered image. Effects should be visible but subtle.

Usage:
    blender --background --python sensor_noise_test.py
"""

import bpy
import random
import time
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_smoke_test" / "camera_realism"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def factory_reset():
    bpy.ops.wm.read_factory_settings()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.cycles.use_denoising = False
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
    bg.inputs["Color"].default_value = (0.1, 0.1, 0.2, 1.0)
    bg.inputs["Strength"].default_value = 0.5
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(bg.outputs["Background"], output.inputs["Surface"])


def create_gray_card_scene(scene):
    """Create a scene with a gray card and colored objects for noise testing."""
    # Sun light
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 100))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.3, 0.0)

    # Gray card (large flat plane at slight angle)
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 15))
    gc = bpy.context.object
    gc.name = "GrayCard"
    gc.rotation_euler = (0.0, 0.0, 0.0)
    gc_mat = bpy.data.materials.new("gray_card")
    gc_mat.use_nodes = True
    bsdf = gc_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    gc.data.materials.append(gc_mat)

    # Gray card on the floor
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -1))
    floor = bpy.context.object
    floor.name = "Floor"
    floor_mat = bpy.data.materials.new("floor")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(floor_mat)

    # Colored spheres
    colors_and_pos = [
        ((1.0, 0.2, 0.2, 1.0), (-8, -8, 8)),
        ((0.2, 1.0, 0.2, 1.0), (8, -8, 8)),
        ((0.2, 0.2, 1.0, 1.0), (-8, 8, 8)),
        ((1.0, 1.0, 0.2, 1.0), (8, 8, 8)),
    ]
    for col, pos in colors_and_pos:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=5, location=pos)
        sphere = bpy.context.object
        mat = bpy.data.materials.new(f"sphere_mat_{pos[0]}_{pos[1]}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = col
            bsdf.inputs["Roughness"].default_value = 0.3
        sphere.data.materials.append(mat)

    # Camera
    bpy.ops.object.camera_add(location=(0, -30, 10))
    cam = bpy.context.object
    cam.name = "NoiseTestCam"
    scene.camera = cam
    cam.rotation_euler = (0.35, 0, 0)

    return cam


def render_and_save(scene, filepath):
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)
    print(f"  Saved: {filepath.name}")


# ---------------------------------------------------------------------------
# Test A: Compositor-based film grain
# ---------------------------------------------------------------------------

def generate_noise_image(width=960, height=540, seed=42):
    """Generate a monochrome noise image in Blender data.

    Returns a bpy.data.images reference with random pixel values.
    The grain is subtle (values centered on 0.5 with small stddev).
    """
    rng = np.random.default_rng(seed)
    # Create noise with mean 0.5, std 0.05 (subtle grain)
    noise = rng.normal(loc=0.5, scale=0.05, size=(height, width))

    # Clamp to [0, 1]
    noise = np.clip(noise, 0.0, 1.0)

    # Create image as 4-channel float
    img = bpy.data.images.new(
        "FilmGrain",
        width=width,
        height=height,
        alpha=True,
        float_buffer=True,
        is_data=False,
    )
    # Pack as RGBA
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, :, 0] = noise
    rgba[:, :, 1] = noise
    rgba[:, :, 2] = noise
    rgba[:, :, 3] = 1.0

    img.pixels.foreach_set(rgba.ravel())
    img.update()
    return img


def test_compositor_grain():
    """Render with compositor-based film grain overlay."""
    print("\n" + "=" * 60)
    print("Test A: Compositor Film Grain")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_gray_card_scene(scene)

    # Generate noise image
    noise_img = generate_noise_image(
        scene.render.resolution_x,
        scene.render.resolution_y,
        seed=42
    )
    print(f"  Generated noise image: {noise_img.size[0]}x{noise_img.size[1]}")

    # Set up compositor
    scene.use_nodes = True
    ng = bpy.data.node_groups.new("FilmGrainCompositor", "CompositorNodeTree")
    scene.compositing_node_group = ng
    ng.nodes.clear()

    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 200)

    # Noise image node
    noise_node = ng.nodes.new("CompositorNodeImage")
    noise_node.image = noise_img
    noise_node.location = (0, -100)

    # Mix grain over render: Overlay blend for natural-looking grain
    mix = ng.nodes.new("CompositorNodeMixRGB")
    mix.location = (250, 200)
    mix.blend_type = "OVERLAY"
    mix.inputs["Fac"].default_value = 0.15  # 15% grain intensity

    # Also test with ADD blend at lower intensity
    mix2 = ng.nodes.new("CompositorNodeMixRGB")
    mix2.location = (250, -50)
    mix2.blend_type = "ADD"
    mix2.inputs["Fac"].default_value = 0.03  # 3% additive grain

    comp = ng.nodes.new("CompositorNodeComposite")
    comp.location = (500, 200)

    # Links
    ng.links.new(rl.outputs["Image"], mix.inputs[1])     # Image base
    ng.links.new(noise_node.outputs["Image"], mix.inputs[2])  # Grain overlay

    ng.links.new(rl.outputs["Image"], mix2.inputs[1])
    ng.links.new(noise_node.outputs["Image"], mix2.inputs[2])

    ng.links.new(mix.outputs["Image"], comp.inputs["Image"])

    fp = OUTPUT_DIR / "noise_compositor_grain.png"
    t0 = time.time()
    render_and_save(scene, fp)
    print(f"  Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Test B: Low sample count (native Cycles noise)
# ---------------------------------------------------------------------------

def test_low_samples():
    """Render with very low sample count to show native Cycles noise."""
    print("\n" + "=" * 60)
    print("Test B: Low Sample Count (Native Noise)")
    print("=" * 60)

    sample_counts = {
        "4": 4,
        "8": 8,
        "16": 16,
        "32": 32,
    }

    for label, samples in sample_counts.items():
        print(f"\n  Samples: {samples}")
        scene = factory_reset()
        clear_scene(scene)
        setup_world(scene)
        create_gray_card_scene(scene)
        scene.cycles.samples = samples
        scene.cycles.use_denoising = False
        scene.use_nodes = False

        fp = OUTPUT_DIR / f"noise_low_samples_{label}.png"
        t0 = time.time()
        render_and_save(scene, fp)
        print(f"    Time: {time.time() - t0:.1f}s")

    # Also render with denoising for comparison
    print("\n  Reference: 32 samples WITH denoising")
    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_gray_card_scene(scene)
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.use_nodes = False
    fp = OUTPUT_DIR / "noise_denoised_32.png"
    t0 = time.time()
    render_and_save(scene, fp)
    print(f"    Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Test C: Clean reference plus combined grain
# ---------------------------------------------------------------------------

def test_clean_reference():
    """Render a clean reference (128 samples + compositor grain)."""
    print("\n" + "=" * 60)
    print("Test C: Clean Reference + Compositor Grain")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_gray_card_scene(scene)
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.use_nodes = False

    fp = OUTPUT_DIR / "noise_clean_reference.png"
    t0 = time.time()
    render_and_save(scene, fp)
    print(f"  Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Test D: Composite grain on clean render
# ---------------------------------------------------------------------------

def test_clean_plus_grain():
    """Clean render + compositor grain for the best of both worlds."""
    print("\n" + "=" * 60)
    print("Test D: Clean Render + Compositor Grain (recommended)")
    print("=" * 60)

    scene = factory_reset()
    clear_scene(scene)
    setup_world(scene)
    create_gray_card_scene(scene)
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True

    noise_img = generate_noise_image(
        scene.render.resolution_x,
        scene.render.resolution_y,
        seed=99,
    )

    scene.use_nodes = True
    ng = bpy.data.node_groups.new("CleanGrainCompositor", "CompositorNodeTree")
    scene.compositing_node_group = ng
    ng.nodes.clear()

    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    noise_node = ng.nodes.new("CompositorNodeImage")
    noise_node.image = noise_img
    noise_node.location = (0, -200)

    mix = ng.nodes.new("CompositorNodeMixRGB")
    mix.location = (250, 0)
    mix.blend_type = "OVERLAY"
    mix.inputs["Fac"].default_value = 0.12

    comp = ng.nodes.new("CompositorNodeComposite")
    comp.location = (450, 0)

    ng.links.new(rl.outputs["Image"], mix.inputs[1])
    ng.links.new(noise_node.outputs["Image"], mix.inputs[2])
    ng.links.new(mix.outputs["Image"], comp.inputs["Image"])

    fp = OUTPUT_DIR / "noise_clean_plus_grain.png"
    t0 = time.time()
    render_and_save(scene, fp)
    print(f"  Time: {time.time() - t0:.1f}s")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SENSOR NOISE AND FILM GRAIN TEST")
    print("================================")
    t_total = time.time()

    test_clean_reference()
    test_compositor_grain()
    test_low_samples()
    test_clean_plus_grain()

    print(f"\n{'=' * 60}")
    print(f"Total time: {time.time() - t_total:.1f}s")
    print(f"Images in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    for f in sorted(OUTPUT_DIR.glob("noise_*")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
