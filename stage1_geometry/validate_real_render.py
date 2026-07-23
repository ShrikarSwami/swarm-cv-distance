"""
Validate the synthetic temporal detection model against real Cycles renders.

Renders a small set of true-scale (0.5m drone) frames at 24mm/2km with:
1. Clear sky background
2. Textured terrain background

Measures actual background noise characteristics (mean, std, spatial
correlation) and compares to the synthetic model's assumptions.

Usage:
    blender --background --python stage1_geometry/validate_real_render.py

Output:
    logs/temporal_detection/real_render_validation.json
    logs/temporal_detection/real_render_*.png (frame images for inspection)
"""

import json
import os
import sys
import time
from pathlib import Path

# Blender's Python doesn't have our venv, so add project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import bpy
import numpy as np
from mathutils import Vector, Matrix

# ---------------------------------------------------------------------------
# Scene parameters — TRUE SCALE, no display inflation
# ---------------------------------------------------------------------------

SENSOR_W_MM = 36.0
FOCAL_MM = 24.0
STANDOFF_M = 2000.0
H_PX, V_PX = 1920, 1080
DRONE_SIZE_M = 0.5
N_DRONES = 20
SWARM_EXTENT_M = 5000.0
SWARM_HEIGHT_M = 1000.0
SAMPLES = 64  # moderate quality for validation
SEED = 42

# Camera position: 2km standoff, ~35° elevation, pointing at swarm center
CAM_ELEV_DEG = 35.0
CAM_AZ_DEG = 0.0


def setup_scene():
    """Clear scene and set render settings."""
    # Clear all objects
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.render.resolution_x = H_PX
    scene.render.resolution_y = V_PX
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"

    return scene


def make_swarm(n_drones, seed):
    """Generate random drone positions in the 5km volume."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-SWARM_EXTENT_M / 2, SWARM_EXTENT_M / 2, size=(n_drones, 2))
    z = rng.uniform(0, SWARM_HEIGHT_M, size=(n_drones, 1))
    return np.hstack([xy, z])


def add_drones(scene, positions, size_m=0.5):
    """Add drone objects (small cubes) at true scale."""
    mesh = bpy.data.meshes.new("drone_mesh")
    # Unit cube, scaled to drone size
    cubeverts = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ]
    cubeedges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    mesh.from_pydata(cubeverts, cubeedges, [])
    mesh.update()

    drones = []
    for i, pos in enumerate(positions):
        obj = bpy.data.objects.new(f"drone_{i:03d}", mesh)
        obj.location = Vector(pos.tolist())
        obj.scale = (size_m, size_m, size_m)
        # Dark drone material
        mat = bpy.data.materials.new(f"drone_mat_{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.1, 0.1, 0.1, 1.0)
            bsdf.inputs["Metallic"].default_value = 0.5
            bsdf.inputs["Roughness"].default_value = 0.5
        obj.data.materials.append(mat)
        bpy.context.collection.objects.link(obj)
        drones.append(obj)
    return drones


def setup_sky_background(scene):
    """Clear sky background — uniform bright color."""
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.5, 0.6, 0.9, 1.0)  # blue sky
    bg.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # Sun light
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = 4.0
    light_data.color = (1.0, 0.95, 0.85)
    light_obj = bpy.data.objects.new("Sun", light_data)
    light_obj.location = (0, 0, 10000)
    light_obj.rotation_euler = (0.3, 0.0, 0.0)
    bpy.context.collection.objects.link(light_obj)


def setup_terrain_background(scene):
    """Terrain background — a large textured plane below the swarm."""
    # World background: darker, earthy
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.3, 0.35, 0.2, 1.0)  # dark green/brown
    bg.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # Add a large ground plane with noise texture
    bpy.ops.mesh.primitive_plane_add(size=20000, location=(0, 0, -50))
    ground = bpy.context.active_object
    ground.name = "ground"

    # Material with noise texture for terrain variation
    mat = bpy.data.materials.new("terrain_mat")
    mat.use_nodes = True
    nodes_mat = mat.node_tree.nodes
    links_mat = mat.node_tree.links

    # Noise texture for terrain variation
    noise = nodes_mat.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 50.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.7

    colorramp = nodes_mat.new("ShaderNodeValToRGB")
    colorramp.color_ramp.elements[0].position = 0.3
    colorramp.color_ramp.elements[0].color = (0.15, 0.12, 0.08, 1.0)  # dark earth
    colorramp.color_ramp.elements[1].position = 0.7
    colorramp.color_ramp.elements[1].color = (0.35, 0.30, 0.15, 1.0)  # light earth

    bsdf = nodes_mat.get("Principled BSDF")
    links_mat.new(noise.outputs["Fac"], colorramp.inputs["Fac"])
    links_mat.new(colorramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9

    ground.data.materials.append(mat)

    # Sun light
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = 4.0
    light_data.color = (1.0, 0.95, 0.85)
    light_obj = bpy.data.objects.new("Sun", light_data)
    light_obj.location = (0, 0, 10000)
    light_obj.rotation_euler = (0.3, 0.0, 0.0)
    bpy.context.collection.objects.link(light_obj)


def add_camera(scene, position, look_at):
    """Add a camera at given position looking at target."""
    cam_data = bpy.data.cameras.new("validation_cam")
    cam_data.lens = FOCAL_MM
    cam_data.sensor_width = SENSOR_W_MM
    cam_data.clip_end = 50000

    cam_obj = bpy.data.objects.new("validation_cam", cam_data)
    cam_obj.location = Vector(position)

    # Point at target
    direction = (Vector(look_at) - cam_obj.location).normalized()
    forward = direction
    up_hint = Vector((0, 0, 1))
    right = forward.cross(up_hint).normalized()
    up = right.cross(forward).normalized()
    mat = Matrix.Identity(4)
    mat[0][0], mat[1][0], mat[2][0] = right.x, right.y, right.z
    mat[0][1], mat[1][1], mat[2][1] = up.x, up.y, up.z
    mat[0][2], mat[1][2], mat[2][2] = -forward.x, -forward.y, -forward.z
    mat[0][3], mat[1][3], mat[2][3] = position[0], position[1], position[2]
    cam_obj.matrix_world = mat

    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    bpy.context.view_layer.update()
    return cam_obj


def render_and_load(scene, filepath):
    """Render and return as numpy array."""
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    # Load back as numpy
    import bpy_extras.io_utils
    img = bpy.data.images.load(filepath)
    # Get pixels as flat array
    pixels = np.array(img.pixels[:])  # (H*W*4,) RGBA float
    img_shape = (V_PX, H_PX, 4)
    pixels = pixels.reshape(img_shape)
    # Convert to grayscale (RGB, drop alpha)
    gray = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    # Scale to 0-255 (Cycles renders are linear 0-1, need tone mapping)
    gray = np.clip(gray * 255, 0, 255)
    bpy.data.images.remove(img)
    return gray


def measure_background_stats(frame, region="center"):
    """Measure noise characteristics in a region of the frame.

    region: 'center' (middle 50%), 'top' (sky region), 'bottom' (terrain).
    Returns dict with mean, std, min, max, and spatial correlation.
    """
    h, w = frame.shape
    if region == "center":
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
    elif region == "top":
        y1, y2 = 0, h // 3
        x1, x2 = 0, w
    elif region == "bottom":
        y1, y2 = 2 * h // 3, h
        x1, x2 = 0, w
    else:
        y1, y2 = 0, h
        x1, x2 = 0, w

    patch = frame[y1:y2, x1:x2]

    # Basic stats
    stats = {
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "min": float(patch.min()),
        "max": float(patch.max()),
        "shape": list(patch.shape),
    }

    # Spatial correlation: how correlated are adjacent pixels?
    # This affects whether frame differencing works (correlated noise cancels)
    if patch.shape[0] > 2 and patch.shape[1] > 2:
        h_diff = np.abs(patch[1:, :].astype(float) - patch[:-1, :].astype(float))
        v_diff = np.abs(patch[:, 1:].astype(float) - patch[:, :-1].astype(float))
        stats["spatial_grad_h_mean"] = float(h_diff.mean())
        stats["spatial_grad_v_mean"] = float(v_diff.mean())
        stats["spatial_grad_h_std"] = float(h_diff.std())
        stats["spatial_grad_v_std"] = float(v_diff.std())

    return stats


def compute_frame_difference(frame1, frame2):
    """Compute absolute frame difference and measure its statistics."""
    diff = np.abs(frame1.astype(float) - frame2.astype(float))
    return {
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "diff_max": float(diff.max()),
        "diff_p99": float(np.percentile(diff, 99)),
        "diff_p999": float(np.percentile(diff, 99.9)),
    }


def main():
    out_dir = Path(project_root) / "logs" / "temporal_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REAL RENDER VALIDATION: true-scale 0.5m drones at 24mm/2km")
    print("=" * 70)
    print(f"Resolution: {H_PX}×{V_PX}, Lens: {FOCAL_MM}mm, Samples: {SAMPLES}")
    print(f"Standoff: {STANDOFF_M}m, Drone size: {DRONE_SIZE_M}m")
    print()

    # Generate swarm
    positions = make_swarm(N_DRONES, SEED)
    center = positions.mean(axis=0)

    # Camera position
    elev_rad = np.radians(CAM_ELEV_DEG)
    az_rad = np.radians(CAM_AZ_DEG)
    cam_pos = center + np.array([
        STANDOFF_M * np.cos(elev_rad) * np.cos(az_rad),
        STANDOFF_M * np.cos(elev_rad) * np.sin(az_rad),
        STANDOFF_M * np.sin(elev_rad),
    ])

    results = {}

    # --- Test 1: Sky background ---
    print("Test 1: Clear sky background")
    print("-" * 40)
    t0 = time.time()

    scene = setup_scene()
    setup_sky_background(scene)
    drones = add_drones(scene, positions, DRONE_SIZE_M)
    cam = add_camera(scene, cam_pos.tolist(), center.tolist())

    sky_frame = render_and_load(scene, str(out_dir / "real_render_sky.png"))
    t_sky = time.time() - t0
    print(f"  Render time: {t_sky:.1f}s")

    sky_stats_center = measure_background_stats(sky_frame, "center")
    sky_stats_top = measure_background_stats(sky_frame, "top")
    sky_diff = compute_frame_difference(
        sky_frame,
        render_and_load(scene, str(out_dir / "real_render_sky_frame2.png"))
    )

    print(f"  Sky (center): mean={sky_stats_center['mean']:.1f}, "
          f"std={sky_stats_center['std']:.2f}")
    print(f"  Sky (top):    mean={sky_stats_top['mean']:.1f}, "
          f"std={sky_stats_top['std']:.2f}")
    if "spatial_grad_h_mean" in sky_stats_center:
        print(f"  Spatial grad: h={sky_stats_center['spatial_grad_h_mean']:.2f}, "
              f"v={sky_stats_center['spatial_grad_v_mean']:.2f}")
    print(f"  Frame diff:   std={sky_diff['diff_std']:.2f}, "
          f"p99={sky_diff['diff_p99']:.1f}")

    results["sky"] = {
        "render_time_s": t_sky,
        "center": sky_stats_center,
        "top": sky_stats_top,
        "frame_diff": sky_diff,
    }

    # Clean up for next test
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)

    # --- Test 2: Terrain background ---
    print("\nTest 2: Terrain background")
    print("-" * 40)
    t0 = time.time()

    scene = setup_scene()
    setup_terrain_background(scene)
    drones = add_drones(scene, positions, DRONE_SIZE_M)
    cam = add_camera(scene, cam_pos.tolist(), center.tolist())

    terrain_frame = render_and_load(scene, str(out_dir / "real_render_terrain.png"))
    t_terrain = time.time() - t0
    print(f"  Render time: {t_terrain:.1f}s")

    terrain_stats_center = measure_background_stats(terrain_frame, "center")
    terrain_stats_top = measure_background_stats(terrain_frame, "top")
    terrain_stats_bottom = measure_background_stats(terrain_frame, "bottom")
    terrain_diff = compute_frame_difference(
        terrain_frame,
        render_and_load(scene, str(out_dir / "real_render_terrain_frame2.png"))
    )

    print(f"  Terrain (center): mean={terrain_stats_center['mean']:.1f}, "
          f"std={terrain_stats_center['std']:.2f}")
    print(f"  Terrain (top):    mean={terrain_stats_top['mean']:.1f}, "
          f"std={terrain_stats_top['std']:.2f}")
    print(f"  Terrain (bottom): mean={terrain_stats_bottom['mean']:.1f}, "
          f"std={terrain_stats_bottom['std']:.2f}")
    if "spatial_grad_h_mean" in terrain_stats_center:
        print(f"  Spatial grad: h={terrain_stats_center['spatial_grad_h_mean']:.2f}, "
              f"v={terrain_stats_center['spatial_grad_v_mean']:.2f}")
    print(f"  Frame diff:   std={terrain_diff['diff_std']:.2f}, "
          f"p99={terrain_diff['diff_p99']:.1f}")

    results["terrain"] = {
        "render_time_s": t_terrain,
        "center": terrain_stats_center,
        "top": terrain_stats_top,
        "bottom": terrain_stats_bottom,
        "frame_diff": terrain_diff,
    }

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON: real render vs synthetic model assumptions")
    print("=" * 70)

    sky_std = sky_stats_center["std"]
    terrain_std = terrain_stats_center["std"]
    ratio = terrain_std / sky_std if sky_std > 0 else float('inf')

    print(f"\n  Synthetic model assumed:  sky σ≈2.0, terrain σ≈12.0, ratio=6.0×")
    print(f"  Real render measured:     sky σ={sky_std:.2f}, terrain σ={terrain_std:.2f}, "
          f"ratio={ratio:.1f}×")
    print()

    if abs(ratio - 6.0) < 1.0:
        print("  → RATIO WITHIN EXPECTATION (within 1× of assumed 6×)")
    elif ratio > 6.0:
        print(f"  → TERRAIN IS HARDER THAN ASSUMED (ratio {ratio:.1f}× vs assumed 6×)")
        print("    Conclusion holds with wider margin — terrain is even worse than modeled.")
    else:
        print(f"  → TERRAIN IS EASIER THAN ASSUMED (ratio {ratio:.1f}× vs assumed 6×)")
        print("    May need to revise contrast thresholds — but direction unchanged.")

    # Spatial correlation check
    if "spatial_grad_h_mean" in sky_stats_center and "spatial_grad_h_mean" in terrain_stats_center:
        sky_grad = sky_stats_center["spatial_grad_h_mean"]
        terrain_grad = terrain_stats_center["spatial_grad_h_mean"]
        print(f"\n  Spatial gradient (adjacent pixel difference):")
        print(f"    Sky:     {sky_grad:.2f}")
        print(f"    Terrain: {terrain_grad:.2f}")
        print(f"    → {'High spatial variation in terrain means frame differencing' if terrain_grad > sky_grad * 2 else 'Spatial variation similar between backgrounds'}")

    # Frame difference comparison
    print(f"\n  Frame-to-frame difference (identical scene, re-rendered):")
    print(f"    Sky diff std:     {sky_diff['diff_std']:.2f}")
    print(f"    Terrain diff std: {terrain_diff['diff_std']:.2f}")
    if sky_diff['diff_std'] > 0:
        diff_ratio = terrain_diff['diff_std'] / sky_diff['diff_std']
        print(f"    Ratio: {diff_ratio:.1f}×")
        print(f"    → This is the ACTUAL noise floor for frame-differencing detection.")

    results["comparison"] = {
        "synthetic_sky_std": 2.0,
        "synthetic_terrain_std": 12.0,
        "synthetic_ratio": 6.0,
        "real_sky_std": float(sky_std),
        "real_terrain_std": float(terrain_std),
        "real_ratio": float(ratio),
        "sky_frame_diff_std": float(sky_diff["diff_std"]),
        "terrain_frame_diff_std": float(terrain_diff["diff_std"]),
    }

    # Save
    out_path = out_dir / "real_render_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)

    return results


if __name__ == "__main__":
    main()
