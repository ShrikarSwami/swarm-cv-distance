"""
Re-render sky background with a proper gradient model (not flat constant).
Uses Blender's built-in Sky Texture node for physical sky.
Renders two consecutive frames to measure real frame-differencing noise.
"""

import sys
import time
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import bpy
import numpy as np
from mathutils import Vector, Matrix

SENSOR_W_MM = 36.0
FOCAL_MM = 24.0
STANDOFF_M = 2000.0
H_PX, V_PX = 1920, 1080
N_DRONES = 20
SWARM_EXTENT_M = 5000.0
SWARM_HEIGHT_M = 1000.0
SAMPLES = 64
SEED = 42

CAM_ELEV_DEG = 35.0


def setup_scene():
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


def make_swarm():
    rng = np.random.default_rng(SEED)
    xy = rng.uniform(-SWARM_EXTENT_M / 2, SWARM_EXTENT_M / 2, size=(N_DRONES, 2))
    z = rng.uniform(0, SWARM_HEIGHT_M, size=(N_DRONES, 1))
    return np.hstack([xy, z])


def add_drones(scene, positions, size_m=0.5):
    mesh = bpy.data.meshes.new("drone_mesh")
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


def setup_physical_sky(scene):
    """Blender Sky Texture — physical sky model with gradient, Rayleigh scattering."""
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # Sky Texture node (Blender's built-in physical sky)
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"  # physical sky model
    sky.sun_elevation = np.radians(45)
    sky.sun_rotation = np.radians(135)

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputWorld")
    links.new(sky.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # No sun light — sky texture provides illumination
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = 2.0
    light_data.color = (1.0, 0.95, 0.85)
    light_obj = bpy.data.objects.new("Sun", light_data)
    light_obj.location = (0, 0, 10000)
    light_obj.rotation_euler = (np.radians(45), 0, np.radians(135))
    bpy.context.collection.objects.link(light_obj)


def add_camera(scene, position, look_at):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = FOCAL_MM
    cam_data.sensor_width = SENSOR_W_MM
    cam_data.clip_end = 50000
    cam_obj = bpy.data.objects.new("cam", cam_data)
    cam_obj.location = Vector(position)

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
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(filepath)
    pixels = np.array(img.pixels[:]).reshape(V_PX, H_PX, 4)
    gray = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    gray = np.clip(gray * 255, 0, 255)
    bpy.data.images.remove(img)
    return gray


def main():
    out_dir = Path(project_root) / "logs" / "temporal_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RENDER: Physical sky model (Nishita) for perturbation test")
    print("=" * 70)

    positions = make_swarm()
    center = positions.mean(axis=0)

    elev_rad = np.radians(CAM_ELEV_DEG)
    cam_pos = center + np.array([
        STANDOFF_M * np.cos(elev_rad),
        0,
        STANDOFF_M * np.sin(elev_rad),
    ])

    # Render with physical sky
    scene = setup_scene()
    setup_physical_sky(scene)
    add_drones(scene, positions)
    add_camera(scene, cam_pos.tolist(), center.tolist())

    t0 = time.time()
    sky_frame = render_and_load(scene, str(out_dir / "real_render_sky_physical.png"))
    t_render = time.time() - t0

    stats = {
        "mean": float(sky_frame.mean()),
        "std": float(sky_frame.std()),
        "min": float(sky_frame.min()),
        "max": float(sky_frame.max()),
        "grad_h": float(np.abs(np.diff(sky_frame, axis=1)).mean()),
        "grad_v": float(np.abs(np.diff(sky_frame, axis=0)).mean()),
    }

    print(f"  Render time: {t_render:.1f}s")
    print(f"  Sky (physical): mean={stats['mean']:.1f}, std={stats['std']:.2f}, "
          f"range=[{stats['min']:.1f}, {stats['max']:.1f}]")
    print(f"  Spatial grad: h={stats['grad_h']:.3f}, v={stats['grad_v']:.3f}")

    # Render second frame (identical scene) for frame-diff measurement
    sky_frame2 = render_and_load(scene, str(out_dir / "real_render_sky_physical_f2.png"))
    diff = np.abs(sky_frame.astype(float) - sky_frame2.astype(float))
    print(f"  Frame diff std: {diff.std():.3f}")

    import json
    with open(out_dir / "physical_sky_stats.json", "w") as f:
        json.dump({"sky_stats": stats, "frame_diff_std": float(diff.std())}, f, indent=2)
    print(f"  Saved to {out_dir / 'physical_sky_stats.json'}")

    # Cleanup
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


if __name__ == "__main__":
    main()
