"""
Render a temporal multi-view clip: N frames × M cameras of a flying swarm.

Renders a boids flight trajectory and captures it from multiple camera
angles. Outputs: per-camera PNG sequences + ground truth NPZ with
per-frame drone positions, camera intrinsics, and extrinsics.

Usage:
    blender --background --python render_sequence.py -- <config.json>

Config:
{
    "clip_name": "seq_001",
    "dataset_root": "dataset_temporal",
    "n_frames": 20,
    "fps": 10,
    "n_views": 12,
    "focal_mm": 24,
    "sensor_width_mm": 36.0,
    "resolution": [1920, 1080],
    "standoff_m": 2000,
    "seed": 42,
    "n_drones": 20,
    "samples": 32
}
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

argv = sys.argv
if "--" not in sys.argv:
    print("Usage: blender --background --python render_sequence.py -- <config.json>")
    sys.exit(1)
args = sys.argv[sys.argv.index("--") + 1:]
config_path = args[0]

with open(config_path) as f:
    cfg = json.load(f)

project_root = str(Path(__file__).resolve().parent)
addon_dir = os.path.join(project_root, "blender_addon")
for p in [project_root, addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import bpy
from mathutils import Vector, Matrix

# Import boids_step and make_swarm from addon/Stage 1
from blender_addon.swarm_scanner import boids_step
from stage1_geometry.scene_config import (
    AREA_KM, HEIGHT_RANGE_M, DRONE_SIZE_M, IMAGE_SIZE
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_FRAMES = cfg.get("n_frames", 20)
FPS = cfg.get("fps", 10)
N_VIEWS = cfg.get("n_views", 12)
FOCAL_MM = cfg.get("focal_mm", 24)
SENSOR_W_MM = cfg.get("sensor_width_mm", 36.0)
H_PX, V_PX = cfg.get("resolution", [1920, 1080])
STANDOFF_M = cfg.get("standoff_m", 2000)
SEED = cfg.get("seed", 42)
N_DRONES = cfg.get("n_drones", 20)
SAMPLES = cfg.get("samples", 32)

DT = 1.0 / FPS  # time step between rendered frames

# Boids parameters (matching addon defaults)
BOIDS_DEFAULTS = {
    "neighbor_radius": 800.0,
    "bound_softness": 500.0,
    "wander_strength": 15.0,
    "max_speed": 30.0,  # m/s
}


def make_swarm(n_drones, seed):
    """Generate random drone positions in the 5km volume."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-AREA_KM * 500, AREA_KM * 500, size=(n_drones, 2))
    z = rng.uniform(0, HEIGHT_RANGE_M, size=(n_drones, 1))
    return np.hstack([xy, z]).astype(np.float64)


def generate_trajectory(n_frames, seed):
    """Run boids sim to generate drone positions at each frame."""
    rng = np.random.default_rng(seed)
    pos = make_swarm(N_DRONES, seed)
    vel = rng.uniform(-5, 5, size=pos.shape)  # small initial velocities

    bounds = np.array([
        [-AREA_KM * 500, -AREA_KM * 500, 0],
        [AREA_KM * 500, AREA_KM * 500, HEIGHT_RANGE_M],
    ])

    trajectory = [pos.copy()]
    for _ in range(n_frames - 1):
        pos, vel = boids_step(
            pos, vel, DT,
            bounds=bounds,
            rng=rng,
            **BOIDS_DEFAULTS,
        )
        trajectory.append(pos.copy())

    return np.array(trajectory)  # (N_FRAMES, N_DRONES, 3)


def setup_scene():
    """Clear scene and configure render settings.

    Uses factory reset to start clean, then rebuilds lighting.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.render.resolution_x = H_PX
    scene.render.resolution_y = V_PX
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    return scene


def setup_lighting(scene):
    """Sun light + sky-colored world background.

    After factory_reset(use_empty=True), there's no world or light.
    Create both explicitly.
    """
    # World background
    world = bpy.data.worlds.new("sky_world")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.5, 0.6, 0.9, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # Sun light (required for Cycles to illuminate objects)
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = 4.0
    light_data.color = (1.0, 0.95, 0.85)
    light_obj = bpy.data.objects.new("Sun", light_data)
    light_obj.location = (0, 0, 10000)
    light_obj.rotation_euler = (0.3, 0.0, 0.0)
    bpy.context.collection.objects.link(light_obj)


def create_drone_mesh():
    """Create a reusable drone mesh (small emissive cube).

    Uses emission shader so drones are visible at sub-pixel sizes.
    This is a detection-verification render, not a photorealistic one.
    The emission ensures Cycles produces a measurable signal even for
    0.5m objects at 2km standoff.
    """
    mesh = bpy.data.meshes.new("drone_mesh")
    verts = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    mat = bpy.data.materials.new("drone_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 100.0

    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    mesh.materials.append(mat)
    return mesh


def add_drones_to_scene(mesh, positions):
    """Add drone objects at given positions. Returns list of objects."""
    drones = []
    for i, pos in enumerate(positions):
        obj = bpy.data.objects.new(f"drone_{i:03d}", mesh)
        obj.location = Vector(pos.tolist())
        obj.scale = (DRONE_SIZE_M, DRONE_SIZE_M, DRONE_SIZE_M)
        bpy.context.collection.objects.link(obj)
        drones.append(obj)
    return drones


def create_cameras(n_views, seed):
    """Place cameras in dome pattern around swarm center."""
    rng = np.random.default_rng(seed + 7777)
    center = np.array([0.0, 0.0, HEIGHT_RANGE_M / 2])
    cameras = []

    for i in range(n_views):
        elev = rng.uniform(20, 50)
        az = 360.0 * i / n_views + rng.uniform(-5, 5)
        elev_rad = np.radians(elev)
        az_rad = np.radians(az)

        cam_pos = center + np.array([
            STANDOFF_M * np.cos(elev_rad) * np.cos(az_rad),
            STANDOFF_M * np.cos(elev_rad) * np.sin(az_rad),
            STANDOFF_M * np.sin(elev_rad),
        ])

        cam_data = bpy.data.cameras.new(f"View_{i:02d}")
        cam_data.lens = FOCAL_MM
        cam_data.sensor_width = SENSOR_W_MM
        cam_data.clip_end = 50000
        cam_obj = bpy.data.objects.new(f"View_{i:02d}", cam_data)
        cam_obj.location = Vector(cam_pos.tolist())

        # Point at swarm center
        direction = (Vector(center.tolist()) - cam_obj.location).normalized()
        forward = direction
        up_hint = Vector((0, 0, 1))
        right = forward.cross(up_hint).normalized()
        up = right.cross(forward).normalized()
        mat = Matrix.Identity(4)
        mat[0][0], mat[1][0], mat[2][0] = right.x, right.y, right.z
        mat[0][1], mat[1][1], mat[2][1] = up.x, up.y, up.z
        mat[0][2], mat[1][2], mat[2][2] = -forward.x, -forward.y, -forward.z
        mat[0][3], mat[1][3], mat[2][3] = cam_pos[0], cam_pos[1], cam_pos[2]
        cam_obj.matrix_world = mat

        bpy.context.collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    return cameras


def render_frame(scene, camera, filepath):
    """Render a single frame from a camera."""
    scene.camera = camera
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def main():
    dataset_root = Path(cfg.get("dataset_root", "dataset_temporal"))
    clip_dir = dataset_root / "clips" / cfg["clip_name"]
    clip_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[{cfg['clip_name']}] Rendering {N_FRAMES} frames × {N_VIEWS} views, "
          f"{N_DRONES} drones, seed={SEED}")

    # Generate trajectory
    trajectory = generate_trajectory(N_FRAMES, SEED)
    print(f"  Trajectory: {trajectory.shape}, "
          f"speed range: {np.linalg.norm(np.diff(trajectory, axis=0), axis=2).mean():.1f} m/frame")

    # Setup scene
    scene = setup_scene()
    setup_lighting(scene)
    mesh = create_drone_mesh()

    # Create cameras
    cameras = create_cameras(N_VIEWS, SEED)
    bpy.context.view_layer.update()

    # Render each view's sequence
    all_K = []
    all_extrinsics = []
    frame_dir = clip_dir / "frames"
    frame_dir.mkdir(exist_ok=True)

    for view_idx, cam_obj in enumerate(cameras):
        view_dir = frame_dir / f"view_{view_idx:02d}"
        view_dir.mkdir(exist_ok=True)

        # Extract intrinsics
        focal_px = FOCAL_MM * H_PX / SENSOR_W_MM
        K = np.array([
            [focal_px, 0, H_PX / 2],
            [0, focal_px, V_PX / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        ext = np.array(cam_obj.matrix_world, dtype=np.float64)
        all_K.append(K)
        all_extrinsics.append(ext)

        for frame_idx in range(N_FRAMES):
            # Remove old drones
            for obj in list(bpy.data.objects):
                if obj.name.startswith("drone_"):
                    bpy.data.objects.remove(obj, do_unlink=True)

            # Add drones at this frame's positions
            positions = trajectory[frame_idx]
            add_drones_to_scene(mesh, positions)
            bpy.context.view_layer.update()

            # Render
            filepath = str(view_dir / f"frame_{frame_idx:04d}.exr")
            render_frame(scene, cam_obj, filepath)

        t_elapsed = time.time() - t_start
        views_done = view_idx + 1
        est_total = t_elapsed / views_done * N_VIEWS
        print(f"  View {view_idx}: done ({views_done}/{N_VIEWS}, "
              f"elapsed {t_elapsed:.0f}s, est total {est_total:.0f}s)")

    # Save ground truth
    gt = {
        "trajectory": trajectory,  # (N_FRAMES, N_DRONES, 3) true-scale positions
        "K": np.array(all_K),      # (N_VIEWS, 3, 3)
        "extrinsics": np.array(all_extrinsics),  # (N_VIEWS, 4, 4)
        "meta": np.array({
            "clip_name": cfg["clip_name"],
            "n_frames": N_FRAMES,
            "fps": FPS,
            "n_drones": N_DRONES,
            "n_views": N_VIEWS,
            "focal_mm": FOCAL_MM,
            "sensor_width_mm": SENSOR_W_MM,
            "resolution": [H_PX, V_PX],
            "standoff_m": STANDOFF_M,
            "drone_size_m": DRONE_SIZE_M,
            "seed": SEED,
            "boids_dt": DT,
        }, dtype=object),
    }
    gt_path = str(clip_dir / "gt.npz")
    np.savez_compressed(gt_path, **gt)

    t_total = time.time() - t_start
    print(f"\n[{cfg['clip_name']}] Done in {t_total:.0f}s")
    print(f"  Trajectory: {gt_path}")
    print(f"  Frames: {frame_dir}")
    print(f"TIMING:{t_total:.1f}")

    # Cleanup
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


if __name__ == "__main__":
    main()
