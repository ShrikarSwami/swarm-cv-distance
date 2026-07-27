"""
Inline dome render — working approach.

Uses bpy.ops.wm.read_factory_settings() (NOT use_empty=True) to preserve
Blender's default material infrastructure. Modifies existing Principled BSDF
for emission instead of creating new materials with nodes.clear().

Root cause of render_sequence.py failure: use_empty=True creates empty
scene where fresh material node trees don't initialize correctly in
Blender 5.x Cycles. The fix is to preserve the default scene and modify
existing materials.
"""

import bpy
import numpy as np
import mathutils
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
N_DRONES = 10
N_VIEWS = 12
N_FRAMES = 3
FPS = 2
FOCAL_MM = 24
SENSOR_W_MM = 36.0
H_PX, V_PX = 480, 270
STANDOFF_M = 2000
SAMPLES = 16
DRONE_SIZE_M = 0.5
HEIGHT_RANGE_M = 1000.0
AREA_KM = 5.0
DT = 1.0 / FPS


def main():
    # Import boids
    project_root = str(Path(__file__).resolve().parent)
    addon_dir = os.path.join(project_root, "blender_addon")
    for p in [project_root, addon_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from blender_addon.swarm_scanner import boids_step

    # Generate trajectory
    rng = np.random.default_rng(SEED)
    xy = rng.uniform(-AREA_KM * 500, AREA_KM * 500, size=(N_DRONES, 2))
    z = rng.uniform(0, HEIGHT_RANGE_M, size=(N_DRONES, 1))
    pos = np.hstack([xy, z]).astype(np.float64)
    vel = rng.uniform(-5, 5, size=pos.shape)
    bounds = np.array([[-2500, -2500, 0], [2500, 2500, 1000]])
    trajectory = [pos.copy()]
    for _ in range(N_FRAMES - 1):
        pos, vel = boids_step(pos, vel, DT, bounds=bounds, rng=rng,
                              neighbor_radius=800, bound_softness=500,
                              wander_strength=15, max_speed=30)
        trajectory.append(pos.copy())
    trajectory = np.array(trajectory)

    # === SCENE SETUP (use factory reset WITHOUT use_empty) ===
    bpy.ops.wm.read_factory_settings()
    scene = bpy.context.scene

    # Remove default Cube and Light
    for name in ["Cube", "Light"]:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Configure render
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = H_PX
    scene.render.resolution_y = V_PX
    scene.render.image_settings.file_format = "OPEN_EXR"

    # Modify existing world (don't create new — preserve node infrastructure)
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    wn = world.node_tree.nodes
    wl = world.node_tree.links
    # Clear existing nodes and set up sky background
    wn.clear()
    bg = wn.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.5, 0.6, 0.9, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    output = wn.new("ShaderNodeOutputWorld")
    wl.new(bg.outputs["Background"], output.inputs["Surface"])

    # Create drone mesh
    mesh = bpy.data.meshes.new("drone_mesh")
    verts = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
             (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    # Create material by MODIFYING existing default BSDF (don't create from scratch)
    mat = bpy.data.materials.new("drone_mat")
    mat.use_nodes = True  # Creates default Principled BSDF
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 100.0
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
    mesh.materials.append(mat)

    # Place cameras in dome
    rng2 = np.random.default_rng(SEED + 7777)
    cam_center = np.array([0.0, 0.0, HEIGHT_RANGE_M / 2])
    cameras = []
    for i in range(N_VIEWS):
        elev = rng2.uniform(20, 50)
        az = 360.0 * i / N_VIEWS + rng2.uniform(-5, 5)
        elev_rad = np.radians(elev)
        az_rad = np.radians(az)
        cam_pos = cam_center + np.array([
            STANDOFF_M * np.cos(elev_rad) * np.cos(az_rad),
            STANDOFF_M * np.cos(elev_rad) * np.sin(az_rad),
            STANDOFF_M * np.sin(elev_rad),
        ])

        cam_data = bpy.data.cameras.new(f"View_{i:02d}")
        cam_data.lens = FOCAL_MM
        cam_data.sensor_width = SENSOR_W_MM
        cam_data.clip_end = 50000
        cam_obj = bpy.data.objects.new(f"View_{i:02d}", cam_data)
        cam_obj.location = mathutils.Vector(cam_pos.tolist())

        direction = (mathutils.Vector(cam_center.tolist()) - cam_obj.location).normalized()
        forward = direction
        up_hint = mathutils.Vector((0, 0, 1))
        right = forward.cross(up_hint).normalized()
        up = right.cross(forward).normalized()
        mat_m = mathutils.Matrix.Identity(4)
        mat_m[0][0], mat_m[1][0], mat_m[2][0] = right.x, right.y, right.z
        mat_m[0][1], mat_m[1][1], mat_m[2][1] = up.x, up.y, up.z
        mat_m[0][2], mat_m[1][2], mat_m[2][2] = -forward.x, -forward.y, -forward.z
        mat_m[0][3], mat_m[1][3], mat_m[2][3] = cam_pos[0], cam_pos[1], cam_pos[2]
        cam_obj.matrix_world = mat_m

        scene.collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    scene.camera = cameras[0]
    bpy.context.view_layer.update()

    # Setup output directory
    clip_dir = Path("dataset_temporal/clips/dome_working")
    clip_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = clip_dir / "frames"
    frame_dir.mkdir(exist_ok=True)

    all_K = []
    all_ext = []

    for vi, cam_obj in enumerate(cameras):
        view_dir = frame_dir / f"view_{vi:02d}"
        view_dir.mkdir(exist_ok=True)

        focal_px = FOCAL_MM * H_PX / SENSOR_W_MM
        K = np.array([[focal_px, 0, H_PX / 2],
                       [0, focal_px, V_PX / 2],
                       [0, 0, 1]], dtype=np.float64)
        ext = np.array(cam_obj.matrix_world, dtype=np.float64)
        all_K.append(K)
        all_ext.append(ext)

        scene.camera = cam_obj

        for fi in range(N_FRAMES):
            # Remove previous drones
            for obj in list(bpy.data.objects):
                if obj.name.startswith("drone_"):
                    bpy.data.objects.remove(obj, do_unlink=True)

            # Add drones
            positions = trajectory[fi]
            for i, pos in enumerate(positions):
                obj = bpy.data.objects.new(f"drone_{i:03d}", mesh)
                obj.location = mathutils.Vector(pos.tolist())
                obj.scale = (DRONE_SIZE_M, DRONE_SIZE_M, DRONE_SIZE_M)
                scene.collection.objects.link(obj)

            bpy.context.view_layer.update()

            filepath = str(view_dir / f"frame_{fi:04d}.exr")
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)

        print(f"  View {vi}: done ({vi + 1}/{N_VIEWS})")

    # Save ground truth
    gt = {
        "trajectory": trajectory,
        "K": np.array(all_K),
        "extrinsics": np.array(all_ext),
        "meta": np.array({
            "clip_name": "dome_working",
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
    print(f"Done. GT: {gt_path}")


if __name__ == "__main__":
    main()
