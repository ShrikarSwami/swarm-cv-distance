"""
Render viewing clip: 6 cameras × 45s × 24fps = 6,480 frames.
MP4/H.264 output. Boids flight sim for smooth motion.
Uses display_scale=20 for visible drones, Standard color management.
"""
import bpy, numpy as np, mathutils, time, subprocess, json, sys, os
from pathlib import Path

project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "blender_addon"))
from blender_addon.swarm_scanner import boids_step

# --- Config ---
SEED = 42
N_DRONES = 20
N_VIEWS = 6
N_FRAMES = 1080  # 45s at 24fps
FPS = 24
DT = 1.0 / FPS
FOCAL_MM = 24
SENSOR_W = 36.0
H_PX, V_PX = 1920, 1080
STANDOFF = 2000
SAMPLES = 32
DRONE_M = 0.5
DISPLAY_SCALE = 20.0
H_MAX = 1000.0
AREA_KM = 5.0
half = AREA_KM * 500
RENDER_MODE = "smoke"  # "smoke" = 10 frames, "full" = all frames

if "--full" in sys.argv:
    RENDER_MODE = "full"
elif "--smoke" in sys.argv:
    RENDER_MODE = "smoke"

n_frames = 10 if RENDER_MODE == "smoke" else N_FRAMES
out_root = Path("renders")
clip_dir = out_root / "swarm_flight"
clip_dir.mkdir(parents=True, exist_ok=True)

# --- Trajectory ---
print(f"Generating {N_FRAMES}-frame boids trajectory...")
t0 = time.time()
rng = np.random.default_rng(SEED)
xy = rng.uniform(-half, half, (N_DRONES, 2))
z = rng.uniform(0, H_MAX, (N_DRONES, 1))
pos = np.hstack([xy, z]).astype(np.float64)
vel = rng.uniform(-5, 5, pos.shape)
bounds = np.array([[-half, -half, 0], [half, half, H_MAX]])
trajectory = [pos.copy()]
for _ in range(N_FRAMES - 1):
    pos, vel = boids_step(pos, vel, DT, bounds=bounds, rng=rng,
                          neighbor_radius=800, bound_softness=500,
                          wander_strength=15, max_speed=30)
    trajectory.append(pos.copy())
trajectory = np.array(trajectory)
print(f"  Trajectory generated in {time.time()-t0:.1f}s")

# --- Scene setup ---
print("Setting up scene...")
bpy.ops.wm.read_factory_settings()
scene = bpy.context.scene
for n in ["Cube", "Light"]:
    o = bpy.data.objects.get(n)
    if o: bpy.data.objects.remove(o, do_unlink=True)

# Color management: Standard (no tonemapping)
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.render.resolution_x = H_PX
scene.render.resolution_y = V_PX

# World background (sky blue)
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
wn, wl = world.node_tree.nodes, world.node_tree.links
wn.clear()
bg = wn.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.5, 0.6, 0.9, 1.0)
bg.inputs["Strength"].default_value = 1.0
out = wn.new("ShaderNodeOutputWorld")
wl.new(bg.outputs["Background"], out.inputs["Surface"])

# Ground plane (green)
bpy.ops.mesh.primitive_plane_add(size=10000, location=(0, 0, 0))
ground = bpy.context.active_object
gmat = bpy.data.materials.new("ground_mat")
gmat.use_nodes = True
gbsdf = gmat.node_tree.nodes.get("Principled BSDF")
gbsdf.inputs["Base Color"].default_value = (0.3, 0.6, 0.2, 1.0)
gbsdf.inputs["Roughness"].default_value = 1.0
ground.data.materials.append(gmat)

# Sun
ld = bpy.data.lights.new("sun", "SUN")
ld.energy = 4.0
lo = bpy.data.objects.new("sun", ld)
lo.location = (0, 0, 10000)
scene.collection.objects.link(lo)

# Drone mesh
mesh = bpy.data.meshes.new("drone_mesh")
verts = [(-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),(0.5,0.5,-0.5),(-0.5,0.5,-0.5),
         (-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5)]
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
mesh.from_pydata(verts, edges, [])
mesh.update()
dmat = bpy.data.materials.new("drone_mat")
dmat.use_nodes = True
dbsdf = dmat.node_tree.nodes.get("Principled BSDF")
dbsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
dbsdf.inputs["Emission Strength"].default_value = 100.0
dbsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
mesh.materials.append(dmat)

# Cameras
rng2 = np.random.default_rng(SEED + 7777)
cam_center = np.array([0.0, 0.0, H_MAX / 2])
cameras = []
for i in range(N_VIEWS):
    elev = rng2.uniform(20, 50)
    az = 360.0 * i / N_VIEWS + rng2.uniform(-5, 5)
    er, ar = np.radians(elev), np.radians(az)
    cam_pos = cam_center + np.array([
        STANDOFF * np.cos(er) * np.cos(ar),
        STANDOFF * np.cos(er) * np.sin(ar),
        STANDOFF * np.sin(er)])

    cd = bpy.data.cameras.new(f"View_{i:02d}")
    cd.lens = FOCAL_MM
    cd.sensor_width = SENSOR_W
    cd.clip_end = 50000
    co = bpy.data.objects.new(f"View_{i:02d}", cd)
    co.location = mathutils.Vector(cam_pos.tolist())
    direction = (mathutils.Vector(cam_center.tolist()) - co.location).normalized()
    up_hint = mathutils.Vector((0, 0, 1))
    right = direction.cross(up_hint).normalized()
    up = right.cross(direction).normalized()
    m = mathutils.Matrix.Identity(4)
    m[0][0],m[1][0],m[2][0] = right.x,right.y,right.z
    m[0][1],m[1][1],m[2][1] = up.x,up.y,up.z
    m[0][2],m[1][2],m[2][2] = -direction.x,-direction.y,-direction.z
    m[0][3],m[1][3],m[2][3] = cam_pos[0],cam_pos[1],cam_pos[2]
    co.matrix_world = m
    scene.collection.objects.link(co)
    cameras.append(co)

bpy.context.view_layer.update()
bpy.context.evaluated_depsgraph_get().update()
print(f"  Scene ready: {len(bpy.data.objects)} objects")

# --- Render loop ---
print(f"\nRendering {RENDER_MODE}: {n_frames} frames × {N_VIEWS} cameras...")
all_frame_times = []

for vi, cam_obj in enumerate(cameras):
    view_dir = clip_dir / f"view_{vi:02d}"
    view_dir.mkdir(exist_ok=True)

    scene.camera = cam_obj
    frame_times = []

    for fi in range(n_frames):
        # Remove old drones
        for obj in list(bpy.data.objects):
            if obj.name.startswith("drone_"):
                bpy.data.objects.remove(obj, do_unlink=True)

        # Add drones at current frame — must use bpy.ops for Cycles evaluation
        positions = trajectory[fi]
        for i, p in enumerate(positions):
            bpy.ops.mesh.primitive_cube_add(size=DRONE_M * DISPLAY_SCALE,
                                            location=p.tolist())
            o = bpy.context.active_object
            o.name = f"drone_{i:03d}"
            o.data.materials.append(dmat)

        bpy.context.view_layer.update()

        # Render to temp PNG
        tmp_path = str(view_dir / f"frame_{fi:04d}.png")
        scene.render.filepath = tmp_path

        t_start = time.time()
        bpy.ops.render.render(write_still=True)
        t_frame = time.time() - t_start
        frame_times.append(t_frame)

        if fi % 100 == 0 or fi == n_frames - 1:
            elapsed = sum(frame_times)
            rate = len(frame_times) / elapsed if elapsed > 0 else 0
            remaining = (n_frames - fi - 1) / rate / 60 if rate > 0 else 0
            print(f"  View {vi} frame {fi:4d}/{n_frames}: {t_frame:.1f}s "
                  f"(avg {sum(frame_times)/len(frame_times):.1f}s, ~{remaining:.0f}min left)")

    all_frame_times.extend(frame_times)

    # Encode to MP4
    mp4_path = str(clip_dir / f"view_{vi:02d}.mp4")
    input_pattern = str(view_dir / "frame_%04d.png")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        mp4_path,
    ]
    print(f"  Encoding view {vi} to MP4...")
    proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"  ✓ {mp4_path}")
    else:
        print(f"  ✗ FFmpeg error: {proc.stderr[-200:]}")

    # Clean up PNGs
    for f in view_dir.glob("frame_*.png"):
        f.unlink()

# --- Summary ---
total_time = sum(all_frame_times)
avg_frame = total_time / len(all_frame_times)
print(f"\n{'='*60}")
print(f"RENDER COMPLETE")
print(f"{'='*60}")
print(f"Total frames: {n_frames * N_VIEWS}")
print(f"Total render time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"Average per frame: {avg_frame:.1f}s")
print(f"Output: {clip_dir}/")
print(f"  6 MP4 files, one per camera angle")
print(f"  {FPS}fps, {H_PX}x{V_PX}, H.264")
