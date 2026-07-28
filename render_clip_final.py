"""
Final viewing clip: horizontal cameras, Nishita sky, textured ground.
EEVEE for speed. 6 angles × 30s × 30fps = 5,400 frames.
1280×720, MP4/H.264.
"""
import bpy, numpy as np, mathutils, time, subprocess, sys, os
from pathlib import Path

project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "blender_addon"))
from blender_addon.swarm_scanner import boids_step

# --- Config ---
SEED = 42
N_DRONES = 20
N_VIEWS = 6
N_FRAMES = 900  # 30s at 30fps
FPS = 30
DT = 1.0 / FPS
FOCAL_MM = 24
SENSOR_W = 36.0
H_PX, V_PX = 1280, 720
STANDOFF = 2000
DRONE_M = 0.5
DISPLAY_SCALE = 20.0
H_MAX = 1000.0
AREA_KM = 5.0
half = AREA_KM * 500
RENDER_MODE = "full"
ENGINE = "BLENDER_EEVEE"

if "--smoke" in sys.argv:
    RENDER_MODE = "smoke"
    N_FRAMES = 10
elif "--full" in sys.argv:
    RENDER_MODE = "full"

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
print(f"  Done in {time.time()-t0:.1f}s")

# --- Scene setup ---
print("Setting up scene...")
bpy.ops.wm.read_factory_settings()
scene = bpy.context.scene
for n in ["Cube", "Light"]:
    o = bpy.data.objects.get(n)
    if o: bpy.data.objects.remove(o, do_unlink=True)

scene.render.engine = ENGINE
scene.render.resolution_x = H_PX
scene.render.resolution_y = V_PX
scene.render.resolution_percentage = 100

# Gradient sky: blue at zenith, lighter at horizon
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
wn, wl = world.node_tree.nodes, world.node_tree.links
wn.clear()
bg = wn.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0
# Gradient texture mapped to Z coordinate for sky gradient
grad = wn.new("ShaderNodeTexGradient")
grad.gradient_type = "LINEAR"
grad.inputs["Vector"].default_value = (0, 0, 1)  # will be overridden by mapping
# Color ramp: dark blue at top → light blue at horizon
ramp = wn.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.15, 0.25, 0.55, 1.0)  # deep blue (zenith)
ramp.color_ramp.elements[1].color = (0.6, 0.75, 0.95, 1.0)   # light blue (horizon)
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[1].position = 1.0
# Mapping to control gradient direction
mapping = wn.new("ShaderNodeMapping")
mapping.inputs["Location"].default_value = (0, 0, -0.5)
mapping.inputs["Scale"].default_value = (1, 1, 2)
# Texture coordinate
texcoord = wn.new("ShaderNodeTexCoord")
wl.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
wl.new(mapping.outputs["Vector"], grad.inputs["Vector"])
wl.new(grad.outputs["Fac"], ramp.inputs["Fac"])
wl.new(ramp.outputs["Color"], bg.inputs["Color"])
out = wn.new("ShaderNodeOutputWorld")
wl.new(bg.outputs["Background"], out.inputs["Surface"])

# Ground with procedural texture
bpy.ops.mesh.primitive_plane_add(size=10000, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "terrain"
gmat = bpy.data.materials.new("terrain_mat")
gmat.use_nodes = True
gbsdf = gmat.node_tree.nodes.get("Principled BSDF")
# Base color from noise texture
wn_nodes = gmat.node_tree.nodes
wn_links = gmat.node_tree.links
noise = wn_nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 50.0
noise.inputs["Detail"].default_value = 8.0
noise.inputs["Roughness"].default_value = 0.6
color_ramp = wn_nodes.new("ShaderNodeValToRGB")
color_ramp.color_ramp.elements[0].color = (0.15, 0.35, 0.1, 1.0)  # dark green
color_ramp.color_ramp.elements[1].color = (0.4, 0.65, 0.25, 1.0)  # light green
color_ramp.color_ramp.elements[0].position = 0.3
color_ramp.color_ramp.elements[1].position = 0.7
wn_links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
wn_links.new(color_ramp.outputs["Color"], gbsdf.inputs["Base Color"])
gbsdf.inputs["Roughness"].default_value = 0.9
ground.data.materials.append(gmat)

# Sun
ld = bpy.data.lights.new("sun", "SUN")
ld.energy = 4.0
ld.color = (1.0, 0.95, 0.9)
lo = bpy.data.objects.new("sun", ld)
lo.location = (0, 0, 10000)
lo.rotation_euler = (np.radians(45), 0, np.radians(135))
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
dbsdf.inputs["Emission Color"].default_value = (0.15, 0.18, 0.22, 1.0)  # near-black, slight blue-grey
dbsdf.inputs["Emission Strength"].default_value = 100.0
dbsdf.inputs["Base Color"].default_value = (0.08, 0.09, 0.12, 1.0)  # dark base
mesh.materials.append(dmat)

# --- HORIZONTAL cameras (show sky + horizon) ---
# Camera at ~1500m altitude, looking ~10° below horizontal
# This puts the horizon in frame with sky above and ground below
cam_center = np.array([0.0, 0.0, H_MAX / 2])
rng2 = np.random.default_rng(SEED + 7777)
cameras = []
for i in range(N_VIEWS):
    # Horizontal placement: low elevation (10-15°), spread around azimuth
    elev = rng2.uniform(8, 15)  # low elevation = more horizontal view
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
    # Look at swarm center (horizontal-ish)
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

# --- Smoke test: 1 frame from camera 0 ---
if RENDER_MODE == "smoke":
    print(f"\nSmoke test: 1 frame from camera 0...")
    scene.camera = cameras[0]
    for obj in list(bpy.data.objects):
        if obj.name.startswith("drone_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    positions = trajectory[0]
    for i, p in enumerate(positions):
        bpy.ops.mesh.primitive_cube_add(size=DRONE_M * DISPLAY_SCALE, location=p.tolist())
        o = bpy.context.active_object
        o.name = f"drone_{i:03d}"
        o.data.materials.append(dmat)
    bpy.context.view_layer.update()

    t_start = time.time()
    scene.render.filepath = str(clip_dir / "smoke_frame.png")
    bpy.ops.render.render(write_still=True)
    t_frame = time.time() - t_start
    print(f"  Frame time: {t_frame:.1f}s")
    print(f"  Extrapolated to {N_FRAMES * N_VIEWS} frames: {t_frame * N_FRAMES * N_VIEWS / 60:.0f} min")
    sys.exit(0)

# --- Full render ---
print(f"\nRendering {RENDER_MODE}: {N_FRAMES} frames × {N_VIEWS} cameras...")
all_frame_times = []

for vi, cam_obj in enumerate(cameras):
    view_dir = clip_dir / f"view_{vi:02d}"
    view_dir.mkdir(exist_ok=True)
    scene.camera = cam_obj
    frame_times = []

    for fi in range(N_FRAMES):
        for obj in list(bpy.data.objects):
            if obj.name.startswith("drone_"):
                bpy.data.objects.remove(obj, do_unlink=True)
        positions = trajectory[fi]
        for i, p in enumerate(positions):
            bpy.ops.mesh.primitive_cube_add(size=DRONE_M * DISPLAY_SCALE, location=p.tolist())
            o = bpy.context.active_object
            o.name = f"drone_{i:03d}"
            o.data.materials.append(dmat)
        bpy.context.view_layer.update()

        tmp_path = str(view_dir / f"frame_{fi:04d}.png")
        scene.render.filepath = tmp_path
        t_start = time.time()
        bpy.ops.render.render(write_still=True)
        t_frame = time.time() - t_start
        frame_times.append(t_frame)

        if fi % 100 == 0 or fi == N_FRAMES - 1:
            elapsed = sum(frame_times)
            rate = len(frame_times) / elapsed if elapsed > 0 else 0
            remaining = (N_FRAMES - fi - 1) / rate / 60 if rate > 0 else 0
            print(f"  View {vi} frame {fi:4d}/{N_FRAMES}: {t_frame:.1f}s "
                  f"(avg {sum(frame_times)/len(frame_times):.1f}s, ~{remaining:.0f}min left)")

    all_frame_times.extend(frame_times)

    # Encode to MP4
    mp4_path = str(clip_dir / f"view_{vi:02d}.mp4")
    input_pattern = str(view_dir / "frame_%04d.png")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", input_pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4_path,
    ]
    print(f"  Encoding view {vi}...")
    proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"  ✓ {mp4_path}")
    else:
        print(f"  ✗ {proc.stderr[-200:]}")

    for f in view_dir.glob("frame_*.png"):
        f.unlink()

total_time = sum(all_frame_times)
avg_frame = total_time / len(all_frame_times)
print(f"\n{'='*60}")
print(f"RENDER COMPLETE")
print(f"{'='*60}")
print(f"Total frames: {N_FRAMES * N_VIEWS}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"Average: {avg_frame:.1f}s/frame")
print(f"Output: {clip_dir}/")
