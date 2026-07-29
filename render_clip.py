"""
M2: Render a single multi-view clip for the dataset.

Renders N views of a static swarm, encodes to FFV1/MKV, saves ground truth
to clip.npz. Measures per-clip wall-clock time for M3 ETA estimation.

Usage:
    blender --background --python render_clip.py -- <clip_config.json>

Clip config:
{
    "clip_name": "desert_001",
    "dataset_root": "dataset",
    "n_views": 6,
    "focal_mm": 50,
    "sensor_width_mm": 36.0,
    "resolution": [1920, 1080],
    "standoff_m": 1000,
    "seed": 42,
    "display_scale": 20.0,
    "environment": "desert",
    "weather": "clear"
}
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

argv = sys.argv
if "--" not in sys.argv:
    print("Usage: blender --background --python render_clip.py -- <config.json>")
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
from mathutils import Vector, Matrix, Euler
import swarm_scanner
from blender_addon.environments import get_environment
from blender_addon.weather import get_weather
from blender_addon.terrain import ProceduralTerrain, configure_desert, configure_forest_floor
from blender_addon.nishita_sky import nishita_sky_replace_hdri

# ---------------------------------------------------------------------------
# Generate swarm using addon
# ---------------------------------------------------------------------------

t_start = time.time()

DISPLAY_SCALE = cfg.get("display_scale", 20.0)
n_drones = cfg.get("n_drones", 20)

# Generate drone positions using make_swarm
from stage1_geometry import multiview_triangulation_test as mvt
import scene_config
scene = bpy.context.scene
positions = mvt.make_swarm(
    n_drones=n_drones,
    area_km=scene_config.AREA_KM,
    height_range_m=scene_config.HEIGHT_RANGE_M,
    seed=cfg["seed"],
)
print(f"[{cfg['clip_name']}] {n_drones} drones, display_scale={DISPLAY_SCALE}")

# Store ground-truth positions at true scale (not inflated)
gt_positions = positions / DISPLAY_SCALE  # back to true 0.5m drone positions

# ---------------------------------------------------------------------------
# Camera placement
# ---------------------------------------------------------------------------

# Remove addon cameras/lights, keep drones
for obj in list(bpy.data.objects):
    if obj.type in ("CAMERA",) or obj.name in ("Cube",):
        bpy.data.objects.remove(obj, do_unlink=True)

# Create emission material
emission_mat = bpy.data.materials.new("drone_emission")
emission_mat.use_nodes = True
ebsdf = emission_mat.node_tree.nodes.get("Principled BSDF")
if ebsdf:
    ebsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    ebsdf.inputs["Emission Strength"].default_value = 100.0
    ebsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)

# Create drones as quadcopter silhouettes
from blender_addon.quadcopter import build_quadcopter_template, create_drones_from_template
qc_scale = DISPLAY_SCALE * 0.5  # visual size matching previous cube
template = build_quadcopter_template(scale=qc_scale, emission_mat=emission_mat)
drones = create_drones_from_template(template, positions)
# Remove template (drones have their own mesh copies)
bpy.data.objects.remove(template, do_unlink=True)

# Verify
drone_count = sum(1 for o in bpy.data.objects if o.name.startswith("drone_"))
print(f"[{cfg['clip_name']}] Drones created: {drone_count}")
print(f"[{cfg['clip_name']}] Emission mat users: {emission_mat.users}")

# --- NEW: Procedural Terrain + Nishita Sky (replaces flat ground + HDRI) ---
env_name = cfg.get("environment", "desert")
weather_name = cfg.get("weather", "clear")

# 1. Create procedural terrain (replaces flat ground plane)
terrain_seed = cfg.get("terrain_seed", cfg["seed"])
terrain = ProceduralTerrain(seed=terrain_seed)
if env_name == "desert":
    configure_desert(terrain)
elif env_name == "forest":
    configure_forest_floor(terrain)
else:
    # Default to desert-like terrain for other environments
    configure_desert(terrain)
terrain_obj = terrain.build(scene)
print(f"[{cfg['clip_name']}] Terrain created: {terrain_obj.name}")

# 2. Apply Nishita physical sky (replaces HDRI + separate sun sync)
nishita_preset_map = {
    "clear": "clear",
    "overcast": "overcast",
    "hazy": "clear",   # map hazy to clear with adjusted params
    "dusk": "dusk",
    "night": "dusk",   # map night to dusk (lowest sun)
}
nishita_preset = nishita_preset_map.get(weather_name, "clear")
nishita_result = nishita_sky_replace_hdri(scene, nishita_preset)
print(f"[{cfg['clip_name']}] Nishita sky: {nishita_preset}")
print(f"[{cfg['clip_name']}] Sun energy: {nishita_result['sun_obj'].data.energy}")
print(f"[{cfg['clip_name']}] Scene exposure: {scene.view_settings.exposure}")

# Debug output
print(f"[{cfg['clip_name']}] Objects: {[obj.name for obj in bpy.data.objects]}")
print(f"[{cfg['clip_name']}] World nodes: {[node.name for node in scene.world.node_tree.nodes] if scene.world and scene.world.use_nodes else 'None'}")

# Place cameras in dome pattern
SWARM_CENTER = Vector(positions.mean(axis=0).tolist())
standoff = cfg["standoff_m"]
n_views = cfg["n_views"]
focal = cfg["focal_mm"]
sensor_w = cfg["sensor_width_mm"]
h_px, v_px = cfg["resolution"]

rng = np.random.default_rng(cfg["seed"] + 7777)
cam_data_list = []
for i in range(n_views):
    elev = rng.uniform(20, 50)
    az = 360.0 * i / n_views + rng.uniform(-5, 5)
    elev_rad = np.radians(elev)
    az_rad = np.radians(az)
    cam_pos = SWARM_CENTER + Vector((
        standoff * np.cos(elev_rad) * np.cos(az_rad),
        standoff * np.cos(elev_rad) * np.sin(az_rad),
        standoff * np.sin(elev_rad),
    ))

    cam_data = bpy.data.cameras.new(f"View_{i:02d}")
    cam_data.lens = focal
    cam_data.sensor_width = sensor_w
    cam_data.clip_end = 50000
    cam_obj = bpy.data.objects.new(f"View_{i:02d}", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Set rotation via matrix_world
    direction = (SWARM_CENTER - cam_obj.location).normalized()
    forward = direction
    up_hint = Vector((0, 0, 1))
    right = forward.cross(up_hint).normalized()
    up = right.cross(forward).normalized()
    mat = Matrix.Identity(4)
    mat[0][0], mat[1][0], mat[2][0] = right.x, right.y, right.z
    mat[0][1], mat[1][1], mat[2][1] = up.x, up.y, up.z
    mat[0][2], mat[1][2], mat[2][2] = -forward.x, -forward.y, -forward.z
    mat[0][3], mat[1][3], mat[2][3] = cam_pos.x, cam_pos.y, cam_pos.z
    cam_obj.matrix_world = mat

    # Force update matrix_world
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get().update()

    cam_data_list.append(cam_obj)

# Force scene evaluation
bpy.context.view_layer.update()
bpy.context.evaluated_depsgraph_get().update()

# ---------------------------------------------------------------------------
# Compositor setup: Object Index pass → EXR (Blender 5.x)
# ---------------------------------------------------------------------------

scene.view_layers[0].use_pass_object_index = True
node_group = bpy.data.node_groups.new("ClipCompositing", "CompositorNodeTree")
scene.compositing_node_group = node_group
# Blender 5.2: compositor node group is separate from the old scene.node_tree;
# the compositor must be activated with use_nodes (default in factory scenes).
scene.use_nodes = True

# ---------------------------------------------------------------------------
# Render each view
# ---------------------------------------------------------------------------

scene.render.engine = "CYCLES"
scene.cycles.samples = cfg.get("samples", 32)
scene.cycles.use_denoising = False
scene.render.resolution_x = h_px
scene.render.resolution_y = v_px
scene.render.resolution_percentage = 100

# Standard color management (no AgX tonemapping) for visible emission
scene.view_settings.view_transform = "Standard"

clip_dir = Path(cfg["dataset_root"]) / "clips" / cfg["clip_name"]
clip_dir.mkdir(parents=True, exist_ok=True)

tmpdir = tempfile.mkdtemp(prefix="m2_clip_")
all_K = []
all_extrinsics = []

for i, cam_obj in enumerate(cam_data_list):
    scene.camera = cam_obj

    # Rebuild compositor for this camera
    node_group.nodes.clear()
    rl = node_group.nodes.new("CompositorNodeRLayers")
    out = node_group.nodes.new("CompositorNodeOutputFile")
    out.directory = tmpdir + os.sep
    out.file_name = f"view_{i:02d}_id_"
    out.file_output_items.clear()
    out.file_output_items.new("FLOAT", "id_")
    out.format.file_format = "OPEN_EXR_MULTILAYER"
    out.format.color_depth = "32"
    node_group.links.new(rl.outputs["Object Index"], out.inputs["id_"])

    # Render
    scene.render.filepath = os.path.join(tmpdir, f"view_{i:02d}.png")
    bpy.ops.render.render(write_still=True)

    # Extract intrinsics and extrinsics
    K_mat = np.array([
        [focal, 0, h_px / 2],
        [0, focal, v_px / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    ext_mat = np.array(cam_obj.matrix_world, dtype=np.float64)

    all_K.append(K_mat)
    all_extrinsics.append(ext_mat)

    print(f"  View {i}: rendered ({h_px}x{v_px})")

# ---------------------------------------------------------------------------
# Encode frames to FFV1/MKV
# ---------------------------------------------------------------------------

# Render the RGB frames as PNGs (we already have them from the render)
# Now encode to MKV using ffmpeg
rgb_frames = sorted(Path(tmpdir).glob("view_*.png"))
if rgb_frames:
    mkv_path = str(clip_dir / "frames.mkv")
    input_pattern = os.path.join(tmpdir, "view_%02d.png")
    # Create a concat file for ffmpeg
    concat_path = os.path.join(tmpdir, "concat.txt")
    with open(concat_path, "w") as f:
        for p in rgb_frames:
            f.write(f"file '{p}'\n")
            f.write("duration 0.033\n")  # 30fps

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c:v", "ffv1", "-level", "3",
        "-coder", "1", "-context", "1",
        mkv_path,
    ]
    proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"  Encoded to {mkv_path}")
    else:
        print(f"  FFV1 encode warning: {proc.stderr[-200:]}")

# ---------------------------------------------------------------------------
# Save ground truth
# ---------------------------------------------------------------------------

K_arr = np.stack(all_K, axis=0)
ext_arr = np.stack(all_extrinsics, axis=0)
# For a static swarm, positions are the same across frames
gt_pos = np.tile(gt_positions[np.newaxis], (1, 1, 1))  # (1, N, 3)

meta = {
    "clip_name": cfg["clip_name"],
    "environment": cfg.get("environment", "desert"),
    "weather": cfg.get("weather", "clear"),
    "seed": cfg["seed"],
    "display_scale": DISPLAY_SCALE,
    "drone_size_m": 0.5,
    "standoff_m": standoff,
    "focal_mm": focal,
    "sensor_width_mm": sensor_w,
    "resolution": cfg["resolution"],
    "n_views": n_views,
    "n_drones": n_drones,
}

npz_path = str(clip_dir / "clip.npz")
np.savez_compressed(
    npz_path,
    K=K_arr,
    extrinsics=ext_arr,
    positions=gt_pos,
    meta=np.array(meta, dtype=object),
)
print(f"  Saved ground truth: {npz_path}")

# ---------------------------------------------------------------------------
# Cleanup and timing
# ---------------------------------------------------------------------------

t_elapsed = time.time() - t_start

# Add to dataset splits
dataset_root = Path(cfg["dataset_root"])
splits_path = dataset_root / "splits.json"
if splits_path.exists():
    with open(splits_path) as f:
        splits = json.load(f)
    splits.setdefault("train", []).append(cfg["clip_name"])
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

# Cleanup Blender
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(mesh)
for ng in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng)

# Cleanup tmpdir
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

print(f"[{cfg['clip_name']}] Done in {t_elapsed:.1f}s")
print(f"TIMING:{t_elapsed:.1f}")
