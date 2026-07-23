"""
M1: Render a single optics-sweep config via Blender headless.

Uses the addon's operators (generate_swarm) for proper scene graph evaluation,
then overrides camera placement per config. Drones rendered at display_scale
(default20x) so they rasterize; pixel measurements divided by scale to get
true apparent size.

Usage:
    blender --background --python blender_addon/render_config.py -- <config.json>
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

argv = sys.argv
if "--" not in argv:
    print("Usage: blender --background --python render_config.py -- <config.json>")
    sys.exit(1)
args = argv[argv.index("--") + 1:]
config_path = args[0]

with open(config_path) as f:
    cfg = json.load(f)

project_root = str(Path(__file__).resolve().parent.parent)
addon_dir = str(Path(__file__).resolve().parent)
for p in [project_root, addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import bpy
import numpy as np
from mathutils import Vector, Matrix

# ---------------------------------------------------------------------------
# Use addon's operators for proper scene graph evaluation
# ---------------------------------------------------------------------------

import swarm_scanner

swarm_scanner.register()
scene = bpy.context.scene

# Configure swarm generation
DISPLAY_SCALE = 20.0
scene.swarm_scan.drone_count = 20
scene.swarm_scan.seed = cfg.get("seed", 42)
scene.swarm_scan.display_scale = DISPLAY_SCALE
scene.swarm_scan.camera_count = 2  # Minimal — we'll override

# Generate swarm (creates drones with proper materials + pass_index)
bpy.ops.swarm.generate_swarm()

# Get drone objects and positions
drones = sorted(
    [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('drone_')],
    key=lambda o: o.name,
)
drone_positions = np.array([o.location[:] for o in drones])  # (N,3) in meters
n_drones = len(drones)
print(f"Generated {n_drones} drones at display_scale={DISPLAY_SCALE}")

# ---------------------------------------------------------------------------
# Camera placement
# ---------------------------------------------------------------------------

SWARM_CENTER = Vector(np.mean(drone_positions, axis=0).tolist())
standoff = cfg["standoff_m"]
sensor_w = cfg["sensor_width_mm"]
focal = cfg["focal_mm"]
n_cams = min(12, max(2, cfg.get("n_cams", 6)))


def compute_camera_positions(standoff_m, n_cams, seed=0):
    """Dome-style camera placement."""
    rng = np.random.default_rng(seed)
    positions = []
    for i in range(n_cams):
        elev = math.radians(rng.uniform(20, 50))
        az = 2 * math.pi * i / n_cams + rng.uniform(-0.15, 0.15)
        slant = standoff_m
        pos = SWARM_CENTER + Vector((
            slant * math.cos(elev) * math.cos(az),
            slant * math.cos(elev) * math.sin(az),
            slant * math.sin(elev),
        ))
        positions.append(pos)
    return positions


# Remove cameras, default cube (addon creates defaults); keep/add light
for obj in list(bpy.data.objects):
    if obj.type == 'CAMERA' or obj.name in ('Cube',):
        bpy.data.objects.remove(obj, do_unlink=True)

# Ensure at least one sun light exists for rendering
has_light = any(o.type == 'LIGHT' for o in bpy.data.objects)
if not has_light:
    light_data = bpy.data.lights.new("Sun", 'SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new("Sun", light_data)
    light_obj.location = (0, 0, 10000)
    bpy.context.collection.objects.link(light_obj)

# Place new cameras with manual rotation (TRACK_TO doesn't evaluate in headless)
cam_positions = compute_camera_positions(standoff, n_cams, seed=cfg.get("seed", 42) + 1000)

cameras = []
for i, pos in enumerate(cam_positions):
    cam_data = bpy.data.cameras.new(f"Camera_{i:02d}")
    cam_data.lens = focal
    cam_data.sensor_width = sensor_w
    cam_data.clip_end = 50000
    cam_obj = bpy.data.objects.new(f"Camera_{i:02d}", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Build camera-to-world matrix directly (rotation_euler not evaluated headless)
    direction = (SWARM_CENTER - pos).normalized()
    forward = direction  # camera -Z points toward target, so forward = direction
    up_hint = Vector((0, 0, 1))
    right = forward.cross(up_hint).normalized()
    up = right.cross(forward).normalized()

    mat = Matrix.Identity(4)
    # Columns: X=right, Y=up, Z=-forward (camera convention)
    mat[0][0], mat[1][0], mat[2][0] = right.x, right.y, right.z
    mat[0][1], mat[1][1], mat[2][1] = up.x, up.y, up.z
    mat[0][2], mat[1][2], mat[2][2] = -forward.x, -forward.y, -forward.z
    mat[0][3], mat[1][3], mat[2][3] = pos.x, pos.y, pos.z

    cam_obj.matrix_world = mat
    cameras.append(cam_obj)

scene.camera = cameras[0]

# Force scene evaluation (required headless — rotation_euler / matrix_world
# are not evaluated until depsgraph runs, which --background skips)
bpy.context.view_layer.update()
bpy.context.evaluated_depsgraph_get().update()

# Diagnostic: verify camera orientations
print(f"Swarm center: {SWARM_CENTER}")
for cam in cameras:
    mat = cam.matrix_world
    cam_pos = Vector((mat[0][3], mat[1][3], mat[2][3]))
    cam_fwd = -Vector((mat[0][2], mat[1][2], mat[2][2])).normalized()
    to_center = (SWARM_CENTER - cam_pos).normalized()
    dot = cam_fwd.dot(to_center)
    print(f"  {cam.name}: pos={cam_pos}, fwd={cam_fwd}, dot={dot:.3f}")

# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons.get('cycles')
if prefs:
    prefs.preferences.compute_device_type = 'NONE'

scene.cycles.samples = 32
scene.cycles.use_denoising = False
scene.render.resolution_x = cfg["h_px"]
scene.render.resolution_y = cfg["v_px"]
scene.render.resolution_percentage = 100

# Compositor: Object Index → File Output (Blender 5.x API)
scene.view_layers[0].use_pass_object_index = True

node_group = bpy.data.node_groups.new("M1Compositing", "CompositorNodeTree")
scene.compositing_node_group = node_group
node_group.nodes.clear()

render_layers = node_group.nodes.new('CompositorNodeRLayers')
render_layers.location = (0, 0)

id_out = node_group.nodes.new('CompositorNodeOutputFile')
id_out.location = (300, 0)
id_out.directory = os.path.join(tempfile.gettempdir(), "m1_render") + os.sep
id_out.file_name = "id_"
id_out.file_output_items.clear()
id_out.file_output_items.new("FLOAT", "id_")
id_out.format.file_format = 'OPEN_EXR_MULTILAYER'
id_out.format.color_depth = '32'

node_group.links.new(render_layers.outputs["Object Index"], id_out.inputs["id_"])

# Grey world background
world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

render_dir = os.path.join(tempfile.gettempdir(), "m1_render")
render_path = os.path.join(render_dir, "id_.exr")
scene.render.filepath = render_dir
bpy.ops.render.render(write_still=True)

# ---------------------------------------------------------------------------
# Extract measurements (via venv Python)
# ---------------------------------------------------------------------------

config_name = f"{cfg['tier']}_{cfg['focal_mm']}mm_{cfg['resolution']}_{cfg['standoff_m']}m_s{cfg.get('seed',42)}"

# Cleanup Blender data
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(mesh)
for ng in list(bpy.data.node_groups):
    if ng.name == "M1Compositing":
        bpy.data.node_groups.remove(ng)

# Call venv Python for EXR extraction
venv_python = os.path.join(project_root, "venv", "bin", "python")
measure_script = os.path.join(project_root, "blender_addon", "measure_id_pass.py")

try:
    proc = subprocess.run(
        [venv_python, measure_script, render_path,
         str(n_drones), str(cfg["standoff_m"]),
         str(cfg["focal_mm"]), str(cfg["sensor_width_mm"]),
         str(cfg["h_px"])],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"measure_id_pass.py failed: {proc.stderr}")
    measurements = json.loads(proc.stdout)

    result = {
        "config_name": config_name,
        "display_scale": DISPLAY_SCALE,
        "render_path": render_path,
        **cfg,
        **measurements,
    }

    # Scale-corrected apparent pixel size (true 0.5m drone)
    # Measured bbox is for inflated (0.5*DISPLAY_SCALE)m mesh
    # True apparent px = measured_px / DISPLAY_SCALE
    # (approximately — breaks down near1px true scale)
    true_apparent_px = measurements["expected_apparent_px"]  # Already computed for true scale
    if measurements["visible_count"] > 0:
        visible = [m for m in measurements["drone_measurements"] if m["visible"]]
        measured_avg_bbox = np.mean([m["bbox_width_px"] for m in visible])
        inflated_apparent_px = 0.5 * DISPLAY_SCALE / (cfg["standoff_m"] * (measurements["theta_um_rad"] * 1e-6))
        result["measured_avg_bbox_px"] = round(float(measured_avg_bbox), 2)
        result["inflated_apparent_px"] = round(float(inflated_apparent_px), 2)
        result["scale_ratio"] = round(float(measured_avg_bbox / inflated_apparent_px), 3) if inflated_apparent_px > 0 else 0

    out_dir = Path(project_root) / "logs" / "m1_renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config_name}_measurements.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Measurements saved: {out_path}")

except Exception as e:
    print(f"ERROR: {e}")
    result = {"error": str(e), "render_path": render_path, "config_name": config_name}

print(f"Done: {config_name}")
