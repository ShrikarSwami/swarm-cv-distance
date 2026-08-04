#!/usr/bin/env python3
"""8-frame test matrix: 2 envs × 2 HDRIs × 2 camera elevations.

Usage:
    blender --background --python test_8frame_matrix.py

Output:
    dataset_smoke_test/matrix_8frame/<frame_name>.png
    dataset_smoke_test/matrix_8frame/<frame_name>_obj_index.exr
"""

import sys, os, math, time as time_module
import bpy
from mathutils import Vector
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
_addon_dir = os.path.join(_project_root, "blender_addon")
for p in [_project_root, _addon_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from blender_addon.environments import get_environment
from blender_addon.weather import get_weather
from blender_addon.hdri import apply as apply_hdri
from blender_addon.quadcopter import build_quadcopter_template, create_drones_from_template

OUTPUT_DIR = Path(_project_root) / "dataset_smoke_test" / "matrix_8frame"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test matrix
CAMERA_CONFIGS = {
    "up":   {"location": (0, -100, 30),  "rotation_x_deg": 80},
    "down": {"location": (0, -100, 80),  "rotation_x_deg": 55},
}
HDRI_PRESETS = ["clear", "dusk"]
ENVIRONMENTS = ["desert", "forest"]

FRAME_INDEX = 0


def build_scene(env_name, hdri_name, cam_config):
    """Build a complete scene: env + HDRI + drones + camera."""
    global FRAME_INDEX

    # Factory reset
    try:
        bpy.ops.wm.read_factory_use_empty(use_empty=True)
    except AttributeError:
        bpy.ops.wm.read_homefile(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"

    # Environment
    env_preset = get_environment(env_name)
    env_preset.apply(scene)

    # Weather
    weather_preset = get_weather("clear")
    weather_preset.apply(scene, hdri_active=True)

    # HDRI
    apply_hdri(scene, hdri_name)

    # Camera
    loc = cam_config["location"]
    rot_x = math.radians(cam_config["rotation_x_deg"])
    bpy.ops.object.camera_add(location=loc)
    cam = [o for o in bpy.data.objects if o.type == "CAMERA"][-1]
    cam.rotation_euler = (rot_x, 0, 0)
    scene.camera = cam

    # Emission material
    mat = bpy.data.materials.new("DroneEm")
    mat.use_nodes = True
    ns = mat.node_tree.nodes
    for n in ns: ns.remove(n)
    e = ns.new("ShaderNodeEmission")
    e.inputs["Strength"].default_value = 2.0
    e.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    o = ns.new("ShaderNodeOutputMaterial")
    o.location = (200, 0)
    mat.node_tree.links.new(e.outputs["Emission"], o.inputs["Surface"])

    # 5 quadcopter drones
    template = build_quadcopter_template(scale=2.0, emission_mat=mat)
    positions = [(i * 10 - 20, 0, 10) for i in range(5)]
    create_drones_from_template(template, positions)
    bpy.data.objects.remove(template, do_unlink=True)

    # Compositor for Object Index EXR
    scene.view_layers[0].use_pass_object_index = True
    node_group = bpy.data.node_groups.new("MatrixCompositing", "CompositorNodeTree")
    scene.compositing_node_group = node_group
    scene.use_nodes = True

    return scene


def render_frame(scene, frame_name, cam_obj):
    """Render one frame, save PNG + EXR, report timing."""
    global FRAME_INDEX

    # Compositor: Object Index → EXR
    node_group = scene.compositing_node_group
    node_group.nodes.clear()

    rl = node_group.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    out = node_group.nodes.new("CompositorNodeOutputFile")
    out.location = (200, 0)
    out.directory = str(OUTPUT_DIR) + "/"
    out.file_name = f"{frame_name}_obj_index"
    out.file_output_items.clear()
    out.file_output_items.new("FLOAT", "obj_index")
    out.format.file_format = "OPEN_EXR_MULTILAYER"
    out.format.color_depth = "32"
    node_group.links.new(rl.outputs["Object Index"], out.inputs["obj_index"])

    # Render PNG
    scene.render.filepath = str(OUTPUT_DIR / f"{frame_name}.png")
    t0 = time_module.time()
    bpy.ops.render.render(write_still=True)
    render_time = time_module.time() - t0

    # Check EXR file size (Blender's Python doesn't have OpenEXR)
    exr_path = OUTPUT_DIR / f"{frame_name}_obj_index.exr"
    exr_size_kb = 0
    if exr_path.exists():
        exr_size_kb = exr_path.stat().st_size // 1024

    return render_time, exr_size_kb


# --- MAIN ---
print("=" * 70)
print("8-FRAME TEST MATRIX")
print("=" * 70)
print(f"  Environments: {ENVIRONMENTS}")
print(f"  HDRI presets: {HDRI_PRESETS}")
print(f"  Camera elevations: {list(CAMERA_CONFIGS.keys())}")
print(f"  Output: {OUTPUT_DIR}")
print()

results = []

for env_name in ENVIRONMENTS:
    for hdri_name in HDRI_PRESETS:
        for cam_name, cam_cfg in CAMERA_CONFIGS.items():
            frame_name = f"{env_name}_{hdri_name}_{cam_name}"
            print(f"\n--- {frame_name} ---")

            scene = build_scene(env_name, hdri_name, cam_cfg)
            cam_obj = [o for o in bpy.data.objects if o.type == "CAMERA"][-1]

            render_time, exr_size_kb = render_frame(scene, frame_name, cam_obj)

            print(f"  Time: {render_time:.2f}s")
            print(f"  EXR size: {exr_size_kb} KB")

            results.append({
                "name": frame_name,
                "env": env_name,
                "hdri": hdri_name,
                "cam": cam_cfg["location"],
                "time": render_time,
                "exr_size_kb": exr_size_kb,
            })

# Summary table
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Frame':<30} {'Time':>6} {'EXR KB':>8} {'Status':>8}")
print("-" * 60)
for r in results:
    status = "✓" if r["exr_size_kb"] > 1000 else "✗"
    print(f"{r['name']:<30} {r['time']:>5.2f}s {r['exr_size_kb']:>7} {status:>8}")

print(f"\nTotal frames: {len(results)}")
print(f"Total render time: {sum(r['time'] for r in results):.1f}s")
print(f"All EXR valid: {all(r['exr_size_kb'] > 1000 for r in results)}")
