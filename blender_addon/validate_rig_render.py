"""
M3 rig-coverage validation, phase 1 of 2 (render side).

Drives the addon's own operators end-to-end (register -> generate swarm ->
place rig), then renders a real object-index pass per rig camera. Coverage
is judged from these renders by validate_rig_report.py (project venv, needs
OpenEXR) -- NOT from frustum math, per the project lesson that idealized
math already missed a real problem once (clip_end).

Renders 3 random-mode rolls (placement is randomized per click, so one roll
proves nothing) plus 1 manual-mode starting layout.

Run:
    "<blender binary>" --background --python blender_addon/validate_rig_render.py
then:
    venv/bin/python blender_addon/validate_rig_report.py
"""

import os
import shutil
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swarm_scanner

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rig_validation")
N_DRONES = 20
N_CAMERAS = 6
# Display scale inflates drone meshes (10 m at 20x) purely so the tiny
# targets rasterize reliably in the ID pass; rig GEOMETRY (frustum fit,
# per-camera visibility, overlap) is what's being validated, and that
# doesn't depend on mesh size. True-scale (0.5 m) detectability is a
# separate, known-open M4 concern -- see PROGRESS.md.
DISPLAY_SCALE = 20.0

shutil.rmtree(OUT_ROOT, ignore_errors=True)
os.makedirs(OUT_ROOT)

swarm_scanner.register()
scene = bpy.context.scene
scene.swarm_scan.drone_count = N_DRONES
scene.swarm_scan.seed = 1
scene.swarm_scan.display_scale = DISPLAY_SCALE
scene.swarm_scan.camera_count = N_CAMERAS
assert bpy.ops.swarm.generate_swarm() == {"FINISHED"}

scene.render.engine = "CYCLES"
scene.cycles.samples = 8
scene.view_layers[0].use_pass_object_index = True

node_group = bpy.data.node_groups.new("RigValidation", "CompositorNodeTree")
scene.compositing_node_group = node_group


def render_rig(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rig = bpy.data.collections[swarm_scanner.RIG_COLLECTION]
    cams = sorted((o for o in rig.objects if o.type == "CAMERA"), key=lambda o: o.name)
    for i, cam_obj in enumerate(cams):
        scene.camera = cam_obj
        scene.render.filepath = os.path.join(out_dir, f"cam{i}_rgb.png")
        node_group.nodes.clear()
        render_layers = node_group.nodes.new("CompositorNodeRLayers")
        id_out = node_group.nodes.new("CompositorNodeOutputFile")
        id_out.directory = out_dir + os.sep
        id_out.file_name = f"cam{i}_id_"
        id_out.file_output_items.clear()
        id_out.file_output_items.new("FLOAT", "id_")
        id_out.format.file_format = "OPEN_EXR_MULTILAYER"
        id_out.format.color_depth = "32"
        node_group.links.new(render_layers.outputs["Object Index"], id_out.inputs["id_"])
        bpy.ops.render.render(write_still=True)


for roll in range(3):
    scene.swarm_scan.camera_mode = "RANDOM"
    assert bpy.ops.swarm.place_cameras() == {"FINISHED"}
    render_rig(os.path.join(OUT_ROOT, f"random_{roll}"))
    print(f"rendered random roll {roll}")

scene.swarm_scan.camera_mode = "MANUAL"
assert bpy.ops.swarm.place_cameras() == {"FINISHED"}
render_rig(os.path.join(OUT_ROOT, "manual"))
print("rendered manual layout")
print(f"RIG RENDERS DONE -> {os.path.normpath(OUT_ROOT)}")
