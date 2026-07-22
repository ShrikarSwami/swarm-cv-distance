"""
Stage 2: build the swarm scene in Blender and render it from the 6-camera
dome rig (see scene_config.py -- switched from a flat ring after the ring's
~2.4deg grazing elevation, combined with an unrelated clip_end bug, made
coverage look far worse than Stage 1 predicted).

Camera intrinsics/positions come directly from mvt.place_dome_of_cameras()
(reusing mvt.Camera, the SAME class Stage 1 uses) so the rig here is
guaranteed to match -- see stage2_render/validate_camera_alignment.py, which
confirmed this position/lens/track-to setup reproduces Stage 1's
Camera.project() to <1px error.

Correspondence shortcut (per project decision, NOT real-world deployable):
each drone's mesh parts are tagged with a unique Blender pass_index, and the
compositor writes an object-index pass to EXR (alongside the RGB render).
Downstream code (stage2_detect) reads the EXR at each YOLO box center to
recover which ground-truth drone a detection belongs to, instead of solving
real cross-view correspondence (epipolar + appearance matching, unsolved).

Run via Blender's own Python, not the project venv:
    "<blender binary>" --background --python stage2_render/render_scene.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_geometry"))

import bpy
import mathutils
import numpy as np

import multiview_triangulation_test as mvt
from scene_config import (
    N_DRONES, AREA_KM, ALTITUDE_SPREAD_M, SWARM_SEED,
    N_CAMERAS, DOME_SLANT_RANGE_M, DOME_ELEV_MIN_DEG, DOME_ELEV_MAX_DEG, LOOK_AT,
    IMAGE_SIZE, FOCAL_PX,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Ground truth: same functions as Stage 1, unchanged ---
drones = mvt.make_swarm(n_drones=N_DRONES, area_km=AREA_KM,
                         altitude_spread_m=ALTITUDE_SPREAD_M, seed=SWARM_SEED)
np.save(os.path.join(OUT_DIR, "ground_truth_positions.npy"), drones)

cameras = mvt.place_dome_of_cameras(N_CAMERAS, DOME_SLANT_RANGE_M, DOME_ELEV_MIN_DEG,
                                     DOME_ELEV_MAX_DEG, look_at=LOOK_AT)
cam_positions = np.array([c.position for c in cameras])
np.save(os.path.join(OUT_DIR, "camera_positions.npy"), cam_positions)

# --- Scene setup ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

scene.render.resolution_x = IMAGE_SIZE[0]
scene.render.resolution_y = IMAGE_SIZE[1]
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = 1.0
scene.render.pixel_aspect_y = 1.0

# Object-index pass (needed for the correspondence shortcut) is only
# available under Cycles in Blender 5.x -- EEVEE doesn't expose it.
# Sample count is kept low since the scene has no realism requirement,
# just needs to render cleanly enough for consistent detection.
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
print(f"Using render engine: {scene.render.engine}")

world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.65, 0.85, 1.0)
scene.world = world

sun_data = bpy.data.lights.new("Sun", type="SUN")
sun_data.energy = 3.5
sun_obj = bpy.data.objects.new("Sun", sun_data)
scene.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (0.9, 0.3, 0.0)

drone_mat = bpy.data.materials.new("DroneBody")
drone_mat.use_nodes = True
bsdf = drone_mat.node_tree.nodes.get("Principled BSDF")
if bsdf is not None:
    bsdf.inputs["Base Color"].default_value = (0.9, 0.1, 0.05, 1.0)


def make_drone(index, position):
    """Simple, non-realistic quadcopter silhouette: body + 4 arms + 4 rotors.
    Shape (not just a bare cube) is meant to give a zero-shot detector a
    recognizable "small aerial vehicle" silhouette -- see stage2_detect notes
    on why a stock COCO-class model can't be used here unmodified.
    """
    body_size = 1.5
    arm_len = 2.5

    bpy.ops.mesh.primitive_cube_add(size=body_size, location=position)
    body = bpy.context.active_object
    body.name = f"drone_{index:02d}_body"
    parts = [body]

    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        arm_pos = (position[0] + dx * arm_len * 0.5, position[1] + dy * arm_len * 0.5, position[2])
        bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=arm_len, location=arm_pos)
        arm = bpy.context.active_object
        arm.rotation_euler = (0, np.pi / 2, np.pi / 4 if dx * dy > 0 else -np.pi / 4)
        parts.append(arm)

        rotor_pos = (position[0] + dx * arm_len, position[1] + dy * arm_len, position[2])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=rotor_pos)
        parts.append(bpy.context.active_object)

    for p in parts:
        p.data.materials.append(drone_mat)
        p.pass_index = index + 1  # 1-indexed; 0 is reserved for background

    return parts


for i, pos in enumerate(drones):
    make_drone(i, tuple(pos))

# --- Blender camera objects, positioned/aimed to match the mvt.Camera rig ---
bl_cameras = []
for i, cam in enumerate(cameras):
    bpy.ops.object.camera_add(location=tuple(cam.position))
    cam_obj = bpy.context.active_object
    cam_obj.name = f"cam_{i}"
    direction = mathutils.Vector(LOOK_AT) - mathutils.Vector(cam.position)
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    cam_data = cam_obj.data
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = FOCAL_PX * cam_data.sensor_width / IMAGE_SIZE[0]
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    # Blender's default clip_end is 1000m -- far shorter than camera-to-drone
    # distances in this scene (up to ~2-3km). Left at default, it silently
    # culls most of the swarm and looks exactly like occlusion. Found this
    # 2026-07-22 after the dome-vs-ring comparison gave suspicious identical
    # results regardless of object size -- see scene_config.py note.
    cam_data.clip_end = 10000.0
    bl_cameras.append(cam_obj)

# --- RGB via standard render output; object-index pass via compositor ---
# (Blender 5.x's File Output node forces OPEN_EXR_MULTILAYER for non-color
# FLOAT sockets, which is why the index pass is a separate EXR rather than
# bundled into the PNG -- verified against a minimal test scene beforehand.)
scene.view_layers[0].use_pass_object_index = True
scene.render.image_settings.file_format = "PNG"

node_group = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
scene.compositing_node_group = node_group

for i, cam_obj in enumerate(bl_cameras):
    scene.camera = cam_obj
    scene.render.filepath = os.path.join(OUT_DIR, f"cam{i}_rgb.png")

    node_group.nodes.clear()
    render_layers = node_group.nodes.new("CompositorNodeRLayers")
    id_out = node_group.nodes.new("CompositorNodeOutputFile")
    id_out.directory = OUT_DIR + os.sep
    id_out.file_name = f"cam{i}_id_"
    id_out.file_output_items.clear()
    id_item = id_out.file_output_items.new("FLOAT", "id_")
    id_out.format.file_format = "OPEN_EXR_MULTILAYER"
    id_out.format.color_depth = "32"
    node_group.links.new(render_layers.outputs["Object Index"], id_out.inputs["id_"])

    bpy.ops.render.render(write_still=True)
    print(f"Rendered camera {i}")

print(f"Rendered {len(bl_cameras)} camera views to {OUT_DIR}")
print(f"Ground truth: {N_DRONES} drones, saved to ground_truth_positions.npy")
