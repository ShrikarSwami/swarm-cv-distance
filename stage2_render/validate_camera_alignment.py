"""
One-off validation: confirm a Blender camera set up with our position/look_at/
focal_px convention actually projects a known 3D point to the same pixel
location that Stage 1's Camera.project() predicts. Run this ONCE before
trusting the full render pipeline -- if this doesn't line up, nothing built
on top of it (YOLO detections fed into triangulate_point) can be trusted
either, since the "ground truth" pixel math and the actual render would be
silently misaligned.

Run via Blender's own Python (not the project venv):
    "<blender binary>" --background --python validate_camera_alignment.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_geometry"))

import bpy
import mathutils
import numpy as np

import multiview_triangulation_test as mvt
from scene_config import IMAGE_SIZE, FOCAL_PX

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CAM_POS = (1200.0, 0.0, 150.0)
LOOK_AT = (0.0, 0.0, 100.0)
DRONE_POS = (300.0, 200.0, 100.0)

# --- Expected pixel location, per Stage 1's math ---
cam = mvt.Camera(CAM_POS, LOOK_AT, image_size=IMAGE_SIZE, focal_px=FOCAL_PX)
expected_px = cam.project(np.array(DRONE_POS))
print(f"EXPECTED pixel (Stage 1 Camera.project): {expected_px}")

# --- Build matching Blender scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

bpy.ops.mesh.primitive_uv_sphere_add(radius=15.0, location=DRONE_POS)
drone_obj = bpy.context.active_object
mat = bpy.data.materials.new("DroneMat")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1, 1, 1, 1)
mat.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 5.0
drone_obj.data.materials.append(mat)

bpy.ops.object.camera_add(location=CAM_POS)
cam_obj = bpy.context.active_object
direction = mathutils.Vector(LOOK_AT) - mathutils.Vector(CAM_POS)
cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
scene.camera = cam_obj

cam_data = cam_obj.data
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.lens = FOCAL_PX * cam_data.sensor_width / IMAGE_SIZE[0]
cam_data.shift_x = 0.0
cam_data.shift_y = 0.0

scene.render.resolution_x = IMAGE_SIZE[0]
scene.render.resolution_y = IMAGE_SIZE[1]
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = 1.0
scene.render.pixel_aspect_y = 1.0
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(OUT_DIR, "validate_alignment.png")

for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        continue
print(f"Using render engine: {scene.render.engine}")

world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
scene.world = world

bpy.ops.render.render(write_still=True)

# --- Find where the bright sphere actually landed in the rendered image ---
# Read back via bpy.data.images (no PIL in Blender's bundled Python).
# Blender's image.pixels is bottom-up (row 0 = bottom); flip to match the
# top-left-origin, row-0-is-top convention Stage 1's Camera class uses.
render_img = bpy.data.images.load(scene.render.filepath)
w, h = render_img.size
pixels = np.array(render_img.pixels[:]).reshape(h, w, 4)
pixels = np.flipud(pixels)  # now row 0 = top, matching image/array convention
gray = pixels[:, :, :3].mean(axis=2)

ys, xs = np.where(gray > 0.15)
if len(xs) == 0:
    print("RESULT: FAIL -- nothing bright found in render, check lighting/material")
else:
    actual_px = (xs.mean(), ys.mean())
    err = np.hypot(actual_px[0] - expected_px[0], actual_px[1] - expected_px[1])
    print(f"ACTUAL pixel (centroid of bright blob in render): {actual_px}")
    print(f"Pixel error: {err:.1f}px")
    print("RESULT: PASS" if err < 15 else "RESULT: FAIL -- camera math misaligned")
