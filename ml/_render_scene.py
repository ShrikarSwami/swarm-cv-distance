"""T2 — Blender-side scene renderer (owner: harness agent).

Renders the 24 angles of one ML scene to PNGs, replicating the T0 plain-background
setup exactly (dark world, white emissive drone cubes, EEVEE, Standard view
transform, no post-processing) so PNG bytes stay on the calibrated 33 KB scale
and the frozen blob detector sees the same bright-blob-on-dark-blob it was
validated against.

Run headless (repo root; outdir receives angle_00.png ... angle_23.png)::

    "/Applications/Blender.app/Contents/MacOS/blender" \\
        --background --python ml/_render_scene.py -- \\
        --scene-dir DIR --outdir OUTDIR [--drone-size 0.5]

Reads `DIR/cameras.json` (views with c2w + K) and `DIR/ground_truth.json`
(positions). Emits one `RENDER_JSON <json>` line per angle plus a final
`RENDER_JSON {"mode": "done", ...}` line to stdout. Exit 0 on success.

This script is deliberately standalone: it imports only stdlib + bpy, so it
runs in Blender's bundled interpreter without needing the repo venv.
"""

import json
import os
import sys
import time

import bpy  # Blender API (not importable outside Blender)

SENSOR_WIDTH_MM = 36.0
BACKGROUND_RGB = (0.03, 0.03, 0.03)
PNG_COLOR = (1.0, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# arg / output helpers
# ---------------------------------------------------------------------------


def parse_args(argv):
    args = {"scene_dir": None, "outdir": None, "drone_size": 0.5}
    it = iter(argv)
    for tok in it:
        if tok == "--scene-dir":
            args["scene_dir"] = next(it)
        elif tok == "--outdir":
            args["outdir"] = next(it)
        elif tok == "--drone-size":
            args["drone_size"] = float(next(it))
    return args


def emit(payload):
    print("RENDER_JSON " + json.dumps(payload), flush=True)


def load_scene(scene_dir):
    with open(os.path.join(scene_dir, "cameras.json")) as f:
        cameras = json.load(f)
    with open(os.path.join(scene_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    return cameras, gt


# ---------------------------------------------------------------------------
# scene construction (mirrors tools/_calibrate_render.py)
# ---------------------------------------------------------------------------


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def build_world():
    world = bpy.data.worlds.new("MLWorld")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = (*BACKGROUND_RGB, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    return world


def build_emissive_material():
    mat = bpy.data.materials.new("DroneEmission")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    em.inputs["Color"].default_value = PNG_COLOR
    em.inputs["Strength"].default_value = 1.0
    return mat


def add_cube(location, size, material):
    bpy.ops.mesh.primitive_cube_add(size=size, location=tuple(location))
    obj = bpy.context.object
    obj.data.materials.append(material)
    return obj


def setup_scene(image_w, image_h):
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(image_w)
    scene.render.resolution_y = int(image_h)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = False
    scene.render.dither_intensity = 0.0
    scene.use_nodes = False
    if hasattr(scene, "view_settings"):
        scene.view_settings.view_transform = "Standard"
    eevee = scene.eevee
    for attr in ("use_bloom", "use_gtao", "use_volumetric", "use_ssr",
                 "use_motion_blur", "use_shadows"):
        if hasattr(eevee, attr):
            try:
                setattr(eevee, attr, False)
            except Exception:
                pass
    scene.world = build_world()
    return scene


def setup_camera(scene, c2w, fx, image_w):
    cam_data = bpy.data.cameras.new("MLCam")
    cam_obj = bpy.data.objects.new("MLCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.lens = fx * SENSOR_WIDTH_MM / image_w  # focal_px -> lens mm
    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000.0
    cam_obj.matrix_world = _to_matrix(c2w)
    return cam_obj


def _to_matrix(c2w):
    import mathutils

    return mathutils.Matrix(c2w)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parse_args(argv)
    if not args["scene_dir"] or not args["outdir"]:
        sys.stderr.write("usage: --scene-dir DIR --outdir OUTDIR [--drone-size S]\n")
        sys.exit(2)

    cameras, gt = load_scene(args["scene_dir"])
    views = cameras["views"]
    outdir = args["outdir"]
    os.makedirs(outdir, exist_ok=True)
    image_w, image_h = cameras["image_size_px"]
    drone_size = args["drone_size"]

    scene = setup_scene(image_w, image_h)
    material = build_emissive_material()
    for pos in gt["positions"]:
        add_cube(pos, drone_size, material)

    total_bytes = 0
    start = time.perf_counter()
    for view in views:
        idx = view["angle_idx"]
        setup_camera(scene, view["c2w"], view["K"][0][0], image_w)
        path = os.path.join(outdir, "angle_%02d.png" % idx)
        scene.render.filepath = path
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        dt = time.perf_counter() - t0
        size = os.path.getsize(path) if os.path.exists(path) else 0
        total_bytes += size
        emit({"mode": "angle", "angle": idx, "tier": view.get("tier"),
              "seconds": round(dt, 6), "png_bytes": size, "png": path})

    wall = time.perf_counter() - start
    emit({"mode": "done", "angles": len(views), "png_bytes_total": total_bytes,
          "render_wall_s": round(wall, 4), "outdir": outdir})
    return 0


if __name__ == "__main__":
    sys.exit(main())
