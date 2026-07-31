"""T0 Blender render helper (owned by the calibration agent).

Builds the EEVEE scenes defined in spec Section 8 T0 and emits one
``CALIB_JSON <json>`` line per rendered frame to stdout.

Run headless (Blender 5.2.0 LTS, engine string 'BLENDER_EEVEE')::

    "/Applications/Blender.app/Contents/MacOS/blender" \\
        --background --python tools/_calibrate_render.py -- --mode timing --outdir DIR
    "/Applications/Blender.app/Contents/MacOS/blender" \\
        --background --python tools/_calibrate_render.py -- --mode empirical --outdir DIR
    "/Applications/Blender.app/Contents/MacOS/blender" \\
        --background --python tools/_calibrate_render.py -- --mode smoke --outdir DIR

Modes
-----
timing    - 10-drone swarm at ~100 m, camera at origin with 90 deg HFOV
            (focal_px = W/2, lens = 18.0 mm). One unmeasured warm-up frame at
            the smallest resolution, then REPS measured frames at each of the
            candidate resolutions {1280, 1920, 2560, 3840} (H = W*9/16).
            Times ONLY the bpy.ops.render.render() call with perf_counter.
empirical - exactly ONE 0.5 m cube at world (0, 0, 100), camera at origin,
            focal_px = 960 at W=1920 (the P7 sanity cell).
smoke     - single 1280x720 frame of the timing scene (fast pipeline check).
"""

import json
import os
import sys
import time

import bpy  # Blender API (not importable outside Blender)

WIDTHS = [1280, 1920, 2560, 3840]
REPS = 3
DRONE_SIZE_M = 0.5
SENSOR_WIDTH_MM = 36.0
BACKGROUND_RGB = (0.03, 0.03, 0.03)
WORLD_TARGET = (0.0, 0.0, 100.0)  # swarm centre / empirical drone position


# ---------------------------------------------------------------------------
# arg / output helpers
# ---------------------------------------------------------------------------


def parse_args(argv):
    args = {"mode": None, "outdir": None}
    it = iter(argv)
    for tok in it:
        if tok == "--mode":
            args["mode"] = next(it)
        elif tok == "--outdir":
            args["outdir"] = next(it)
    return args


def emit(payload):
    print("CALIB_JSON " + json.dumps(payload), flush=True)


def render(scene, outdir, name):
    """Render one still, timing only the render call. Returns (seconds, bytes, path)."""
    path = os.path.join(outdir, name + ".png")
    scene.render.filepath = path
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    dt = time.perf_counter() - t0
    if not os.path.exists(path):  # robustness: Blender may have altered the path
        import glob

        matches = sorted(glob.glob(os.path.join(outdir, name) + "*"))
        path = matches[-1] if matches else path
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return dt, size, path


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def build_world():
    world = bpy.data.worlds.new("CalibWorld")
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
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs["Strength"].default_value = 1.0
    return mat


def add_cube(location, material):
    bpy.ops.mesh.primitive_cube_add(size=DRONE_SIZE_M, location=location)
    obj = bpy.context.object
    obj.data.materials.append(material)
    return obj


def setup_scene():
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = False
    scene.render.dither_intensity = 0.0
    scene.use_nodes = False
    # Deterministic linear-ish output: disable filmic view transform + EEVEE
    # post-processing so a tiny emissive blob is a hard bright region (no bloom
    # halo inflating the P7 pixel-extent measurement).
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


def setup_camera(scene, target):
    cam_data = bpy.data.cameras.new("CalibCam")
    cam_obj = bpy.data.objects.new("CalibCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    empty = bpy.data.objects.new("CalibTarget", None)
    scene.collection.objects.link(empty)
    empty.location = target
    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return cam_obj, cam_data


def set_focal_px(scene, cam_data, width):
    """focal_px = W/2 (90 deg HFOV); lens_mm = focal_px * 36.0 / W = 18.0 for all W."""
    focal_px = width / 2.0
    cam_data.lens = focal_px * SENSOR_WIDTH_MM / width
    return focal_px


def build_timing_scene(scene):
    import numpy as np

    rng = np.random.default_rng(0)
    material = build_emissive_material()
    drones = []
    for _ in range(10):
        x = float(rng.uniform(-30.0, 30.0))
        y = float(rng.uniform(-30.0, 30.0))
        z = float(rng.uniform(70.0, 130.0))
        drones.append(add_cube((x, y, z), material))
    return drones


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def run_smoke(args):
    scene = setup_scene()
    cam_obj, cam_data = setup_camera(scene, WORLD_TARGET)
    build_timing_scene(scene)
    w, h = 1280, 720
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    set_focal_px(scene, cam_data, w)
    dt, size, path = render(scene, args["outdir"], "smoke")
    emit({"mode": "smoke", "width": w, "height": h,
          "seconds": round(dt, 6), "png_bytes": size, "png_path": path})


def run_timing(args):
    outdir = args["outdir"]
    scene = setup_scene()
    cam_obj, cam_data = setup_camera(scene, WORLD_TARGET)
    build_timing_scene(scene)

    # Unmeasured warm-up at the smallest resolution (compiles EEVEE shaders).
    w0, h0 = WIDTHS[0], int(WIDTHS[0] * 9 / 16)
    scene.render.resolution_x = w0
    scene.render.resolution_y = h0
    set_focal_px(scene, cam_data, w0)
    dt, size, path = render(scene, outdir, "warmup")
    emit({"mode": "warmup", "width": w0, "height": h0,
          "seconds": round(dt, 6), "png_bytes": size, "png_path": path})

    # Measured reps.
    for w in WIDTHS:
        h = int(w * 9 / 16)
        scene.render.resolution_x = w
        scene.render.resolution_y = h
        set_focal_px(scene, cam_data, w)
        for rep in range(REPS):
            dt, size, path = render(scene, outdir, "timing_w%d_r%d" % (w, rep))
            emit({"mode": "timing", "width": w, "height": h, "rep": rep,
                  "seconds": round(dt, 6), "png_bytes": size, "png_path": path})


def run_empirical(args):
    outdir = args["outdir"]
    scene = setup_scene()
    cam_obj, cam_data = setup_camera(scene, WORLD_TARGET)
    material = build_emissive_material()
    add_cube(WORLD_TARGET, material)
    w, h = 1920, 1080
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    set_focal_px(scene, cam_data, w)
    dt, size, path = render(scene, outdir, "empirical")
    emit({"mode": "empirical", "width": w, "height": h,
          "seconds": round(dt, 6), "png_bytes": size, "png_path": path})


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parse_args(argv)
    mode = args["mode"]
    if mode == "smoke":
        run_smoke(args)
    elif mode == "timing":
        run_timing(args)
    elif mode == "empirical":
        run_empirical(args)
    else:
        sys.stderr.write("Unknown mode: %r\n" % mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
