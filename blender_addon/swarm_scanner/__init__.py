"""
Swarm Scan: interactive Blender addon for the swarm-cv-distance project.

Scope pivot 2026-07-22 (see PROGRESS.md): the remaining project work is this
addon -- a live UI panel driving a navigable 3D viewport -- not more batch
render-and-analyze scripts. Milestones:

  1. Swarm generator + live viewport (THIS FILE's current state)
  2. Lightweight boids-style flight sim
  3. Camera rig UI (random dome placement or manual place+aim) -- the pending
     D_MAX decision (candidates 3688/3949/4358m) lands here, not before
  4. "Scan" mode: run the existing Stage 1 triangulation pipeline against
     live swarm + camera state, visualize the distance map as an overlay

Shared constants and the swarm distribution come from stage1_geometry/
(scene_config.py, multiview_triangulation_test.py) -- single source of
truth, imported by repo-relative path. This means the addon must be loaded
from its in-repo location (see blender_addon/dev_load.py); Blender's
"Install..." button copies the file elsewhere and breaks that import, which
the panel will report rather than silently duplicating the constants.
"""

bl_info = {
    "name": "Swarm Scan",
    "author": "swarm-cv-distance project",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Swarm Scan",
    "description": "Generate and inspect a drone swarm; later: flight sim, camera rig, triangulation scan",
    "category": "Object",
}

import json
import os
import shutil
import subprocess
import sys

import blf
import bpy
import gpu
import numpy as np
from bpy.props import EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from gpu_extras.batch import batch_for_shader

# --- Repo-relative import of Stage 1's shared config + swarm distribution ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STAGE1_DIR = os.path.join(_REPO_ROOT, "stage1_geometry")
if _STAGE1_DIR not in sys.path:
    sys.path.insert(0, _STAGE1_DIR)

try:
    import multiview_triangulation_test as mvt
    import scene_config
    _IMPORT_ERROR = None
except ImportError as exc:  # loaded outside the repo (e.g. via Install...)
    mvt = None
    scene_config = None
    _IMPORT_ERROR = str(exc)

SWARM_COLLECTION = "Swarm Drones"
DRONE_MESH_NAME = "swarm_drone_mesh"
DRONE_MATERIAL_NAME = "SwarmDroneBody"

# Footprint of the drone asset as originally modeled in render_scene.py
# (rotor centers at +/-2.5m plus 0.35m rotor radius); used to rescale the
# same proportions down to DRONE_SIZE_M.
_ASSET_NATIVE_FOOTPRINT_M = 5.7


class SwarmScanSettings(PropertyGroup):
    drone_count: IntProperty(
        name="Drones",
        description="Number of drones to generate",
        default=20, min=2, max=500,
    )
    # --- M2 flight-sim tunables (read live every tick, so sliders take
    # effect mid-flight) ---
    neighbor_radius: FloatProperty(
        name="Neighbor Radius",
        description=(
            "Cohesion neighborhood in meters: each drone steers toward the "
            "average position of drones within this radius (local flocking, "
            "not whole-swarm centroid). Separation distance scales with it "
            "(25%), so this slider sets the overall flocking scale"
        ),
        default=600.0, min=50.0, max=2500.0,
    )
    bound_softness: FloatProperty(
        name="Bound Softness",
        description=(
            "Width in meters of the soft leash at the edges of the "
            "5km x 5km x 1km volume: turn-back steering ramps up across this "
            "distance instead of hard-clamping at the boundary"
        ),
        default=400.0, min=50.0, max=2000.0,
    )
    wander_strength: FloatProperty(
        name="Wander",
        description="Random steering perturbation (m/s^2) so motion doesn't look mechanical",
        default=2.0, min=0.0, max=20.0,
    )
    sim_speed: FloatProperty(
        name="Speed",
        description="Maximum drone speed in m/s",
        default=25.0, min=1.0, max=300.0,
    )
    # --- M3 camera rig ---
    camera_count: IntProperty(
        name="Cameras",
        description="Number of observing cameras in the rig",
        default=6, min=2, max=12,
    )
    camera_mode: EnumProperty(
        name="Placement",
        description="How rig cameras are positioned",
        items=[
            ("RANDOM", "Random",
             "Randomized dome placement around the current swarm's bounding "
             "volume (jittered azimuths, random elevations); each click "
             "re-rolls a new arrangement"),
            ("MANUAL", "Manual",
             "Evenly spaced dome as a starting layout; move/rotate cameras "
             "by hand. Each stays auto-aimed at the swarm via a track-to "
             "constraint until you release it (Toggle Auto-Aim)"),
        ],
        default="RANDOM",
    )
    formation: EnumProperty(
        name="Formation",
        description="Swarm formation preset",
        items=[
            ("RANDOM_CLOUD", "Random Cloud",
             "Uniform 3D distribution across the full scene volume "
             "(same distribution as Stage 1's make_swarm)"),
            # Light-show-style presets come later, per the milestone plan.
        ],
        default="RANDOM_CLOUD",
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed for the formation (same seed = same swarm)",
        default=1, min=0,
    )
    display_scale: FloatProperty(
        name="Display Scale",
        description=(
            "Viewport-only size multiplier for the drone meshes. At true scale "
            "(1.0) a 0.5 m drone is invisible across a 5 km scene. Drone "
            "POSITIONS are always true-scale; only mesh size is exaggerated, "
            "so later scan/triangulation milestones are unaffected"
        ),
        default=20.0, min=1.0, max=500.0,
    )
    # --- M4 scan ---
    scan_seed: IntProperty(
        name="Scan Noise Seed",
        description=(
            "Random seed for the synthetic pixel-noise layered on top of real "
            "ID-pass centroids (same seed = same noise draw). Detection "
            "availability itself (which camera sees which drone) comes from "
            "the real render, not this seed"
        ),
        default=1, min=0,
    )


def _get_or_make_material():
    mat = bpy.data.materials.get(DRONE_MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(DRONE_MATERIAL_NAME)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.9, 0.1, 0.05, 1.0)
        # Solid-shading viewport color, so drones read as red without
        # switching to rendered shading.
        mat.diffuse_color = (0.9, 0.1, 0.05, 1.0)
    return mat


def _build_drone_mesh(footprint_m):
    """Build one quadcopter mesh (body + 4 arms + 4 rotors) at the given
    footprint and return the mesh datablock. Same asset proportions as
    stage2_render/render_scene.py's make_drone, joined into a single mesh so
    N drones are N objects sharing one mesh (cheap to instance and navigate).
    """
    s = footprint_m / _ASSET_NATIVE_FOOTPRINT_M
    body_size = 1.5 * s
    arm_len = 2.5 * s
    rotor_r = 0.35 * s

    parts = []
    bpy.ops.mesh.primitive_cube_add(size=body_size, location=(0, 0, 0))
    body = bpy.context.active_object
    parts.append(body)

    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.15 * s, depth=arm_len,
            location=(dx * arm_len * 0.5, dy * arm_len * 0.5, 0),
        )
        arm = bpy.context.active_object
        arm.rotation_euler = (0, np.pi / 2, np.pi / 4 if dx * dy > 0 else -np.pi / 4)
        parts.append(arm)

        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=rotor_r, location=(dx * arm_len, dy * arm_len, 0),
        )
        parts.append(bpy.context.active_object)

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()

    joined = bpy.context.active_object
    mesh = joined.data
    mesh.name = DRONE_MESH_NAME
    mesh.materials.clear()
    mesh.materials.append(_get_or_make_material())

    # Keep only the mesh datablock; the template object goes away and
    # per-drone objects are created around the shared mesh instead.
    bpy.data.objects.remove(joined, do_unlink=True)
    return mesh


def _clear_swarm():
    coll = bpy.data.collections.get(SWARM_COLLECTION)
    if coll is not None:
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll)
    old_mesh = bpy.data.meshes.get(DRONE_MESH_NAME)
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def _extend_viewport_clip():
    """Blender's default viewport clip_end is 1000 m -- the exact default that
    silently culled the 5 km scene during Stage 2 batch rendering (see
    PROGRESS.md lesson). Extend every 3D viewport so the whole swarm is
    actually visible while navigating. Iterates bpy.data.screens rather than
    open windows so it also works in background mode -- that way a .blend
    saved headlessly opens with usable clip distances too.
    """
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.clip_end = max(space.clip_end, 20000.0)


class SWARM_OT_generate(Operator):
    bl_idname = "swarm.generate_swarm"
    bl_label = "Generate Swarm"
    bl_description = "Replace the current swarm with a freshly generated one"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if _IMPORT_ERROR is not None:
            self.report(
                {"ERROR"},
                "Cannot import stage1_geometry (load the addon from its "
                f"in-repo location via dev_load.py): {_IMPORT_ERROR}",
            )
            return {"CANCELLED"}

        settings = context.scene.swarm_scan

        # A running flight sim holds references to the objects being replaced;
        # signal it to stop before tearing them down (it exits cleanly on its
        # next timer tick).
        context.window_manager.swarm_sim_running = False

        # Only RANDOM_CLOUD exists so far; the enum is the extension point
        # for the light-show presets milestone.
        positions = mvt.make_swarm(
            n_drones=settings.drone_count,
            area_km=scene_config.AREA_KM,
            height_range_m=scene_config.HEIGHT_RANGE_M,
            seed=settings.seed,
        )

        _clear_swarm()
        mesh = _build_drone_mesh(scene_config.DRONE_SIZE_M * settings.display_scale)

        coll = bpy.data.collections.new(SWARM_COLLECTION)
        context.scene.collection.children.link(coll)
        for i, pos in enumerate(positions):
            obj = bpy.data.objects.new(f"drone_{i:03d}", mesh)
            obj.location = tuple(pos)
            obj.pass_index = i + 1  # 0 = background; reused by the scan milestone
            coll.objects.link(obj)

        _extend_viewport_clip()
        self.report(
            {"INFO"},
            f"Generated {settings.drone_count} drones across "
            f"{scene_config.AREA_KM:g} km x {scene_config.AREA_KM:g} km x "
            f"{scene_config.HEIGHT_RANGE_M / 1000:g} km",
        )
        return {"FINISHED"}


# --- M2: boids-style flight sim -------------------------------------------
#
# Live viewport behavior ONLY: the sim animates the drone Objects' locations
# through a modal timer; it does not feed back into stage1_geometry's
# make_swarm or the D_MAX calibration, which stay static-snapshot-based.

_SIM_DT = 1.0 / 30.0        # timer tick, seconds
_COHESION_GAIN = 0.008      # 1/s^2 -- accel per meter of offset from local mean
_SEPARATION_GAIN = 6.0      # m/s^2 -- max repulsion accel per crowding neighbor
_BOUND_GAIN = 10.0          # m/s^2 per unit of normalized leash penetration
_SEP_FRACTION = 0.25        # separation distance = this fraction of neighbor radius


def _scene_bounds():
    half = scene_config.AREA_KM * 500.0
    lo = np.array([-half, -half, 0.0])
    hi = np.array([half, half, scene_config.HEIGHT_RANGE_M])
    return lo, hi


def boids_step(pos, vel, dt, neighbor_radius, bound_softness,
               wander_strength, max_speed, rng, bounds):
    """One vectorized flight-sim step. Pure numpy (no bpy) so it can be
    exercised headlessly; the modal operator is just a thin driver around it.
    Returns updated (pos, vel).
    """
    lo, hi = bounds
    n = len(pos)
    not_self = ~np.eye(n, dtype=bool)

    diff = pos[None, :, :] - pos[:, None, :]        # diff[i, j] = pos[j] - pos[i]
    dist = np.maximum(np.linalg.norm(diff, axis=2), 1e-6)

    # Cohesion: steer toward the LOCAL neighborhood mean, not the swarm
    # centroid -- drones with no neighbors in radius get no cohesion pull.
    neighbors = (dist < neighbor_radius) & not_self
    counts = neighbors.sum(axis=1)
    has_nb = counts > 0
    local_mean = np.zeros_like(pos)
    local_mean[has_nb] = (neighbors @ pos)[has_nb] / counts[has_nb, None]
    accel = np.where(has_nb[:, None], (local_mean - pos) * _COHESION_GAIN, 0.0)

    # Separation: push away from anyone inside the crowding distance,
    # weighted up as they get closer, averaged so magnitude stays bounded.
    sep_dist = neighbor_radius * _SEP_FRACTION
    crowding = (dist < sep_dist) & not_self
    n_crowd = np.maximum(crowding.sum(axis=1), 1)
    away = -diff / dist[:, :, None]                 # unit vector j -> i
    weight = np.where(crowding, 1.0 - dist / sep_dist, 0.0)
    accel += _SEPARATION_GAIN * (away * weight[:, :, None]).sum(axis=1) / n_crowd[:, None]

    # Soft leash at the volume edges: steering ramps up across the
    # bound_softness margin and keeps growing past the boundary -- never a
    # hard position clamp.
    m = bound_softness
    excess_lo = np.clip((lo + m) - pos, 0.0, None) / m
    excess_hi = np.clip(pos - (hi - m), 0.0, None) / m
    accel += _BOUND_GAIN * (excess_lo - excess_hi)

    # Wander: small random accel so the motion doesn't look mechanical.
    accel += rng.normal(0.0, wander_strength, size=pos.shape)

    vel = vel + accel * dt
    speed = np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9)
    vel = vel * np.minimum(1.0, max_speed / speed)
    return pos + vel * dt, vel


class SWARM_OT_sim_start(Operator):
    bl_idname = "swarm.sim_start"
    bl_label = "Start Flight Sim"
    bl_description = "Fly the swarm live in the viewport (boids-style); Esc or Stop ends it"

    _timer = None

    def execute(self, context):
        # Reached only when there's no real UI event to invoke with (e.g.
        # background mode, where modal timers never fire anyway) -- decline
        # instead of letting Blender log "Invalid operator call".
        self.report({"WARNING"}, "Flight sim needs an interactive session")
        return {"CANCELLED"}

    def invoke(self, context, event):
        if _IMPORT_ERROR is not None:
            self.report({"ERROR"}, f"stage1_geometry import failed: {_IMPORT_ERROR}")
            return {"CANCELLED"}
        coll = bpy.data.collections.get(SWARM_COLLECTION)
        if coll is None or len(coll.objects) < 2:
            self.report({"ERROR"}, "Generate a swarm first")
            return {"CANCELLED"}

        self._objs = sorted(coll.objects, key=lambda o: o.name)
        self._pos = np.array([obj.location[:] for obj in self._objs])
        self._rng = np.random.default_rng()
        directions = self._rng.normal(size=self._pos.shape)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
        self._vel = directions * (0.5 * context.scene.swarm_scan.sim_speed)

        wm = context.window_manager
        wm.swarm_sim_running = True
        self._timer = wm.event_timer_add(_SIM_DT, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        wm = context.window_manager
        if not wm.swarm_sim_running or event.type == "ESC":
            return self._finish(wm)
        if event.type == "TIMER":
            s = context.scene.swarm_scan
            try:
                self._pos, self._vel = boids_step(
                    self._pos, self._vel, _SIM_DT,
                    s.neighbor_radius, s.bound_softness,
                    s.wander_strength, s.sim_speed,
                    self._rng, _scene_bounds(),
                )
                for obj, p in zip(self._objs, self._pos):
                    obj.location = p
            except ReferenceError:
                # swarm was deleted/regenerated under us -- stop cleanly
                return self._finish(wm)
        # PASS_THROUGH so orbit/pan/zoom stay fully usable mid-flight
        return {"PASS_THROUGH"}

    def _finish(self, wm):
        wm.swarm_sim_running = False
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        return {"FINISHED"}


class SWARM_OT_sim_stop(Operator):
    bl_idname = "swarm.sim_stop"
    bl_label = "Stop Flight Sim"
    bl_description = "Stop the running flight sim (drones stay where they are)"

    def execute(self, context):
        context.window_manager.swarm_sim_running = False
        return {"FINISHED"}


# --- M3: camera rig -------------------------------------------------------

RIG_COLLECTION = "Swarm Cameras"
AIM_EMPTY_NAME = "Swarm Aim"
_RIG_ELEV_MIN_DEG = 20.0   # below ~20 deg the view goes edge-on (M1-era lesson)
_RIG_ELEV_MAX_DEG = 50.0   # above ~50 deg cameras crowd the zenith and lose parallax
_RIG_FIT_MARGIN = 1.15     # slack so the volume isn't wall-to-wall in frame
_RIG_CLIP_END = 50000.0    # camera-to-far-corner can exceed 10km at this scale


def _swarm_snapshot():
    """Current drone positions (whatever the flight sim has done to them),
    or None if there's no swarm."""
    coll = bpy.data.collections.get(SWARM_COLLECTION)
    if coll is None or len(coll.objects) == 0:
        return None
    return np.array([obj.location[:] for obj in sorted(coll.objects, key=lambda o: o.name)])


def derive_dome_positions(swarm_pos, n_cameras, randomize, rng=None):
    """Camera positions on a dome around the swarm's actual bounding volume.

    Standoff is re-derived from scratch for whatever the swarm currently
    occupies (nothing inherited from the old 2km-scene rig): for each
    camera's elevation, the slant range is the minimum distance at which the
    whole volume fits in BOTH the horizontal and vertical FOV (computed from
    scene_config's real intrinsics), times a fit margin.

    Returns (positions (n,3), center (3,)).
    """
    center = 0.5 * (swarm_pos.min(axis=0) + swarm_pos.max(axis=0))
    r_xy = float(np.linalg.norm((swarm_pos - center)[:, :2], axis=1).max())
    z_half = float(np.abs(swarm_pos[:, 2] - center[2]).max())
    r_xy = max(r_xy, 100.0)   # degenerate/tiny swarms still get a sane rig
    z_half = max(z_half, 50.0)

    w, h = scene_config.IMAGE_SIZE
    tan_h = (0.5 * w) / scene_config.FOCAL_PX   # tan of half the horizontal FOV
    tan_v = (0.5 * h) / scene_config.FOCAL_PX

    if randomize:
        rng = rng or np.random.default_rng()
        azimuths = (np.arange(n_cameras) * 2 * np.pi / n_cameras
                    + rng.uniform(0, 2 * np.pi)
                    + rng.uniform(-0.26, 0.26, n_cameras))   # +/-15 deg jitter
        elevations = np.radians(rng.uniform(_RIG_ELEV_MIN_DEG, _RIG_ELEV_MAX_DEG, n_cameras))
    else:
        azimuths = np.arange(n_cameras) * 2 * np.pi / n_cameras
        elevations = np.radians(np.linspace(_RIG_ELEV_MIN_DEG, _RIG_ELEV_MAX_DEG, n_cameras))

    positions = []
    for az, el in zip(azimuths, elevations):
        d_h = r_xy / tan_h
        # Apparent half-height of the volume from elevation el: horizontal
        # depth foreshortened by sin, vertical extent by cos.
        d_v = (r_xy * np.sin(el) + z_half * np.cos(el)) / tan_v
        d = _RIG_FIT_MARGIN * max(d_h, d_v)
        positions.append(center + d * np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ]))
    return np.array(positions), center


def _clear_rig():
    coll = bpy.data.collections.get(RIG_COLLECTION)
    if coll is not None:
        for obj in list(coll.objects):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if isinstance(data, bpy.types.Camera) and data.users == 0:
                bpy.data.cameras.remove(data)
        bpy.data.collections.remove(coll)
    old_aim = bpy.data.objects.get(AIM_EMPTY_NAME)
    if old_aim is not None:
        bpy.data.objects.remove(old_aim, do_unlink=True)


class SWARM_OT_place_cameras(Operator):
    bl_idname = "swarm.place_cameras"
    bl_label = "Place Cameras"
    bl_description = "Replace the camera rig around the current swarm (Random re-rolls each click)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if _IMPORT_ERROR is not None:
            self.report({"ERROR"}, f"stage1_geometry import failed: {_IMPORT_ERROR}")
            return {"CANCELLED"}
        swarm_pos = _swarm_snapshot()
        if swarm_pos is None:
            self.report({"ERROR"}, "Generate a swarm first")
            return {"CANCELLED"}

        settings = context.scene.swarm_scan
        positions, center = derive_dome_positions(
            swarm_pos, settings.camera_count,
            randomize=(settings.camera_mode == "RANDOM"),
        )

        _clear_rig()
        coll = bpy.data.collections.new(RIG_COLLECTION)
        context.scene.collection.children.link(coll)

        aim = bpy.data.objects.new(AIM_EMPTY_NAME, None)
        aim.empty_display_type = "PLAIN_AXES"
        aim.empty_display_size = 150.0
        aim.location = center
        coll.objects.link(aim)

        for i, pos in enumerate(positions):
            cam_data = bpy.data.cameras.new(f"rigcam_{i:02d}")
            cam_data.sensor_fit = "HORIZONTAL"
            cam_data.sensor_width = 36.0
            cam_data.lens = (scene_config.FOCAL_PX * cam_data.sensor_width
                             / scene_config.IMAGE_SIZE[0])
            cam_data.clip_end = _RIG_CLIP_END
            cam_data.display_size = 200.0  # meters; default-size icons vanish at 5km scale

            cam_obj = bpy.data.objects.new(f"rigcam_{i:02d}", cam_data)
            cam_obj.location = pos
            # Auto-aim at the swarm; survives hand-moving the camera. Release
            # per-camera via Toggle Auto-Aim to rotate manually.
            track = cam_obj.constraints.new("TRACK_TO")
            track.target = aim
            track.track_axis = "TRACK_NEGATIVE_Z"
            track.up_axis = "UP_Y"
            coll.objects.link(cam_obj)

        # Keep scene render settings in lockstep with the rig intrinsics so
        # any later render (validation, M4 scan) matches Stage 1's math.
        context.scene.render.resolution_x = scene_config.IMAGE_SIZE[0]
        context.scene.render.resolution_y = scene_config.IMAGE_SIZE[1]
        context.scene.render.resolution_percentage = 100

        _extend_viewport_clip()
        self.report({"INFO"},
                    f"Placed {settings.camera_count} cameras "
                    f"({'randomized' if settings.camera_mode == 'RANDOM' else 'even'} dome)")
        return {"FINISHED"}


class SWARM_OT_toggle_aim(Operator):
    bl_idname = "swarm.toggle_aim"
    bl_label = "Toggle Auto-Aim (Selected)"
    bl_description = (
        "Mute/unmute the track-to constraint on selected rig cameras: muted "
        "cameras can be rotated by hand, unmuted ones re-aim at the swarm"
    )

    def execute(self, context):
        rig = bpy.data.collections.get(RIG_COLLECTION)
        if rig is None:
            self.report({"ERROR"}, "No camera rig")
            return {"CANCELLED"}
        toggled = 0
        for obj in context.selected_objects:
            if obj.name in rig.objects and obj.type == "CAMERA":
                for con in obj.constraints:
                    if con.type == "TRACK_TO":
                        con.mute = not con.mute
                        toggled += 1
        if toggled == 0:
            self.report({"WARNING"}, "Select one or more rig cameras first")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Toggled auto-aim on {toggled} camera(s)")
        return {"FINISHED"}


# --- M4: scan pipeline + viewport overlay ----------------------------------
#
# Detection source is the object-index EXR pass (centroid of each drone's
# ID-pass footprint from a REAL Cycles render, so occlusion/out-of-frame is
# whatever the render actually shows -- not frustum math), with Stage 1's
# synthetic pixel-noise model layered on top of those centroids to stand in
# for real detector localization error (see PROGRESS.md's subpixel finding
# for why an actual YOLO run isn't viable at this scene scale). Triangulation
# reuses stage1_geometry's triangulate_point()/reconstruct_swarm() unchanged.
#
# The EXR readback runs in a subprocess against the project's venv, not
# Blender's own Python: bpy.data.images.load() was tried first and could not
# read this project's custom-named "id_" multilayer pass (loads as a 0x0
# TARGA-typed image) -- OpenEXR direct reads, already validated for this
# exact pass by validate_rig_report.py, are the reliable path.

_SCAN_WORKER_PATH = os.path.join(_REPO_ROOT, "blender_addon", "scan_worker.py")
_SCAN_VENV_PYTHON = os.path.join(_REPO_ROOT, "venv", "bin", "python")
_SCAN_TMP_DIR = os.path.join(_REPO_ROOT, "logs", "scan_tmp")
_SCAN_SAMPLES = 8  # Cycles samples for ID-pass renders; matches M3's validated rig-coverage renders

_SCAN_EDGE_COLOR_CORRECT = (0.15, 0.85, 0.2, 0.9)
_SCAN_EDGE_COLOR_WRONG = (0.9, 0.15, 0.15, 0.9)

# Populated by SWARM_OT_scan, read by the draw handlers and the panel. Plain
# module state (not a bpy property) -- the result is a nested structure
# (per-drone positions, an edge list) that doesn't map cleanly onto Blender's
# ID-property system and is only ever produced/consumed within this session.
_LAST_SCAN = None
_draw_handle_3d = None
_draw_handle_2d = None


def _camera_pose_P(cam_obj, image_size, focal_px):
    """Projection matrix from the camera's ACTUAL rendered world transform,
    not a re-derived look-at target -- correct even for hand-rotated
    manual-mode cameras with auto-aim muted. Mirrors mvt.Camera's
    convention (z-forward, x-right, y-down) but reads real world axes.
    """
    position = np.array(cam_obj.matrix_world.translation[:], dtype=float)
    m = cam_obj.matrix_world.to_3x3()
    right = np.array(m.col[0][:]); right /= np.linalg.norm(right)
    up = np.array(m.col[1][:]); up /= np.linalg.norm(up)
    forward = -np.array(m.col[2][:]); forward /= np.linalg.norm(forward)
    R = np.vstack([right, -up, forward])
    cx, cy = image_size[0] / 2, image_size[1] / 2
    K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]])
    t = -R @ position
    P = K @ np.hstack([R, t.reshape(3, 1)])
    return P.tolist()


def _render_id_pass_exrs(scene, cams, out_dir):
    """Render each camera's real object-index pass to an EXR: the same
    compositor graph (Object Index -> OutputFile, OPEN_EXR_MULTILAYER)
    validated by validate_rig_render.py in M3. Returns exr paths in cam order.
    """
    scene.render.resolution_x, scene.render.resolution_y = scene_config.IMAGE_SIZE
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.cycles.samples = _SCAN_SAMPLES
    scene.view_layers[0].use_pass_object_index = True

    node_group = bpy.data.node_groups.new("SwarmScanCompositing", "CompositorNodeTree")
    scene.compositing_node_group = node_group
    try:
        exr_paths = []
        for i, cam_obj in enumerate(cams):
            scene.camera = cam_obj
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
            exr_paths.append(os.path.join(out_dir, f"cam{i}_id_.exr"))
        return exr_paths
    finally:
        # Removing a datablock clears any properties pointing at it, so
        # scene.compositing_node_group goes back to None afterward.
        bpy.data.node_groups.remove(node_group)


class SWARM_OT_scan(Operator):
    bl_idname = "swarm.scan"
    bl_label = "Run Scan"
    bl_description = (
        "Render each rig camera's object-index pass, triangulate drone "
        "positions from the real ID-pass centroids plus Stage 1's synthetic "
        "pixel noise, and overlay the resulting distance graph vs ground truth"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        if _IMPORT_ERROR is not None:
            self.report({"ERROR"}, f"stage1_geometry import failed: {_IMPORT_ERROR}")
            return {"CANCELLED"}
        if not os.path.exists(_SCAN_VENV_PYTHON):
            self.report({"ERROR"},
                        f"Project venv not found at {_SCAN_VENV_PYTHON} (needed for OpenEXR readback)")
            return {"CANCELLED"}

        swarm_coll = bpy.data.collections.get(SWARM_COLLECTION)
        if swarm_coll is None or len(swarm_coll.objects) < 2:
            self.report({"ERROR"}, "Generate a swarm first")
            return {"CANCELLED"}
        rig_coll = bpy.data.collections.get(RIG_COLLECTION)
        cams = sorted(
            (o for o in (rig_coll.objects if rig_coll else []) if o.type == "CAMERA"),
            key=lambda o: o.name,
        )
        if len(cams) < 2:
            self.report({"ERROR"}, "Place a camera rig (>= 2 cameras) first")
            return {"CANCELLED"}

        drones = sorted(swarm_coll.objects, key=lambda o: o.name)
        settings = context.scene.swarm_scan

        shutil.rmtree(_SCAN_TMP_DIR, ignore_errors=True)
        os.makedirs(_SCAN_TMP_DIR)

        exr_paths = _render_id_pass_exrs(context.scene, cams, _SCAN_TMP_DIR)

        manifest = {
            "drones": [{"id": obj.pass_index, "position": list(obj.location)} for obj in drones],
            "cameras": [
                {
                    "name": cam_obj.name,
                    "P": _camera_pose_P(cam_obj, scene_config.IMAGE_SIZE, scene_config.FOCAL_PX),
                    "exr_path": exr_path,
                }
                for cam_obj, exr_path in zip(cams, exr_paths)
            ],
            "pixel_noise_std": scene_config.SCAN_PIXEL_NOISE_STD_PX,
            "d_max": scene_config.D_MAX,
            "near_threshold_frac": scene_config.SCAN_NEAR_THRESHOLD_FRAC,
            "noise_seed": settings.scan_seed,
        }
        manifest_path = os.path.join(_SCAN_TMP_DIR, "manifest.json")
        result_path = os.path.join(_SCAN_TMP_DIR, "result.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        try:
            proc = subprocess.run(
                [_SCAN_VENV_PYTHON, _SCAN_WORKER_PATH, manifest_path, result_path],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            self.report({"ERROR"}, "scan_worker timed out")
            return {"CANCELLED"}
        if proc.returncode != 0:
            self.report({"ERROR"}, f"scan_worker failed: {proc.stderr[-500:]}")
            return {"CANCELLED"}

        with open(result_path) as f:
            result = json.load(f)

        global _LAST_SCAN
        _LAST_SCAN = result

        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

        near = result["near_threshold_accuracy"]
        msg = (
            f"Scan: {result['n_triangulated']}/{result['n_total']} triangulated, "
            f"{result['overall_accuracy'] * 100:.1f}% adjacency accuracy"
        )
        if near is not None:
            msg += f", {near * 100:.1f}% near-threshold"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


def _draw_scan_edges():
    if not _LAST_SCAN:
        return
    by_id = {d["id"]: d for d in _LAST_SCAN["drones"]}
    correct_pts, wrong_pts = [], []
    for e in _LAST_SCAN["edges"]:
        a, b = by_id.get(e["i"]), by_id.get(e["j"])
        if a is None or b is None or a["est_position"] is None or b["est_position"] is None:
            continue
        pts = correct_pts if e["correct"] else wrong_pts
        pts.append(a["est_position"])
        pts.append(b["est_position"])
    if not correct_pts and not wrong_pts:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    for pts, color in ((correct_pts, _SCAN_EDGE_COLOR_CORRECT), (wrong_pts, _SCAN_EDGE_COLOR_WRONG)):
        if not pts:
            continue
        batch = batch_for_shader(shader, 'LINES', {"pos": pts})
        shader.uniform_float("color", color)
        batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_scan_hud():
    if not _LAST_SCAN:
        return
    r = _LAST_SCAN
    lines = [f"Swarm Scan: {r['n_triangulated']}/{r['n_total']} drones triangulated"]
    lines.append(
        f"Adjacency accuracy: {r['overall_accuracy'] * 100:.1f}%"
        if r["overall_accuracy"] is not None else "Adjacency accuracy: n/a"
    )
    lines.append(
        f"Near-D_MAX accuracy: {r['near_threshold_accuracy'] * 100:.1f}% (D_MAX={r['d_max']:.0f}m)"
        if r["near_threshold_accuracy"] is not None
        else f"Near-D_MAX accuracy: n/a (D_MAX={r['d_max']:.0f}m)"
    )

    font_id = 0
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.size(font_id, 14)
    for i, line in enumerate(reversed(lines)):
        blf.position(font_id, 20, 20 + i * 20, 0)
        blf.draw(font_id, line)


class SWARM_PT_panel(Panel):
    bl_label = "Swarm Scan"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Swarm Scan"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.swarm_scan

        if _IMPORT_ERROR is not None:
            box = layout.box()
            box.label(text="stage1_geometry import failed", icon="ERROR")
            box.label(text="Load via blender_addon/dev_load.py")
            return

        layout.prop(settings, "drone_count")
        layout.prop(settings, "formation")
        layout.prop(settings, "seed")
        layout.prop(settings, "display_scale")
        layout.operator(SWARM_OT_generate.bl_idname, icon="OUTLINER_OB_POINTCLOUD")

        coll = bpy.data.collections.get(SWARM_COLLECTION)
        if coll is not None:
            layout.label(text=f"Current swarm: {len(coll.objects)} drones")

        box = layout.box()
        box.label(text="Flight Sim", icon="FORCE_WIND")
        box.prop(settings, "neighbor_radius")
        box.prop(settings, "bound_softness")
        box.prop(settings, "wander_strength")
        box.prop(settings, "sim_speed")
        if context.window_manager.swarm_sim_running:
            box.operator(SWARM_OT_sim_stop.bl_idname, icon="PAUSE")
        else:
            box.operator(SWARM_OT_sim_start.bl_idname, icon="PLAY")

        box = layout.box()
        box.label(text="Camera Rig", icon="OUTLINER_OB_CAMERA")
        box.prop(settings, "camera_count")
        box.prop(settings, "camera_mode")
        box.operator(SWARM_OT_place_cameras.bl_idname, icon="CON_CAMERASOLVER")
        rig = bpy.data.collections.get(RIG_COLLECTION)
        if rig is not None:
            n_cams = sum(1 for o in rig.objects if o.type == "CAMERA")
            box.label(text=f"Current rig: {n_cams} cameras")
            box.operator(SWARM_OT_toggle_aim.bl_idname, icon="ORIENTATION_GIMBAL")

        box = layout.box()
        box.label(text="Scan", icon="VIEWZOOM")
        box.prop(settings, "scan_seed")
        box.operator(SWARM_OT_scan.bl_idname, icon="RENDER_STILL")
        if _LAST_SCAN is not None:
            box.label(text=f"Triangulated: {_LAST_SCAN['n_triangulated']}/{_LAST_SCAN['n_total']}")
            acc = _LAST_SCAN["overall_accuracy"]
            box.label(text=f"Accuracy: {acc * 100:.1f}%" if acc is not None else "Accuracy: n/a")
            near = _LAST_SCAN["near_threshold_accuracy"]
            box.label(text=f"Near-threshold: {near * 100:.1f}%" if near is not None else "Near-threshold: n/a")


_CLASSES = (
    SwarmScanSettings,
    SWARM_OT_generate,
    SWARM_OT_sim_start,
    SWARM_OT_sim_stop,
    SWARM_OT_place_cameras,
    SWARM_OT_toggle_aim,
    SWARM_OT_scan,
    SWARM_PT_panel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.swarm_scan = PointerProperty(type=SwarmScanSettings)
    # Runtime-only flag (WindowManager props aren't saved into .blend files):
    # True while the modal sim operator is alive; clearing it stops the sim.
    bpy.types.WindowManager.swarm_sim_running = bpy.props.BoolProperty(default=False)

    global _draw_handle_3d, _draw_handle_2d
    _draw_handle_3d = bpy.types.SpaceView3D.draw_handler_add(
        _draw_scan_edges, (), 'WINDOW', 'POST_VIEW')
    _draw_handle_2d = bpy.types.SpaceView3D.draw_handler_add(
        _draw_scan_hud, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _draw_handle_3d, _draw_handle_2d, _LAST_SCAN
    if _draw_handle_3d is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle_3d, 'WINDOW')
        _draw_handle_3d = None
    if _draw_handle_2d is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle_2d, 'WINDOW')
        _draw_handle_2d = None
    _LAST_SCAN = None

    del bpy.types.WindowManager.swarm_sim_running
    del bpy.types.Scene.swarm_scan
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
