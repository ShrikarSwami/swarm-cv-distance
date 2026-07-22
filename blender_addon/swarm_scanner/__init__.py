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

import os
import sys

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

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


_CLASSES = (
    SwarmScanSettings,
    SWARM_OT_generate,
    SWARM_OT_sim_start,
    SWARM_OT_sim_stop,
    SWARM_PT_panel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.swarm_scan = PointerProperty(type=SwarmScanSettings)
    # Runtime-only flag (WindowManager props aren't saved into .blend files):
    # True while the modal sim operator is alive; clearing it stops the sim.
    bpy.types.WindowManager.swarm_sim_running = bpy.props.BoolProperty(default=False)


def unregister():
    del bpy.types.WindowManager.swarm_sim_running
    del bpy.types.Scene.swarm_scan
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
