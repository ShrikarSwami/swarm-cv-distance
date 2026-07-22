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


_CLASSES = (SwarmScanSettings, SWARM_OT_generate, SWARM_PT_panel)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.swarm_scan = PointerProperty(type=SwarmScanSettings)


def unregister():
    del bpy.types.Scene.swarm_scan
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
