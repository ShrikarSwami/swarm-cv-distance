"""Quadcopter drone geometry builder.

Builds a simple quadcopter silhouette mesh from bpy.ops primitives,
intended for use as a stand-in for true drone geometry in rendered datasets.
"""

import bpy
import math
from mathutils import Vector


def build_quadcopter_template(scale: float, emission_mat) -> bpy.types.Object:
    """Create a quadcopter template object by joining primitives.

    Builds one master quadcopter centered at origin, arms at 45° diagonals,
    thin rotor discs at arm tips, with the given emission material.

    Args:
        scale: Overall size multiplier (e.g. DISPLAY_SCALE * 0.5 for pipeline)
        emission_mat: Blender material to assign to the whole drone

    Returns:
        A Blender object whose mesh can be shared by duplicates.
    """
    s = scale
    parts = []

    def _cube(name, scale_xyz, loc=(0, 0, 0), rot=(0, 0, 0)):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        obj = bpy.context.active_object
        obj.scale = scale_xyz
        bpy.ops.object.transform_apply()
        if any(r != 0 for r in rot):
            obj.rotation_euler = rot
            bpy.ops.object.transform_apply()
        obj.name = name
        return obj

    # Central body
    parts.append(_cube("qc_body", (0.18 * s, 0.14 * s, 0.08 * s)))

    # Four diagonal arms
    arm_len = 0.35 * s
    arm_w = 0.04 * s
    arm_h = arm_w

    for angle_deg in (45, 135, 225, 315):
        rad = math.radians(angle_deg)
        # Create arm as elongated box, rotate along diagonal
        half = arm_len / 2
        arm = _cube(
            f"qc_arm_{angle_deg}",
            (arm_w, arm_len, arm_h),
            loc=(half * math.cos(rad), half * math.sin(rad), 0),
            rot=(0, 0, rad - math.radians(45)),
        )
        parts.append(arm)

    # Four rotor discs at arm tips
    rotor_r = 0.12 * s
    for angle_deg in (45, 135, 225, 315):
        rad = math.radians(angle_deg)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=rotor_r,
            depth=0.02 * s,
            location=(arm_len * math.cos(rad), arm_len * math.sin(rad), 0.02 * s),
        )
        rotor = bpy.context.active_object
        rotor.name = f"qc_rotor_{angle_deg}"
        parts.append(rotor)

    # Join all parts into one mesh
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()

    joined = parts[0]
    joined.name = "QuadcopterTemplate"

    # Assign material
    joined.data.materials.clear()
    joined.data.materials.append(emission_mat)

    return joined


def create_drones_from_template(
    template: bpy.types.Object,
    positions: list,
    base_name: str = "drone",
    start_index: int = 0,
) -> list:
    """Create drone objects by duplicating the template at each position.

    Args:
        template: Quadcopter template object (mesh shared by all duplicates).
        positions: List of (x, y, z) location vectors.
        base_name: Object name prefix.
        start_index: Starting index for naming and pass_index.

    Returns:
        List of created drone objects.
    """
    drones = []
    for i, pos in enumerate(positions):
        idx = start_index + i
        # Duplicate the template
        bpy.ops.object.select_all(action="DESELECT")
        template.select_set(True)
        bpy.context.view_layer.objects.active = template
        bpy.ops.object.duplicate()
        dup = bpy.context.active_object
        dup.location = Vector(pos) if not isinstance(pos, Vector) else pos
        dup.name = f"{base_name}_{idx:03d}"
        dup.pass_index = idx + 1  # 1-based pass index for object index pass
        drones.append(dup)

    return drones
