"""
Nishita physical sky model for Blender 5.x renders.

Replaces the downloaded-HDRI approach (blender_addon/hdri.py) with Blender's
built-in ShaderNodeTexSky using the Nishita multiple-scattering model.
This eliminates network dependencies, baked-in vs. lamp sun desync, and
the 3-preset limit of the HDRI system.

All exposure calibration values were determined empirically by rendering
test frames with the Nishita sky at each preset's default parameters and
checking that sky pixels do not clip (max RGB < 0.95 in scene linear).

Usage:
    from blender_addon.nishita_sky import apply_to_scene, list_presets

    # Apply a preset
    result = apply_to_scene(bpy.context.scene, "clear")
    # result == {"world": ..., "sky_node": ..., "sun_obj": ...}
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class NishitaPreset:
    """Configuration for a Nishita physical sky preset.

    All angular values are in degrees for human readability; conversion
    to radians happens in ``apply_to_scene()``.

    Attributes:
        name: Short identifier (e.g. "clear", "overcast", "dusk").
        sun_elevation: Sun angle above the horizon in degrees.
            90 = zenith, 0 = horizon, negative = below horizon.
        sun_rotation: Sun azimuth in degrees (0 = +Y, increases clockwise).
        sun_disk_size: Angular radius of the sun disk in radians.
            Real sun ~0.0093 rad (~0.53 deg).  Larger values produce a
            visibly bigger sun disc in the sky.
        turbidity: Atmosphere turbidity (1.0-10.0+).  Low = clear alpine
            sky, high = hazy/overcast appearance.
        ground_albedo: Ground reflectivity (0.0-1.0).  Higher values
            bounce more light back into the sky (brightens the sky
            from below).
        sun_intensity: Multiplier for the sun component in the Nishita
            model.  Reduces the sky's overall brightness while preserving
            the spatial gradient and colour.
        exposure_compensation: Scene exposure offset (in EV stops) applied
            to ``scene.view_settings.exposure``.  Compensates for the
            Nishita model's high default scene-linear radiance.
        strength: World Background node strength multiplier.  Another
            knob for overall brightness (multiplies the entire sky output).
    """

    name: str
    sun_elevation: float
    sun_rotation: float
    sun_disk_size: float = 0.0093
    turbidity: float = 2.2
    ground_albedo: float = 0.3
    sun_intensity: float = 1.0
    exposure_compensation: float = 0.0
    strength: float = 1.0
    # Nishita density parameters
    air_density: float = 1.0
    aerosol_density: float = 1.0
    ozone_density: float = 1.0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# Exposure and strength values were calibrated by rendering test frames
# and verifying no sky clipping (max pixel < 0.95 in scene linear).

CLEAR = NishitaPreset(
    name="clear",
    sun_elevation=55.0,
    sun_rotation=135.0,
    turbidity=2.0,
    sun_intensity=1.0,
    exposure_compensation=-1.5,
    strength=1.0,
    air_density=1.0,
    aerosol_density=0.5,
    ozone_density=1.0,
)

OVERCAST = NishitaPreset(
    name="overcast",
    sun_elevation=40.0,
    sun_rotation=180.0,
    turbidity=9.0,
    sun_intensity=0.3,
    exposure_compensation=-0.5,
    strength=1.0,
    air_density=2.0,
    aerosol_density=5.0,
    ozone_density=0.5,
)

DUSK = NishitaPreset(
    name="dusk",
    sun_elevation=8.0,
    sun_rotation=270.0,
    turbidity=3.5,
    sun_intensity=1.0,
    exposure_compensation=-1.0,
    strength=1.0,
    air_density=1.0,
    aerosol_density=2.0,
    ozone_density=1.0,
)

#: Registry of all presets keyed by name.
_PRESETS: Dict[str, NishitaPreset] = {
    "clear": CLEAR,
    "overcast": OVERCAST,
    "dusk": DUSK,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_to_scene(scene, preset_name: str) -> dict:
    """Apply a Nishita sky preset to *scene*.

    This creates or updates:
    1. The world node tree with a Nishita sky texture
    2. A sun lamp with matching rotation and energy

    The world background chain is:
        ShaderNodeTexSky -> ShaderNodeBackground -> ShaderNodeOutputWorld

    The sun lamp energy is derived from ``preset.sun_intensity`` so that
    shadow direction is always synchronised with the sky's sun position.

    Args:
        scene: Blender scene (``bpy.types.Scene``).
        preset_name: Key into ``_PRESETS`` ("clear", "overcast", "dusk").

    Returns:
        dict with keys ``world``, ``sky_node``, ``sun_obj``.

    Raises:
        ValueError: If *preset_name* is not recognised.
    """
    import bpy

    if preset_name not in _PRESETS:
        raise ValueError(
            f"Unknown Nishita preset {preset_name!r}. "
            f"Available: {list(_PRESETS.keys())}"
        )

    preset = _PRESETS[preset_name]

    # ---- World node tree ----
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # Output node
    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (600, 0)

    # Background node
    bg = nodes.new("ShaderNodeBackground")
    bg.location = (300, 0)
    bg.inputs["Strength"].default_value = preset.strength

    # Sky Texture node (Nishita multiple-scattering)
    sky = nodes.new("ShaderNodeTexSky")
    sky.location = (0, 0)
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = True

    # Set sky parameters (all in radians, converted from degrees)
    sky.sun_elevation = math.radians(preset.sun_elevation)
    sky.sun_rotation = math.radians(preset.sun_rotation)
    sky.sun_size = preset.sun_disk_size
    sky.sun_intensity = preset.sun_intensity
    sky.turbidity = preset.turbidity
    sky.ground_albedo = preset.ground_albedo
    sky.air_density = preset.air_density
    sky.aerosol_density = preset.aerosol_density
    sky.ozone_density = preset.ozone_density

    # Wire: Sky Texture -> Background -> Output
    links.new(sky.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # ---- Scene exposure ----
    scene.view_settings.exposure = preset.exposure_compensation

    # ---- Sun lamp ----
    sun_obj = _get_or_create_sun(scene)
    _configure_sun(sun_obj, preset)

    return {
        "world": world,
        "sky_node": sky,
        "sun_obj": sun_obj,
    }


def get_preset(name: str) -> NishitaPreset:
    """Return the preset dataclass for *name* (read-only)."""
    if name not in _PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Available: {list(_PRESETS.keys())}"
        )
    return _PRESETS[name]


def list_presets():
    """Return sorted list of available preset names."""
    return sorted(_PRESETS.keys())


def nishita_sky_replace_hdri(scene, preset_name: str) -> dict:
    """Compatibility wrapper -- drop-in for ``hdri.apply()``.

    Same signature as ``apply_to_scene()`` but also explicitly removes any
    Environment Texture nodes from the world (to undo an earlier HDRI
    setup).  Call this from the render pipeline when migrating from
    ``hdri.apply()`` to ``nishita_sky.apply_to_scene()``.

    Returns the same dict as ``apply_to_scene()``.

    Example integration in ``render_clip.py``:
        # Old:
        # import blender_addon.hdri as hdri
        # hdri.apply(scene, "clear")

        # New:
        import blender_addon.nishita_sky as nishita_sky
        nishita_sky.nishita_sky_replace_hdri(scene, "clear")
    """
    import bpy

    # Run the normal Nishita apply (this clears and rebuilds the world tree)
    result = apply_to_scene(scene, preset_name)

    # Remove any leftover HDRI image data from previous runs
    for img in bpy.data.images:
        if img.filepath.endswith(".exr") and "polyhaven" in img.filepath.lower():
            bpy.data.images.remove(img)

    for light in bpy.data.lights:
        pass  # no cleanup needed; _get_or_create_sun preserves existing sun

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_create_sun(scene):
    """Return an existing sun light or create a new one."""
    import bpy

    # Look for existing sun
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            return obj

    # Create new sun
    light_data = bpy.data.lights.new("Sun", "SUN")
    sun_obj = bpy.data.objects.new("Sun", light_data)
    if sun_obj.name not in scene.collection.objects:
        scene.collection.objects.link(sun_obj)
    return sun_obj


def _configure_sun(sun_obj, preset: NishitaPreset):
    """Set sun lamp rotation and energy to match the preset."""
    import bpy

    # Energy derived from sun_intensity.  A sun_intensity of 1.0
    # corresponds to ~5 W/m²/sr for reasonable exposure behaviour.
    sun_obj.data.energy = preset.sun_intensity * 5.0

    # Color — slightly warm for direct sun
    sun_obj.data.color = (1.0, 0.95, 0.88)

    # Position far above
    sun_obj.location = (0.0, 0.0, 10000.0)

    # Rotation: Blender convention — X rotation is elevation
    # (90 - elev) degrees about X, Z rotation is azimuth.
    sun_obj.rotation_euler = (
        math.radians(90.0 - preset.sun_elevation),
        0.0,
        math.radians(preset.sun_rotation),
    )
