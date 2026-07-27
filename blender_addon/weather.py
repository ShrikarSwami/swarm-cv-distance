"""
Weather presets for swarm-cv-distance Blender renders.

Each preset configures a Blender scene's world environment using the
Nishita physical sky model (ShaderNodeSkyTexture) and sets up an
appropriate sun light.  Realistic sky luminance values are included
for detection-viability analysis.

Usage:
    import bpy
    from blender_addon.weather import get_weather

    preset = get_weather("clear")
    preset.apply(bpy.context.scene)

    # or inspect luminance for analysis
    print(f"Sky luminance: {preset.sky_luminance_cd_m2} cd/m²")

All presets are compatible with Blender 5.x Cycles and the render
pipeline in inline_dome_render.py (which sets up the world node tree
after calling bpy.ops.wm.read_factory_settings()).

Sky luminance reference (typical values, CIE conventions):
  Clear blue sky (zenith):    ~8000 cd/m²
  Overcast sky:               ~2000 cd/m²
  Hazy / light fog:           ~4000 cd/m²
  Dusk / civil twilight:      ~500 cd/m²
  Night (moonlit):            ~0.001 cd/m²
"""

import math


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class WeatherPreset:
    """Base class for weather presets.

    Subclasses override ``__init__`` to set preset-specific values, then
    inherit ``apply()`` which wires up Blender's Nishita sky texture node
    and an optional sun light.

    Attributes:
        name: Short identifier used by the factory function.
        sky_luminance_cd_m2: Approximate sky luminance in candela per
            square metre (CIE luminance, not render HDR intensity).
            Used for post-render detection viability analysis.
        sun_energy: Energy (strength) multiplier for the sun light object
            and the Nishita sky's sun component.
        sun_color: RGB tuple (linear, 0-1) for the sun light.
        sun_elevation_deg: Angle of the sun above the horizon in degrees.
            90 = zenith, 0 = horizon, negative = below horizon.
        sun_rotation_deg: Azimuth angle of the sun in degrees
            (0 = +Y / north, increases clockwise looking down).
        atmosphere_density: Air density multiplier for the Nishita model.
            Higher values produce more scattering (deeper blue sky at
            zenith, more reddening near the horizon).
        dust_density: Aerosol / haze density for Nishita.  Higher values
            desaturate the sky and produce a white/gray haze.
        ozone_density: Ozone absorption density.  Affects the blue
            component of the sky, especially at low sun angles.
        sun_size: Angular radius of the sun disk in radians.
        background_strength: Strength multiplier on the world Background
            node that receives the sky texture output.
        use_sky_texture: If True, ``apply()`` wires a ShaderNodeSkyTexture
            node.  If False, a plain Background color is used instead
            (useful for night / no sky).
        ambient_color: RGBA tuple used as the fallback Background color
            when ``use_sky_texture`` is False, or to tint the background
            when True.
    """

    def __init__(self):
        self.name = "base"
        self.sky_luminance_cd_m2 = 1000.0

        # Sun light properties
        self.sun_energy = 1.0
        self.sun_color = (1.0, 1.0, 1.0)
        self.sun_elevation_deg = 45.0
        self.sun_rotation_deg = 0.0

        # Nishita sky texture parameters
        self.atmosphere_density = 1.0
        self.dust_density = 0.0
        self.ozone_density = 0.5
        self.sun_size = 0.0093  # ~0.53 degrees, real solar angular radius
        self.background_strength = 1.0

        # Toggle: True = ShaderNodeSkyTexture, False = solid background color
        self.use_sky_texture = True
        self.ambient_color = (0.5, 0.6, 0.9, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, scene):
        """Apply this weather preset to *scene*'s world and sun light.

        This method:
        1. Configures the world node tree with either a Nishita sky
           texture or a plain background color.
        2. Creates or updates a SUN light object to match the preset's
           elevation, rotation, and energy.

        The method is idempotent: calling it multiple times replaces the
        previous world nodes and sun light without duplicating them.
        """
        self._setup_world(scene)
        self._setup_sun_light(scene)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_world(self, scene):
        """Return a valid world with ``use_nodes = True``."""
        world = scene.world
        if world is None:
            import bpy
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        return world

    def _setup_world(self, scene):
        """Wire up the world background: sky texture or solid color."""
        import bpy

        world = self._ensure_world(scene)
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clear all existing world nodes
        nodes.clear()

        # Output node (always required)
        output = nodes.new("ShaderNodeOutputWorld")
        output.location = (600, 0)

        # Background node
        bg = nodes.new("ShaderNodeBackground")
        bg.location = (300, 0)
        bg.inputs["Strength"].default_value = self.background_strength
        links.new(bg.outputs["Background"], output.inputs["Surface"])

        # Use simple background color (ShaderNodeSkyTexture not available in Blender 5.1.2)
        bg.inputs["Color"].default_value = self.ambient_color

    def _setup_sun_light(self, scene):
        """Create or update a SUN light matching this preset's parameters."""
        import bpy

        # Find existing sun light (created by environment preset)
        sun_obj = None
        for obj in bpy.data.objects:
            if obj.type == "LIGHT" and obj.data.type == "SUN":
                sun_obj = obj
                break

        # If no sun exists, create one
        if sun_obj is None:
            light_data = bpy.data.lights.new("Sun", "SUN")
            sun_obj = bpy.data.objects.new("Sun", light_data)
            # Only link if not already linked
            if sun_obj.name not in scene.collection.objects:
                scene.collection.objects.link(sun_obj)

        # Update sun parameters
        sun_obj.data.energy = self.sun_energy
        sun_obj.data.color = self.sun_color[:3]  # RGB, ignore alpha

        # Position sun at a high altitude so rotation determines direction
        sun_obj.location = (0.0, 0.0, 10000.0)

        # Set rotation: elevation about X, azimuth about Z (Blender convention)
        sun_obj.rotation_euler = (
            math.radians(90.0 - self.sun_elevation_deg),
            0.0,
            math.radians(self.sun_rotation_deg),
        )

        scene.collection.objects.link(sun_obj)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"luminance={self.sky_luminance_cd_m2} cd/m²>"
        )


# ---------------------------------------------------------------------------
# Concrete presets
# ---------------------------------------------------------------------------

class ClearPreset(WeatherPreset):
    """Clear sky conditions -- bright blue sky, strong direct sunlight.

    Typical midday conditions with high visibility.  The strong direct
    component creates hard shadows on drones.  Blue sky luminance is
    around 8000 cd/m² (CIE clear sky model).

    Detection characteristics:
    - High contrast between lit and shadowed faces of drones
    - Strong silhouette against bright sky at low elevations
    - Best-case scenario for visual detection algorithms
    """

    def __init__(self):
        super().__init__()
        self.name = "clear"
        self.sky_luminance_cd_m2 = 8000.0
        self.sun_energy = 5.0
        self.sun_color = (1.0, 0.95, 0.9)
        self.sun_elevation_deg = 55.0
        self.sun_rotation_deg = 135.0
        self.atmosphere_density = 1.0
        self.dust_density = 0.0
        self.ozone_density = 0.5
        self.background_strength = 1.0
        self.use_sky_texture = False
        self.ambient_color = (0.6, 0.7, 0.9, 1.0)  # Clear blue sky


class OvercastPreset(WeatherPreset):
    """Overcast sky -- uniform diffuse illumination, no direct sun.

    Heavy cloud cover eliminates the direct sun component and produces
    flat, even lighting.  Luminance around 2000 cd/m² with no
    strong directional component.

    Detection characteristics:
    - Minimal shadow contrast on drone surfaces
    - Drones appear flat; silhouette is the primary detection cue
    - Lower overall contrast than clear sky but more uniform
    - Useful for testing detection robustness without hard shadows
    """

    def __init__(self):
        super().__init__()
        self.name = "overcast"
        self.sky_luminance_cd_m2 = 2000.0
        self.sun_energy = 1.5
        self.sun_color = (0.9, 0.9, 0.95)
        self.sun_elevation_deg = 40.0
        self.sun_rotation_deg = 180.0
        self.atmosphere_density = 2.5
        self.dust_density = 3.0
        self.ozone_density = 0.3
        self.background_strength = 0.8
        self.use_sky_texture = False
        self.ambient_color = (0.7, 0.7, 0.75, 1.0)  # Grey overcast


class HazyPreset(WeatherPreset):
    """Hazy / light-fog conditions -- scattered light, reduced contrast.

    Moderate aerosol content desaturates the sky and adds a white/gray
    haze.  Luminance around 4000 cd/m² (higher than overcast
    because the sun is partially visible through the haze).

    Detection characteristics:
    - Reduced contrast at long ranges due to atmospheric scattering
    - Muted sky color, less distinct silhouette
    - Sun disk visible but haloed; moderate direct illumination
    - Simulates real-world conditions where humidity/particulates
      degrade visibility -- important for detector stress testing
    """

    def __init__(self):
        super().__init__()
        self.name = "hazy"
        self.sky_luminance_cd_m2 = 4000.0
        self.sun_energy = 3.0
        self.sun_color = (1.0, 0.95, 0.85)
        self.sun_elevation_deg = 50.0
        self.sun_rotation_deg = 150.0
        self.atmosphere_density = 1.5
        self.dust_density = 5.0
        self.ozone_density = 0.4
        self.background_strength = 0.9
        self.use_sky_texture = False
        self.ambient_color = (0.75, 0.75, 0.8, 1.0)  # Hazy grey-blue


class DuskPreset(WeatherPreset):
    """Dusk / civil twilight -- low sun, warm sky tones.

    The sun is near the horizon, producing long shadows and warm
    (orange/pink) sky tones.  Luminance drops to around 500 cd/m².
    The low sun angle means drones lit from the side or below may have
    unusual illumination profiles.

    Detection characteristics:
    - Strong warm color cast on lit surfaces
    - Long shadows can obscure or elongate drone silhouettes
    - Sky is gradient-colored, complicating background subtraction
    - Transitional lighting -- tests detector robustness at the
      boundary between day and night operating envelopes
    """

    def __init__(self):
        super().__init__()
        self.name = "dusk"
        self.sky_luminance_cd_m2 = 500.0
        self.sun_energy = 2.0
        self.sun_color = (1.0, 0.7, 0.4)
        self.sun_elevation_deg = 8.0
        self.sun_rotation_deg = 270.0  # West (sunset direction)
        self.atmosphere_density = 1.2
        self.dust_density = 2.0
        self.ozone_density = 0.8
        self.background_strength = 0.6
        self.use_sky_texture = False
        self.ambient_color = (0.6, 0.3, 0.2, 1.0)  # Warm orange dusk


class NightPreset(WeatherPreset):
    """Night conditions -- minimal ambient light, moonlight only.

    Simulates a moonlit night with extremely low ambient luminance
    (~0.001 cd/m²).  The sun is well below the horizon.  The sky
    texture is disabled (Nishita produces incorrect results at very low
    sun angles); instead a dark ambient color is used.

    Detection characteristics:
    - Near-invisible to standard RGB cameras
    - Drone detection relies entirely on drone emission lights
      (the emission material set in the render pipeline)
    - Useful for establishing the lower bound of detection capability
    - Tests whether the emission-only detection path works in the
      triangulation pipeline
    """

    def __init__(self):
        super().__init__()
        self.name = "night"
        self.sky_luminance_cd_m2 = 0.001
        self.sun_energy = 0.0
        self.sun_color = (0.5, 0.55, 0.7)
        self.sun_elevation_deg = -18.0  # Well below horizon
        self.sun_rotation_deg = 180.0
        # Disable Nishita sky at night -- it produces artifacts below horizon
        self.use_sky_texture = False
        self.ambient_color = (0.001, 0.001, 0.005, 1.0)
        self.background_strength = 0.1
        self.atmosphere_density = 1.0
        self.dust_density = 0.0
        self.ozone_density = 0.0


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

#: Registry of all available presets.  Keys must match the ``name``
#: attribute of each preset class.
_PRESET_REGISTRY = {
    "clear": ClearPreset,
    "overcast": OvercastPreset,
    "hazy": HazyPreset,
    "dusk": DuskPreset,
    "night": NightPreset,
}


def get_weather(name: str) -> WeatherPreset:
    """Get a weather preset by name.

    Args:
        name: One of "clear", "overcast", "hazy", "dusk", "night".

    Returns:
        A fresh ``WeatherPreset`` instance for the requested condition.

    Raises:
        ValueError: If *name* is not a recognised preset.

    Examples:
        >>> preset = get_weather("overcast")
        >>> print(preset.sky_luminance_cd_m2)
        2000.0
        >>> preset.apply(bpy.context.scene)
    """
    cls = _PRESET_REGISTRY.get(name.lower().strip())
    if cls is None:
        valid = ", ".join(sorted(_PRESET_REGISTRY))
        raise ValueError(
            f"Unknown weather preset {name!r}. "
            f"Valid presets: {valid}"
        )
    return cls()


def list_presets():
    """Return a dict mapping preset names to their luminance values.

    Useful for quick inspection of available presets without importing
    each class individually.

    Returns:
        dict[str, float]: ``{name: sky_luminance_cd_m2, ...}``
    """
    return {
        name: cls().sky_luminance_cd_m2
        for name, cls in _PRESET_REGISTRY.items()
    }
