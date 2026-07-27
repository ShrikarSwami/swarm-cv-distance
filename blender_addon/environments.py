"""
Environment presets for swarm-cv-distance render pipeline.

Provides four distinct visual environments (desert, grassland, forest, city)
that can be applied to a Blender scene. Each preset configures:
  - Ground plane (5km x 5km) with terrain-appropriate color/material
  - Sky background color
  - Sun light (energy, color, rotation)
  - Optional atmosphere fog

All assets are procedural (no external textures). Designed for Blender 5.x
Cycles renderer. Compatible with inline_dome_render.py scene setup.

Usage:
    from blender_addon.environments import get_environment

    scene = bpy.context.scene
    env = get_environment("desert")
    env.apply(scene)
"""

import math
import mathutils


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class EnvironmentPreset:
    """Base class for environment presets.

    Provides a uniform API: instantiate a preset, then call ``apply(scene)``
    to configure ground, sky, sun, and atmosphere on an existing Blender scene.

    Subclasses override ``__init__`` to set palette and lighting values, and
    may override ``apply`` to add environment-specific elements (e.g. fog
    volumes, extra lights).
    """

    def __init__(self):
        self.name = "base"

        # Ground
        self.ground_color = (0.5, 0.5, 0.5, 1.0)
        self.ground_roughness = 0.8
        self.ground_scale = 5000.0  # meters (5 km x 5 km)

        # Sky / world background
        self.sky_color = (0.5, 0.6, 0.9, 1.0)
        self.sky_strength = 1.0

        # Sun light
        self.sun_energy = 3.0
        self.sun_color = (1.0, 1.0, 1.0)
        # Rotation as Euler XYZ radians (elevation ~45 deg, azimuth ~30 deg)
        self.sun_elevation_deg = 45.0
        self.sun_azimuth_deg = 30.0

        # Optional atmosphere (fog density, 0 = disabled)
        self.fog_density = 0.0
        self.fog_color = (0.8, 0.8, 0.8, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_world(self, scene):
        """Return (world, bg_node) ensuring the world node tree exists."""
        world = scene.world
        if world is None:
            import bpy
            world = bpy.data.worlds.new("World")
            scene.world = world
        world.use_nodes = True
        tree = world.node_tree
        # Remove all existing nodes for a clean start
        tree.nodes.clear()
        return world, tree

    def _setup_world_background(self, scene):
        """Create world background nodes with the preset sky color."""
        import bpy

        world, tree = self._ensure_world(scene)
        nodes = tree.nodes
        links = tree.links

        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = self.sky_color
        bg.inputs["Strength"].default_value = self.sky_strength
        bg.location = (0, 0)

        output = nodes.new("ShaderNodeOutputWorld")
        output.location = (300, 0)

        links.new(bg.outputs["Background"], output.inputs["Surface"])
        return bg

    def _create_ground_plane(self, scene):
        """Create a large ground plane mesh with a simple diffuse material."""
        import bpy

        half = self.ground_scale / 2.0

        # Create plane mesh
        mesh = bpy.data.meshes.new(f"{self.name}_ground")
        # 4 corners at z=0
        verts = [
            (-half, -half, 0.0),
            ( half, -half, 0.0),
            ( half,  half, 0.0),
            (-half,  half, 0.0),
        ]
        faces = [(0, 1, 2, 3)]
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        # Material: Principled BSDF with base color and roughness
        mat = bpy.data.materials.new(f"{self.name}_ground_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = self.ground_color
            bsdf.inputs["Roughness"].default_value = self.ground_roughness
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Specular IOR Level"].default_value = 0.5
        mesh.materials.append(mat)

        # Object
        obj = bpy.data.objects.new(f"{self.name}_ground", mesh)
        obj.location = (0.0, 0.0, 0.0)
        scene.collection.objects.link(obj)
        return obj

    def _configure_sun(self, scene):
        """Create or reconfigure a sun light matching this preset."""
        import bpy

        # Find existing sun or create one
        sun_obj = None
        for obj in bpy.data.objects:
            if obj.type == "LIGHT" and obj.data.type == "SUN":
                sun_obj = obj
                break

        if sun_obj is None:
            light_data = bpy.data.lights.new("Sun", "SUN")
            sun_obj = bpy.data.objects.new("Sun", light_data)
            scene.collection.objects.link(sun_obj)

        light_data = sun_obj.data
        light_data.energy = self.sun_energy
        light_data.color = self.sun_color[:3]

        # Position sun high above scene (visual only; direction comes from rotation)
        sun_obj.location = (0.0, 0.0, 10000.0)

        # Set rotation: elevation -> Blender rotation_euler
        elev = mathutils.Euler((0.0, 0.0, 0.0))
        elev.x = math.radians(self.sun_elevation_deg)
        elev.y = math.radians(self.sun_azimuth_deg)
        sun_obj.rotation_euler = elev

        return sun_obj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, scene):
        """Apply this environment preset to a Blender scene.

        Args:
            scene: A bpy.types.Scene (typically ``bpy.context.scene``).

        Actions:
            1. Sets world background color and strength.
            2. Creates a 5 km x 5 km ground plane at z=0.
            3. Configures (or creates) a sun light.
            4. Calls ``_apply_extra(scene)`` for subclass-specific additions.

        Returns:
            dict with keys ``"world"``, ``"ground"``, ``"sun"`` pointing to
            the created/configured Blender objects.
        """
        bg = self._setup_world_background(scene)
        ground_obj = self._create_ground_plane(scene)
        sun_obj = self._configure_sun(scene)

        self._apply_extra(scene)

        return {
            "world": scene.world,
            "ground": ground_obj,
            "sun": sun_obj,
        }

    def _apply_extra(self, scene):
        """Override in subclasses for environment-specific additions.

        Default implementation applies fog via volume scatter if
        ``self.fog_density > 0``.
        """
        if self.fog_density > 0:
            self._apply_fog(scene)

    def _apply_fog(self, scene):
        """Add a volume scatter node to the world for atmospheric fog."""
        import bpy

        world = scene.world
        if world is None or not world.use_nodes:
            return
        tree = world.node_tree
        nodes = tree.nodes
        links = tree.links

        # Find the output node
        output = None
        for n in nodes:
            if n.type == "OUTPUT_WORLD":
                output = n
                break
        if output is None:
            return

        vol = nodes.new("ShaderNodeVolumeScatter")
        vol.inputs["Color"].default_value = self.fog_color
        vol.inputs["Density"].default_value = self.fog_density
        vol.location = (0, -200)

        links.new(vol.outputs["Volume"], output.inputs["Volume"])

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Desert
# ---------------------------------------------------------------------------

class DesertPreset(EnvironmentPreset):
    """Hot, arid desert environment.

    Characteristics:
      - Sandy tan ground with warm undertones
      - Hazy pale-blue sky (dust scattering)
      - Strong, warm-tinted sun (slightly yellow-orange)
      - High energy to simulate harsh midday light
      - Light atmospheric haze (desert dust)

    Typical scenario: open flat terrain, high visibility, drone swarm
    operating over arid region.
    """

    def __init__(self):
        super().__init__()
        self.name = "desert"

        # Warm sand tones — sun-baked desert floor
        self.ground_color = (0.82, 0.72, 0.52, 1.0)
        self.ground_roughness = 0.9

        # Hazy, warm sky — dust particles scatter shorter wavelengths
        self.sky_color = (0.78, 0.72, 0.55, 1.0)
        self.sky_strength = 1.2

        # Hot desert sun — slightly warm-shifted, high energy
        self.sun_energy = 4.0
        self.sun_color = (1.0, 0.95, 0.82)
        self.sun_elevation_deg = 60.0
        self.sun_azimuth_deg = 25.0

        # Light atmospheric dust haze
        self.fog_density = 0.00001
        self.fog_color = (0.85, 0.80, 0.65, 1.0)


# ---------------------------------------------------------------------------
# Grassland
# ---------------------------------------------------------------------------

class GrasslandPreset(EnvironmentPreset):
    """Temperate grassland / prairie environment.

    Characteristics:
      - Medium green ground with earthy undertone
      - Clear blue sky with slight green horizon tint
      - Neutral white sun at moderate energy
      - No atmospheric haze (clear, dry air)

    Typical scenario: open meadow or steppe, moderate visibility,
    mid-latitude drone operations.
    """

    def __init__(self):
        super().__init__()
        self.name = "grassland"

        # Natural grass green — not too saturated, slightly earthy
        self.ground_color = (0.42, 0.55, 0.28, 1.0)
        self.ground_roughness = 0.85

        # Clear sky with subtle green near horizon
        self.sky_color = (0.52, 0.65, 0.90, 1.0)
        self.sky_strength = 1.0

        # Neutral daylight sun
        self.sun_energy = 3.5
        self.sun_color = (1.0, 0.98, 0.95)
        self.sun_elevation_deg = 50.0
        self.sun_azimuth_deg = 30.0

        # No fog — clear open air
        self.fog_density = 0.0
        self.fog_color = (0.8, 0.8, 0.8, 1.0)


# ---------------------------------------------------------------------------
# Forest
# ---------------------------------------------------------------------------

class ForestPreset(EnvironmentPreset):
    """Dense temperate forest canopy environment.

    Characteristics:
      - Dark olive-green ground (shadowed forest floor)
      - Cool blue-grey sky visible through canopy gaps
      - Dimmer, cooler sun (filtered through trees)
      - Notable atmospheric haze (moisture, particulates)
      - Slightly elevated fog for depth perception

    Typical scenario: dense woodland, reduced visibility between drones,
    challenging triangulation due to occlusion.
    """

    def __init__(self):
        super().__init__()
        self.name = "forest"

        # Dark forest floor — deep green-brown
        self.ground_color = (0.28, 0.35, 0.18, 1.0)
        self.ground_roughness = 0.95

        # Cool, overcast sky — forest canopy filters direct sunlight
        self.sky_color = (0.45, 0.52, 0.62, 1.0)
        self.sky_strength = 0.8

        # Dimmer, cooler sun — filtered light
        self.sun_energy = 2.5
        self.sun_color = (0.9, 0.93, 1.0)
        self.sun_elevation_deg = 55.0
        self.sun_azimuth_deg = 40.0

        # Visible atmospheric haze — moisture in forest air
        self.fog_density = 0.00003
        self.fog_color = (0.65, 0.70, 0.72, 1.0)

    def _apply_extra(self, scene):
        """Forest: add subtle ambient fill light (simulates scattered sky light)."""
        import bpy

        # Soft ambient fill to simulate inter-reflected canopy light
        fill_data = bpy.data.lights.new("ForestFill", "SUN")
        fill_data.energy = 1.0
        fill_data.color = (0.6, 0.7, 0.65)
        fill_obj = bpy.data.objects.new("ForestFill", fill_data)
        fill_obj.location = (0.0, 0.0, 8000.0)
        # Slight downward tilt (simulates scattered canopy light from above)
        fill_obj.rotation_euler = mathutils.Euler((math.radians(70), 0.0, 0.0))
        scene.collection.objects.link(fill_obj)

        super()._apply_extra(scene)


# ---------------------------------------------------------------------------
# City
# ---------------------------------------------------------------------------

class CityPreset(EnvironmentPreset):
    """Urban / cityscape environment.

    Characteristics:
      - Grey concrete/asphalt ground
      - Muted grey-blue sky (urban haze / smog)
      - Moderate sun with urban-tinted color
      - Moderate atmospheric haze (pollution, dust)
      - Slightly elevated specular on ground (wet/asphalt)

    Typical scenario: urban drone operations, reflective surfaces,
    potential GPS multipath. Ground plane represents cityscape at
    macro scale (rooftops/terrain level).
    """

    def __init__(self):
        super().__init__()
        self.name = "city"

        # Grey asphalt/concrete — urban terrain
        self.ground_color = (0.55, 0.55, 0.55, 1.0)
        self.ground_roughness = 0.7  # Slightly smoother (paved surface)

        # Muted urban sky — haze and pollution
        self.sky_color = (0.6, 0.62, 0.7, 1.0)
        self.sky_strength = 1.0

        # Urban sun — slightly warm, moderate energy
        self.sun_energy = 3.0
        self.sun_color = (1.0, 0.92, 0.85)
        self.sun_elevation_deg = 40.0
        self.sun_azimuth_deg = 35.0

        # Urban haze / smog layer
        self.fog_density = 0.00002
        self.fog_color = (0.72, 0.72, 0.74, 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PRESETS = {
    "desert": DesertPreset,
    "grassland": GrasslandPreset,
    "forest": ForestPreset,
    "city": CityPreset,
}


def get_environment(name: str) -> EnvironmentPreset:
    """Get an environment preset by name.

    Args:
        name: One of ``"desert"``, ``"grassland"``, ``"forest"``, ``"city"``.

    Returns:
        A new instance of the requested :class:`EnvironmentPreset`.

    Raises:
        ValueError: If ``name`` is not a recognized preset.

    Examples:
        >>> env = get_environment("desert")
        >>> env.name
        'desert'

        >>> env = get_environment("forest")
        >>> env.apply(bpy.context.scene)
        {'world': <World ...>, 'ground': <Object ...>, 'sun': <Object ...>}
    """
    cls = _PRESETS.get(name.lower().strip())
    if cls is None:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(
            f"Unknown environment {name!r}. Available: {available}"
        )
    return cls()


def list_environments() -> list[str]:
    """Return sorted list of available environment preset names."""
    return sorted(_PRESETS.keys())
