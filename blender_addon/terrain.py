"""
Procedural terrain generation for swarm-cv-distance renders.

Replaces the flat ground plane with height-mapped terrain using
Perlin noise displacement via ``mathutils.noise``.  All materials are
procedural (Blender shader nodes, no external textures).

Usage::

    from blender_addon.terrain import ProceduralTerrain, configure_desert

    terrain = ProceduralTerrain(seed=42)
    configure_desert(terrain)
    obj = terrain.build(bpy.context.scene)
"""

from __future__ import annotations

import math
import mathutils
import mathutils.noise

# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

# Number of fBm octaves.  More octaves = finer detail.
_NUM_OCTAVES = 6


def _fbm_noise(x: float, y: float, seed: int, noise_scale: float,
               num_octaves: int = _NUM_OCTAVES) -> float:
    """Compute fractal Brownian motion noise at world coordinate (x, y).

    Returns a value in [0, 1].
    """
    nx = x / noise_scale
    ny = y / noise_scale

    h = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    # Use seed as a z-offset to decorrelate different seeds
    z_offset = seed * 100.0

    for octave in range(num_octaves):
        pos = (nx * frequency, ny * frequency, z_offset + octave * 50.0)
        h += amplitude * mathutils.noise.noise(pos)
        max_amplitude += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    # Normalise to [-1, 1], then map to [0, 1]
    h /= max_amplitude
    h = h * 0.5 + 0.5
    h = max(0.0, min(1.0, h))  # Clamp for safety
    return h


# ---------------------------------------------------------------------------
# Preset colour palettes
# ---------------------------------------------------------------------------

COLOR_PALETTES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "desert": {
        "low":  (0.62, 0.48, 0.22, 1.0),   # darker sand (valleys)
        "mid":  (0.82, 0.72, 0.52, 1.0),   # mid sand (slopes)
        "high": (0.92, 0.84, 0.65, 1.0),   # light sand (ridges)
    },
    "forest": {
        "low":  (0.08, 0.12, 0.06, 1.0),   # very dark forest floor (valleys)
        "mid":  (0.20, 0.28, 0.14, 1.0),   # mid green (slopes)
        "high": (0.45, 0.55, 0.32, 1.0),   # lighter green (ridges)
    },
}

# ---------------------------------------------------------------------------
# Preset configurators
# ---------------------------------------------------------------------------


def configure_desert(terrain: "ProceduralTerrain") -> "ProceduralTerrain":
    """Apply desert preset parameters to *terrain*.

    - Wider, taller features (noise_scale=300, height=30 m)
    - Warm sandy color palette
    """
    terrain.noise_scale = 300.0
    terrain.height = 30.0
    terrain.color_palette = COLOR_PALETTES["desert"]
    return terrain


def configure_forest_floor(terrain: "ProceduralTerrain") -> "ProceduralTerrain":
    """Apply forest-floor preset parameters to *terrain*.

    - Finer features with moderate height (noise_scale=150, height=35 m)
    - Cool earthy green-brown palette with higher contrast
    """
    terrain.noise_scale = 150.0
    terrain.height = 35.0
    terrain.color_palette = COLOR_PALETTES["forest"]
    return terrain


# ---------------------------------------------------------------------------
# Terrain builder
# ---------------------------------------------------------------------------


class ProceduralTerrain:
    """Procedurally generated terrain with noise-based height displacement.

    The ``build()`` method creates a subdivided grid mesh with vertex
    heights computed from fractal Perlin noise, applies a procedural
    node-based material that maps height to colour, and returns the
    resulting Blender object.

    Parameters
    ----------
    seed : int
        PRNG seed controlling the noise pattern.  Different seeds produce
        uncorrelated terrain.
    noise_scale : float
        Wavelength of the largest noise features (metres).  Smaller values
        produce more densely-packed hills.
    height : float
        Maximum elevation above the base plane (metres).
    detail : int
        Number of subdivisions along each axis (grid resolution is
        ``(detail+1) x (detail+1)`` vertices).  128 gives a 129x129 grid
        (~16k vertices), which is fast to generate while showing adequate
        detail.
    size : float
        Side length of the square terrain patch (metres).
    color_palette : dict
        Dict with keys ``"low"``, ``"mid"``, ``"high"`` mapping to RGBA
        tuples for the procedural material.
    """

    def __init__(self, seed: int = 0, noise_scale: float = 200.0,
                 height: float = 50.0, detail: int = 128,
                 size: float = 5000.0):
        self.seed = seed
        self.noise_scale = noise_scale
        self.height = height
        self.detail = detail
        self.size = size
        self.color_palette = COLOR_PALETTES["desert"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, scene) -> "bpy.types.Object":
        """Create the terrain mesh and return it linked into *scene*.

        Steps:
          1. Build vertex list with noise-displaced heights.
          2. Create mesh via ``from_pydata`` (reliable in background mode).
          3. Apply smooth shading to all faces.
          4. Create and assign a procedural node-based material.
          5. Link the object into *scene*'s default collection.

        Returns:
            The newly created Blender mesh object.
        """
        import bpy

        half = self.size / 2.0
        step = self.size / self.detail

        # (1) Build vertex list --------------------------------------------
        verts = []
        for i in range(self.detail + 1):
            for j in range(self.detail + 1):
                x = -half + i * step
                y = -half + j * step
                z = _fbm_noise(x, y, self.seed, self.noise_scale) * self.height
                verts.append((x, y, z))

        # (2) Build quad-face index list ------------------------------------
        faces = []
        for i in range(self.detail):
            row = i * (self.detail + 1)
            for j in range(self.detail):
                a = row + j
                b = a + 1
                c = row + self.detail + 1 + j + 1
                d = row + self.detail + 1 + j
                faces.append((a, b, c, d))

        # (3) Create mesh ---------------------------------------------------
        mesh = bpy.data.meshes.new("Terrain")
        mesh.from_pydata(verts, [], faces)
        for poly in mesh.polygons:
            poly.use_smooth = True
        mesh.update()

        # (4) Create object and link ----------------------------------------
        obj = bpy.data.objects.new("Terrain", mesh)
        scene.collection.objects.link(obj)

        # (5) Apply material ------------------------------------------------
        self._create_material(mesh)

        return obj

    # ------------------------------------------------------------------
    # Material
    # ------------------------------------------------------------------

    def _create_material(self, mesh: "bpy.types.Mesh") -> None:
        """Build a node-based procedural material keyed on height.

        Node graph (simplified)::

            TexCoord (Object)  -->  SeparateXYZ (Z)
                                        |
                                        v
                                   MapRange  [0, height] -> [0, 1]
                                        |
                                      +---+
                                      |   |
                                      v   v
                                 ColorRamp  NoiseTex --> ColorRamp
                                      |         (micro detail)
                                      |   |
                                      v   v
                                    MixRGB (multiply, 15%)
                                        |
                                        v
                                   Principled BSDF  -->  Output
        """
        import bpy

        mat = bpy.data.materials.new("TerrainMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # -- Nodes ----------------------------------------------------------

        # 1. Texture Coordinate (Object space -> vertex position)
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-1100, 0)

        # 2. Separate XYZ  ->  extract Z (height)
        sep_xyz = nodes.new("ShaderNodeSeparateXYZ")
        sep_xyz.location = (-900, 0)

        # 3. Map Range  ->  normalise height to [0, 1]
        map_range = nodes.new("ShaderNodeMapRange")
        map_range.location = (-700, 0)
        map_range.inputs["From Min"].default_value = 0.0
        map_range.inputs["From Max"].default_value = self.height
        map_range.inputs["To Min"].default_value = 0.0
        map_range.inputs["To Max"].default_value = 1.0
        # ColorRamp will clamp at 0-1, no need for explicit clamp here

        # 4. Color Ramp  ->  height -> colour
        color_ramp = nodes.new("ShaderNodeValToRGB")
        color_ramp.location = (-500, 0)
        ramp = color_ramp.color_ramp
        # Set up three stops: low (0.0), mid (0.5), high (1.0)
        ramp.elements[0].position = 0.0
        ramp.elements[0].color = self.color_palette["low"]
        ramp.elements[-1].position = 1.0
        ramp.elements[-1].color = self.color_palette["high"]
        mid = ramp.elements.new(0.5)
        mid.color = self.color_palette["mid"]

        # 5. Noise Texture (micro-detail)
        noise_tex = nodes.new("ShaderNodeTexNoise")
        noise_tex.location = (-700, -350)
        noise_tex.inputs["Scale"].default_value = 500.0
        noise_tex.inputs["Detail"].default_value = 2.0
        noise_tex.inputs["Roughness"].default_value = 0.5

        # 6. ColorRamp for noise (subtle grey variation)
        noise_ramp = nodes.new("ShaderNodeValToRGB")
        noise_ramp.location = (-500, -350)
        nr = noise_ramp.color_ramp
        nr.elements[0].color = (0.92, 0.92, 0.92, 1.0)  # slight darkening
        nr.elements[0].position = 0.0
        nr.elements[-1].color = (1.0, 1.0, 1.0, 1.0)  # identity
        nr.elements[-1].position = 1.0

        # 7. MixRGB  ->  blend noise into base colour
        mix = nodes.new("ShaderNodeMixRGB")
        mix.location = (-250, 0)
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 0.15  # subtle (15 %)

        # 8. Principled BSDF
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Roughness"].default_value = 0.85
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.0

        # 9. Output
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (200, 0)

        # -- Connections ----------------------------------------------------

        # Height chain
        links.new(tex_coord.outputs["Object"], sep_xyz.inputs["Vector"])
        links.new(sep_xyz.outputs["Z"], map_range.inputs["Value"])
        links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], mix.inputs["Color1"])

        # Noise chain
        links.new(noise_tex.outputs["Fac"], noise_ramp.inputs["Fac"])
        links.new(noise_ramp.outputs["Color"], mix.inputs["Color2"])

        # Final
        links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        # Assign
        mesh.materials.append(mat)

    def __repr__(self) -> str:
        return (
            f"<ProceduralTerrain seed={self.seed} noise_scale={self.noise_scale} "
            f"height={self.height} detail={self.detail} size={self.size}>"
        )
