"""HDRI environment lighting for Blender renders.

Downloads HDRI skies from Poly Haven on first use, caches in assets/hdris/,
and applies them to Blender's world node tree for image-based lighting.
"""

import ssl
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from urllib.request import urlopen, Request
from urllib.error import URLError


def _create_ssl_context():
    """Create an SSL context using certifi certificates if available."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return ctx


@dataclass
class HDPreset:
    """HDRI preset with sun orientation parameters."""
    name: str
    asset_id: str
    sun_azimuth: float  # degrees
    sun_elevation: float  # degrees
    strength: float


PRESETS: Dict[str, HDPreset] = {
    "clear": HDPreset(
        name="clear",
        asset_id="blue_photo_studio",
        sun_azimuth=45.0,
        sun_elevation=60.0,
        strength=0.15,  # Optimal: good contrast, shadows visible
    ),
    "overcast": HDPreset(
        name="overcast",
        asset_id="kloofendal_overcast_puresky",  # Genuine overcast: soft, diffuse lighting
        sun_azimuth=0.0,
        sun_elevation=45.0,
        strength=0.8,
    ),
    "dusk": HDPreset(
        name="dusk",
        asset_id="kiara_9_dusk",
        sun_azimuth=90.0,
        sun_elevation=15.0,
        strength=1.2,
    ),
}


def _get_cache_dir() -> Path:
    """Return project-scoped cache directory for HDRIs."""
    # assets/hdris/ relative to this file's parent (blender_addon/)
    cache_dir = Path(__file__).parent.parent / "assets" / "hdris"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_hdri(asset_id: str) -> Path:
    """Download HDRI from Poly Haven with verification.

    Args:
        asset_id: Poly Haven asset ID (e.g., "blue_photo_studio")

    Returns:
        Path to downloaded .exr file

    Raises:
        RuntimeError: If download fails or verification fails
    """
    cache_dir = _get_cache_dir()
    target_path = cache_dir / f"{asset_id}.exr"

    # Check cache first
    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return target_path

    # Fetch file info from Poly Haven API
    api_url = f"https://api.polyhaven.com/files/{asset_id}"
    _ssl_ctx = _create_ssl_context()
    try:
        req = Request(api_url, headers={"User-Agent": "curl/7.88.1"})
        with urlopen(req, context=_ssl_ctx) as response:
            import json
            file_info = json.loads(response.read())
    except URLError as e:
        raise RuntimeError(f"Failed to fetch file info from Poly Haven: {e}")

    # Get download URL for 2k resolution .exr
    if "hdri" not in file_info:
        raise RuntimeError(f"No HDRI data available for {asset_id}")

    hdri_info = file_info["hdri"]
    available_resolutions = [k for k in hdri_info.keys() if k.endswith("k")]
    if not available_resolutions:
        raise RuntimeError(f"No HDRI resolutions available for {asset_id}")

    # Use '2k' if available, otherwise largest available
    if "2k" in hdri_info:
        resolution = "2k"
    else:
        resolution = sorted(available_resolutions, key=lambda x: int(x[:-1]))[-1]

    exr_info = hdri_info[resolution].get("exr")
    if not exr_info or "url" not in exr_info:
        raise RuntimeError(f"No EXR download URL for {asset_id} at {resolution}")

    download_url = exr_info["url"]
    expected_bytes = exr_info.get("size")

    # Download to temp file
    try:
        req = Request(download_url, headers={"User-Agent": "curl/7.88.1"})
        with urlopen(req, context=_ssl_ctx) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is None:
                raise RuntimeError("Response missing Content-Length header")

            expected_bytes = int(content_length)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp_file:
                downloaded_bytes = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                    downloaded_bytes += len(chunk)

                tmp_path = Path(tmp_file.name)

            # Verify byte count
            if downloaded_bytes != expected_bytes:
                tmp_path.unlink()
                raise RuntimeError(
                    f"Download incomplete: got {downloaded_bytes} bytes, "
                    f"expected {expected_bytes} bytes"
                )

            # Verify file size > 1MB
            if downloaded_bytes < 1_000_000:
                tmp_path.unlink()
                raise RuntimeError(
                    f"Downloaded file too small: {downloaded_bytes} bytes"
                )

            # Move to final location
            tmp_path.rename(target_path)
            return target_path

    except URLError as e:
        raise RuntimeError(f"Failed to download HDRI from {download_url}: {e}")


def apply(scene, preset_name: str):
    """Apply HDRI environment texture to scene's world.

    Args:
        scene: Blender scene
        preset_name: Name of HDRI preset ("clear", "overcast", "dusk")

    Raises:
        ValueError: If preset_name not found
    """
    import bpy

    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown HDRI preset '{preset_name}'. "
            f"Available: {list(PRESETS.keys())}"
        )

    preset = PRESETS[preset_name]

    # Download HDRI (uses cache if available)
    hdri_path = download_hdri(preset.asset_id)

    # Get or create world
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True

    # Clear existing nodes
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # Create node tree: Output → Background → Environment Texture → Mapping → Texture Coordinate
    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (600, 0)

    bg = nodes.new("ShaderNodeBackground")
    bg.location = (300, 0)
    bg.inputs["Strength"].default_value = preset.strength
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (0, 0)
    env_tex.image = bpy.data.images.load(str(hdri_path))
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-200, 0)
    import math
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(preset.sun_azimuth))
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-400, 0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    # Set sun lamp rotation to match HDRI
    _set_sun_lamp_rotation(scene, preset)


def _set_sun_lamp_rotation(scene, preset: HDPreset):
    """Set sun lamp rotation to match HDRI's dominant light direction.

    Args:
        scene: Blender scene
        preset: HDRI preset with sun_azimuth and sun_elevation
    """
    import bpy
    import math

    # Find existing sun light
    sun_obj = None
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.data.type == "SUN":
            sun_obj = obj
            break

    if sun_obj is None:
        return  # No sun lamp to rotate

    # Convert to Blender rotation convention
    # Blender camera looks down -Z, sun rotation is elevation about X, azimuth about Z
    sun_obj.rotation_euler = (
        math.radians(90.0 - preset.sun_elevation),
        0.0,
        math.radians(preset.sun_azimuth),
    )
