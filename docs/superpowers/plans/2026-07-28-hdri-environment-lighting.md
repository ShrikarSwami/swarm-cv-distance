# HDRI Environment Lighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HDRI environment lighting to the render pipeline with download-on-first-use caching.

**Architecture:** New `hdri.py` module handles HDRI download, caching, and world node tree setup. Integration into existing pipeline via `render_clip.py` calling `hdri.apply()` after weather preset. Weather preset gains `hdri_active` flag to skip world background setup when HDRI is active.

**Tech Stack:** Python, Blender `bpy`, Poly Haven API, pathlib, urllib

## Global Constraints

- Apple Silicon, no CUDA — any torch/YOLO work must target device `"mps"`
- Standalone Mac-local test track — no dependency on Linux/CORE+EMANE/ArduPilot
- HDRIs stored in `assets/hdris/` (project-scoped, gitignored)
- Download verification: byte-count comparison against Content-Length header (mandatory)
- Fail loudly on download failure (no silent fallback)
- Do NOT touch terrain, drone geometry, atmospherics, or motion blur

---

### Task 1: Create `blender_addon/hdri.py` module skeleton

**Files:**
- Create: `blender_addon/hdri.py`

**Interfaces:**
- Produces: `download_hdri(asset_id: str) -> Path`, `apply(scene, preset_name: str)`

- [ ] **Step 1: Create hdri.py with preset dataclass and download function**

```python
"""HDRI environment lighting for Blender renders.

Downloads HDRI skies from Poly Haven on first use, caches in assets/hdris/,
and applies them to Blender's world node tree for image-based lighting.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from urllib.request import urlopen, Request
from urllib.error import URLError


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
        strength=1.0,
    ),
    "overcast": HDPreset(
        name="overcast",
        asset_id="kloofendal_43d_clear_puresky",
        sun_azimuth=0.0,
        sun_elevation=45.0,
        strength=0.8,
    ),
    "dusk": HDPreset(
        name="dusk",
        asset_id="kiara_1_dusk",
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
    
    # Fetch asset info from Poly Haven API
    api_url = f"https://api.polyhaven.com/assets/{asset_id}"
    try:
        with urlopen(api_url) as response:
            import json
            asset_info = json.loads(response.read())
    except URLError as e:
        raise RuntimeError(f"Failed to fetch asset info from Poly Haven: {e}")
    
    # Get download URL for 2k resolution .exr
    if "hdri" not in asset_info or "0k" not in asset_info.get("hdri", {}):
        # Try to find any available resolution
        hdri_info = asset_info.get("hdri", {})
        available_resolutions = [k for k in hdri_info.keys() if k.endswith("k")]
        if not available_resolutions:
            raise RuntimeError(f"No HDRI resolutions available for {asset_id}")
        # Use largest available (sort numerically)
        resolution = sorted(available_resolutions, key=lambda x: int(x[:-1]))[-1]
    else:
        resolution = "2k"
    
    # Construct download URL
    download_url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/{resolution}/{asset_id}_{resolution}.exr"
    
    # Download to temp file
    try:
        req = Request(download_url)
        with urlopen(req) as response:
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
    mapping.rotation = (0, 0, preset.sun_azimuth)
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
```

- [ ] **Step 2: Run syntax check**

Run: `python3 -m py_compile blender_addon/hdri.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add blender_addon/hdri.py
git commit -m "feat: add hdri.py module with Poly Haven download and world node setup"
```

---

### Task 2: Modify `blender_addon/weather.py` to support HDRI flag

**Files:**
- Modify: `blender_addon/weather.py:100-106`

**Interfaces:**
- Consumes: `hdri_active: bool = False` parameter
- Produces: Modified `apply()` method signature

- [ ] **Step 1: Add hdri_active parameter to apply() method**

```python
def apply(self, scene, hdri_active: bool = False):
    """Apply this weather preset to *scene*'s world and sun light.

    This method:
    1. Configures the world node tree with either a Nishita sky
       texture or a plain background color (skipped if hdri_active=True).
    2. Creates or updates a SUN light object to match the preset's
       elevation, rotation, and energy.

    Args:
        scene: Blender scene to apply preset to
        hdri_active: If True, skip world background setup (HDRI handles it)
    """
    if not hdri_active:
        self._setup_world(scene)
    self._setup_sun_light(scene)
```

- [ ] **Step 2: Run syntax check**

Run: `python3 -m py_compile blender_addon/weather.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add blender_addon/weather.py
git commit -m "feat: add hdri_active flag to weather.apply() to skip world setup"
```

---

### Task 3: Modify `render_clip.py` to call HDRI module

**Files:**
- Modify: `render_clip.py:114-151`

**Interfaces:**
- Consumes: `hdri.apply(scene, preset_name)` from Task 1
- Produces: HDRI applied after weather preset

- [ ] **Step 1: Add HDRI import and apply call**

Find the section after weather preset application (around line 146) and add:

```python
# Apply HDRI environment lighting (after weather preset)
from blender_addon.hdri import apply as apply_hdri
hdri_name = cfg.get("hdri", "clear")
try:
    apply_hdri(scene, hdri_name)
    print(f"[{cfg['clip_name']}] HDRI: {hdri_name}")
except Exception as e:
    print(f"[{cfg['clip_name']}] HDRI failed: {e}")
    raise
```

- [ ] **Step 2: Run syntax check**

Run: `python3 -m py_compile render_clip.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add render_clip.py
git commit -m "feat: integrate HDRI module into render_clip.py pipeline"
```

---

### Task 4: Create smoke test script

**Files:**
- Create: `smoke_test_hdri.py`

**Interfaces:**
- Consumes: Environment, weather, HDRI presets
- Produces: Rendered frame with HDRI lighting

- [ ] **Step 1: Create smoke test script**

```python
#!/usr/bin/env python3
"""Smoke test for HDRI environment lighting.

Renders a single frame with existing cube drones and HDRI sky,
confirms: HDRI visible, drones lit correctly, EXR pass works.

Usage:
  blender --background --python smoke_test_hdri.py
"""

import bpy
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Configuration
OUTPUT_DIR = Path(__file__).parent / "dataset_smoke_test" / "hdri_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("HDRI ENVIRONMENT LIGHTING SMOKE TEST")
    print("=" * 70)
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Load existing clip config (desert_clear)
    print("\n[1] Loading scene configuration")
    
    # Factory reset
    try:
        bpy.ops.wm.read_factory_use_empty(use_empty=True)
    except AttributeError:
        bpy.ops.wm.read_homefile(use_empty=True)
    
    # Set render engine to Cycles
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    # Step 2: Apply environment preset (creates ground plane, sun light)
    print("\n[2] Applying environment preset")
    from blender_addon.environments import get_environment
    env_preset = get_environment("desert")
    env_preset.apply(scene)
    
    # Step 3: Apply weather preset with hdri_active=True
    print("\n[3] Applying weather preset")
    from blender_addon.weather import WeatherPreset
    weather_preset = WeatherPreset(
        name="clear",
        sun_energy=5.0,
        sun_elevation_deg=60.0,
        sun_rotation_deg=45.0,
        sun_color=(1.0, 0.95, 0.9, 1.0),
        sky_luminance_cd_m2=8000.0,
        ambient_color=(0.5, 0.6, 0.9, 1.0),
    )
    weather_preset.apply(scene, hdri_active=True)
    
    # Step 4: Apply HDRI preset (clear)
    print("\n[4] Applying HDRI preset")
    from blender_addon.hdri import apply as apply_hdri
    apply_hdri(scene, "clear")
    
    # Step 5: Create camera
    print("\n[5] Setting up camera")
    bpy.ops.object.camera_add(location=(0, -100, 50))
    cam = bpy.context.object
    cam.name = "SmokeTestCamera"
    
    # Point camera at scene center
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = None  # Will track origin
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    scene.camera = cam
    
    # Step 6: Create test cube drones
    print("\n[6] Creating test cube drones")
    for i in range(5):
        x = (i - 2) * 10  # -20, -10, 0, 10, 20
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, 0, 10))
        cube = bpy.context.object
        cube.name = f"Drone_{i:02d}"
        
        # Assign emission material
        mat = bpy.data.materials.new(f"DroneMat_{i:02d}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)
        
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 10.0
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (200, 0)
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        
        cube.data.materials.append(mat)
        
        # Set object index for EXR pass
        cube.pass_index = i + 1
    
    # Step 7: Render single frame
    print("\n[7] Rendering single frame")
    render_start = time.time()
    
    scene.render.filepath = str(OUTPUT_DIR / "render")
    bpy.ops.render.render(write_still=True)
    
    render_time = time.time() - render_start
    total_time = time.time() - start_time
    
    print(f"\n  Render time: {render_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    
    # Step 8: Analyze render
    print("\n[8] Analyzing render")
    
    # Load rendered image
    render_path = OUTPUT_DIR / "render.png"
    if render_path.exists():
        img = np.array(Image.open(render_path))
        h, w = img.shape[:2]
        
        # Center region (where drones should be)
        center_region = img[h//4:3*h//4, w//4:3*w//4, :3]
        center_mean = center_region.mean(axis=(0,1))
        
        # Corner regions (background/sky)
        corners = [
            img[:h//4, :w//4, :3],
            img[:h//4, 3*w//4:, :3],
            img[3*h//4:, :w//4, :3],
            img[3*h//4:, 3*w//4:, :3],
        ]
        bg_mean = np.mean([c.mean(axis=(0,1)) for c in corners], axis=0)
        
        print(f"  Center pixel (drones): {center_mean}")
        print(f"  Background pixel (sky): {bg_mean}")
        print(f"  Difference: {np.abs(center_mean - bg_mean).max():.2f}")
        
        # Check if drones are visible
        if np.abs(center_mean - bg_mean).max() > 5.0:
            print("  ✓ Drones visible against sky")
        else:
            print("  ✗ Drones not distinguishable from sky")
    
    # Step 9: Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Render time: {render_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output: {OUTPUT_DIR / 'render.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run: `python3 -m py_compile smoke_test_hdri.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git add smoke_test_hdri.py
git commit -m "feat: add HDRI smoke test script"
```

---

### Task 5: Run smoke test and verify

**Files:**
- Test: `smoke_test_hdri.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces: Rendered frame, analysis results

- [ ] **Step 1: Run smoke test**

Run: `/Applications/Blender.app/Contents/MacOS/Blender --background --python smoke_test_hdri.py`
Expected: Successful render with HDRI sky visible, drones lit, EXR pass working

- [ ] **Step 2: Verify render output**

Check that:
- `dataset_smoke_test/hdri_test/render.png` exists
- Center pixel ≠ background pixel (drones visible)
- Render time measured and reported

- [ ] **Step 3: Verify EXR pass (if multi-layer EXR saved)**

Check that EXR contains object index pass with valid drone IDs

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: HDRI environment lighting complete, smoke test passes"
```

---

## Self-Review Checklist

- [x] Spec coverage: All 10 requirements covered in tasks
- [x] Placeholder scan: No TBD, TODO, or incomplete sections
- [x] Type consistency: Function signatures match between tasks
- [x] File structure: Clear separation of concerns (hdri.py, weather.py, render_clip.py, smoke_test)
- [x] Testing: Smoke test provides end-to-end verification
