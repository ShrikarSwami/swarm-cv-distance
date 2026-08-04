# HDRI Environment Lighting Design

## Overview

Switch the scan/render pipeline from EEVEE to Cycles and add HDRI environment lighting. This is step 1 of the 4b realism stack.

**Date:** 2026-07-28
**Status:** Design complete, ready for implementation

## Requirements

1. Confirm `scene.render.engine = 'CYCLES'` is set in `render_clip.py` and `batch_queue.py`
2. Replace gradient-sky node tree with HDRI environment texture
3. Source 2-3 free HDRI skies from Poly Haven (clear, overcast, dusk)
4. Store HDRIs in `assets/hdris/` (project-scoped, gitignored)
5. Download on first use with file size/checksum verification
6. Fail loudly if download fails (no silent fallback)
7. Render ONE single frame smoke test with existing cube drones
8. Confirm: HDRI sky visible, drones lit correctly, object-index EXR pass works
9. Measure wall-clock for single frame
10. Do NOT touch terrain, drone geometry, atmospherics, or motion blur

## Design

### 1. New `hdri.py` Module

**Location:** `blender_addon/hdri.py`

**HDRI Presets:**

| Name | Poly Haven Asset ID | Sun Azimuth | Sun Elevation | Strength |
|------|---------------------|-------------|---------------|----------|
| clear | blue_photo_studio | 45° | 60° | 1.0 |
| overcast | kloofendal_43d_clear_puresky | 0° | 45° | 0.8 |
| dusk | kiara_1_dusk | 90° | 15° | 1.2 |

**Note:** Sun elevation matters for dusk (low sun = long shadows). Both azimuth and elevation are stored per preset to correctly orient the sun lamp.

**Functions:**

```python
def download_hdri(asset_id: str) -> Path:
    """Download HDRI to assets/hdris/ with verification."""
    # Check cache first (assets/hdris/{asset_id}.exr)
    # Fetch from Poly Haven API: https://api.polyhaven.com/assets/{asset_id}
    # Get download URL for 2k resolution .exr
    # Download to temp file
    # Verify: downloaded_bytes == Content-Length header
    # Verify: file size > 1MB
    # Move to final location
    # Raise RuntimeError if download fails or verification fails

def apply(scene, preset_name: str):
    """Apply HDRI environment texture to scene's world."""
    # Get or create world
    # Clear existing nodes
    # Create node tree:
    #   Output → Background → Environment Texture → Mapping → Texture Coordinate
    #   Background.Strength = preset.strength
    #   Environment Texture.image = load HDRI
    #   Mapping.rotation = (0, 0, preset.sun_azimuth)
    # Set sun lamp rotation_euler = (90 - preset.sun_elevation, 0, preset.sun_azimuth)
```

**Node Tree Structure:**

```
Texture Coordinate (Generated)
    ↓
Mapping (Rotation: 0, 0, sun_azimuth)
    ↓
Environment Texture (HDRI image)
    ↓
Background (Strength: preset_strength)
    ↓
Output
```

### 2. Pipeline Integration

**Integration order:** Environment → Weather → HDRI → Render

**render_clip.py changes:**

```python
from blender_addon.hdri import apply as apply_hdri

# After weather preset application
hdri_name = cfg.get("hdri", "clear")
apply_hdri(scene, hdri_name)
```

**weather.py changes:**

```python
def apply(self, scene, hdri_active: bool = False):
    """Apply weather preset to scene.
    
    Args:
        scene: Blender scene
        hdri_active: If True, skip world background setup (HDRI handles it)
    """
    if not hdri_active:
        self._setup_world(scene)
    self._setup_sun_light(scene)
```

**Sun lamp consistency:**

- HDRI provides image-based lighting
- Keep existing sun lamp from weather.py for directional shadows
- `hdri.apply()` explicitly sets sun lamp's `rotation_euler` from preset's stored `sun_azimuth` and `sun_elevation`
- This ensures shadows point in same direction as HDRI's dominant light

### 3. Smoke Test

**Script:** `smoke_test_hdri.py`

**Steps:**

1. Load existing clip config (desert_clear)
2. Apply environment preset
3. Apply weather preset with `hdri_active=True`
4. Apply HDRI preset (clear)
5. Render single frame at 1280x720
6. Save RGB as PNG, EXR as multilayer
7. Measure wall-clock time
8. Analyze:
   - Center pixel value (cube drones)
   - Background pixel value (HDRI sky)
   - EXR pass channels (object index)
   - Shadow direction consistency with HDRI light direction

**Success criteria:**

- Center pixel ≠ background pixel (drones visible)
- EXR pass contains valid object IDs
- Shadows point toward HDRI's bright region
- Wall-clock time measured and reported (no threshold — real gating comes at full-stack settings)

## Files to Modify

1. `blender_addon/hdri.py` (new)
2. `blender_addon/weather.py` (add hdri_active parameter)
3. `render_clip.py` (call hdri.apply())
4. `smoke_test_hdri.py` (new)

## Files NOT Modified

- `batch_queue.py` (calls render_clip.py, no changes needed)
- `blender_addon/environments.py` (ground plane setup unchanged)
- Drone geometry, terrain, atmospherics, motion blur (separate tasks)

## Testing

1. Run `smoke_test_hdri.py`
2. Visually inspect render: HDRI sky visible, shadows correct
3. Check EXR pass: object IDs present
4. Measure wall-clock time
5. Commit after successful test
