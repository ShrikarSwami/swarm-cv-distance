# Pipeline Status — 2026-07-27

## What's Built

### Render Pipeline
- `render_clip.py` — renders a single multi-view clip (N cameras × M frames)
- `smoke_test.py` — renders 5 test clips across environments/weather
- `batch_queue.py` — resumable job runner with configurable output root
- `config_generator.py` — generates full sweep (env × weather × formation × camera × seed)

### Environment Presets (`blender_addon/environments.py`)
- Desert, grassland, forest, city — procedural, 5km×5km ground plane
- Each sets: ground color, sky background, sun light, world nodes
- Fog disabled (`fog_density=0.0`) — volumetric scatter at 0.00001 density makes Cycles renders black in 5km scenes

### Weather Presets (`blender_addon/weather.py`)
- Clear (8000 cd/m²), overcast (2000), hazy (4000), dusk (500), night (0.001)
- Modifies sun energy and world background color
- All presets produce visually distinct frames when tested in isolation

### Formation Presets (`blender_addon/formations.py`)
- Grid, sphere, herringbone, light-show shapes (circle, star, spiral, line)
- Conforms to existing `make_swarm()` interface

## What's Verified

### Geometry (12-camera rig, 2km standoff)
- Standoff: 2000m (matches M1 validated config)
- **20/20 drones have ≥2 views** (triangulation minimum met)
- Mean 4.4 views per drone
- Distribution: 0 drones with 0 views, 0 with 1 view, 5 with 2, 3 with 3, 3 with 4, 1 with 5, 6 with 6, 2 with 8
- 100% of sightlines are terrain-backed (ground plane behind every drone)
- Camera altitudes: 1244m–1979m

### Render Settings
- Samples: 32 (validated)
- Frames: 20 (validated)
- Resolution: 1920×1080
- Estimated time: ~9s per clip → 2400 clips ≈ 6 hours

## What's NOT Working (Open Bug)

### Objects invisible to camera — world background renders, nothing else does

**Symptoms:**
- World background renders correctly (different colors for different environments/weather)
- Ground plane exists in scene with correct mesh data (4 verts, 1 poly, correct normal)
- Ground plane material has correct emission settings (verified via debug output)
- Drones exist in scene with correct positions
- `hide_render=False` on all objects and collections
- **ID pass (Object Index) shows max=0.0 — NO objects visible to camera at all**
- RGB render shows only the world background color (flat, no variation)

**What I've verified is NOT the cause:**
- ~~Fog/Volume Scatter~~ — disabled, confirmed not the issue
- ~~Camera rotation/matrix_world~~ — isolated tests with identical matrix construction produce visible renders
- ~~Collection linking (context.collection vs scene.collection)~~ — both work in isolated tests
- ~~swarm_scanner interference~~ — isolated test with swarm_scanner produces visible renders
- ~~Compositor setup~~ — isolated test with compositor produces visible renders
- ~~Resolution/sample count~~ — tested at both 200×200/4 samples and 1920×1080/32 samples
- ~~display_scale~~ — tested with 1.0 and 20.0
- ~~Environment preset fog_density~~ — set to 0.0, confirmed

**What remains unknown:**
- Why the EXACT same code produces visible renders in an isolated script but not in render_clip.py
- The only structural difference is that render_clip.py reads a config file and runs via `blender --background --python`
- The isolated test (`test_single_frame.py`) uses the SAME config values and produces the SAME flat output
- But EARLIER isolated tests (`debug_dome.py`, `test_with_swarm.py`, `test_world_only.py`) produced VISIBLE renders
- The difference between working and non-working isolated tests has not been identified

**Next debugging step:**
- The issue appears to be that the ground plane and drones exist in the scene but are not being ray-traced by Cycles
- Check if there's a material node issue (Principled BSDF inputs might not be connecting correctly)
- Check if the objects are being excluded from the render by some Blender internal state
- Compare `bpy.data.objects` at render time between working and non-working scripts

## Commands for Tomorrow

```bash
# Generate sweep config
python config_generator.py --output sweep_config.json --output-root /Volumes/1TB/drone_footage

# Run batch render
python batch_queue.py --config sweep_config.json --output-root /Volumes/1TB/drone_footage
```

**DO NOT RUN until the object visibility bug is fixed.** All clips would render as flat world-background colors.
