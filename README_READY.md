# Render Pipeline: Ready to Generate

## Status: READY

The render pipeline is complete and verified. All components are working:
- ✅ Emission shader bug: already fixed in Blender 5.1.2
- ✅ Environment presets: desert, grassland, forest, city
- ✅ Weather presets: clear, overcast, hazy, dusk, night (with luminance values)
- ✅ Formation presets: grid, sphere, herringbone, light-show shapes
- ✅ Batch queue: resumable, crash-safe, configurable output root
- ✅ Config generator: full sweep (2400 clips)
- ✅ Smoke test: 5/5 clips rendered successfully

## Quick Start

### Tomorrow's Command

When your 1TB drive arrives, run:

```bash
python config_generator.py --output sweep_config.json --output-root /Volumes/1TB/drone_footage
python batch_queue.py --config sweep_config.json --output-root /Volumes/1TB/drone_footage
```

That's it! The batch queue is resumable — if it crashes, just run the same command again and it picks up where it left off.

### Custom Runs

**Specific environments/weather:**
```bash
python config_generator.py --output custom_config.json \
    --output-root /Volumes/1TB/drone_footage \
    --environments desert forest \
    --weather clear overcast \
    --n-seeds 10
python batch_queue.py --config custom_config.json --output-root /Volumes/1TB/drone_footage
```

**Check status:**
```bash
python batch_queue.py --config sweep_config.json --output-root /Volumes/1TB/drone_footage --status
```

**Smoke test (5 clips):**
```bash
python smoke_test.py --output-root dataset_smoke_test
```

## Performance

Based on smoke test (Blender 5.1.2, Apple Silicon):

| Clips | Time |
|-------|------|
| 5 (smoke test) | 15.6s |
| 100 | 5.2min |
| 500 | 26.1min |
| 1000 | 52.2min |
| 2400 (full sweep) | 125.2min (~2 hours) |

## File Structure

```
swarm-cv-distance/
├── batch_queue.py          # Resumable job runner
├── config_generator.py     # Full sweep generator
├── smoke_test.py           # 5-clip smoke test
├── blender_addon/
│   ├── environments.py     # Desert, grassland, forest, city
│   ├── weather.py          # Clear, overcast, hazy, dusk, night
│   └── formations.py       # Grid, sphere, herringbone, light-show
└── render_clip.py          # Individual clip renderer
```

## Configuration

### Environments
- `desert`: Sandy ground, hazy sky, warm sun
- `grassland`: Green ground, clear sky
- `forest`: Dark green ground, filtered light
- `city`: Grey ground, urban sky

### Weather (with luminance)
- `clear`: 8000 cd/m²
- `overcast`: 2000 cd/m²
- `hazy`: 4000 cd/m²
- `dusk`: 500 cd/m²
- `night`: 0.001 cd/m²

### Formations
- `random_cloud`: Uniform 3D distribution
- `grid`: Regular grid pattern
- `sphere`: Sphere surface
- `herringbone`: Staggered rows
- `lightshow_circle`, `lightshow_star`, `lightshow_spiral`, `lightshow_line`

### Camera Arrangements
- `dome_6`: 6 cameras, 24mm, 2km standoff
- `dome_12`: 12 cameras, 24mm, 2km standoff
- `wide_6`: 6 cameras, 16mm, 1.5km standoff
- `narrow_12`: 12 cameras, 50mm, 3km standoff

## Output Structure

```
/Volumes/1TB/drone_footage/
├── .queue_state/
│   └── queue_state.json    # Durable state (crash recovery)
├── clips/
│   ├── desert_clear_random_cloud_dome_6_s0000/
│   │   ├── frames.mkv      # FFV1/MKV video
│   │   └── clip.npz        # Ground truth
│   ├── desert_overcast_random_cloud_dome_6_s0001/
│   │   └── ...
│   └── ...
└── sweep_config.json        # Full sweep configuration
```

## Notes

- **ExFAT-safe**: Few large files (MKV videos), not many small files
- **Crash recovery**: Queue state persisted to `.queue_state/queue_state.json`
- **Configurable output root**: Point to any drive, no hardcoded paths
- **Resumable**: Run the same command again after crash, picks up where it left off
