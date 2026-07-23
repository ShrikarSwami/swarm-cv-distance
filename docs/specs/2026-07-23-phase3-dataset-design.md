# Phase 3 Design: Synthetic Multi-View Dataset + ML Distance Estimation

**Date:** 2026-07-23
**Status:** Approved for implementation (M1–M2 continuous)

## 1. Parameter Grid

### Primary derived quantity

**Angular resolution** [µrad/px] = sensor_width / (focal_length × h_pixels). This determines apparent drone size regardless of sensor format.

**Apparent pixel size** on 0.5m drone at standoff `d`: `px = 0.5 / (d × θ_rad/px)`

**Sanity check:** Two configs with identical angular resolution must produce identical apparent pixel sizes. If not, the FOV math is wrong.

### Sensor classes (real specs)

| Class | Sensor width | Example bodies |
|---|---|---|
| Full-frame | 36.0 mm | Canon R5 (8192×5464, 45MP), Nikon Z8 |
| APS-C | 23.5 mm | Canon R7 (6000×4000, 24MP), Nikon Z50 |
| 1-inch | 13.2 mm | Sony RX10 IV (5472×3648) |
| 1/2.3-inch | 6.17 mm | Nikon P1000 (4608×3456) |

### Focal lengths (actual, not equivalent)

| Label | Full-frame | APS-C | 1-inch | 1/2.3" |
|---|---|---|---|---|
| F0 | 24 mm | 24 mm | 24 mm | 24 mm |
| F1 | 50 mm | 50 mm | 50 mm | 50 mm |
| F2 | 100 mm | 100 mm | 100 mm | 100 mm |
| F3 | 200 mm | 200 mm | 200 mm | 200 mm |
| F4 | 400 mm | 400 mm | 400 mm | 400 mm |
| F5 | 800 mm | 800 mm | — | — |
| F6 | 1200 mm | — | — | — |
| P1000 max | — | — | — | 539 mm (actual) |

### Standoff distances

| Label | Distance | Notes |
|---|---|---|
| S0 | 500 m | Close-range airborne |
| S1 | 750 m | Near-boundary airborne |
| S2 | 1,000 m | Short-range airborne |
| S3 | 2,000 m | Mid-range airborne / close ground |
| S4 | 3,000 m | Transition zone |
| S5 | 5,000 m | Current scene scale (M3/M4 baseline) |
| S6 | 7,500 m | Extended ground range |
| S7 | 10,000 m | Outer limit |

### Resolution tiers

| Label | Pixels | Class |
|---|---|---|
| R0 | 1280×720 | Budget |
| R1 | 1920×1080 | Standard |
| R2 | 6000×4000 (~24MP) | High-res |
| R3 | 8192×5464 (~45MP) | Ultra-res |

## 2. Platform Tiers

### Tier A — Ground post (unconstrained optics)

- Sensor classes: full-frame + APS-C
- Focal lengths: all labeled for that sensor class
- Standoffs: S0–S7
- Resolutions: R0–R3
- Mains power, active cooling, tripod-stable
- Includes 800mm/1200mm + 45MP combinations
- Unconstrained: if ground doesn't work here, it doesn't work period

### Tier B — Airborne observer (UAS-constrained)

- Sensor classes: APS-C + 1-inch
- Focal lengths: up to 200 mm actual (flag >200mm as "exceeds realistic UAS payload")
- Standoffs: S0–S5 (flag >5km as "unlikely airborne range")
- Resolutions: R0–R2
- Weight/power/stability constrained

### Tier C — Cheap commodity (both regimes)

- Sensor classes: 1/2.3" + 1-inch
- Focal lengths: up to P1000 max (539mm actual)
- Standoffs: S0–S7
- Resolutions: R0–R1
- Floor-tier baseline

## 3. Render Methodology

Renders measure what analytics cannot: occlusion, edge-on viewing, FOV-boundary dropout, clipping.

**Approach:** Static snapshot using `make_swarm()` + ID-pass pipeline. For each render config: generate swarm → place cameras → render Object Index pass → extract per-drone centroid from EXR → measure pixel size and visibility.

**Targets:** ~20–30 boundary configs per platform tier. 2–3 anchor configs for model validation. At least 1–2 low-elevation ground configs to settle edge-on question.

**Cost:** ~1–2 min per config at 1080p Cycles. ~30 configs ≈ 1 hour.

## 4. Metrics and Decision Framework

### Detector-class bands (rules of thumb, not hard physics)

| Band | Threshold | Caveats |
|---|---|---|
| Bounding-box detector (YOLO/RT-DETR scale) | ≥8 px | COCO-scale training heuristic; detector-specific |
| Centroid with known target size | ≥3–5 px | Viable with known dimensions and clean background; degrades with terrain clutter |
| Sub-pixel template/temporal | <3 px | Requires known target size, clean background (sky), or temporal consistency. Drone against cluttered terrain ≠ drone against sky |

### Decision output

Per platform tier: maximum standoff at which apparent pixel size exceeds each detector-class band. Visualized as a curve: max range vs. pixel threshold, one curve per tier.

### Temporal integration note

Moving drone across frames is detectable at smaller per-frame apparent size than single-frame threshold implies. Could extend effective range by ~1.5–2×. Flagged for revisitation once dataset temporal characteristics are known. Not modeled in M1.

## 5. On-Disk Dataset Schema

### Directory layout

```
dataset/
├── metadata.json          # scene config, D_MAX, camera catalog
├── splits.json            # train/val/test clip assignments
├── clips/
│   ├── desert_001/
│   │   └── clip.npz       # K, extrinsics, positions, meta
│   ├── desert_002/
│   │   └── clip.npz
│   └── ...
```

### clip.npz keys

- `K`: float64, shape `(n_views, 3, 3)` — camera intrinsics
- `extrinsics`: float64, shape `(n_views, 4, 4)` — camera-to-world transforms
- `positions`: float32, shape `(n_frames, n_drones, 3)` — Blender world-space XYZ
- `meta`: dict — environment, weather, formation, seed, display_scale, drone_size_m, standoff, focal_length, sensor_width, resolution

### Derived quantities (computed on demand, not stored)

- Pairwise distances: derived from positions
- Adjacency matrix: distances thresholded at D_MAX (tunable at eval time)

### Loader function

```python
def load_clip(clip_path, d_max=3949.0):
    data = np.load(clip_path, allow_pickle=True)
    positions = data['positions']  # (F, N, 3)
    distances = np.linalg.norm(positions[:, :, None, :] - positions[:, None, :, :], axis=-1)
    adjacency = distances <= d_max
    return {
        'K': data['K'],
        'extrinsics': data['extrinsics'],
        'positions': positions,
        'distances': distances,
        'adjacency': adjacency,
        'meta': data['meta'].item(),
    }
```

### ExFAT compatibility

- No journaling: each clip written atomically (npz first, then video)
- Poor small-file performance: no loose frames on master drive
- Lossless master: FFV1/MKV (verified bit-exact before M3)
- Training frames decoded from video to scratch/local SSD, never stored on ExFAT master

### Resumability

Each clip is self-contained. Crash mid-clip loses that clip only; pipeline resumes from next unrendered index.

## 6. Deliverables

### M1

1. Analytical sweep script with sanity check
2. ~30 boundary renders
3. Decision report (per-tier max range at each pixel threshold)
4. Dataset schema spec (this document)
5. Updated PROGRESS.md

### M2

1. Schema implementation (loader, FFV1/MKV verification)
2. ~20-clip smoke-test dataset (≥2 environments, ≥2 weather conditions)
3. Ground truth validation (project positions → pixel coords, compare to rendered)
4. Per-clip render cost and M3 ETA

## 7. Assumptions and Caveats

- Detector-class thresholds are rules of thumb from COCO-scale training, not measured constraints
- Sub-pixel detection assumes clean background or temporal consistency (not unconditional)
- D_MAX = 3949m is provisional (85% target, recalibrated for 5km scene)
- Temporal integration could extend range by ~1.5–2× (not modeled, flagged for revisitation)
- Sensor specs from manufacturer datasheets; real-world performance may vary
- P1000 539mm is actual focal length at max zoom, not 35mm equivalent
