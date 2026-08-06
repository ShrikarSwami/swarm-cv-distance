# Model Details — Swarm Reconstruction Tool

This document describes both reconstruction backends: the **geometric baseline**
(working, default) and the **learned T6 model** (in development, not yet
passing G2). It is written for the Chief Scientist or any engineer evaluating
the pipeline.

---

## Geometric Backend (default)

### Pipeline

```
PNG images (V views)
    │
    ▼
blob detection (scikit-image Laplacian of Gaussian)
    │  ─ detects bright drones against dark sky at known scale
    │  ─ drone_size_m=0.5, focal_px=2666.67, standoff_m≈139
    │
    ▼
epipolar correspondence (b3_correspondence.py)
    │  ─ computes fundamental matrices from camera poses (K, R, t)
    │  ─ matches detections across view pairs via epipolar distance
    │  ─ assembles multi-view tracks (≥2 views per track)
    │  ─ epipolar_threshold = 3.0 px
    │
    ▼
DLT triangulation (b5_triangulation.py)
    │  ─ Direct Linear Transform on each track
    │  ─ SVD-based solution, filtered by reprojection error < 50 px
    │
    ▼
3D positions (N, 3) ENU metres
```

### Camera Rig

- 24 views: 8 ground-level (elevation ~0°), 8 mid-level (~30°), 8 aerial (~60°)
- 1920×1080 resolution, ~50mm-equivalent focal length (2666.67 px)
- OpenCV camera convention (+X right, +Y down, +Z forward)
- World frame: ENU (East, North, Up) in metres
- Standoff distance: ~139 m from swarm centre

### Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| Detection speed | ~0.1 s/view |
| Triangulation speed | < 1 s for 24 views, ~50 drones |
| Median position error | ~0.8–1.5 m (varies with n_drones and view count) |
| Detector recall | ~0.85–0.95 at 5 px radius |
| Count error | Typically undercounts (ghost tracks filtered by epipolar consistency) |

### Limitations

- **No identity:** Tracks are anonymous — the correspondence solver groups
  detections that are geometrically consistent, but cannot link a track to a
  specific drone ID across frames.
- **Blob detector:** Laplacian of Gaussian detects bright blobs at a known
  scale. It works well on clean rendered drones against a dark sky but is not
  robust to real-world clutter, occlusion, or varying illumination.
- **DLT only:** No nonlinear refinement (Levenberg-Marquardt) in this path.
  The frozen control uses pure DLT for deterministic, reproducible results.
- **No per-point confidence:** The geometric backend does not produce
  confidence estimates for individual detections.

---

## Learned Backend (T6 Voxel-Fusion Model)

### Status: NOT YET PASSING G2

As of 2026-08-06, the learned model fails the G2 acceptance gate. It is
included as a stub so the pluggable interface is wired from day one, but
`reconstruct()` currently returns empty results.

See `ml/FIX_QUEUE.md` in the main repository for the current diagnosis and
active fix queue.

### Architecture

```
PNG images (V views)
    │
    ▼
Shared-weight 2D CNN encoder (stride 4)
    │  ─ 2× stride-2 Conv2d stages → (V, 64, 270, 480) feature maps
    │  ─ 2× stride-1 refinement stages
    │
    ▼
Voxel back-projection
    │  ─ Each feature pixel → world 3D point via camera intrinsics (K)
    │    and extrinsics (w2c_R, w2c_t)
    │  ─ Bilinear sampling (F.grid_sample) populates a shared 64³ voxel grid
    │
    ▼
Symmetric pooling (mean + max, concatenated)
    │  ─ Joint-permutation invariant: sorting (view, camera) pairs by pose
    │    key guarantees bitwise-identical output for any input ordering
    │
    ▼
Small 3D CNN decoder
    │  ─ 2× Conv3d (64→32 channels) with GroupNorm + ReLU
    │  ─ 1×1 Conv3d → Softplus head (values ≥ 0)
    │
    ▼
3D occupancy heatmap (64, 64, 64)
    │
    ▼
extract_positions()
    │  ─ 3×3×3 local maxima above threshold
    │  ─ Clustering (merge within 1.5 voxels)
    │  ─ Per-peak soft-argmax over 5×5×5 neighbourhood
    │
    ▼
3D positions (K, 3) ENU metres + per-peak confidences
```

### Key Design Decisions

1. **Joint-permutation invariance:** The model is invariant to arbitrary
   reordering of (view, camera) pairs by sorting on a pose-derived canonical
   key before pooling. The pairing IS load-bearing: mispairing a view with
   the wrong camera changes which world points are sampled, and hence the
   fused output.

2. **Mean+max pooling:** Both mean and max are computed across views,
   concatenated. Max captures strong single-view evidence; mean captures
   multi-view consensus.

3. **Softplus head:** `Softplus(x) = log(1 + exp(x))` guarantees strictly
   positive output (≥ 0), satisfying the contract while remaining
   differentiable everywhere.

4. **Soft-argmax refinement:** Each detected peak is refined to sub-voxel
   precision by a weighted centroid over its 5×5×5 neighbourhood.

### Current Failure Mode (2026-08-06)

The model produces a **diffuse, low-contrast output** with hundreds of
spurious local maxima:

| Metric | Baseline (stride-8, count term) | FIX-06 (no count term) |
|--------|-------------------------------|------------------------|
| count_err | +280 to +356 | +451 to +485 |
| median_err_m | 1.14 m | 0.58 m |
| peak-to-background | 2.9:1 | 15.8:1 |

**Root cause (discovered 2026-08-06):** The count loss term (measuring total
output mass) dominated the MSE shape term by 4–6 orders of magnitude. The
model learned to suppress ALL output globally rather than form peaks at
correct positions.

**Current state:** FIX-06 (count_weight=0 control) proved the model CAN form
sharp peaks (15.8:1 contrast) and localise well (0.58 m) when the count term
is removed. FIX-07 (normalized per-drone count term) is queued to balance
the two terms.

See the main repository's `ml/FIX_QUEUE.md` for the active fix queue.

### Expected Performance (when trained)

| Metric | Target |
|--------|--------|
| median_err_m | < 1.0 m |
| count_err per scene | within [-1, +1] |
| peak-to-background | > 10:1 |
| Inference speed | ~0.5–1.0 s/scene (MPS) |

---

## Comparison

| Property | Geometric | Learned (target) |
|----------|-----------|-------------------|
| Works now? | ✅ Yes | ❌ No (G2 not passing) |
| Per-point confidence | ❌ No | ✅ Yes (peak heatmap values) |
| Robust to clutter | ❌ No (blob detector) | ✅ Yes (learned features) |
| Requires GPU? | ❌ No | ✅ Yes (torch, MPS/CUDA) |
| Requires training? | ❌ No | ✅ Yes (~600+ steps) |
| Identity across frames? | ❌ No | ❌ No (not in scope) |
| Speed (24 views) | ~2 s | ~1 s (with GPU) |

---

## References

- Chen et al., "Countering Large-Scale Drone Swarm Attack by Efficient
  Splitting" (IEEE TVT 2022) — the GA/PSO pipeline this tool feeds.
- `docs/superpowers/specs/ml_contracts.md` — frozen I/O contract for the
  learned model.
- `docs/handoff_summary_cv_distance_pivot_20260722.md` — project design
  rationale.
