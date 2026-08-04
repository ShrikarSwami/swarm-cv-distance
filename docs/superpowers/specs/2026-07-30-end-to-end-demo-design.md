# End-to-End Demo — Design Specification

**Date:** 2026-07-30  
**Status:** Approved  
**Revision:** R2 (incorporates 14 corrections from design review)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Bundle Format](#2-bundle-format)
3. [Blender Addon — Export + Framing Guard](#3-blender-addon--export--framing-guard)
4. [Pixel Detector](#4-pixel-detector)
5. [Reconstruction App](#5-reconstruction-app)
6. [Headless Sweep Harness](#6-headless-sweep-harness)
7. [Analytic Sweep (Stage 1)](#7-analytic-sweep-stage-1)
8. [Cross-Validation: Phase 1 vs Phase 2](#8-cross-validation-phase-1-vs-phase-2)
9. [Detection Viability Constraint](#9-detection-viability-constraint)
10. [PROGRESS.md Specification](#10-progressmd-specification)

---

## 1. Architecture Overview

```
Blender addon                  Bundle file              Reconstruction app
-------------                  -----------              ------------------
generate swarm         ->      images/video      ->    detect drones in pixels
place N cameras                camera poses             solve correspondence
render synced frames           intrinsics               triangulate
export bundle                  ground truth             grade vs ground truth
                                                        3D view + error overlay
```

Plus a headless sweep harness for automated batch analysis. All scoring uses the
identical evaluation code (B4) — the app and harness produce the same metrics so
numbers are comparable.

**Key interface:** the bundle format (Section 2). Everything upstream writes it;
everything downstream reads it. Frozen once written.

---

## 2. Bundle Format

### Directory structure

```
<bundle_root>/
  manifest.json
  views/
    cam_00/
      frame_0000.png
      frame_0001.png
      ...
    cam_01/
      ...
  poses.json
  ground_truth.json              (optional — flag in manifest)
  object_index/                  (optional)
    cam_00_id_.exr
    cam_01_id_.exr
    ...
```

### manifest.json

```json
{
  "bundle_version": "1.0",
  "scene_id": "demo-2026-07-30-01",
  "format": "png",
  "n_views": 8,
  "n_frames": 1,
  "frame_indices": [0],
  "image_size_px": [1920, 1080],
  "focal_px": 2666.67,
  "sensor_width_mm": 36.0,
  "units": "meters",
  "has_ground_truth": true,
  "coverage_pct": 100.0,
  "sync_convention": "all cameras render same frame indices",
  "generated_by": {
    "software": "swarm-cv-distance blender-addon",
    "commit": "e03ad35",
    "seed_swarm": 42,
    "seed_rig": 123,
    "standoff_m": 2000.0,
    "n_drones": 5,
    "geometry_class": "mixed",
    "drone_size_m": 2.0
  }
}
```

- `format`: `"png"` or `"mp4"`. Video bundles use `frame_indices` to declare
  which frames to extract. Frame indices are the canonical reference regardless
  of storage format.
- `coverage_pct`: computed by the framing guard before export.
- All camera-specific render settings (resolution, focal) are central in the
  manifest to prevent cross-view inconsistency.

### poses.json

```json
{
  "convention": "blender_c2w",
  "views": [
    {
      "view_idx": 0,
      "K": [[2666.67, 0.0, 960.0], [0.0, 2666.67, 540.0], [0.0, 0.0, 1.0]],
      "c2w": [[...4x4 Blender camera-to-world matrix...]],
      "w2c_R": [[...3x3...]],
      "w2c_t": [0.0, 0.0, 0.0]
    }
  ]
}
```

- `K`, `c2w` are always present. `w2c_R`, `w2c_t` are derivable from c2w but
  included for convenience.
- `convention` tag: `"blender_c2w"` (Blender +X right, +Y up, -Z forward).
  The app converts to OpenCV frame on import.

### ground_truth.json

```json
{
  "drone_ids": [0, 1, 2, 3, 4],
  "positions": [
    [[x0, y0, z0], [x1, y1, z1], ...],
    ...
  ]
}
```

- `positions`: (n_frames, n_drones, 3) world coordinates in ENU meters.
- Optional. Omitted for real footage. The `has_ground_truth` manifest flag tells
  the app whether to show the grading panel.

### Schema validation

A `bundle_schema.py` module in the repo root defines Pydantic models for all
three JSON files. Validation is a single call per file. The app and the addon
export share this module.

### Constraints

- **Ground truth never reaches the solver.** The solver function signature
  (`B5.triangulate_dlt(tracks, rig, detections)`) takes no truth reference.
  Ground truth is loaded by the grader only. Same discipline that kept B3 honest.
- **Frame sync is the caller's responsibility.** `manifest.json` states the
  frame indices explicitly. Unsynchronized frames break triangulation in a way
  that looks like noise rather than an error.

---

## 3. Blender Addon — Export + Framing Guard

### New operator: SWARM_OT_export_bundle

Neighbors the existing `SWARM_OT_scan` and `SWARM_OT_place_cameras` operators in
`blender_addon/swarm_scanner/__init__.py`.

**Workflow:**

1. **Generate swarm** (existing). User sets drone count, formation, seed.
2. **Place cameras** (existing). User sets camera count, geometry, standoff.
3. **Export bundle** (new). The operator:
   a. **Framing check:** calls `compute_framing_coverage_detailed()` from
      `stage1_geometry.b1_scene_rig`. If coverage < 100%, refuses to export
      with an error message listing each view whose coverage is below 100%.
      The user must adjust standoff, focal, or camera placement.
   b. **Render:** renders synchronized frames across all cameras. Cycles, same
      settings as the existing `_render_id_pass_exrs()` but renders full RGB
      frames too.
   c. **Write bundle:** creates the bundle directory, writes manifest.json,
      poses.json, ground_truth.json, views/cam_NN/frame_MMMM.png, and (if
      available) the Object Index EXR pass.
   d. **Report:** prints the bundle path, estimated wall-clock for render, and
      a coverage summary.

**Why 100% coverage for export vs 95% for the sweep flag:**

- **Export:** rendering is irreversible — a frame a camera does not see is
  permanently missing. There is no second chance to reframe.
- **Sweep flag (95%):** the analytic pipeline adds synthetic pixel noise to
  ground-truth projections. When a drone is outside the frame, it drops silently
  from that view. The remaining views still triangulate it. Below 95% coverage
  (experience from Gate 0b), the geometric dropout degrades quality measurably
  and the row is flagged as biased.

### Framing guard implementation

```python
from stage1_geometry.b1_scene_rig import (
    compute_framing_coverage_detailed,
    compute_required_standoff,
)

overall, per_view = compute_framing_coverage_detailed(truth, rig)
if overall < 1.0:
    missing_views = [v for v, cov in per_view.items() if cov < 1.0]
    raise RuntimeError(
        f"Coverage {overall*100:.0f}%: views {missing_views} see <100% of drones. "
        f"Increase standoff (requires ≥{compute_required_standoff(truth, focal_px):.0f}m) "
        "or adjust camera placement."
    )
```

### CLI variant

A `blender_addon/export_bundle_cli.py` script for headless rendering in CI.
Same logic, no Blender UI. Called via `blender -b -P export_bundle_cli.py -- <args>`.

---

## 4. Pixel Detector

A lightweight detection module in `stage1_geometry/detect_blobs.py`. No external
ML dependencies.

**Algorithm:**

```python
def detect_blobs(
    rgb: NDArray[np.uint8],         # (H, W, 3)
    drone_size_m: float,
    focal_px: float,
    standoff_m: float,
    image_width_px: int = 1920,
) -> Detections:
```

1. **Luminance:** `Y = 0.299*R + 0.587*G + 0.114*B`
2. **Threshold:** OTSU auto-threshold on luminance. Falls back to 0.1 if OTSU
   fails (uniform background).
3. **Connected components:** scikit-image `measure.label()`, 8-connectivity.
4. **Centroids:** moment-based (M01/M00, M10/M00) per labeled region.
5. **Filtration by expected apparent size:**

```python
expected_apparent_px = drone_size_m * focal_px / standoff_m
min_px = 3
max_px = 3 * max(expected_apparent_px, 3.0)
```

   - Reject components < `min_px` (noise).
   - Accept components within [`min_px`, `max_px`].
   - Count components > `max_px` as `merged_detections` and include their
     centroids (they represent occlusion that merged two drone blobs, which is a
     real failure mode).

**Validation against Object Index EXR:**

```python
id_pass = read_object_index_exr(exr_path)   # existing blender_addon/measure_id_pass.py
pixel_dets = detect_blobs(rgb_frame, ...)

recall = len(matched_drones) / len(id_pass_drones)
fp = len(pixel_dets) - recall * len(id_pass_drones)
centroid_error = mean_distance(matched_centroids)
```

Each sweep row includes: `detector_recall`, `fp_per_frame`, `centroid_error_px`,
`merged_detections`.

**Unit test for apparent size:**

```python
def test_apparent_size():
    """Assert pinhole symmetry: doubling standoff halves apparent size,
    doubling focal doubles it."""
    a1 = apparent_px(drone=0.5, standoff=1000, focal=2666.67)
    a2 = apparent_px(drone=0.5, standoff=2000, focal=2666.67)
    assert abs(a2 * 2 - a1) < 1e-6  # double standoff → half apparent px
    a3 = apparent_px(drone=0.5, standoff=1000, focal=5333.33)
    assert abs(a3 - a1 * 2) < 1e-6  # double focal → double apparent px
```

---

## 5. Reconstruction App

Standalone Python file: `reconstruction_app.py`. Run as:

```
python reconstruction_app.py [<bundle_path>]
```

- Starts FastAPI on `localhost:8820`.
- Opens browser to the Three.js single-page app.
- Without `<bundle_path>`, the page shows an upload drop zone.

### API

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/` | GET | — | Single HTML page (Three.js from CDN) |
| `POST /api/upload` | POST | multipart `.zip` bundle | `{"bundle_id", "manifest", "views", "n_frames"}` |
| `POST /api/run` | POST | `{"bundle_id", "view_indices": [0,2,5], "epipolar_threshold": 3.0, "match_threshold_m": 1.5}` | `{"reconstruction", "grading", "detection_quality"}` |
| `POST /api/export` | POST | `{"bundle_id", "result_id"}` | Bundle result debug JSON |

- `view_indices`: list of camera indices to use. Empty list = all views.
  Enables the contrast between ground-level cameras vs elevation-spread cameras
  for the same scene.
- `epipolar_threshold`: default 3.0 (same value as the sweep). Displayed
  prominently in the UI. Included in exported results for reproducibility.
- `match_threshold_m`: default 1.5 (3 drone-lengths at reference 0.5m).
  Displayed alongside the epipolar threshold.

### Backend pipeline (inside the app)

```
bundle → parse manifest + poses → for selected views:
    read images → detect blobs → Detections
    read poses → collect K, R, t for selected views → CameraRig slice
    solve_correspondence → Tracks
    triangulate_dlt → Reconstruction
    evaluate_reconstruction(Reconstruction, ground_truth, threshold=1.5m) → Grade
    return to frontend
```

No rewriting. Imports B2, B3, B4, B5 directly from `stage1_geometry`.

- `B2` for projection (not used in the app — images come from the bundle)
- `B3` for correspondence
- `B4` for scoring: precision, recall, F1, ghost count, missed count, median/p95
  error. Ghost definition = triangulated point with no close ground-truth drone
  (> match_threshold_m). Same as the sweep.
- `B5` for triangulation

### Frontend

Single HTML file, Three.js from CDN (`unpkg.com/three@0.160`), no build step.

**Layout:**

- Left: 3D viewport (THREE.Scene, OrbitControls, axes helper)
- Right sidebar:
  - Bundle info (scene_id, n_views, n_frames, focal_px, standoff)
  - View selector (N checkboxes, one per camera, pre-selected = all)
  - Epipolar threshold slider + displayed value (default 3.0)
  - Match threshold field (default 1.5m / 3 drone-lengths)
  - **Run** button
  - When results exist:
    - Grade panel: precision, recall, F1, median/p95 error, ghost count, missed
      count
    - Detection quality panel (if Object Index present): detector recall, FP,
      centroid error, merged detections
  - Timeline scrubber (when n_frames > 1)

**3D view rendering:**

- Ground-truth drones: green spheres (if `has_ground_truth`)
- Reconstructed drones: blue spheres
- Error vectors: lines from true to reconstructed, colored by magnitude (green
  < 1m, yellow 1-3m, red > 3m)
- Ghosts (reconstructed with no match): red spheres, dashed outline
- Missed drones (truth with no reconstruction): faded gray spheres
- Camera frustums: thin wireframe for each selected view

**Speed note:** the geometric solve runs in <100ms. No live rendering — Blender
prerenders the frames. The app is genuinely interactive.

### Upload UI

When no bundle path is given on the CLI, the page shows a drag-and-drop zone
(accepts `.zip`). Uploaded bundles are extracted to a server-managed temp
directory. The user can upload a bundle, select views, run the pipeline, and
download the result package. CLI path remains as a convenience for direct use.

---

## 6. Headless Sweep Harness

Single script: `stage1_geometry/sweep_b.py`.

Imports the same B2/B3/B4/B5 pipeline as the app. No UI. Outputs to
`logs/sweep_b_results.csv` and a formatted report to stdout.

### Sweep axes

| Axis | Values |
|------|--------|
| n_views | 2, 4, 6, 8, 10, 12 |
| geometry_class | all_ground, mixed, surround |
| noise_std_px | 0, 1, 3 |
| n_drones | 5 (fixed for this sweep) |

Total: 6 × 3 × 3 = 54 configs. Each runs 20 trials with independent noise seeds
to average Monte Carlo variance.

### Output CSV columns

| Column | Description |
|--------|-------------|
| n_views | Camera count |
| geometry_class | Camera placement |
| noise_std | Pixel noise standard deviation |
| n_drones | Swarm size (always 5 for this sweep) |
| focal_px | Camera focal length in pixels |
| standoff_m | Standoff distance in meters |
| coverage_pct | Framing coverage (flag if < 95%) |
| n_matched | Number of drones with match to truth |
| recall | Fraction of truth drones matched |
| ghost_count | Triangulated points with no truth match |
| precision | True positives / (TP + FP) |
| f1 | Harmonic mean of recall and precision |
| median_err_m | Median 3D position error of matched drones |
| p95_err_m | 95th percentile 3D error |
| frame_idx | Frame index (for multi-frame bundles) |
| match_threshold_m | Match distance used (1.5m) |

Rendered-run sweeps add:

| Column | Description |
|--------|-------------|
| detector_recall | Fraction of ID-pass drones detected by pixel detector |
| fp_per_frame | False positives per frame from pixel detector |
| centroid_error_px | Mean centroid localization error vs ID pass |
| merged_detections | Count of above-threshold blobs (occlusion merging) |

### Coverage flagging

- Rows with `coverage_pct < 95%` are flagged with a `⚠` prefix in the printed
  report. They are included in the CSV but marked `coverage_warning: true`.
- Justification for 95% threshold: Gate 0b showed coverage at 60-85% degraded
  the matched set and inflated error by up to 2× compared to runs with > 95%
  coverage. 95% is the empirically observed inflection point below which results
  are no longer comparable.

### Minimum inter-drone spacing constraint

**Enforced at generation time**, not at scoring. The swarm generator rejects
placements where any two drones are closer than `2 × match_threshold_m`:

```python
def generate_swarm_truth(
    n_drones, n_frames, area_km, height_range_m,
    seed, min_spacing_m=3.0,  # 2 × 1.5m default match threshold
) -> SwarmTruth:
```

Rejection sampling: for each candidate drone position, if it lands within
`min_spacing_m` of any existing drone, resample up to `max_rejection_attempts`
before falling back to jitter. Reports the achieved minimum spacing as a
warning if it falls below `2 × match_threshold_m` despite rejection sampling.

The scoring-time check is retained as a soft guardrail (warn, don't crash):

```python
min_spacing = min_pairwise_distance(truth.positions[frame])
if match_threshold_m >= 0.5 * min_spacing:
    logger.warning(
        f"match_threshold ({match_threshold_m}m) >= 0.5 × min spacing "
        f"({min_spacing}m) — wrong-drone matching possible"
    )
```

`min_spacing` is reported as a column in every output row.

---

## 7. Analytic Sweep (Stage 1)

This is the first deliverable — run *before* the bundle format freeze (it's
independent).

**Script:** `stage1_geometry/sweep_b.py` (same file as the headless harness,
with `--mode=analytic` flag or automatic detection of input source).

### Two-scale sweep

The analytic sweep runs at two scales:

| Scale | Area | Max offset (R) | Purpose |
|-------|------|----------------|---------|
| Full | AREA_KM=5.0 | ~3535m | Full geometry characterization |
| Matched | AREA_KM=0.3 | ~212m | Apples-to-apples cross-validation vs Phase 2 rendered runs |

Both use the same sweep axes (n_views, geometry_class, noise_std) and produce
separate outputs in `logs/sweep_b/` with `_full` and `_matched` suffixes.

**Process (at each scale):**

1. Generate `SwarmTruth` with 5 drones, configurable AREA_KM, HEIGHT_RANGE_M=1000.0,
   `min_spacing_m=3.0` (2 × default match threshold — enforced at generation time)
2. Generate `CameraRig` for each (n_views, geometry_class) pair
3. `project_swarm_to_detections(truth, rig, noise_std)` → synthetic Detections
4. `solve_correspondence` → Tracks
5. `triangulate_dlt` → Reconstruction
6. `evaluate_reconstruction` with match threshold 1.5m → Grade
7. Output to CSV + formatted report

**Outputs (committed to `logs/sweep_b/`):**
- `sweep_b_analytic_results_full.csv` (AREA_KM=5.0)
- `sweep_b_analytic_results_matched.csv` (AREA_KM=0.3)
- `sweep_b_report.md` — formatted summary
- `sweep_b_error_vs_views.png` — the key plot

### Prediction

Error decreases monotonically with camera count, with diminishing returns above
8 views. The mixed geometry should outperform all-ground at low camera counts
due to better elevation diversity.

---

## 8. Cross-Validation: Phase 1 vs Phase 2

**The constraint that makes this meaningful:**

From the detection viability derivation (Section 9), a rendered run can only
work when `drone_size * W / (2 * R) >= 3px`. This means Phase 2 (rendered) runs
require a different parameter range than Phase 1 (analytic).

**Cross-validation is performed at matched configurations:**

| Parameter | Phase 1 (analytic, full) | Phase 1 (analytic, matched) | Phase 2 (rendered) |
|-----------|-------------------------|----------------------------|-------------------|
| AREA_KM | 5.0 | **0.3** | **0.3** |
| n_views | 2, 4, 6, 8, 10, 12 | 2, 4, 6, 8, 10, 12 | 2, 4, 6, 8, 10, 12 |
| geometry | all_ground, mixed, surround | all_ground, mixed, surround | mixed (default) |
| n_drones | 5 | 5 | 5 |
| drone_size | 0.5m (geometric point) | 0.5m (geometric point) | 2.0m (cube asset for visibility) |
| match_threshold | 1.5m | 1.5m | 1.5m |

The matched Phase 1 column (AREA_KM=0.3) is the primary cross-validation
comparand. If Phase 2 rendered errors diverge from Phase 1 matched at the same
geometry, the difference is attributed to non-idealities in rendered images
(noise, occlusion, detection error) that the analytic model does not capture.

The full Phase 1 sweeps (AREA_KM=5.0) cover the entire parameter space for the
core geometry result and are independent of detection viability constraints.

---

## 9. Detection Viability Constraint

Framing and detection oppose each other:

```
framing (coverage) requires:  f <= W * S / (2 * R)   [1]
apparent size:                 a = d * f / S          [2]
```

Substituting [1] into [2] at the minimum standoff:

```
a_max = d * W / (2 * R)
```

**Maximum apparent drone size depends only on drone size / swarm radius × image
width.** Standoff and focal cancel.

### Worked examples (W = 1920px)

| Drone size | Swarm radius (R) | Max apparent | Viable for detector (>3px)? |
|------------|-----------------|-------------|-----------------------------|
| 0.5m | ~1414m (AREA_KM=2.0) | 0.33px | No |
| 2.0m | ~1414m (AREA_KM=2.0) | 1.34px | No |
| 0.5m | ~212m (AREA_KM=0.3) | 2.26px | Marginal |
| **2.0m** | **~212m (AREA_KM=0.3)** | **9.06px** | **Yes — comfortable** |
| 5.0m | ~707m (AREA_KM=1.0) | 6.79px | Yes |

### Design rule

For rendered validation runs, target:

```
d * W / (2 * R) >= 5 px
```

With headroom for robustness, aim for 8-10 px.

### What this means for the deliverable

- The analytic sweep (Stage 1) is **not limited by this constraint** — it adds
  synthetic pixel noise to ideal projections and works at any scale.
- The rendered validation (Phase 2) operates at **AREA_KM=0.3, drone_size=2.0m**,
  giving 9.06px maximum apparent size — well above the 3px noise floor.
- **This does not imply anything about operational detection range at true
  scale.** The 0.5m-drone-at-5km scale requires sub-pixel detection (< 1px)
  which is a separate problem (Component 5 in the original architecture). The
  rendering validation is a test of the geometry pipeline, not a detection
  feasibility demonstration.

---

## 10. PROGRESS.md Specification

The file lives at `docs/PROGRESS.md` and is updated at the end of every session
before the context fills. It must be readable cold by someone with no prior
context.

### Required sections

### 1. Current State (one paragraph)
What works, what is being worked on right now. Should answer "where does this
project stand" in 3-5 sentences.

### 2. Gate Status Table
Every gate and step, with status and the commit that closed it:

| Item | Status | Closed by |
|------|--------|-----------|
| Gate 0a — focal comparison | Open/Closed/Blocked | `<commit>` |
| Gate 0b — range sweep | Open/Closed/Blocked | `<commit>` |
| ... | ... | ... |

An item is open unless evidence closed it.

### 3. Verified Numbers
Every current headline figure, each with: the value, the config that produced it,
the date, and the commit. Any number without a run behind it does not go here.

### 4. Prediction vs Observed Log
Append-only. For each experiment: what was predicted from theory beforehand, what
was observed, the ratio, and whether it matched.

| Experiment | Predicted | Observed | Ratio | Match? |
|------------|-----------|----------|-------|--------|
| 0a error ratio (2667/1400) | 0.525 (1/focal) | 0.526 | 1.002 | ✓ |

### 5. Open Questions and Blockers
Numbered, each with what would resolve it.

### 6. Disqualified and Known-Bad
Things not to retry:

- `multiview_triangulation_test.py` correspondence is an index oracle — don't use
  it where identity-ambiguous detection is assumed
- Reprojection error is not accuracy — never substitute for 3D position error
- The CameraRig frame-conversion bug and its fix (reference commit)
- Fixed 500px detection ceiling (replaced with range-adaptive sizing)
- Frame-async data breaks triangulation silently — bundle format declares sync
  convention explicitly

### 7. Conventions and Constants
Handedness, extrinsics direction, principal point, image origin, focal_px
definition, bundle format version. Single source of truth for conventions.

### 8. Session Log
Two or three lines per session: what was attempted, what landed, what failed.
Failures included.

---

## Appendix: Match Threshold Specification

| Property | Value | Reason |
|----------|-------|--------|
| Default | 1.5m (3 drone-lengths at 0.5m) | Empirically above the p95 error at all tested configs, below minimum inter-drone spacing |
| Guard | `match_threshold < 0.5 * min_inter_drone_distance` | Prevents wrong-neighbor matching |
| Display | In drone-lengths and meters | Scale-independent meaning; editable by user |
| Scope | Same absolute value for ALL runs (analytic + rendered) | Ensures cross-validation is valid |
| Export | Included in every exported result and CSV row | Reviewer can reproduce any demo number |

## Appendix: Epipolar Threshold Specification

| Property | Value | Reason |
|----------|-------|--------|
| Default | 3.0 px | Calibrated by the existing Stage 1 pipeline |
| Display | Prominently in app, below view selectors | User sees the value at all times |
| Export | Included in exported results and every CSV row | Reproducibility |
| Guard | Not constrained by values slider can reach | User can explore but the default is published |

## Appendix: Coverage Thresholds

| Context | Threshold | Rationale |
|---------|-----------|-----------|
| Bundle export (render) | 100% | Rendering is irreversible — missed drones are permanently missing |
| Sweep flag | 95% | Gate 0b showed measurable degradation below this level. 95% is the empirically observed inflection point |
| Both | Pass is pass; explain the difference | 100% is a strict gate; 95% is a label. Both are defensible; the spec documents the distinction |
