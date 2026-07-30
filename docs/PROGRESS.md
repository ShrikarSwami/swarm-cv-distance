# PROGRESS — Swarm CV Distance End-to-End Demo

## 1. Current State

Task 1 (B4 unfreeze) is complete. B4 scoring now classifies every track into four
categories: matched, imprecise, false_track, missed. Seven hand-constructed control
tests pass, including fixtures for each category. The gate check at 1px noise,
matched scale shows ghosts peak at 2 views (all imprecise, no false tracks), with
the mechanism being positional error from DLT, not correspondence ambiguity.

All 41 prediction tests pass (2 new tests added for imprecise and false_track
controls). All acceptance commands exit 0.

---

## 2. Gate Status Table

| Item | Status | Closed by |
|------|--------|-----------|
| Gate 0a — focal comparison | Closed | `bc6b4b9` (Stage A fix) |
| Gate 0b — range sweep | Closed | `e03ad35` (Stage B) |
| B1 SwarmGen + framing | Closed | `e03ad35` |
| B2 Projection | Closed | `e03ad35` |
| B3 Correspondence | Closed | `e03ad35` |
| B4 Scoring (unfreeze) | Closed | `bcc6dc7` |
| B5 Triangulation | Closed | `e03ad35` |
| 7. Analytic sweep | Closed | `01162d4` |
| 2. Bundle format + schema | Closed | `be4ed7b` |
| 3. Blender addon export + framing guard | Closed | `f8e1a00` |
| 4. Pixel detector | Closed | `52a6322` |
| 6. Headless harness | Closed | `c1e6031` |
| 5. Reconstruction app | Closed | `954430a` |
| 8. Cross-validation | In progress | Pending rendered sweep |
| #30a Render wall-clock | Closed | `614f444` (estimated) |
| #30b Rendered sweep sizing | Closed | `614f444` (estimated) |
| #30 Rendered sweep | Open | | |

---

## 3. Verified Numbers

- **7. Analytic sweep (2026-07-30):**
  - Matched scale (0.3km): all 18 zero-noise configs achieve recall=1.0 (perfect);
    at 1px noise, configs with n_views >= 6 reach recall=1.0; at 3px noise, only
    n_views >= 12 with surround geometry achieves recall > 0.6.
  - Full scale (5.0km): all 54 configs have coverage < 95% (swarm spread > camera
    FOV at 2000m standoff); despite this, 2 zero-noise configs (8v and 10v all_ground)
    still achieve recall=1.0.
  - Prediction validated: at matched scale with 1px noise, median error decreases
    monotonically with n_views (all_ground: 0.96m at 2v -> 0.45m at 12v), with
    diminishing returns above 8 views (2.6% improvement from 8v to 12v).
  - Prediction partially validated: surround geometry outperforms all_ground at 2
    views for 1px noise (recall 0.83 vs 0.38), confirming elevation diversity benefit.
  - 108 rows per CSV, 19 columns (median_err_m reported as mean + std across 20 trials).
- **#30a Render wall-clock (2026-07-30):**
  - Average 11.3s per 1920x1080 Cycles frame at 128 samples on Apple Silicon (M4).
  - Full rendered sweep at AREA_KM=0.3, mixed geometry: 42 views × ~15s ≈ 10.5 min.
  - Sweep is feasible at this refresh rate.
- **4. Pixel detector (2026-07-30):**
  - `apparent_px` formula validated: `0.5 * 2666.67 / 1000 = 1.333` — symmetry confirmed
    (doubling standoff halves apparent size, doubling focal doubles it).
  - Detection pipeline passes: 41/41 prediction tests (3 TestPixelDetector tests
    previously skipped now all PASS, 2 new B4 control tests added).
- **Task 1: B4 four-category classification (2026-07-30):**
  - Gate check at 1px noise, matched scale, mixed geometry:
    - 2 views: matched=4.0, imprecise=1.0, false_track=0.0, n_ghost=1.0
    - 4 views: matched=5.0, imprecise=0.0, false_track=0.0, n_ghost=0.0
    - 6 views: matched=5.0, imprecise=0.0, false_track=0.0, n_ghost=0.0
    - 8 views: matched=5.0, imprecise=0.0, false_track=0.0, n_ghost=0.0
  - all_ground at 2 views: imprecise=2.6 (highest ghost count)
  - False tracks essentially absent at all camera counts for 5-drone configs
  - Mechanism: DLT from 2 noisy rays produces positional error, NOT correspondence ambiguity
  - Seven control tests pass: perfect, scrambled, single_view, known_offset, ghosts, imprecise, false_track
- **Task 2: Intersection-set metrics (2026-07-30):**
  - Intersection_n at 1px noise: all_ground=1, mixed=3, surround=2
  - all_ground 6v uptick: survives but intersection_n=1 makes it meaningless
  - mixed: monotonically decreasing (no upticks)
  - surround 6v/12v upticks: survive (same 2 drones, real error variation)
  - Intersection very small (1-3 drones) → error estimates have high variance
- **Task 3: Full-scale sweep dropped (2026-07-30):**
  - Required standoff for 5.0km area: 6846m (compute_required_standoff)
  - At 7000m standoff: coverage = 96.7% (still below 100%)
  - At fixed 2000m standoff: coverage < 95% for ALL configs
  - Decision: drop full-scale sweep, keep matched-scale (0.3km) only
  - Reason: swarm too spread out for camera FOV at practical distances
- **Task 4: Ghost assertion restored (2026-07-30):**
  - 2v all_ground 1px: imprecise > 0 (DLT from 2 noisy rays → positional error)
  - 8v all_ground 1px: imprecise = 0 (more baselines eliminate ghosts)
  - Original prediction refuted: ghosts peak at 2v, not at 4+ views
  - Mechanism: positional error from poor triangulation geometry, NOT correspondence ambiguity
- **Density sweep: false_track never appears (2026-07-30):**
  - false_track = 0 at ALL trials (5, 10, 15 drones), ALL geometries, ALL camera counts
  - Even at 15 drones × 2 views × 1px noise: every ghost is imprecise, not false_track
  - Correspondence is trivially solved by epipolar geometry at all tested densities
  - Camera count buys triangulation precision only, not correspondence
  - Prediction refuted: false_track does NOT become nonzero at 15 drones

---

## 4. Prediction vs Observed Log

| Experiment | Predicted | Observed | Ratio | Match? |
|------------|-----------|----------|-------|--------|
| Ghost behavior vs camera count | Ghost count is zero or near-zero at 2–3 views (Hungarian one-to-one matching forces bijection with no room for spurious tracks), rises at 4+ views as combinatorics create overlapping tracks, then falls as views increase further. Consequence: if ghosts cannot arise at low camera counts, wrong pairings surface as position error instead, and the camera-count curve tells a different story than one that includes ghost-driven qualitative regimes. | Zero noise: ghosts=0 at all configs. 1px noise: ghosts peak at 2 views (all_ground 2.85), decrease rapidly to zero at 6+ views. 3px noise: ghosts peak at 4 views (4.8), then decrease gradually. Wrong pairings at 2 views surface as ghosts AND position error simultaneously — DLT from 2 noisy rays can place the triangulated point far enough from truth to exceed match_threshold. **Four-category split (Task 1):** At 2 views, ghosts are ALL imprecise (correct identity, bad position). Zero false tracks at any camera count. Mechanism: DLT from 2 noisy rays → positional error, NOT correspondence ambiguity. | | Partial: zero-noise behavior confirmed (ghosts=0 everywhere). Low-noise regime reverses the predicted shape — ghosts peak at 2 views, not at 4+. The one-to-one Hungarian constraint prevents spurious TRACKS but not spurious 3D positions from 2-view triangulation. **Mechanism refuted:** original prediction said correspondence ambiguity causes ghosts; actual cause is positional error from poor triangulation geometry. |
| Spec Section 7: error vs views | Error decreases monotonically with camera count, with diminishing returns above 8 views. Mixed geometry outperforms all_ground at low camera counts due to elevation diversity. | Matched scale 1px noise: 0.96m (2v) -> 0.45m (12v) monotonically; 2.6% improvement 8v->12v. Surround recall 0.83 > all_ground 0.38 at 2v. | 2.1x (error), 2.2x (recall) | Validated |
| Intersection-set monotonicity | Upticks at all_ground 6v and surround 6v/12v are artifacts of comparing different drone sets | all_ground intersection_n=1 (meaningless); mixed intersection_n=3 (monotonic); surround intersection_n=2 (upticks survive) | N/A | Partial: upticks real but intersection too small for statistical meaning |
| Density: false_track vs n_drones | false_track remains zero at 5 drones, becomes nonzero at 15 drones and low camera counts. The density at which false_track appears falls as camera count rises. | false_track = 0 at ALL trials (5, 10, 15 drones), ALL geometries, ALL camera counts at 1px noise. Even at 15 drones × 2 views, every ghost is imprecise (correct identity, bad position), not false_track. | N/A | **Refuted:** false_track never appears. Correspondence is trivially solved by epipolar geometry. Camera count buys triangulation precision only. |

---

## 5. Open Questions and Blockers

1. **Phase 2 rendered sweep (#30):** Not run yet. Estimated ~10.5 min for 6 configs
   (n_views 2,4,6,8,10,12, mixed geometry, AREA_KM=0.3, 128 samples, 1920x1080).
   Requires creating Blender scenes, rendering, and processing through the headless
   harness. Cross-validation compares against the analytic matched-scale CSV already
   in `logs/sweep_b/`.

2. **Cross-validation (Section 8):** Frame-comparison scripts and metric tables are
   not yet built. The comparison between analytic (matched) and rendered Phase 2
   results requires matching n_views-level rows and computing error ratios.

---

## 6. Disqualified and Known-Bad

- `multiview_triangulation_test.py` correspondence is an index oracle — don't use
  it where identity-ambiguous detection is assumed
- Reprojection error is not accuracy — never substitute for 3D position error
- The CameraRig frame-conversion bug and its fix (reference `d7bc4eb`)
- Fixed 500px detection ceiling (replaced with range-adaptive sizing)
- Frame-async data breaks triangulation silently — bundle format declares sync
  convention explicitly

---

## 7. Conventions and Constants

| Convention | Value | Source |
|------------|-------|--------|
| Handedness | Blender (+X right, +Y up, -Z forward) | `CONVENTION_TAG = "blender_c2w"` |
| Extrinsics direction | camera-to-world (c2w) in bundle; converted to w2c for solver | `data_contract.py` |
| Principal point | (960, 540) | `PRINCIPAL_POINT` |
| Image origin | top-left | OpenCV convention |
| focal_px definition | `f * W / sensor_width_mm` | `make_K()` |
| Bundle format version | `1.0` | spec Section 2 |
| Default focal | 2666.67 px | `DEFAULT_FOCAL_PX` |
| Image size | 1920 × 1080 | `IMAGE_SIZE` |

---

## 8. Session Log

| Date | Attempt | What landed | What failed |
|------|---------|-------------|-------------|
| 2026-07-30 | Session 1 — orchestrated build | PROGRESS.md created, orchestrator started | |
| 2026-07-30 | Session 2 — analytic sweep | B-Sweep analytic sweep complete: `de6e21c` — sweep_b.py, CSV, plot, report in logs/sweep_b/ | |
| 2026-07-30 | Session 2b — column fix | median_err_mean → median_err_m, .gitignore fix: `01162d4` | |
| 2026-07-30 | Session 3 — bundle schema | `bundle_schema.py` with Pydantic v2 models (BundleManifest, BundlePoses, BundleGroundTruth, CameraView) + `bundle_minimal()` fixture + `validate_file()` methods. Commit `be4ed7b`. | |
| 2026-07-30 | Session 4 — export + framing guard | `SWARM_OT_export_bundle` operator in addon + `export_bundle_cli.py`. Framing check at 100% threshold with exact spec error message. RGB + ID EXR rendering per view. Bundle JSON files via `bundle_schema`. Commit `f8e1a00`. |
| 2026-07-30 | Session 5 — pixel detector | `detect_blobs.py` with luminance threshold, OTSU, connected components, centroid extraction, size filtration. Unit tests restored from skip to pass (39/39). Edge cases covered. Commit `52a6322`. | |
| 2026-07-30 | Session 6 — headless harness | `sweep_b.py` extended with `--mode=headless`, `--bundle-dir`, `--output`, `--test-synthetic`. New functions: `bundle_poses_to_rig`, `load_ground_truth`, `detect_from_bundle_views`, `_compute_detector_quality`, `process_bundle`, `_run_headless`, `_create_synthetic_bundle`. Full pipeline (detection -> correspondence -> triangulation -> evaluation) runs on bundle directories. Synthetic bundle test passes. Commit `c1e6031`. | |
| 2026-07-30 | Session 7 — reconstruction app | `reconstruction_app.py` created: FastAPI backend (status/upload/run/export endpoints) with inline Three.js frontend. Pipeline: detection -> correspondence -> triangulation -> grading. 3D viz: truth/reconstructed/ghost/missed drones, error vectors, camera frustums. Sidebar: bundle info, view selector, param sliders, grade panel, timeline scrubber. All 39/39 prediction tests pass. Commit `954430a`. | |
| 2026-07-30 | Session 8 — render measurement + final update | Render wall-clock: 11.3s/frame @ 1920x1080 Cycles 128spp. Full sweep ~10.5 min. Cross-validation analytics matched-scale data available for Phase 2 comparison. | |
| 2026-07-30 | Session 9 — B4 unfreeze + four-category classification | Task 1 complete: B4 scoring classifies tracks into matched/imprecise/false_track/missed. Seven control tests pass (7/7). 41/41 test suite passes. Gate check: ghosts peak at 2v (all imprecise), mechanism is positional error not ambiguity. Commit `bcc6dc7`. |
| 2026-07-30 | Session 9b — intersection-set metrics | Task 2 complete: sweep CSV now has intersection_n and intersection_median_err_m columns. Sweep uses same truth for all n_views. Intersection very small (1-3 drones at 1px noise). Mixed geometry monotonically decreasing; surround upticks survive but intersection_n=2. Commit `48136ea`. |
| 2026-07-30 | Session 9c — drop full-scale sweep | Task 3 complete: full-scale sweep (5.0km) dropped — required standoff 6846m exceeds practical camera range. Matched-scale (0.3km) only. Report and plot updated. Commit `f0430c5`. |
| 2026-07-30 | Session 9d — ghost assertion restored | Task 4 complete: test_ghosts_at_low_camera_count added. 2v all_ground 1px: imprecise > 0. 8v: imprecise = 0. Original prediction refuted. 42/42 tests pass. Commit `3ab7b63`. |
| 2026-07-30 | Session 9e — density sweep + corrections | Density sweep at n_drones=5,10,15. false_track=0 at ALL configs — correspondence trivially solved. Full-scale explanation corrected (threshold artifact). n_drones added as sweep axis. Prediction refuted. | |
