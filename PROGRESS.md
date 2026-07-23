# Progress tracker — swarm-cv-distance

Goal: an interactive Blender addon — a live UI panel driving a navigable 3D
viewport — where a simulated drone swarm can be generated, flown, "scanned"
by a configurable camera rig, and the triangulated distance map visualized
against ground truth. Deeper improvements (real detector training,
correspondence-problem solving, integration with the Linux swarm sim) remain
deferred to future chats.

## Process note (added 2026-07-22, after the clip_end debugging detour)

Going forward: **one task at a time, commit after each, stop and report
back before starting the next.** The clip_end bug earlier this session
turned into a multi-step, multi-hypothesis chain (occlusion misdiagnosis
-> dome rig change -> "worse, not better" -> sample-count test -> object-size
test -> actual root cause) all inside a single uninterrupted run, with no
commit checkpoints along the way. That made it hard to tell what was
actually validated vs. still-in-progress, and separately, an attached
PROGRESS.md update went missing across two interrupted turns because there
was no committed checkpoint to recover from. Small, committed, checked-in
steps fix both problems.

## Scope pivot (2026-07-22): interactive addon, not batch scripts

The remaining work is a proper Blender addon (`blender_addon/swarm_scanner/`),
not more offline render-and-analyze scripts. The Stage 1 / Stage 2 batch work
below is **not discarded** — its validated pieces are the addon's foundation:

- `Camera` class with pinhole projection math, validated against real Blender
  renders to <1px (`stage1_geometry/multiview_triangulation_test.py`,
  `stage2_render/validate_camera_alignment.py`)
- Empirical D_MAX calibration methodology (percentiles of the scene's own
  pairwise-distance distribution, targeting Chen et al.'s ~80-90%
  reachability) — `stage1_geometry/sweep_dmax.py`
- Quadcopter drone asset (primitives, joined mesh) — originally
  `stage2_render/render_scene.py`, now rebuilt shared-mesh-style in the addon
- Dome-style camera placement logic (`place_dome_of_cameras`)
- Triangulation pipeline: `triangulate_point()` / `reconstruct_swarm()` /
  `evaluate()`, to be reused unchanged by the scan milestone

## Milestones

- [x] **M1 — Swarm generator + live viewport** (2026-07-22): addon skeleton
      with bl_info + View3D sidebar panel (`Swarm Scan` tab); drone-count
      slider (2-500), formation dropdown (Random Cloud only for now,
      light-show presets later), seed, and a Generate Swarm operator.
      Positions come from Stage 1's `make_swarm` (imported, not copied —
      verified identical output in the headless test). Drones are N objects
      sharing one mesh for cheap instancing. Display Scale property
      (default 20x) exaggerates mesh size for visibility at 5km scale —
      positions are always true-scale, flagged in the tooltip so later scan
      milestones aren't affected. Generate also extends every viewport's
      clip_end to 20km (the clip_end lesson, applied to the viewport).
      Headless test covers registration, bounds, position-match vs Stage 1,
      regeneration-replaces, clean unregister. Load via
      `blender_addon/dev_load.py` (NOT Blender's Install button — the addon
      imports stage1_geometry by repo-relative path; the panel reports this
      if broken). Demo file: `logs/swarm_demo.blend` (gitignored).
      Interactive orbit/pan/zoom feel: needs a human check — drones are
      plain mesh objects with no handlers, so navigation is Blender-native,
      but confirm before calling M1 fully closed.
- [x] **M2 — Lightweight flight sim** (2026-07-22): boids-style, running
      live in the viewport via a modal timer operator (30 Hz) with a
      Start/Stop button; Esc also stops it, and viewport navigation stays
      fully usable mid-flight (PASS_THROUGH). Behaviors: cohesion toward
      the local-neighborhood mean (not swarm centroid), separation inside
      25% of neighbor radius, soft leash at the 5km x 5km x 1km edges
      (steering ramps across a margin, never a hard clamp), random wander.
      Tunables exposed as sliders and read live every tick (take effect
      mid-flight): Neighbor Radius, Bound Softness, Wander, Speed. Core
      step is pure numpy (`boids_step`, no bpy) so it's headlessly
      testable: at 500 drones, 5.7 ms/step vs the 33 ms 30fps budget;
      after 30 sim-seconds — speed cap held, 0 drones outside bounds,
      min pairwise 22.8m (no stacking), spread maintained (no collapse).
      Explicitly viewport-only: no changes to stage1_geometry, which
      stays static-snapshot-based. Regenerating the swarm auto-stops a
      running sim. "Looks organic" is numerically proxied
      (moving/cohesive/bounded/uncollapsed) — final feel check is human,
      same as M1's navigation check.
- [x] **M3 — Camera rig UI** (2026-07-22): panel box with camera-count
      slider (2-12) and two placement modes. Random: dome around the
      CURRENT swarm's bounding volume (works on live sim positions, not
      the generation params) — standoff re-derived from scratch per
      camera: minimum slant range fitting the whole volume in both FOVs
      (from scene_config intrinsics) x1.15 margin, elevation 20-50deg
      (edge-on lesson at the low end, parallax loss at the high end),
      jittered azimuths; each click re-rolls. Manual: same fit math but
      even/deterministic as a starting layout for hand-editing; every
      camera auto-aims at a "Swarm Aim" empty via TRACK_TO, and a Toggle
      Auto-Aim operator mutes the constraint on selected cameras so they
      can be rotated by hand. Rig cameras carry the validated intrinsics
      (36mm sensor-width convention) and clip_end=50km. Coverage
      validated on REAL ID-pass renders, not frustum math
      (blender_addon/validate_rig_render.py + validate_rig_report.py,
      rerunnable): 3 random rolls + manual layout, all 20/20 drones
      >=2-camera visible at the first-guess derivation (bar: >=18/20).
      Note: per-camera counts run 10-20/20 because far drones (~10km)
      drop below ID-pass rasterization even at 20x display scale — first
      concrete signal of the true-scale detectability concern parked for
      M4. **D_MAX correction: not needed for M3 after all — it's consumed
      in M4's adjacency thresholding. D_MAX = 3949m (85% target) locked
      as provisional for M4.**
- [x] **M4 — Scan mode** (2026-07-22): "Run Scan" operator renders each rig
      camera's real object-index pass (Cycles, same compositor graph M3
      validated: Object Index -> OutputFile, OPEN_EXR_MULTILAYER), takes each
      drone's detection as the centroid of its ID-pass footprint -- so
      occlusion/out-of-frame comes from the actual render, not frustum math --
      then layers Stage 1's synthetic pixel-noise model on top (per the scope
      decision below) and reuses `triangulate_point()`/`reconstruct_swarm()`/
      the adjacency-agreement logic unchanged. D_MAX = 3949m (85% target,
      recalibrated for the 5km scene) is now live in `scene_config.py`,
      replacing the stale 1574m placeholder. Viewport overlay: green/red 3D
      lines between triangulated drones for each graph edge present in either
      the true or estimated adjacency (green = correctness agrees, red =
      false positive/negative), plus a HUD readout (triangulated count,
      overall adjacency accuracy, near-D_MAX-band accuracy -- the hardest,
      decision-boundary case). Detour mid-implementation: Blender 5.x removed
      `CompositorNodeMath` (compositor math nodes are now unified with
      `ShaderNodeMath`); turned out not to matter here, since centroid
      extraction is done in numpy, not compositor nodes. Real detour:
      `bpy.data.images.load()` can't read back this project's custom-named
      "id_" multilayer EXR pass (loads as 0x0/TARGA) -- worked around by
      running the EXR-read + triangulation step as a `venv/bin/python`
      subprocess (`blender_addon/scan_worker.py`), reusing the OpenEXR
      package M3's `validate_rig_report.py` already depends on (now listed in
      `requirements.txt`). Camera poses for triangulation are read directly
      from each rig camera's actual `matrix_world` (not re-derived from a
      look-at target), so hand-rotated manual-mode cameras triangulate
      correctly too. Validated end-to-end via the real operator, not a
      synthetic stand-in (`blender_addon/validate_scan_pipeline.py`,
      rerunnable): 4 seeds, 19-20/20 triangulated, 100% adjacency accuracy
      (both overall and near-threshold), mean distance error 5-7m against a
      3949m D_MAX -- comfortably resolved. Scope decision (made ahead of
      implementation): detection source is the object-index EXR pass
      (centroid of each drone's ID-pass footprint), NOT a real YOLO run --
      see the subpixel finding below for why. Display Scale (whatever the
      swarm was generated with) is reused for scan rendering too, same
      reasoning as M3's rig-coverage validation: true-scale (0.5m) drones are
      subpixel and wouldn't rasterize in the ID pass at all.

      **Closing the loop on the 100% accuracy figure (2026-07-22):** the
      validation run above used only `SCAN_PIXEL_NOISE_STD_PX = 2.0`, never
      swept -- checked whether that number holds across Stage 1's original
      noise range (0.5-8px pixel-noise std) before treating it as a
      quotable result. Reused 3 real rendered configs (real occlusion, no
      re-render needed since noise is added post-render) and swept noise
      with 15 trials/level:

      | noise (px) | reconstructed | overall acc. | near-threshold acc. |
      |---|---|---|---|
      | 0.5 | 19.7/20 | 100.0% | 100.0% |
      | 1.0 | 19.7/20 | 100.0% | 99.9% |
      | 2.0 (validation default) | 19.7/20 | 99.9% | 99.4% |
      | 3.0 | 19.7/20 | 99.9% | 99.5% |
      | 5.0 | 19.7/20 | 99.6% | 98.4% |
      | 8.0 | 19.7/20 | 99.6% | 98.3% |

      Real result, not a low-noise artifact: near-threshold accuracy shows a
      genuine, monotonic downward trend (100% -> 98.3%) as noise increases.
      It does NOT reproduce the 90-98% band from Stage 1's original
      `sweep.py` runs, though -- those used the old 2km scene, ring cameras,
      and a different D_MAX, so they aren't apples-to-apples with M4's real
      5km rig. Two things hold M4's number up even at 8px: (1) reconstructed
      count stays flat (19.7/20) across all noise levels, because M4
      determines visibility from real render occlusion, not a noise-coupled
      synthetic drop probability the way Stage 1 did; (2) the near-threshold
      band (+/-395m, 10% of D_MAX=3949m) is wide relative to this rig's
      actual triangulation error even at 8px noise (single-to-low-double-digit
      meters), so noise rarely pushes a distance estimate across the
      395m-wide boundary needed to flip an edge.
- [x] **Tooling** (2026-07-22): double-clickable "Swarm Scanner.app" at
      repo root (minimal unsigned bundle wrapping the dev_load.py launch;
      finds Blender across Steam//Applications/~/Applications, passes extra
      args through so `.../MacOS/swarm_scanner --background` smoke-tests
      the same code path) + STARTUP.txt with the manual command as
      fallback/debugging reference. Both must stay inside the repo
      (repo-relative imports). Headless smoke test passed; Finder
      double-click structurally identical but human-confirm on first use.

## Finding: drones are subpixel at true scale from derived standoffs (2026-07-22)

A real optics/range result out of M3's render validation, not a pipeline
bug: at the FOV-fit standoff distances the 5km x 5km x 1km volume forces
(~5.5-7km slant, up to ~10km to far drones), a drone rendered with 20x
display inflation (10m mesh) covers only ~1.4px -- so a TRUE-scale 0.5m
drone is far below one pixel (~0.07px) at scene_config's 1920x1080 /
1400px-focal intrinsics. Consequences:
- Real detector-on-render (the original Stage 2 YOLO plan) cannot work at
  this scene scale + camera spec -- there is nothing to detect. This is
  why M4 uses ID-pass centroids + synthetic noise instead; the pivot is
  recorded here deliberately rather than routed around silently.
- Making real detection viable would need some combination of: longer
  focal length / narrower FOV (trades against per-camera coverage),
  higher resolution sensors, closer standoff (trades against whole-swarm
  framing -- would need cameras that don't each see the whole volume, and
  therefore a coverage-planning story), or a detector operating on
  sub-pixel cues. That trade study is future-chat material; the
  triangulation-accuracy question M4 answers is independent of it.

## Scene facts (current)

- Bounds: **5km x 5km x 1km** (real requirement from outside this chat;
  `scene_config.py` is the single source of truth)
- Drone size: **0.5m footprint, assumption not confirmed spec** (Intel
  Shooting Star reference ~38cm, "a little bigger") — `DRONE_SIZE_M`
- D_MAX: **3949m (85% target reachability)**, live in `scene_config.py` and
  consumed by M4's adjacency thresholding — the old 1574m value was stale
  (2km scene)
- Camera count is adjustable in the addon (2-12, default 6); coverage at
  6 validated by real renders in M3

## Lessons learned (keep for reference)

- Blender defaults bite at km scale: camera `clip_end` (render) and viewport
  `clip_end` both default to 1000m and silently cull beyond it — looks
  exactly like occlusion or missing objects. Both are now handled explicitly.
- Validation method that caught it: compare real ID-pass coverage against
  idealized frustum-math predictions, and use sample-count/object-size
  probes to distinguish bug classes (rendering noise vs geometry vs culling).
  Reuse this when validating the M3 rig.
- Stage 1's point-projection math has no occlusion model — a flat ring at
  near-swarm altitude views a wide/thin swarm nearly edge-on; dome-style
  elevation spread is the right default.

## Deferred to future chats

- Light-show-style formation presets (dropdown is the extension point)
- Correspondence problem for real (non-synthetic-ID) multi-view matching
- Integration with the Linux CORE+EMANE swarm sim / real distance logs
- Training or fine-tuning a detector specifically on drone imagery
- Camera rig placement optimization beyond "does it achieve target
  coverage" (e.g. cost/practicality of an actual observer-platform count)
- Noise model that scales with apparent object pixel size rather than flat
  px (from old Stage 1 backlog; still worth doing, matters more at 5km)

## Phase 3 — temporal multi-view detection + triangulation

**Revised plan (post-M1):** M1 ruled out per-frame object detection
(YOLO etc.) at true scale. The detection backbone is now frame
differencing (classical, not learned). The pipeline becomes: frame
differencing → multi-camera association → triangulation → adjacency
inference. The dataset and milestones are rebuilt around this.

### M1 — Optics/standoff trade study [CLOSED 2026-07-23]

**Status: M1 closed.** Full findings in `docs/M1_optics_findings.md`.

**Headline:** Per-frame object detection is infeasible at true scale (0.32px
apparent size). Temporal detection via frame differencing is viable under
realistic deployment conditions, contingent on drone producing flux≥8 at
the sensor (2× margin below the assumed flux≈16). The flux≈16 estimate
needs empirical validation before becoming a design commitment.

**What M1 settled:**
- No camera config achieves ≥2-view coverage AND detectable true-scale pixels
- Temporal detection works on both sky and terrain backgrounds (static texture
  cancels in frame differencing)
- Holds under realistic perturbation (jitter ≤1.5px, shimmer ≤2.0)
- Conclusion flips to "marginal" at flux≈8–10 (2× below assumed)
- Viewing geometry (sky fraction ~47% for ground cameras) is moot — frame
  diff eliminates static texture regardless
- Multi-camera fusion does NOT lower per-camera flux requirement. Fusion
  buys false-positive rejection (true drone appears in all N cameras,
  noise appears in ~1), not sensitivity. Per-camera threshold remains
  flux≥5 for SNR≥3.

**What M1 did NOT settle:**
- Actual drone reflectance / sensor flux (flux≈16 is derived, not measured)
- Real platform jitter characteristics (ground mast vs airborne)
- Multi-camera temporal fusion (single-camera analysis only)
- Matched-filter detection (template correlation, not tested)

**Core finding:** The binding constraint for triangulation is ≥2-camera
overlap (every drone seen by ≥2 cameras), NOT pixel resolution. The
≥2-view requirement eliminates all narrow-FOV configs — 800mm at 10km
gives 11px on target but needs hundreds of cameras for overlap.

Simulation-verified ≥2-view camera counts (dome arrangement, 20 drones
in 5km×5km×1km):

| Config | 500m | 1km | 2km | 3km | 5km | 7.5km | 10km |
|---|---|---|---|---|---|---|---|
| 24mm FF | 30✗ | 30✗ | **12✓** | **5✓** | **3✓** | **2✓** | **2✓** |
| 50mm FF | 30✗ | 30✗ | 30✗ | 30✗ | 25✗* | **4✓** | **2✓** |
| 100mm+ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

✓ = all 20 drones seen by ≥2 cameras. ✗ = <30 cameras insufficient.
*50mm @ 5km: 25 cams in sim but still exceeds12-cam ground-post limit.

**True-scale apparent sizes at coverage-passing standoffs:**

| Config | Standoff | Cams | Apparent px | Detector class |
|---|---|---|---|---|
| 24mm FF 1920 | 2km | 12 | 0.32px | sub-pixel |
| 24mm FF 6000 | 2km | 9 | 1.00px | sub-pixel (floor) |
| 24mm FF 8192 | 2km | 9 | 1.37px | sub-pixel |
| 24mm FF 1920 | 5km | 3 | 0.23px | sub-pixel |
| 50mm FF 1920 | 7.5km | 4 | 0.18px | sub-pixel |
| 50mm FF 8192 | 5km | 12 | 1.14px | sub-pixel (floor) |

**Only 24mm and 50mm achieve ≥2-view coverage with practical camera counts.**
Every narrower lens fails at every standoff. But even the configs that pass
coverage give ≤1.37px at true scale — sub-pixel for bbox/centroid detectors.

**M3 config chosen: 24mm full-frame, 2km standoff, 12 cameras.**
- ≥2-view coverage: ✓ (simulation-verified)
- True-scale apparent size: 0.32px (sub-pixel)
- At 20x display scale: ~6.4px in ID pass renders (detectable only with inflation)
- This is the only ground-post config satisfying both ≥2-view AND ≤12 cameras

### M1 correction — coverage-vs-resolution is the binding constraint (2026-07-23)

**Headline: No ground-post (or any-tier) configuration satisfies ≥2-camera
dome coverage AND a detectable true-scale pixel size simultaneously for a
5km×5km×1km swarm of 0.5m drones.**

The optics sweep's decision boundary table used 1D linear tiling for camera
counts, which underestimates by 2-10× vs the actual 2D dome coverage
simulation. When both constraints are checked against the dome simulation:

- **bbox detector (≥8px):** zero configs pass both constraints at any tier,
  any standoff (100m–10km), any sensor class.
- **centroid detector (≥3px):** zero configs pass both constraints.
- **subpixel/temporal (≥1px):** exactly 3 configs barely pass, all requiring
  6000+ pixel sensors:
  - 24mm FF 8192px @ 2km: 1.37px, 9 cams (ground-post, needs high-res body)
  - 50mm FF 8192px @ 5km: 1.14px, 12 cams (ground-post, marginal)
  - 24mm FF 6000px @ 2km: 1.00px, 9 cams (ground-post, at detection floor)

**Why close standoffs don't help:** At <500m, pixel size improves but the
swarm's angular extent explodes (at 100m, ±2.5km subtends ~180°). Even 30
dome cameras can't achieve ≥2-view coverage — the volume is simply too large
to surround from nearby.

**Why long lenses don't help:** Every lens ≥100mm fails coverage at every
standoff with ≤30 cameras. The dome simulation shows min_views=0 for all
100mm+, 200mm, 400mm, 800mm, and 1200mm FF configs across 500m–10km.

**The physics:** Coverage requires wide FOV (≥24mm). Wide FOV at the only
standoffs where dome coverage works (≥2km for 24mm) produces ≤1.4px on a
0.5m target. The two constraints pull in opposite directions and cannot be
simultaneously satisfied.

**M3 dataset was abandoned** because the chosen config (24mm FF / 2km / 12
cams / 20× display scale) renders drones at ~11.6px only because of 20×
display inflation. At true scale they are 0.58px — a detector trained on
inflated targets learns something operationally meaningless.

### Minimum changes to make per-frame detection feasible (ranked by cost)

1. **Increase assumed drone size** (lowest cost): A 5m drone (10× current
   0.5m assumption) at 24mm FF / 2km gives 5.8px — near centroid threshold.
   A 12.5m drone gives 8px+ (bbox threshold). Cost: changes operational
   assumptions (Intel Shooting Star is ~38cm). This is a modeling choice,
   not a physics change. If the target swarm uses larger airframes (military
   Group 3-4 UAS, 2-5m wingspan), this reframes the problem meaningfully.

2. **Reduce swarm extent** (moderate cost): A 1km×1km×500m swarm at 24mm FF
   / 1km standoff gives ≥2-view with ~5 cams AND ≥3px centroid detection.
   Cost: changes the operational scenario from "wide-area swarm" to
   "localized formation." The5km assumption comes from the broader project;
   if the real scenario allows a smaller observable area, everything works.

3. **Closer standoff + abandon full-volume coverage** (moderate cost):
   Cameras at 200-300m give 2-6px on 0.5m drones, but can't see the full
   5km volume. Could track sub-swarms or individual drones. Cost: loses
   the "see everything at once" requirement; needs a coverage-planning
   story for which drones are observed by which cameras.

4. **Hybrid rig: wide for coverage + narrow for resolution** (higher cost):
   2-3 wide cameras (24mm) for global coverage and drone counting, plus
   8-10 narrow cameras (100-200mm) pointed at known high-interest regions.
   Cost: needs a priori knowledge of where to point the narrow cameras, or
   a wide→narrow handoff system. Fundamentally a two-stage detection
   architecture.

5. **Motion/streak detection across frames** (highest conceptual cost,
   lowest hardware cost): Instead of per-frame object detection, detect
   drone motion over time as pixel-level streaks or consistent point
   displacements across frames. Works at <1px per frame if the drone moves
   enough between frames. Cost: completely different detector architecture
   (optical flow / background subtraction, not YOLO). Abandons the
   "single-frame detection" paradigm. But the optics_sweep.py already
   notes this as a 1.5-2× extension to detector thresholds.

**Hard constraint maintained:** No display_scale inflation in training data
or reported metrics. Any visualization inflation is explicitly separated
from the dataset pipeline.

### Temporal detection investigation (2026-07-23)

**Corrected finding (after real-render validation and perturbation testing):**

Temporal detection via frame differencing works on both sky and terrain
backgrounds at true scale, under realistic deployment perturbations,
contingent on flux≥8 at the sensor.

The initial synthetic analysis showed a 6× background dependence (sky easy,
terrain hard). This was an artifact: the sky was rendered as a flat constant
(σ=0.000), not a physical sky. Re-rendered with Blender's physical sky
model (σ=18.24), the frame-differencing noise ratio is 1.1× — identical on
both backgrounds. Static texture cancels in frame differencing.

Perturbation test (camera jitter, cloud shadows, atmospheric shimmer,
combined effects) shows the flux≥5 threshold holds under all individual
perturbations on both backgrounds. Combined moderate perturbations
(jitter=1.0px + shimmer=5.0) push the threshold to flux≥15 on terrain.

**Load-bearing assumption:** flux≈16 for a real 0.5m drone at 2km. This
is derived from assumed reflectance (10%), atmospheric transmission (~0.85),
and sensor characteristics — not measured. Sensitivity: at flux=8 (2×
below assumed), individual perturbations still pass but combined moderate
effects start failing. The conclusion flips to "marginal" at flux≈8–10.

Full analysis in `docs/M1_optics_findings.md`. Scripts:
`temporal_detection.py`, `validate_real_render.py`, `analyze_real_render.py`,
`perturbation_test.py`, `rerender_sky.py`, `viewing_geometry.py`.

**Sanity check passed:** Same angular resolution → same apparent pixel size
(validated across full-frame and APS-C sensor classes).

**Render pipeline validated:** Blender 5.x headless Cycles with ID-pass
compositing works, but requires: (1) addon operators for scene graph
evaluation (standalone objects don't expose Object Index pass), (2)
`bpy.context.view_layer.update()` before render to evaluate camera rotations,
(3) venv Python for EXR extraction (Blender's bundled Python lacks OpenEXR).

**Detector-class thresholds flagged as rules of thumb** —8px (YOLO-scale),
3–5px (centroid with known size), <3px (sub-pixel/temporal). All are
approximations with stated assumptions about background clutter and target
knowledge.

**Schema decided:** `gt.npz` stores positions, K, extrinsics, meta only.
Distances and adjacency computed on demand with tunable D_MAX. No loose
frames on disk (FFV1/MKV master, decode to scratch for training).

### M2 — Schema implementation + smoke test (2026-07-23) [superseded]

Original M2 implemented the single-frame clip schema (`dataset_schema.py`)
and rendered 20 smoke-test clips. This work is not wasted — the schema,
FFV1/MKV pipeline, and render infrastructure carry forward — but the
single-frame dataset design is superseded by M1's finding that per-frame
detection doesn't work. The dataset must now contain **frame sequences**
with temporal continuity, not single stills.

### Revised Phase 3 milestones (post-M1)

**Pipeline architecture (revised):**

```
Multi-camera video → Frame differencing (per camera) → Detections (per frame)
  → Multi-camera association (epipolar matching) → Triangulation
    → Pairwise distances → Adjacency matrix → GA/PSO
```

Frame differencing is classical. The ML component fits where classical
methods hit limits: false-positive rejection in cluttered scenes,
correspondence in dense swarms, and temporal association across occlusions.

#### M3 — Temporal multi-view dataset [NOT STARTED]

**What the dataset must contain:**
- **Frame sequences**, not stills: 10–30 frame clips at 10–30 fps
  (drone must move ≥0.3 px between frames for frame differencing to work)
- **Motion continuity**: drones fly realistic trajectories (boids sim
  already generates these via M2 flight sim)
- **Multi-view synchronization**: all cameras capture the same frame
  (already validated in M3 camera rig)
- **Ground truth tracks**: 3D position of each drone at each frame
  (not just start/end — full trajectory for triangulation validation)
- **Per-frame ground truth visibility**: which drones are visible in
  which cameras at which frames (from ID-pass, already validated in M4)

**Render config:** 24mm FF, 2km standoff, 12 cameras (M1's validated
coverage config). True scale (0.5m drones, no display inflation).
32 Cycles samples (sufficient for frame differencing — noise is handled
by the temporal pipeline, not the renderer).

**Dataset size target**: ~500 clips × 20 frames × 12 views = 120,000
rendered frames. At ~0.5s/frame (24mm, 32 samples, 1920×1080): ~17 hours
render time. Batch overnight.

**Schema update**: extend `clip.npz` to store per-frame positions (N_frames,
N_drones, 3) instead of single positions. Add frame timestamps.

#### M4 — Detection pipeline validation [NOT STARTED]

**What M4 validates (classical pipeline, no learning):**

1. **Per-camera frame differencing**: on rendered sequences, measure
   detection SNR per drone per frame. Compare against perturbation test
   predictions (flux≥5 → SNR≥3). This is the empirical flux validation.

2. **Multi-camera association**: match detections across camera views using
   epipolar geometry. Measure association accuracy (correct matches / total
   matches) as a function of drone count and noise level.

3. **Triangulation from temporal detections**: feed associated multi-view
   detections into existing `triangulate_point()`/`reconstruct_swarm()`.
   Compare against ground-truth tracks. Report distance error and
   adjacency agreement vs D_MAX.

4. **End-to-end accuracy**: from raw multi-camera video frames to
   adjacency matrix. Report: detection rate (fraction of drones detected
   per frame), false-positive rate, triangulation error, adjacency
   agreement.

**This is the critical milestone.** If the classical pipeline (frame diff
→ epipolar match → triangulate) achieves acceptable accuracy, ML is
optional refinement. If it doesn't, ML is needed and we know exactly
where.

#### M5 — ML augmentation (if needed) [NOT STARTED]

ML fits where classical methods hit limits. Three candidates, in order
of likely impact:

1. **False-positive rejection** (lowest risk): A binary classifier that
   takes a candidate detection (patch around a bright spot in the
   difference image) and decides "drone" vs "noise/shimmer". Classical
   frame differencing produces candidates; ML filters them. Training data:
   M4's rendered sequences with ground-truth drone/noise labels. Simple
   CNN, small model, fast inference. **This is where YOLO-style detection
   re-enters** — not as a per-frame detector (that failed at 0.32px) but
   as a post-differencing classifier on candidate patches.

2. **Cross-camera correspondence** (medium risk): Learning to match
   detections across views when epipolar constraints are noisy. At 0.32px,
   detection centroids have ~1px uncertainty, which maps to large 3D
   uncertainty. A learned matcher could use temporal trajectory shape
   (the drone's path across frames) as a signature for association.
   Training data: M4's associated detections with ground-truth matches.

3. **End-to-end temporal detector** (highest risk, highest ceiling):
   Replace frame differencing + classical matching with a spatiotemporal
   model (3D CNN or transformer over the multi-camera video cube).
   Detects drones directly from raw frames without the two-stage
   pipeline. Requires large training set. Only justified if stages 1-2
   prove insufficient.

**M5 is conditional.** Run M4 first. If the classical pipeline achieves
≥90% detection rate with ≤5% false-positive rate, M5 is optional
optimization, not a requirement.
