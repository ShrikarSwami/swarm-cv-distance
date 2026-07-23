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

## Phase 3 — synthetic multi-view dataset + ML distance estimation

### M1 — Optics/standoff trade study (2026-07-23)

**Core finding:** The detection boundary depends on platform tier, detector
threshold, and — critically — how many cameras you can field. The analytical
sweep computed 736 configs across3 platform tiers,4 sensor classes,8 focal
lengths,8 standoffs, and4 resolutions. Key results (all respecting practical
camera-count limits):

| Tier | Max cameras | Bbox (≥8px) | Centroid (≥3px) | Sub-pixel (≥1px) |
|---|---|---|---|---|
| A (ground post) | 12 | 10km (800mm,11 cams) | 10km (400mm,6 cams) | 10km (100mm,2 cams) |
| B (airborne UAS) | 6 | 750m (50mm,6 cams) | 3km (100mm,6 cams) | 5km (50mm,3 cams) |
| C (cheap commodity) | 8 | — | 500m (24mm,6 cams) | 10km (100mm,8 cams) |

**The coverage/resolution tension:** Narrower FOV (longer focal length) gives
more pixels on target but requires more cameras for full swarm coverage. The
analytical model shows the camera-count constraint is the binding factor for
every tier — without it, the P1000's 539mm zoom achieves detection at 10km,
but requires 43–240 cameras for full coverage.

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

### M2 — Schema implementation + smoke test (2026-07-23)

[In progress]
