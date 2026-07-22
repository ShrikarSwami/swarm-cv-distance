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
- [ ] **M2 — Lightweight flight sim**: boids-style (stay near neighbors,
      don't stray past scene bounds, gentle wander) — explicitly not real
      formation-holding.
- [ ] **M3 — Camera rig UI**: adjustable count; random dome placement or
      manual place+aim. **The pending D_MAX decision lands here** (candidates
      from the 5km recalibration: 80% -> 3688m, 85% -> 3949m, 90% -> 4358m),
      along with re-running the near-threshold edge-accuracy sweep once the
      rig exists. Per the clip_end lesson: validate rig coverage against
      real render/ID-pass output, not idealized frustum math alone.
- [ ] **M4 — Scan mode**: run the existing triangulation pipeline against
      current swarm + camera state, visualize the resulting distance map as
      a viewport overlay vs ground truth.

## Scene facts (current)

- Bounds: **5km x 5km x 1km** (real requirement from outside this chat;
  `scene_config.py` is the single source of truth)
- Drone size: **0.5m footprint, assumption not confirmed spec** (Intel
  Shooting Star reference ~38cm, "a little bigger") — `DRONE_SIZE_M`
- D_MAX: **not yet locked at this scale** — old 1574m value is stale (was
  calibrated on the 2km pancake scene); candidates above await decision at M3
- 6 cameras was the locked count on the old scene; camera count is
  adjustable in the addon, so M3 should revisit rather than assume

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
