# Progress tracker — swarm-cv-distance

Goal for this chat: a working demo loop — N simulated drones flying in a
3D scene, 5-6 (adjustable) virtual cameras "scan" the swarm, and the
system estimates inter-drone distances from those camera views, comparable
against simulation ground truth. Deeper improvements (real detector
training, correspondence-problem solving, integration with the Linux
swarm sim) are explicitly deferred to future chats.

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

## Scale change (2026-07-22)

**New scene bounds: 5km x 5km x 1km (height).** Up from the original
2km-wide, ~50m-thick placeholder. This is a real requirement from outside
this chat, not a tuning choice — treat prior calibration built on the old
scale as invalidated, not as a starting point to nudge.

**Drone size: assumed ~0.5m (50cm) footprint**, based on "a little bigger
than viral light-show drones" -- reference point is the Intel Shooting
Star, a commonly-used light-show drone, at 384x384x93mm (~38cm footprint).
This is an assumption, not a confirmed spec -- flag to the user if an
exact number becomes available, and treat as a named constant
(`DRONE_SIZE_M` or similar) so it's a one-line change if it needs revising.

**What this invalidates (needs redoing, not scaling):**
- D_MAX = 1574m was calibrated against the *old* 2km-scene distance
  distribution. Meaningless at 5km scale -- needs full recalibration
  against the new scene's own pairwise-distance distribution, same
  empirical method as before (target ~80-90% reachability, matching
  Chen et al.'s reported range), not a naive linear scale-up.
- `make_swarm()` (`stage1_geometry/multiview_triangulation_test.py`)
  currently generates a thin horizontal "pancake" (wide xy spread, ~50m z
  jitter). A genuine 5km x 5km x 1km volume needs real 3D distribution
  across the full height range, not the same pancake logic with bigger
  numbers -- this is a structural function change, not a constant change.
- The dome camera rig (1201m slant range, 25-45deg elevation spread) was
  sized for the old 2km scene. Needs re-derivation for 5km scale, and --
  per the clip_end lesson -- whatever comes out of that math needs to be
  confirmed against a real Blender render's actual ID-pass coverage
  before being trusted, not assumed correct because the frustum math
  says so.
- Apparent drone pixel size in frame will change with both the larger
  scene (longer likely camera-to-drone ranges) and the larger assumed
  drone size (partially offsetting). Needs re-checking once the new
  rig exists, not assumed to net out to roughly the same ~5px as before.

## Stage 1 — pure geometry

- [x] Synthetic pinhole-camera triangulation harness
      (`stage1_geometry/multiview_triangulation_test.py`)
- [x] Camera-count sweep (2/3/4/6 cams x noise levels) -- **methodology
      still valid, needs re-running against the new scale once
      `make_swarm` is updated**
- [ ] **STALE, needs redo at 5km x 5km x 1km scale:** D_MAX empirical
      calibration (was 1574m / 85% target, valid only for the old 2km
      scene)
- [ ] **NEEDS STRUCTURAL CHANGE:** `make_swarm()` -- extend from thin
      horizontal pancake to genuine 3D volume distribution across the
      full 1km height
- [ ] Near-threshold edge-accuracy check -- re-run once D_MAX is
      recalibrated
- [ ] Noise model that scales with apparent object pixel size rather
      than flat px (still not done, still worth doing, now more
      important given the bigger range variance at 5km scale)

## Stage 2 — real renders, real detector

- [ ] **STALE, needs redo at new scale:** Blender scene (drone swarm
      previously matched the old 2km/20-drone/~100m-altitude
      assumptions)
- [ ] **NEEDS UPDATE:** drone 3D asset -- resize to ~0.5m assumed
      footprint (previous asset was quadcopter-silhouette primitives,
      sized for the old scene; geometry can likely be reused, just
      needs correct scale)
- [x] Lesson learned, keep for reference: flat-ring rig at
      near-swarm-altitude views a wide/thin swarm nearly edge-on and
      *looks* like severe self-occlusion; in this project's case it
      was actually two compounding issues -- (1) a genuine geometry
      problem (2.4deg elevation) and (2) an unrelated Blender
      `clip_end`=1000m default silently culling geometry beyond that
      range. Confirmed via: shortfall was identical regardless of
      render sample count or object size (rules out rendering noise
      and confirms it's not a size/occlusion effect at the pixel
      level). **This general validation method (check real ID-pass
      coverage against idealized frustum-math predictions, and use
      sample-count/object-size probes to distinguish bug classes) is
      worth reusing when re-deriving the rig for the new scale** -- the
      specific rig parameters (1201m slant range, 25-45deg) are not
      reusable at 5km.
- [ ] Re-derive camera rig for 5km x 5km x 1km scale (dome-style
      approach still likely right, parameters need rework), validate
      against real render coverage, not idealized math alone
- [ ] Render frames from each camera
- [ ] Run YOLOv8 (ultralytics, mps device) on renders to get real 2D
      detections, replacing Stage 1's simulated pixel noise
- [ ] Correspondence: use Blender's object-index EXR pass as a
      ground-truth shortcut for this demo (explicitly NOT a
      real-world-deployable solution -- note in code comments).
      Render pipeline groundwork from the old scene exists and is
      largely reusable (RGB PNG + object-index EXR via compositor,
      Cycles required for the index pass) -- rebuild against new scene,
      don't assume old renders are usable.
- [ ] Feed real detections into Stage 1's unchanged
      `triangulate_point()` / `reconstruct_swarm()` / `evaluate()`
- [ ] Re-measure real detector noise (vs. Stage 1's assumed flat 8px)
      and recalibrate D_MAX if the real number differs meaningfully
- [ ] Demo output: some visual (rendered scene + overlaid estimated vs.
      true distances, or a simple before/after graph comparison) that
      makes the result legible at a glance

## Deferred to future chats

- Correspondence problem for real (non-synthetic-ID) multi-view matching
- Integration with the Linux CORE+EMANE swarm sim / real distance logs
- Training or fine-tuning a detector specifically on drone imagery
- Camera rig placement optimization beyond "does it achieve target
  coverage" (e.g. cost/practicality of an actual observer-platform count)
