# Progress tracker — swarm-cv-distance

Goal for this chat: a working demo loop — N simulated drones flying in a
3D scene, 5-6 (adjustable) virtual cameras "scan" the swarm, and the
system estimates inter-drone distances from those camera views, comparable
against simulation ground truth. Deeper improvements (real detector
training, correspondence-problem solving, integration with the Linux
swarm sim) are explicitly deferred to future chats.

## Stage 1 — pure geometry (DONE)

- [x] Synthetic pinhole-camera triangulation harness
      (`stage1_geometry/multiview_triangulation_test.py`)
- [x] Camera-count sweep (2/3/4/6 cams x noise levels), 20 trials/cell
- [x] Empirical D_MAX calibration against Stage 1 scene's own distance
      distribution (20 drones, 2km spread, ~100m altitude)
- [x] Near-threshold edge-accuracy check (the metric that actually
      matters, vs. misleadingly-robust overall accuracy)
- [x] **Decision: D_MAX = 1574m (85% target reachability), 6 cameras**
      as the Stage 2 default, camera count adjustable (5-6 range)
- [ ] Flagged, not yet done: noise model that scales with apparent
      object pixel size rather than flat px, to more realistically
      anticipate Stage 2's real detector error before it's measured

## Stage 2 — real renders, real detector (IN PROGRESS)

- [x] Blender scene: drone swarm matching Stage 1's calibration
      assumptions (20 drones / 2km spread / ~100m altitude)
- [x] Simple drone 3D asset (quadcopter-silhouette primitives: body +
      4 arms + 4 rotors, not realistic, just recognizable as "small
      aerial object")
- [x] 6-camera rig placement resolved: **dome**, not a flat ring.
      First attempt used a flat ring (Stage 1's placeholder, 150m
      height) and showed only ~8/20 drones with >=2-camera overlap —
      initially misread as severe real self-occlusion from viewing the
      swarm edge-on (2.4deg elevation). Root cause was actually a bug
      (Blender's default camera `clip_end`=1000m silently culling
      geometry beyond that range, in a scene where camera-to-drone
      distances reach 2-3km) — confirmed because the shortfall was
      identical regardless of render sample count or object size, which
      ruled out both occlusion and rendering noise. Fixed `clip_end`,
      and switched to a dome rig (elevation spread 25-45deg across
      cameras, slant range held ~1201m so drone apparent size doesn't
      shrink further). Result: 20/20 drones ever visible, 19/20 with
      >=2-camera overlap, matching Stage 1's ~19.6/20 prediction.
- [x] Render frames from each camera (RGB PNG + object-index EXR pass,
      `stage2_render/render_scene.py`)
- [ ] Run YOLOv8 (ultralytics, mps device) on renders to get real 2D
      detections, replacing Stage 1's simulated pixel noise
- [ ] Correspondence: use Blender's object-index EXR pass as a
      ground-truth shortcut for this demo (explicitly NOT a
      real-world-deployable solution -- noted as a known simplification
      in code comments). Render pipeline for this is built and verified;
      not yet wired to real YOLO detections.
- [ ] Feed real detections into Stage 1's unchanged
      `triangulate_point()` / `reconstruct_swarm()` / `evaluate()`
- [ ] Re-measure real detector noise (vs. Stage 1's assumed 8px) and
      recalibrate D_MAX if the real number differs meaningfully
- [ ] Demo output: some visual (rendered scene + overlaid estimated vs.
      true distances, or a simple before/after graph comparison) that
      makes the result legible at a glance

## Deferred to future chats

- Correspondence problem for real (non-synthetic-ID) multi-view matching
- Integration with the Linux CORE+EMANE swarm sim / real distance logs
- Training or fine-tuning a detector specifically on drone imagery
- Camera rig placement optimization (currently a simple ring placeholder)
