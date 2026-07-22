# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project summary

This is the camera-based distance-estimation sub-track of a broader drone-swarm-splitting
research project. The broader project (see `~/Projects/drone-swarm-splitting`, a separate repo)
implements Chen et al., "Countering Large-Scale Drone Swarm Attack by Efficient Splitting" (IEEE
TVT 2022): it models a hostile drone swarm as a graph and uses GA/PSO to find *critical nodes*
whose removal splits the swarm below a size threshold. That existing code assumes the adjacency
matrix is already known. This sub-project exists because in a real scenario you can't query a
hostile swarm's radios for link quality — you can only observe it externally (camera) and must
infer connectivity. The approach: estimate 3D drone positions via multi-camera triangulation,
compute pairwise distances, threshold at `d_max` (the assumed comms range) to produce an inferred
adjacency matrix, then feed that into the existing GA/PSO code unchanged.

Full narrative background, the design decisions behind this pivot, and open questions are in
`docs/handoff_summary_cv_distance_pivot_20260722.md` — read that for context this file
deliberately omits.

## Stage 1 / Stage 2 split

- `stage1_geometry/` — pure numpy/scipy triangulation, no rendering, no GPU, no external assets.
  Synthetic ground-truth drone positions -> synthetic pinhole camera projection -> simulated
  pixel-noise "detections" -> DLT triangulation -> compare reconstructed pairwise distances and
  the resulting adjacency matrix against ground truth.
- `stage2_render/` — Blender `bpy` scene generation and rendering of the same camera rig, with
  actual drone models, to replace Stage 1's simulated pixel noise with real rendered images.
- `stage2_detect/` — YOLOv8 (`ultralytics`) detection on Stage 2 renders, feeding into Stage 1's
  `triangulate_point()` / `reconstruct_swarm()` / `evaluate()` unchanged.

These are kept separate deliberately: it isolates *geometry error* (is the camera count/placement
good enough, given a noise model) from *detector error* (is YOLO's real localization error within
what the geometry can tolerate). Don't merge these stages or start wiring Stage 2 before Stage 1's
camera-count/noise sweep has been reviewed — that sweep is what determines the camera rig Stage 2
needs to replicate.

## Constraints

- Apple Silicon, no CUDA. Any torch/YOLO work in Stage 2 must target device `"mps"`, not assume a
  CUDA stack.
- Standalone Mac-local test track. Must run without the existing Linux/CORE+EMANE/ArduPilot swarm
  simulation environment — no dependency on that stack or its logs.
- `ultralytics` and `bpy` are Stage 2 dependencies only — do not install them until Stage 1 results
  are validated (see `requirements.txt`).

## Relationship to the GA/PSO track

This project's output is a reconstructed adjacency matrix (drone pairs within `d_max` of each
other, per the camera-based distance estimate). That matrix is meant to plug into the existing
GA/PSO critical-node search in the `drone-swarm-splitting` repo as a drop-in replacement for the
simulation-derived adjacency matrix it currently uses. Do not duplicate the GA/PSO code here —
this repo's scope ends at producing the adjacency matrix.
