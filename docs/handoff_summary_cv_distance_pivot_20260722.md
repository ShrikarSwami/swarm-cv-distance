# Handoff: pivot to camera-based distance estimation for swarm-split critical-node detection

Prepared 2026-07-22. Covers a design/planning session (chat, not code
execution) that pivots a sub-track of the broader drone-swarm-splitting
research toward vision-based connectivity inference, and hands off to a
new Claude Code session for implementation.

## Background: the broader project

The project's main line of work (covered in other session summaries --
`session_summary_gps_disable_phases1-7*`, `session_summary_gpsspoof_scale*`,
and the `swarm_run_*.log` files) is a Linux/CORE+EMANE-based simulation of
a drone swarm's communication network, plus ArduPilot SITL integration to
test GPS jamming/spoofing effects on individual nodes, at scales from 1 to
55 simulated nodes.

Separately, the team is implementing the approach from **Chen et al.,
"Countering Large-Scale Drone Swarm Attack by Efficient Splitting"** (IEEE
Trans. Vehicular Technology, 2022) -- a defensive counter-swarm method
that models a hostile drone swarm as an undirected graph, identifies
*critical nodes* (nodes whose removal splits the graph into disconnected
components below a size threshold) via genetic algorithm (GA) and
particle swarm optimization (PSO), and validates that disabling those
nodes prevents the swarm from reaching flight-state consensus. A working
Python implementation of the paper's GA/PSO critical-node search already
exists (per the meeting transcript referenced below) and has been
demonstrated at various swarm sizes up to 155 nodes.

## What changed in this session: the sensing-model pivot

A team meeting transcript (provided as project context, not further
summarized here) discussed a key limitation: the existing GA/PSO code
assumes the adjacency matrix (who's connected to whom) is *known* --
in the simulation work, it's derived from actual measured packet
loss/RTT. But in a real counter-swarm scenario, you can't query a hostile
swarm's radios for link quality. You can only observe the swarm
externally (camera, radar) and must **infer** connectivity.

The team's discussion converged on: **use inter-drone distance as a proxy
for connection likelihood** -- assume two drones are linked if they're
within some threshold distance `d_max` (derived from the known comms
protocol's effective range), and unlinked otherwise. This transforms the
sensing problem from "measure the graph directly" to "estimate 3D
positions, compute pairwise distances, threshold into an adjacency
matrix" -- which then feeds into the *existing, unchanged* GA/PSO
critical-node code.

Key ideas surfaced in the transcript, relevant to implementation:
- **Multi-view/multi-camera triangulation** is the intended approach for
  getting distance without communicating with the target swarm -- single-
  camera depth is unreliable, especially at altitude/range.
- **Relative (non-unit) distance is enough.** If all drones in a
  homogeneous hostile swarm are assumed the same physical size, you can
  derive a scalar size estimate from a few views and express distances as
  multiples of that unit, without needing absolute calibration.
- **Radar was discussed as a complementary/alternative sensor** -- good
  for absolute range, but can't attribute which specific link exists
  between two drones the way geometric proximity inference can.
- **A configurable threshold ("knob"), not a hard physical model**, was
  the team's explicit preference for `d_max` -- mirroring how the
  simulation work already uses a packet-loss threshold to decide whether
  an edge counts as "connected."

Known failure modes flagged for the design (not yet solved, just noted):
- Drones spatially close but not actually linked (directional vs.
  omnidirectional real-world comms) would make a pure distance-threshold
  graph an over-estimate of the true comm graph.
- Occlusion and depth ambiguity from a single viewpoint -- motivates the
  multi-camera requirement.
- **Correspondence problem**: knowing which 2D detection in camera A's
  image is the same physical drone as a detection in camera B's image.
  Not solved in this session; flagged as a separate open problem. For
  early testing, synthetic-scene ground-truth object IDs are a reasonable
  shortcut; a real system would need epipolar-constraint + appearance
  matching or similar.

## Decision: test the pivot on Mac, standalone, before touching the Linux swarm sim

Per direct clarification in this session, the CV/distance-estimation work
is being developed and tested **independently** of the existing
Linux/CORE+EMANE/ArduPilot pipeline, on a Mac (Apple Silicon, no CUDA --
any ML work should target `mps`, not assume a CUDA stack). It does not
need to connect to the existing swarm-sim logs yet. Scope is explicitly
**just the CV distance-estimation piece in isolation**: synthetic scenes,
ground-truth vs. estimated distance -- not the full pipeline, not a
stakeholder demo.

Approach chosen: **multi-camera/multi-view triangulation** (as opposed to
single-image object-size heuristics or monocular depth estimation).

### Two-stage plan (deliberately separated to isolate error sources)

**Stage 1 -- pure geometry, no rendering, no GPU.** Synthetic 3D drone
positions (ground truth) + synthetic pinhole cameras at known
poses, project to 2D, add simulated pixel-level detection noise,
triangulate back via DLT, compare reconstructed pairwise distances (and
the resulting thresholded adjacency matrix) against ground truth. This
isolates "is the viewing geometry/camera count good enough" from "is the
detector good enough," which would otherwise be tangled together.

A working standalone script for this stage was written and delivered
this session: `multiview_triangulation_test.py` (pure `numpy`/`scipy`,
no external assets, runs anywhere). It supports sweeping camera count (2/
3/4/6) and detection-noise level, and reports both raw distance error and
**adjacency agreement** -- whether the reconstructed graph places edges
in the same locations as ground truth given a `d_max` cutoff, which is
the metric that actually matters for whether the downstream GA/PSO
critical-node search would find the same nodes on real vs. ground-truth
data. `D_MAX` in the script is currently a placeholder, not a validated
value -- needs to be derived from the target comms protocol's real
effective range before results are trusted.

**Stage 2 -- real images, real model (not yet started).** Planned:
Blender's `bpy` Python API to script a matching camera rig and render
actual drone-model images; a real object detector (YOLOv8 via
`ultralytics`, `mps` device) run on those renders in place of Stage 1's
simulated noise; reuse Stage 1's `triangulate_point()`/`reconstruct_swarm()`
/`evaluate()` unchanged, so Stage 1 (synthetic noise model) and Stage 2
(actual detector error) are directly comparable.

## Tooling decision: Claude Code, not chat-based file delivery

Team decided to move implementation work into Claude Code rather than
continuing to receive one-off scripts in chat, specifically because this
phase needs local execution (running Blender, launching a Python venv,
iterating on rendered output) that a chat interface can't do. Also
discussed: community "meta-skills" for finding/installing relevant Claude
Code skills (e.g. Skill Finder, Find Skills, Anthropic's bundled
skill-creator) -- worth having available for this project, but nothing
was installed or committed to in this session.

## Not yet done / open questions for the next session

- `D_MAX` (comms-range cutoff) needs a real value, not the script's
  placeholder -- should be derived from whatever comms protocol the
  target swarm is assumed to use (802.11-class LOS range was used
  elsewhere in the broader project's QualNet parameters, Table I of the
  Chen et al. paper, as a reference point, but this hasn't been
  re-derived for the vision-based context specifically).
- Stage 1 has not yet been run/swept -- camera count and noise-tolerance
  tradeoffs are unknown until the sweep described in the Claude Code
  init prompt is actually executed.
- Correspondence problem (matching detections across camera views to the
  same physical drone) is unsolved; Stage 2 will need at least a
  placeholder solution (e.g. Blender ground-truth object IDs) to proceed,
  with a note that this isn't a real-world-deployable solution on its own.
- No decision yet on camera rig geometry beyond the "ring around the
  swarm centroid" placeholder in the Stage 1 script -- real deployment
  would presumably use a small number of friendly observing drones/posts
  at unknown, not evenly-spaced, positions.
- Stage 2 (Blender rendering + YOLO) has not been started at all.

## Source material

- `Countering_Large-Scale_Drone_Swarm_Attack_by_Efficient_Splitting.pdf`
  (Chen et al., IEEE TVT 2022) -- the source paper for the GA/PSO
  critical-node approach this whole track builds toward feeding.
- Team meeting transcript (provided as chat context this session,
  discussing the camera/radar/distance-proxy idea) -- not separately
  saved as a project file; key points extracted above.
- `multiview_triangulation_test.py` -- Stage 1 deliverable from this
  session, included alongside this handoff doc.
