# Brief: Multi-View 3D Swarm Reconstruction Proof

## The one-sentence question

Can we recover a drone swarm's 3D structure from nothing but images of it taken
from several angles, and how does that recovery degrade as we take cameras away?

Nothing else. Not realism, not splitting algorithms, not detection quality.
Just: pictures in, 3D graph out, scored honestly against truth.

---

## Scope

**In scope**
- Synthetic swarms with exactly known 3D positions
- Rendering / projecting them from a varying number of camera angles (2 to 10+)
- Solving correspondence across views without being told which blob is which drone
- Triangulating 3D positions from matched detections
- Scoring the result against Blender ground truth
- Producing the camera-count vs. accuracy relationship

**Explicitly out of scope — do not start, do not "prepare for"**
- Visual realism (terrain, sky, lighting). A parallel effort owns this. Do not
  touch those worktrees, do not wait on them, do not import their changes.
- The critical-node splitting algorithm. Reconstructed positions get scored and
  stop there. Nothing is fed downstream.
- Real pixel-based drone detection beyond what Phase 2 specifies.

---

## Phase 0 — Inspect before building

Do not assume repo structure. Before writing any code:

1. Confirm the correct base commit — the last commit **before** today's realism
   work began. Report the hash and commit message for sign-off.
2. Create a new isolated git worktree and branch off that commit. This must not
   collide with the parallel realism worktrees on the same repo.
3. Inspect and report what actually exists today:
   - the Blender addon (boids sim, camera rig UI)
   - the batch queue and config generator
   - the dataset schema — specifically the exact format of stored positions,
     camera intrinsics, and extrinsics
   - the Object Index EXR ground-truth pass
4. Report the camera model conventions in use: coordinate handedness, whether
   extrinsics are world-to-camera or camera-to-world, sensor size / focal length
   units, principal point convention, and image origin (top-left vs bottom-left).

Getting these conventions wrong silently is the single most likely source of a
result that looks plausible and is entirely wrong. This project has already been
burned by exactly this class of bug — stale focal length, incorrect camera
projection math. Write the conventions down in PROGRESS.md and verify them with
a hand-checked round-trip before proceeding.

**Stop here for review before Phase 1.**

---

## Phase 1 — Analytic reconstruction (no rendering)

Prove the math works with zero render cost.

### Build

- **Swarm generator**: N drones at known 3D positions in a bounded volume.
  Start N = 3–5.
- **Camera rig**: K cameras placed around the swarm with known intrinsics and
  extrinsics. Use the *same conventions as the Blender rig* so Phase 2 is a
  drop-in, not a rewrite.
- **Projection**: project true 3D positions through each camera to get 2D image
  points. Verify this by hand: take one drone, one camera, compute the expected
  pixel coordinate manually, and confirm the code agrees. Record both numbers.
- **Correspondence solver — epipolar consistency.** Not nearest-reprojection
  matching. Nearest-reprojection would trivially succeed here by using the
  ground-truth identity you will not have in Phase 2, and would prove nothing.
  - Compute the fundamental matrix for each camera pair from known K, R, t.
  - For a seed camera pair, candidate matches are detections lying within a
    threshold of the epipolar line.
  - Extend candidate matches into remaining views: triangulate the pair,
    reproject into each other view, accept the nearest detection within
    threshold, reject the track if reprojection error is too high.
  - Resolve competing assignments with a global assignment step (Hungarian or
    equivalent), not greedy-first-match.
- **Triangulation**: linear DLT across all views in a track, then optional
  nonlinear refinement minimizing total reprojection error.

### The failure mode to expect and measure

With few cameras and dense swarms, epipolar constraints admit **ghost points** —
3D positions that are perfectly consistent with detections in every view but
correspond to no real drone. This is the same phenomenon as phantom particles in
3D particle tracking velocimetry, and it is the actual reason two cameras will
perform badly. Not triangulation precision. Ghosts.

This matters for the deliverable's story: the camera-count curve will likely be
dominated by ghost suppression, and should improve sharply from 2 to 3 to 4
cameras and then flatten. Count and report ghosts explicitly as their own
quantity. If they do not appear at all, that is suspicious and needs
investigating, not celebrating.

### Noise

Run noiseless first — solely to confirm the logic is correct. A noiseless pass
is a checkpoint, not a result. Immediately follow with a sweep of small Gaussian
pixel noise on the 2D points. Noiseless-only numbers do not go in the deliverable.

**Stop for review once noiseless small-swarm reconstruction verifies.**

---

## Scoring — two numbers, never collapsed into one

Report these separately, always, in every result table.

**1. Correspondence accuracy**
- fraction of recovered tracks where every member detection belongs to the same
  true drone
- counts of: correct tracks, ghost tracks (match no real drone), merged tracks
  (mix two drones), missed drones (no track recovered)

**2. Position error, in drone-lengths**
- computed only over correctly-corresponded tracks
- matched estimate-to-truth by global assignment on 3D distance
- report **median and 95th percentile**, not mean alone

Drone-lengths because that is the unit that determines whether a reconstruction
is good enough to reason about swarm structure at all.

These stay separate because a good position error can conceal broken
correspondence that got lucky, and correspondence is the thing actually being
tested. Collapsing them into one "accuracy" figure is how this project gets
burned again.

**3. Visual side-by-side** — true swarm vs. reconstructed swarm, per run.
Qualitative only. Catches structural failures a scalar hides: mirrored swarms,
systematic offsets, collapsed geometry.

---

## Mandatory tautology checks

Run all of these before any result is reported as real. Each one is designed to
make a fake success visibly fail.

1. **Single-camera control.** Run the full pipeline with one camera. It must
   fail — 3D position is not recoverable from one view. If it produces a good
   answer, ground truth is leaking somewhere in the pipeline. Find it.
2. **Scrambled-correspondence control.** Deliberately shuffle the correspondence
   assignments. Position error must explode. If it stays low, the error metric
   is not measuring what it claims to measure.
3. **Known-offset check.** Displace one true drone by a known amount. The
   reported error for that drone must change by that amount.
4. **Object Index label stripping (Phase 2, critical).** The Object Index EXR
   pass hands you perfect correspondence for free — it literally labels which
   pixels belong to which drone. It may be used **only** to extract detection
   centroids. Labels must be discarded before the correspondence solver ever
   sees them. If labels reach the solver, correspondence accuracy will be 100%
   by construction and the entire result is tautological. This is the same shape
   of failure as the exact-zero divergence in the spoof-mode scale test.

---

## Phase 2 — Real pixels

- Render scenes with the existing low-realism pipeline. Cube drones, simple
  lighting. Do not improve visuals.
- Extract detection centroids from the Object Index pass, strip labels, feed the
  identical solver from Phase 1.
- Compare Phase 2 numbers against Phase 1 at matched configurations. They should
  be close. If Phase 2 is substantially *better* than analytic Phase 1, that is
  not good news — it means ground truth is leaking. Investigate before
  proceeding.
- Measure and report wall-clock for one scene before committing to a full sweep.

---

## Phase 3 — The deliverable

A grid, not a single curve:

- **camera count**: 2, 3, 4, 6, 8, 10+
- **swarm size**: small (3–5), medium (10–15)
- **noise level**: at least two settings

At every grid point, report: correspondence accuracy, ghost count, median
position error in drone-lengths, 95th percentile position error.

Expected narrative to confirm or falsify: accuracy is limited by correspondence
ambiguity rather than triangulation precision, ghosts dominate at low camera
counts, and denser swarms need more cameras to reach the same accuracy. If the
data says otherwise, the data wins and gets written up as-is.

---

## Working discipline

- One task per turn. Commit after each verified milestone.
- Cheap smoke test before every expensive run. Wall-clock measured before any
  long render or sweep.
- PROGRESS.md updated continuously as working memory, not at the end.
- Stop at every phase boundary for review.
- **Every result verified with real evidence** — actual sampled values, actual
  positions, actual images inspected. Never "ran without error." Never
  "looks right."
- Suspiciously clean results — exact zeros, perfect scores, tight convergence —
  are treated as bugs until proven otherwise.
- Falsified results and inconclusive findings get written down, not smoothed over.

The governing question at every checkpoint: **can these two numbers
simultaneously be true?**
