# Spec: Learned Multi-View Swarm Reconstruction

**Date:** 2026-07-31
**Repo:** `ShrikarSwami/swarm-cv-distance`
**Status:** ready to launch, T0 first
**Operating rules:** `docs/superpowers/ORCHESTRATOR_ML.md`
**Extends:** `2026-07-30-end-to-end-demo-design.md`, which remains source of truth
for the geometric track.

---

## 1. The question

Can a learned model recover the 3D positions of a drone swarm from a variable
number of rendered camera views with known pose, and how does accuracy depend on
(a) how many views it gets and (b) where those views are placed?

The geometric track (epipolar + DLT) already answers a related question and
serves as the control. This spec adds a learned model and evaluates both on
identical data with identical metrics.

## 2. Scope

**In scope**
- Synthetic swarms, 5 to 60 drones, static single timestep
- Plain uniform backgrounds only
- Known camera intrinsics and extrinsics, supplied to the model
- Learned end-to-end 3D position regression, permutation-invariant
- Comparison against the existing geometric baseline on the same scenes
- Interactive UI for scene and angle selection

**Not in scope**
- Real imagery, real drones, real deployment
- Unposed images (camera pose estimation from image content)
- Temporal / video input
- Photorealistic environments
- Beating the geometric baseline. Matching or characterising it is sufficient.

**Stated scope limits, to appear in the writeup**
- Plain backgrounds are the easiest possible detection case. All numbers here are
  an upper bound on performance, not an estimate of field performance.
- Camera pose is given, not estimated. Real deployment would need to solve that.
- Synthetic-to-real gap is unmeasured and unbounded by this work.

**This is a feasibility proof, not a product.** The bar is a defensible finding,
not a perfect one. Where a choice trades rigour against scope, prefer narrowing
scope over weakening rigour: fewer densities, fewer compositions, fewer scenes
are all acceptable. Loosening a gate is not.

## 3. Pre-registered predictions

Log predicted, observed, ratio, match. Falsified predictions go in the report.

| ID | Prediction | Basis |
|---|---|---|
| P1 | Mixed-tier compositions beat all-ground at equal view count, gap widens with density | Geometric finding #4: upticks for all_ground, coplanar cameras are ill-conditioned |
| P2 | mAP rises with view count, knee between 3 and 5 views | Diminishing returns once triangulation is over-determined |
| P3 | ML does not beat geometric at low density | Geometric is near-optimal under near-perfect detection |
| P4 | ML degrades more gracefully than geometric above ~50 drones | Geometric false-track threshold formula; learned model has no explicit correspondence step to fail |
| P5 | Count error grows monotonically with density | Occlusion and blob merging |
| P6 | At least one tier-composition will appear anomalously good by chance | ~161 compositions swept; multiple comparisons |
| P7 | FALSIFIED. Predicted 4.8px at R=100/W=1920, measured 6.38px, ratio
  1.33. Cause: antialiasing coverage floor. A 4.8px square rasterizes across
  ~6px at OTSU threshold. The formula describes geometric projection; the
  detector consumes the AA-widened blob. Under-predicts in the safe
  direction. |
| P7b | **OPEN** (Ruling 2026-07-31) — NOT resolved. Two candidate models,
  and the data cannot yet separate them:
  (A) **additive**: measured_extent(OTSU) = a_max + ~1.0–1.2px (P7b as
      registered, the +1.2 upper edge from the T0 references);
  (B) **multiplicative**: measured_extent = ~1.33 × a_max (the P7 ratio).
  The two are indistinguishable at the P7b verification scale (R=50m, a_max
  9.6px): OTSU bbox is the most quantized measure in the table (integers only:
  6 and 12 px), and an integer bbox cannot tell a true 10.8px from 12px, so
  the cleanest-looking measure is also the least discriminating.
  **Contradiction from T0 references:** T0's well-resolved reference renders
  (48px @ 10m, 19.2px @ 25m) reportedly matched the formula within ~+1px. At
  a_max 19.2, (A) predicts 20.4px and (B) predicts 25.5px; T0 reported ~20px.
  Either those references used a different measurement convention (which must
  be stated), or (B) is already falsified by T0 data. Two points cannot
  distinguish additive from multiplicative, and both models were fit AFTER the
  P7 prediction failed — a fitted curve, not a validated one.
  Does not block anything: at R=50 the measured extent is directly observed,
  and the encoder-stride constraint holds either way. |
| P7c | **QUEUED** (Ruling 2026-07-31 — do not run now). One render at R=25m
  (a_max 19.2px), where the models diverge by ~5px. PREDICT BOTH VALUES
  BEFORE MEASURING: (A) additive ≈ 20.4px, (B) multiplicative ≈ 25.5px.
  Then reconcile against the T0 reference points. |

P6 is the guard, not a finding. See T8.

P7 is the T0 sanity check. A measured extent far from the formula means the
render, the scene scale, or the measurement is wrong, and it must be resolved
before any dataset generation.

## 4. Architecture

Multi-view voxel fusion. Chosen over flat MLP regression because swarm position
prediction is a **set** prediction problem with no natural ordering.

1. Shared-weight 2D CNN encoder per view
2. Back-project per-view features into a shared voxel grid over the swarm volume,
   using known intrinsics and extrinsics
3. Pool across views (mean and max, concatenated) at each voxel
4. Small 3D CNN decodes fused volume to a 3D occupancy heatmap
5. Positions = local maxima, soft-argmax for differentiability

Encoder stride constraint. At a_max ~10px, encoder stride must not exceed
8, leaving ~1.2 feature pixels per drone. Stride-16 leaves 0.4 and the
drone vanishes before the 3D head. Use at most 3 downsampling stages, or
dilated convolutions at full stride. This is a hard architectural
constraint from T0, not a tuning choice.

**Why this and not flat regression**
- Permutation invariant by construction, no Hungarian matching in the loss
- Variable view count handled by the pooling step, no architecture change between
  a 2-view run and an 8-view run
- Multi-view geometry is an inductive bias rather than something the network must
  rediscover, which matters at this dataset scale

**Training target:** 3D Gaussian heatmap centred on each ground-truth drone
position. Derived directly from known scene geometry. **No Object Index pass is
required for the ML track.** The Blender emission-shader bug does not gate this
work.

**Loss:** MSE on heatmap, plus a count-regularising term. Revisit if count error
dominates.

## 5. Camera tiers

Tier by elevation angle relative to swarm centroid:

| Tier | Elevation | Real-world analogue | Deployment cost |
|---|---|---|---|
| `ground` | looking up, elev < -20 deg | fixed ground sensors | cheap |
| `level` | -20 to +20 deg | peer drones at altitude | moderate |
| `aerial` | looking down, elev > +20 deg | overwatch above swarm | expensive |

Azimuth is sampled to a controlled spread per composition so that "4 ground
cameras" does not silently mean "4 ground cameras clustered on one side."
Azimuth spread is logged per run and reported as a covariate.

**Render budget: 8 per tier, 24 angles per scene.** Pure-tier and mixed
compositions both reach 8 views, so composition comparisons are balanced.
Measured render cost is 0.11s/frame, so the extra 6 angles cost ~0.5s per
scene. Delete the pure-tier cap note; it no longer applies.
Resolves open item 3.

**One model, many compositions.** The model trains on randomly sampled subsets
(random count, random tier mix) and is evaluated across all compositions. Do not
train per composition.

## 6. Metrics

Shared harness, both tracks, identical code path. `ml/metrics.py` is built once,
before either track produces a number, and frozen on acceptance.

- Hungarian matching of predicted to ground-truth points under threshold tau
- Precision / recall / F1 at tau
- **AP@tau** using heatmap peak value as confidence, PR curve integrated
- **mAP** averaged over tau in {0.5, 1, 2, 5} m — headline number
- **Median position error** on matched pairs, metres — comparable to geometric track
- **Chamfer distance**, threshold-free set-level check
- **Count error**, predicted N vs true N

Headline graph: mAP vs number of views, one curve per tier composition.

Any metric compared across configurations is computed on the intersection set or
labelled non-comparable. Inherited rule, non-negotiable.

Report all results against a_max in pixels-per-drone, not swarm radius in
metres. a_max depends only on d/R and W; absolute metres never enter.
"Reconstruction works down to N pixels per drone" lets any reader map their
own drone size, swarm extent, and sensor onto the curve, and pre-empts the
objection that a 50m-radius swarm is small. The finding was never about
metres.

## 7. Storage layout

All data on internal SSD. The external drive is NOT used.

```
~/swarm_ml/
  scenes/00/00001/
    ground_truth.json
    cameras.json
    angle_00.png ... angle_23.png
  manifest.jsonl
~/swarm_ml_packed/
  shard_0000.npz
  splits.json
```

Measured: 33KB/PNG at 1920x1080 plain background.
24 angles x 33KB = ~800KB/scene.
5,000 scenes  ~= 4 GB
20,000 scenes ~= 16 GB
Internal free: 99 GiB. Not a constraint.

CRITICAL for T4: shards store PNG BYTES, not decoded arrays. Raw uint8 at
1080p is 6.2MB/image; packing naively makes 5,000 scenes 560GB and fills
the disk. Decode on load, multiple dataloader workers.

/Volumes/My Passport/ is read-only NTFS. No agent writes to it or deletes
anything on it.

## 8. Tasks

A component is done when its acceptance command exits 0, run by the orchestrator,
never because an agent reported success.

### T0 — Calibration and detectability gate

Measure, do not assume. This task **chooses the operating point** rather than
measuring a pre-chosen one. Standoff and focal length cancel in the apparent-size
relation, so apparent size depends only on drone size, image width, and swarm
radius.

Outputs a single JSON of measured constants containing:

- **a_max table** over the grid
  `swarm_radius {50, 100, 200, 400, 800} m` x `image_width {1280, 1920, 2560, 3840}`
  where `a_max = drone_size * image_width / (2 * swarm_radius)`, drone_size 0.5 m
- **measured EEVEE seconds per frame** at each candidate resolution, M4 Pro
- **PNG bytes per render**, plain background, at each candidate resolution
- **empirical verification of one cell**: render it, measure actual blob pixel
  extent, confirm it matches the formula (P7)
- free space: internal SSD and external drive
- external drive confirmed HDD or SSD

**Gate G0: at least one grid cell with a_max >= 2 px at an acceptable render
time.** If no cell qualifies, halt and renegotiate drone size or swarm extent.
Escalate the table either way; the human picks the operating cell.

Acceptance: `python tools/calibrate.py --out calib.json` exits 0, all fields
populated, a_max table complete, empirical cell measured, gate reported
explicitly as pass or fail.

### T1 — Scene and camera generator

Random swarm within the chosen radius, N in [5,60], plain background. 24 cameras,
8 per tier (PATCH 3), controlled azimuth spread. Writes `ground_truth.json` +
`cameras.json`. Minimum inter-drone spacing enforced **at generation time**, not
asserted afterwards, so unlucky seeds cannot crash a multi-day run.

Acceptance: generates 3 scenes, schema-validates, camera tiers verified by
recomputing elevation from extrinsics rather than trusting the label.

Operating cells, both at W=1920, f=2666.67px (repo default, ~40deg HFOV):
    primary   R=50m,  standoff 139m, a_max 9.6px
    secondary R=100m, standoff 278m, a_max 4.8px

Standoff derives from the framing constraint: standoff = R * 2f/W = 2.778*R.
Do NOT use the 90deg-HFOV convention from T0's empirical cell. HFOV is a
free parameter; a_max only requires that the swarm fills the frame, so any
(focal, standoff) pair satisfying that gives identical apparent size. Keep
the repo focal — 90deg is near-fisheye and no real sensor uses it — and
stay consistent with the frozen geometric track.

Scenes are tagged with their cell. Generate both cells in the campaign: each split's
seed range is split into disjoint per-cell sub-ranges — primary takes the lower half,
secondary the upper half (test 0-499/500-999, val 1000-1499/1500-1999, train
2000-3499/3500-4999). The seed-keying harness property (T2) makes this mandatory: a
seed rendered under one cell can never be rendered again under the other.

Reserved seed ranges, fixed permanently before any data is generated:
    seeds 0-999      TEST. Locked. Not touched until final evaluation.
    seeds 1000-1999  VAL. Iteration and model selection.
    seeds 2000+      TRAIN. Extendable indefinitely.

Campaign extension adds train seeds only. Test contamination is structurally
impossible rather than procedurally avoided. Re-evaluating on the same
holdout after every data increment inflates confidence without improving
generalisation; the val set absorbs that, the test set is read rarely.

Scenes are seed-indexed and manifest-driven, so `--target 5000` after a
2000-scene run adds 3000 and regenerates nothing.

### T2 — Resumable render harness

- JSONL manifest, atomic append after each scene
- Control file polled at scene boundary: `RUNNING` / `PAUSED` / `STOP`
- Resume is idempotent, skips scenes already in manifest
- Drive-presence check before each scene, auto-pause rather than crash
- Temp path + atomic rename, so unplug never leaves a half-written scene
- Detached via `nohup`, wrapped in `caffeinate`

**KNOWN HARNESS PROPERTY (2026-07-31, scratch-test verified):** scene directories are
keyed by **seed alone**, not seed+cell (`scene_dir(root, seed)`). Rendering the same
seed under a second `--cell` is a **silent no-op**: the seed is already in the manifest,
so `pending_seeds` returns empty, the harness logs "target reached", exits 0, and
creates nothing. There is no error anywhere — a primary-only dataset would be produced
silently. Mitigation, in force for this campaign: **disjoint seed ranges per cell** —

    test   0-499 primary / 500-999 secondary
    val    1000-1499 primary / 1500-1999 secondary
    train  2000-3499 primary / 3500-4999 secondary

Any future campaign extension must use a **fresh seed range**; never re-render existing
seeds under a different cell. Adding the secondary cell to a seed set already rendered
primary would silently produce nothing.

**Built and smoke-tested by an agent. The full campaign is launched by the human,
detached, outside any agent session.**

Acceptance: start a 10-scene run, `kill -9` mid-scene, resume, verify no
duplicate and no truncated scene. Unmount the drive mid-run, verify clean pause.

### T3 — Status page (pinned tab)

Local page: scenes done, measured rate, ETA computed from measured rate rather
than estimated, pause and resume buttons writing the control file.

Acceptance: pause from UI, confirm generator halts at next scene boundary,
resume, confirm continuation.

### T4 — Dataset packing and splits

Pack to shards on internal SSD. **Split by scene seed**, never by image.

Acceptance: assert zero scene-seed overlap across train/val/test (G1).

Reserved seed ranges, fixed permanently before any data is generated:
    seeds 0-999      TEST. Locked. Not touched until final evaluation.
    seeds 1000-1999  VAL. Iteration and model selection.
    seeds 2000+      TRAIN. Extendable indefinitely.

Campaign extension adds train seeds only. Test contamination is structurally
impossible rather than procedurally avoided. Re-evaluating on the same
holdout after every data increment inflates confidence without improving
generalisation; the val set absorbs that, the test set is read rarely.

Scenes are seed-indexed and manifest-driven, so `--target 5000` after a
2000-scene run adds 3000 and regenerates nothing.

### T5 — Geometric baseline on identical scenes

Reuse the existing epipolar + DLT path unchanged. It is the control and is
frozen.

**Detection source: run `detect_blobs.py` on the same PNGs the ML model sees.**
No Object Index pass, no ground-truth 2D projections. Both tracks consume
identical pixels. Detector recall on plain backgrounds is **measured and
reported**, not assumed perfect; if it is near-perfect, that is a stated
condition of the comparison.

If the geometric track is dropped for scope reasons, T5 and the ML-vs-geometric
sections of T7 and T10 drop with it, and G4 becomes inapplicable. That is a human
decision, not an agent decision.

Acceptance: baseline mAP curve produced over the same composition sweep, using
the frozen `ml/metrics.py`.

### T6 — Model and training loop

Voxel fusion per Section 4. MPS backend. Random view-subset sampling during
training. Cheap smoke test before any long run. Wall-clock estimate required
before full training.

**Overfit gate first (G2): the model must drive loss to near zero on 8 scenes
before any full training run begins.** If it cannot overfit a handful of scenes,
the architecture or the target construction is wrong and more data will not fix
it.

Full training is launched detached by the human, not inside an agent session.

Acceptance: G2 passes, then full training completes and checkpoints load.

### T7 — Evaluation sweep

Both tracks, all compositions, all densities, shared frozen harness.

**SIZED (Ruling 2026-07-31).** The naive full sweep is
~161 compositions x 6 density bins x 500 scenes = ~480,000 forward passes,
roughly 40h on MPS — longer than rendering and training combined. Do NOT
launch the naive version. Sized instead: **100 test scenes per cell, 3 density
bins (low ~10, mid ~35, high ~55, straddling the ~50-drone false-track
threshold)** = ~48,000 forward passes. This sizing is fixed in the spec now;
it is not a tuning choice.

Acceptance: results table + mAP-vs-views plots, one curve per composition; G4
evaluated and reported.

### T8 — Held-out confirmation of composition winners

Any composition that looks anomalously good gets re-tested on a scene set not
used anywhere in T7. P6 says at least one will look good by chance.

Acceptance: winner confirmed on held-out set, or explicitly reported as not
reproducing (G3).

### T9 — Reconstruction UI

Select scene, then either pick exact angles, pick a random N, or pick a random
number of random angles. Runs model, renders predicted vs ground-truth 3D overlay
plus metrics for that run. Checks external drive is mounted at startup.

Acceptance: end-to-end from scene selection to overlaid reconstruction.

### T10 — `docs/RESULTS_ML.md`

Written by the human, not an agent. Question, method, controls, mAP vs views per
composition, ML vs geometric, density behaviour, prediction log with
falsifications, scope limits from Section 2.

## 9. Gate table

| Gate | Condition | Action if failed |
|---|---|---|
| G0 | At least one a_max cell >= 2 px at acceptable render time | Halt, renegotiate drone size or swarm extent |
| G1 | Zero scene-seed overlap across train/val/test | Halt, fix split |
| G2 | Model drives loss to near zero on 8 scenes | Halt, architecture or target construction is wrong |
| G3 | Composition winner reproduces on held-out set | Report as non-reproducing |
| G4 | `ml_median_error >= 0.9 * geometric_median_error` at density <= 20 and views >= 4 | Halt, investigate leak before believing the result |

G4 is the equivalent of the geometric track's `phase2 >= phase1 * 0.9` check. A
learned model beating well-conditioned triangulation under near-perfect detection
at low density is a bug signal, not a win.

### Gate status (Session 15e, 2026-08-02)

**G2 — NOT PASSED; ML TRACK PAUSED (documented negative result).** The overfit gate
fails and the cause is structural, not a bug.

- Best checkpoint (600 steps, weighted MSE `pos_weight=500`, `target_sigma=2.0`):
  aggregate median_err 1.0196 m, count error +241..+354 (gate requires per-scene
  count_err in [-1,+1]).
- **Structural cause:** at encoder stride 8 with a_max ~10 px/drone, each drone
  occupies ~1.2 feature pixels, so the encoder cannot produce a peak narrower than
  its receptive field. PATCH 5 set stride <= 8 as a floor; the model is at the floor.
- **Measured evidence (Agent H, `ml/extract_debug.py`, orchestrator-run
  `/tmp/extract_debug_acceptance.txt`):** the model learned POSITION but not
  SHARPNESS. Absolute-threshold sweep from 1e-3*max to 0.7*max removes zero peaks
  (count error stays +241..+354 through 0.5*max); true-drone-location voxels sit at
  2.4:1 over background and 98.3% exceed 0.5*pred_max; but 1,012 local maxima share
  the same value band as 231 true drones — no threshold separates them. The diffuse
  field is a genuine architecture limit, not an extraction defect.
- **Do not attempt further fixes in V1.** Cost is unbounded (76 min/epoch; five loss
  variants already tried without a clean long run) and the comparison it would enable
  is saturated (baseline mAP 0.98, detector recall 0.985-1.000; G4 says a low-density
  win is a leak signal anyway).
- **Named path forward (NOT ATTEMPTED IN V1):** higher feature resolution — stride <= 4,
  dilated convolutions, or an FPN-style upsampling head — plus multi-hour training runs.

**G4 — NOT ATTEMPTED IN V1.** Depends on the ML track, which is not attempted in V1
(see G2 above). Marked NOT ATTEMPTED, not passed; do not present it as untried.

## 10. Execution model

### Orchestrator and subagents

Built via a Claude Code orchestrator loop per
`docs/superpowers/ORCHESTRATOR_ML.md`. The orchestrator reads this spec, spawns
one **superpowers subagent** per component with exclusive file ownership, and
verifies completion by running the acceptance command itself. A subagent's report
is never evidence.

File ownership map and frozen paths live in `ORCHESTRATOR_ML.md`. Summary:
- Entire geometric track (`stage1_geometry/**`) is FROZEN. It is the control.
- `ml/metrics.py`, `calib.json`, `ml/splits.json`, and all prediction tests are
  frozen once their acceptance passes.
- `docs/RESULTS_ML.md` is human-owned. No agent writes findings.

### Skills

- **findskills** — run at session start to surface any relevant skill not
  anticipated here. Cheap, and this project has already benefited from tooling
  discovered mid-build.
- **superpowers** — subagent fan-out. One agent per component, ownership enforced
  by the orchestrator, which rejects any diff touching a path the agent does not
  own.
- **claude-mem** — cross-session context, since this campaign spans days and
  includes multi-day detached renders. **`docs/PROGRESS.md` remains the authority
  for gate state.** Memory is convenience, not evidence. Never resolve a gate from
  memory; re-run the command.

### Prediction tests

`tests/test_predictions_ml.py` is written **before any component is built** and
escalated for approval. It encodes geometry and invariants as assertions so
architecture bugs fail loudly instead of surfacing as mediocre accuracy after
hours of training. Contents specified in `ORCHESTRATOR_ML.md`; the two highest
value are back-projection round-trip and permutation invariance.

### Long runs

No run expected to exceed roughly 10 minutes executes inside an agent session.
The rendered sweep on the geometric track hung when run this way. Agents build
and smoke-test; the human launches detached.

## 11. Working discipline

Inherited, unchanged.

- Every claim backed by a run in that session. No numbers carried forward.
- Predict before running. Log predicted, observed, ratio, match.
- Never loosen an acceptance criterion, tau set, or match threshold to make code
  pass. Escalate.
- Suspiciously clean results are bugs until proven otherwise: exact zeros,
  perfect scores, a metric unchanged after a parameter change.
- Falsified predictions and partial results go in the report, not a footnote.
- One task per turn. Commit at each milestone. `PROGRESS.md` updated throughout as
  working memory, not at the end.
- Wall-clock estimate before expensive operations. Cheap smoke test before long
  renders or training runs.

## 12. Open items

1. **RESOLVED.** External drive HDD vs SSD unconfirmed. Changes the storage
   argument if SSD. T0 resolves.
   - Item 1 resolved: external drive is HDD, read-only NTFS, dropped from plan.
2. **RESOLVED.** Operating point (swarm radius, image width) chosen from the T0
   a_max table by the human, not by an agent.
   - Item 2 resolved: two cells per PATCH 4.
3. **RESOLVED.** Whether to raise to 8 cameras per tier depends on measured
   render time. Decide after T0.
   - Item 3 resolved: 8 per tier per PATCH 3.
4. Whether the geometric baseline stays in scope is a human decision. Default is
   that it stays: it is the control, the adapter is roughly an hour of work, and
   it is the first comparison a reviewer will ask for.
5. Merged blobs at high density have no analytic equivalent. Watch count error and
   recall together.
6. Azimuth spread is a covariate that could masquerade as a tier effect. Logged
   per run, reported.
7. Plain background may make the task trivially easy, compressing differences
   between compositions. If all compositions score near-identically, that is the
   likely cause, and it is a finding about the experiment, not about geometry.
