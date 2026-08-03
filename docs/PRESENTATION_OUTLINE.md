# Presentation Outline — Camera-Based Drone Swarm Reconstruction

> **For Chief Scientist review.** Slide-level outline, not prose. Every number
> traceable to `logs/ml_sweep/` (full test set: 1000 scenes, 29000 rows).

---

## Slide 1 — Title

**From Pixels to Adjacency: Camera-Based Distance Estimation for Drone Swarm Splitting**

Subtitle: Recovering 3D positions and pairwise connectivity from multi-view imagery

---

## Slide 2 — The question and why it matters

Chen et al.'s GA/PSO critical-node attack assumes perfect positional data from simulation.
In a real scenario, a hostile swarm's radio link state is not observable — you can only
*see* it.

**This work:** estimate 3D drone positions from multi-camera imagery, compute pairwise
distances, threshold at the assumed comms range, and produce an inferred adjacency matrix
as a drop-in for the GA/PSO code — without simulation access.

---

## Slide 3 — Method (pipeline diagram)

```
Multi-camera renders (24 cameras, dome rig)
        │
        ▼
  detect_blobs          ← blob detector on plain-background renders
        │
        ▼
  solve_correspondence   ← epipolar geometry (±3 px threshold)
        │
        ▼
  triangulate_dlt       ← Direct Linear Transform on matched multi-view detections
        │
        ▼
  pairwise distances    ← Hungarian matching against ground truth
        │
        ▼
  mAP@tau / median position error / adjacency accuracy
```

Frozen pipeline, wiring only. Detection runs on rendered PNGs — the same pixels any
real detector would see. Ground truth is used for recall scoring and evaluation only,
never as pipeline input. Metrics through a single frozen `ml/metrics.py` implementation
built before either track produced numbers.

---

## Slide 4 — Headline results

| Metric | Value | Condition |
|---|---|---|
| mAP (avg over tau 0.5/1/2/5 m) | **0.9859** | 500 test scenes, primary cell, V=8, mixed |
| Median matched position error | **0.0315 m** | Primary cell, V=8, ground |
| Reconstruction count accuracy | count_err ≤ +0.75 | Primary V=8 across compositions |
| Detector recall (24 views) | 0.984–0.998 | Across density bins, scoring-only |

Full test set (1000 scenes). Primary cell: R=50 m, a_max 9.6 px/drone, standoff 139 m.
Secondary cell: R=100 m, a_max 4.8 px/drone, standoff 278 m.

---

## Slide 5 — How many camera angles do you need?

**Knee at V=3.** The largest single-step mAP gain is always at 3 views (7 of 8
composition/cell combinations; secondary/level at V=5).

| V | Primary mAP (mixed) | Secondary mAP (mixed) |
|---|---|---|
| 2 | 0.9439 | 0.9331 |
| **3** | **0.9709** | **0.9650** |
| 4 | 0.9765 | 0.9737 |
| 5 | 0.9803 | 0.9788 |
| 8 | 0.9859 | 0.9863 |

Diminishing returns past V=4. Going from 3→8 views adds only ~1.5 mAP points.

---

## Slide 6 — Does camera placement matter?

**Only at V=2–3, and only when view count is low.** The mixed-tier composition (interleaved
ground/level/aerial) beats all-ground cameras at low view counts; the gap widens with drone
density.

| Scenario | V=2 delta mAP | V=8 delta mAP |
|---|---|---|
| Primary high density (55 drones) | +0.0829 | +0.0003 |
| Secondary high density (55 drones) | +0.0729 | +0.0006 |
| Primary low density (10 drones) | +0.0390 | +0.0008 |

All compositions saturate near 0.98 by V=8. **Practical implication:** cheap ground
cameras suffice given enough of them. Mixed placement buys margin only when you have
very few views.

---

## Slide 7 — Where it breaks: count error vs density

The baseline over-predicts drone count as density grows — false tracks from coincidental
epipolar alignments create phantom drones.

| Cell | Low (~10 drones) | Mid (~35 drones) | High (~55 drones) |
|---|---|---|---|
| Primary | −0.06 | +0.26 | +1.24 |
| Secondary | −0.06 | +0.31 | +1.50 |

Monotonic in both cells (P5 confirmed). At high density the baseline overshoots by
~1–2 drones — below the ~50-drone false-track threshold that defines the high-density
bin. This is the regime where learned false-positive rejection would help most.

---

## Slide 8 — The ML track: built, structural limit found

A voxel-fusion model was trained on the same scenes. It learned drone *positions* but
not *sharpness* — a diffuse volume with ~1,000 spurious local maxima vs 231 true drones.

**Root cause (measured, not guessed):** at encoder stride 8, each 0.5 m drone occupies
~1.2 feature pixels — the encoder cannot produce a peak narrower than its receptive
field. This is a physics/architecture limit, not a training bug.

**G2 gate stays failed** (count error +241 to +354, gate requires ±1).

**Named path forward (not run in V1):**
- Higher feature resolution (stride ≤ 4, dilated convs, or FPN upsampling head)
- Multi-hour training runs (best attempt was 600 steps at 76 min/epoch)

---

## Slide 9 — Scope limits

*Stated on their own slide, not buried.*

- **Plain backgrounds are the easiest possible detection case.** Detector recall is
  0.984–0.998. All numbers here are an *upper bound* on field performance.
- **Near-perfect recall compresses composition differences.** By V=8 every composition
  hits ~0.98 mAP — the study's composition signal is visible only at low view counts
  because that is where the detector's margin matters.
- **Camera pose is given, not estimated.** Real deployment would need pose estimation;
  this work measures reconstruction conditional on known pose.
- **Synthetic-to-real gap is unmeasured.** Real imagery, real drones, real deployment
  are out of scope for this feasibility phase.
- **Static single timestep only.** 5–60 drones, no temporal input.

---

## Slide 10 — What comes next

1. **Temporal detection** (frame differencing on real multi-camera video) — replaces the
   blob detector; works at <1 px apparent size if the drone moves between frames.
2. **Correspondence without oracle** — epipolar + temporal consistency matching across
   anonymous detections (the correspondence debt this work deferred via Blender object IDs).
3. **Learned false-positive rejection** — the regime where the classical baseline's
   count error grows (high density) is exactly where a small classifier would help.
4. **Integration with GA/PSO** — the reconstructed adjacency matrix drops directly into
   the existing critical-node search (separate repo, unchanged).

---

## Appendix — Data summary

| Parameter | Value |
|---|---|
| Test scenes | 1000 (500 primary + 500 secondary) |
| Cells | Primary R=50 m (a_max 9.6 px), Secondary R=100 m (a_max 4.8 px) |
| Camera rig | 24 cameras, dome, 24 mm FF equiv, W=1920, f=2666.67 px |
| Sweep grid | 7 view counts × 4 compositions × 2 cells × ~500 scenes |
| Sweep wall-clock | 91.9 min (14 parallel jobs) |
| Frozen test suite | 55/55 passing |
| Primary source | `logs/ml_sweep/eval_sweep_report.md`, `eval_sweep_summary.json` |
| Plots | `logs/ml_sweep/plot_mAP_vs_views.png`, `plot_median_err_vs_views.png` |
