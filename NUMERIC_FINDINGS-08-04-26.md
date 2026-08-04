# Numeric Findings — Camera-Based Swarm Reconstruction

**As of 2026-08-02.** Geometric method (epipolar + DLT). No learned model in
any number below.

> Compiled from session reports. Values marked (verify) should be checked
> against `logs/` before use in a paper or deck.

---

## 1. Camera rig

| Parameter | Value |
|---|---|
| Cameras rendered per scene | 24 (dome rig) |
| Per tier | 8 ground / 8 level / 8 aerial |
| Image resolution | 1920 x 1080 |
| Focal length | 2666.67 px (~40 deg HFOV, repo default) |
| Tier definition | by elevation angle to swarm centroid |
| `ground` | elev < -20 deg (looking up) — cheap, fixed sensors |
| `level` | -20 to +20 deg — peer drones at altitude |
| `aerial` | elev > +20 deg (looking down) — expensive overwatch |

Tier labels verified by recomputing elevation from extrinsics, not trusted
from the label.

---

## 2. Operating cells

Standoff derives from the framing constraint `standoff = R * 2f/W = 2.778*R`.
Focal and standoff cancel in the apparent-size relation.

| Cell | Swarm radius | Standoff | a_max | Drones/scene |
|---|---|---|---|---|
| Primary | 50 m | 139 m | 9.6 px | 5-60 |
| Secondary | 100 m | 278 m | 4.8 px | 5-60 |

**Apparent size:** `a_max = drone_size * W / (2 * R)`, drone_size 0.5 m.
Standoff and focal cancel. Report results against a_max in pixels-per-drone,
not metres — a_max depends only on `d/R` and `W`.

**Encoder constraint (ML track):** at a_max ~10 px, encoder stride must not
exceed 8. Stride-16 leaves 0.4 feature pixels per drone and the drone
vanishes before the 3D head.

---

## 3. Dataset

| Parameter | Value |
|---|---|
| Total scenes | 5,000 |
| Test / val / train | 1,000 / 1,000 / 3,000 |
| Seed ranges | test 0-999, val 1000-1999, train 2000-4999 |
| Per-cell split | primary lower half, secondary upper half of each range |
| Cell balance | 500/500 test, 500/500 val, 1500/1500 train |
| Images | 120,000 (5,000 x 24) |
| On disk, raw | 5.2 GB |
| Packed shards | 5.215 GB, 157 shards @ 32 scenes |
| npz overhead | 1.35% |
| Backgrounds | plain, uniform |
| Timestep | static, single |

**Render performance (Blender EEVEE, M4 Pro):**

| Metric | Value |
|---|---|
| Per frame | 0.11 s at 1920x1080 |
| Per scene (24 frames + startup) | ~3.2-3.6 s |
| Full campaign wall-clock | 4h 23m (21:21 -> 01:44) |
| Bytes per scene | ~1,360 KB |
| Bytes per PNG at N=60 | ~57 KB |
| Bytes per PNG, single cube | ~33 KB |
| Packing 5,000 scenes | 31.1 s |

**Render time by resolution (measured, T0):**

| Resolution | s/frame | PNG bytes |
|---|---|---|
| 1280x720 | 0.06 | 16,714 |
| 1920x1080 | 0.11 | 33,441 |
| 2560x1440 | 0.19 | 56,009 |
| 3840x2160 | 0.45 | 118,436 |

---

## 4. a_max grid (T0, gate G0 PASS — 12 cells >= 2 px)

`a_max = 0.5 * W / (2 * R)` px

| R \ W | 1280 | 1920 | 2560 | 3840 |
|---|---|---|---|---|
| 50 m | 6.4 | 9.6 | 12.8 | 19.2 |
| 100 m | 3.2 | 4.8 | 6.4 | 9.6 |
| 200 m | 1.6 | 2.4 | 3.2 | 4.8 |
| 400 m | 0.8 | 1.2 | 1.6 | 2.4 |
| 800 m | 0.4 | 0.6 | 0.8 | 1.2 |

**Measured vs formula (P7 — OPEN, unresolved):** measured OTSU extent exceeds
the geometric projection. Two candidate models, neither validated:

| a_max | measured OTSU bbox | additive (+1.2) | multiplicative (x1.33) |
|---|---|---|---|
| 4.8 px (R=100) | 6 px | 6.0 | 6.4 |
| 9.6 px (R=50) | 12 px | 10.8 | 12.8 |

Two points cannot separate the models; both were fit after P7 failed. T0
reference renders (48 px @ 10 m, 19.2 px @ 25 m) reportedly matched within
~+1 px, which favours additive. **P7c queued:** one render at R=25 m
(a_max 19.2 px) where the models diverge by ~5 px. Predict both before
measuring.

Does not block: at R=50 the extent is directly observed at 12 px.

---

## 5. Headline results (geometric)

| Metric | Value | Condition |
|---|---|---|
| Median position error | **0.0315 m** | primary, V=8, ground |
| mAP (tau 0.5/1/2/5 m) | **0.9859** | primary, V=8, mixed, n=500 |
| mAP, all 24 views | 0.9890 primary / 0.9923 secondary | n=500 |
| Adjacency F1 | **0.982** | d_max >= 25 m, 200 scenes, V=8 mixed |
| Adjacency recall | **1.000** | all d_max — zero false negatives |
| Detector recall | 0.984-0.998 | across density bins, 24 views |
| Count error | <= +0.75 | primary V=8 across compositions |

---

## 6. mAP vs view count (mixed composition)

| V | Primary | Secondary |
|---|---|---|
| 2 | 0.9439 | 0.9331 |
| **3** | **0.9709** | **0.9650** |
| 4 | 0.9765 | 0.9737 |
| 5 | 0.9803 | 0.9788 |
| 8 | 0.9859 | 0.9863 |
| 24 | 0.9890 | 0.9923 |

**Knee at V=3** — largest single-step gain, 7 of 8 composition/cell
combinations (secondary/level knees at V=5). V=3 -> V=8 adds only ~1.5 mAP
points.

Monotonic across all 8 cells on the full test set. The V=7 -> 8 dip seen at
n=200 (~0.001) washed out at n=1000, upgrading P2 from PARTIAL to MATCH.

---

## 7. Camera placement (mixed minus all-ground, delta mAP)

| Scenario | V=2 | V=8 |
|---|---|---|
| Primary, high density (~55) | +0.0829 | +0.0003 |
| Primary, mid (~35) | +0.0640 | — |
| Primary, low (~10) | +0.0390 | +0.0008 |
| Secondary, high (~55) | +0.0729 | +0.0006 |
| Secondary, mid (~35) | +0.0577 | — |
| Secondary, low (~10) | +0.0255 | — |

Gap widens monotonically with density in both cells independently. All
compositions converge to ~0.98 by V=8.

**Operational reading:** cheap ground cameras suffice given enough of them.
Mixed placement buys margin only at very low view count.

---

## 8. Count error vs density

| Cell | ~10 drones | ~35 drones | ~55 drones |
|---|---|---|---|
| Primary | -0.06 | +0.26 | +1.24 |
| Secondary | -0.06 | +0.31 | +1.50 |

Monotonic in both cells (P5 MATCH). Overshoot at high density is consistent
with the false-track onset predicted by the geometric track's threshold
formula:

```
expected false candidates per point = n_drones * (2 * epipolar_threshold_px) / image_height
```

Epipolar threshold in use: 3.0 px.

---

## 9. Adjacency (200 scenes, V=8 mixed, primary)

| d_max | Precision | Recall | F1 |
|---|---|---|---|
| 10 m | 0.958 | 1.000 | 0.977 |
| 25 m | 0.967 | 1.000 | 0.982 |
| 50 m | 0.968 | 1.000 | 0.983 |
| 100 m | 0.969 | 1.000 | 0.984 |
| 200 m | 0.969 | 1.000 | 0.984 |
| 500 m | 0.969 | 1.000 | 0.984 |

Recall 1.000 everywhere: 3 cm position error against a 10 m minimum threshold
is a ~300x margin, so no true edge can fall below it. Precision is bounded by
count error — spurious drones create spurious edges — and improves as d_max
grows because extra edges matter less in a denser graph.

**For Chen et al.:** the GA/PSO critical-node search needs true edges present.
All are. Failure is in the harmless direction.

---

## 10. ML track (paused, G2 FAILED)

| Metric | Value |
|---|---|
| Position error, 8 overfit scenes | 0.985 m (gate < 1.0 m) |
| Count error | +286 to +364 (gate +/-1) |
| Spurious local maxima | ~1,012 vs 231 true drones |
| Peak-to-background ratio | 2.4:1 |
| True-loc voxel mean | 0.0987 |
| Volume mean | 0.0499 |
| Background median | 0.0416 |
| True-loc voxels above volume p99 | 51.5% |
| pred_max vs true-loc mean | 1.66x (strongest voxels NOT at drones) |

**Cause:** at encoder stride 8 each drone occupies ~1.2 feature pixels; the
encoder cannot produce a peak narrower than its receptive field. Structural,
not a training bug.

**Extraction threshold refuted as primary cause:** raising it 300x removed
zero peaks; at 0.7 x pred_max count collapses but median error explodes to
36.8 m because true-drone voxels (~0.099) fall below the threshold (0.115).
No threshold separates signal from spurious — they occupy the same value band.

**MPS timings (M4 Pro, torch 2.12.1, clean window):**

| V | forward | fwd+back | peak MPS |
|---|---|---|---|
| 2 | 0.218 s | 0.820 s | 1,081 MiB |
| 4 | 0.251 s | 1.010 s | 1,757 MiB |
| 8 | 0.327 s | 1.298 s | 3,090 MiB |

Batch growth superlinear (1.3 / 2.7 / 7.0 / 21.9 / 87 s for B=1/2/4/6/8), so
batch=1 is both fastest per scene and near-constant memory.

`aten::grid_sampler_2d_backward` has no MPS implementation in torch 2.12.1;
training requires `PYTORCH_ENABLE_MPS_FALLBACK=1`.

Epoch time, 3,000 scenes: 76 min at V<=8 (measured, io 0.008 s/step, prefetch
hides decode). 5.5 h at V<=24 before the sampling cap.

**Path forward, not attempted in V1:** higher feature resolution (stride <= 4,
dilated convs, or FPN upsampling head), plus multi-hour runs. Best attempt was
600 steps.

---

## 11. Evaluation sweep

| Parameter | Value |
|---|---|
| Test scenes | 1,000 (500 primary + 500 secondary) |
| Grid | 29 cells (7 view counts x 4 compositions) |
| Rows | 29,000 |
| Wall-clock | 91.9 min, 14 parallel jobs |
| Density bins | ~10 / ~35 / ~55 drones |
| Frozen test suite | 55/55 |

**Prediction outcomes:**

| ID | Prediction | Verdict |
|---|---|---|
| P1 | mixed beats all-ground, gap widens with density | PARTIAL — confirmed at low V, ceiling at high V |
| P2 | mAP rises, knee at V=3-5 | MATCH (at n=1000; PARTIAL at n=200) |
| P5 | count error monotonic with density | MATCH |
| P7 | measured extent within 20% of a_max | FALSIFIED |
| P7b | additive vs multiplicative AA model | OPEN — P7c queued |

---

## 12. Web backend verification (T9b)

Seed 0, views [0,1,2,3,4,5], recall_radius_px 5.0. CLI vs API, bitwise
identical on every field:

| Metric | Value |
|---|---|
| n_true / n_pred | 52 / 54 |
| count_err | +2 |
| mAP | 0.9629629629629629 |
| median_err_m | 0.03563070863599636 |
| chamfer_m | 0.09219196863936603 |
| detector_recall | 0.9834834574855584 |

Expected: both paths execute the same frozen `process_scene` ->
`ml.metrics.evaluate`.

---

## 13. Scope limits

Every number above is an **upper bound**, not a field estimate.

- **Plain backgrounds** are the easiest possible detection case. Detector
  recall 0.984-0.998.
- **Near-perfect recall compresses composition differences.** All compositions
  reach ~0.98 by V=8, so the tier signal is only visible at low view count.
- **Camera pose is given, not estimated.** Results are conditional on known
  pose.
- **Synthetic-to-real gap unmeasured.**
- **Static single timestep**, 5-60 drones, no temporal input.
- **Correspondence unvalidated under realistic detection** — the solver works
  on anonymous epipolar constraints, but only under near-perfect detection on
  plain backgrounds.

---

## 14. Values to verify against logs

Reported in session summaries; confirm before publication.

- Slide-4 headline conditions: median error 0.0315 m is quoted at V=8
  **ground**, mAP 0.9859 at V=8 **mixed**. Different compositions in the same
  table — intentional or an inconsistency?
- Count error `<= +0.75` at V=8 vs `+1.24` at high density in the sweep — the
  first is averaged across compositions, the second is a density bin. Confirm
  they are not being read as contradictory.
- P1 mid-density V=8 deltas were not reported; only low and high.
- Detector recall band 0.984-0.998 vs 0.9873 mean in the T5 acceptance —
  different scene counts, confirm which is quoted where.
