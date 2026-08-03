# RESULTS_ML — Multi-View Swarm Reconstruction: Results

> **DRAFT — geometric baseline section; pending human review/approval**
>
> Per spec T10 (`docs/superpowers/specs/2026-07-31-ml-swarm-reconstruction-design.md`, §8)
> and `docs/superpowers/ORCHESTRATOR_ML.md`, `docs/RESULTS_ML.md` is human-owned and no
> agent writes findings. This baseline draft of the **geometric baseline** section has been
> written at the explicit dispatch of the human, who will review it. The **ML track section
> is NOT ATTEMPTED IN V1** (G2 failed on a structural cause, Session 15e ruling) and contains
> **no numbers** — it is marked as such below.
>
> **Status summary:** GEOMETRIC BASELINE — complete. ML TRACK — NOT ATTEMPTED IN V1
> (structural G2 failure; documented negative result). ML-vs-geometric comparison / G4 —
> NOT ATTEMPTED IN V1 (depends on the ML track).

**Source of every number below:** T7 evaluation sweep, FULL test split (1000 scenes, 29000
rows, wall-clock 5513.6 s = 91.9 min, commit `b3deba4` for `select_scenes` full test-split
coverage, `1b960e7` for report title, error-bar extension at this session).
Primary source: `logs/ml_sweep/eval_sweep_report.md`. Raw per-cell data for verification:
`logs/ml_sweep/eval_sweep_summary.json`, `logs/ml_sweep/eval_sweep_results.csv`,
`logs/ml_sweep/eval_sweep_rows.jsonl`. Plots: `logs/ml_sweep/plot_mAP_vs_views.png` and
`logs/ml_sweep/plot_median_err_vs_views.png`. Each figure below carries an inline source
tag naming the report section and/or the summary-JSON key it was read from.

---

## 1. The question

Does multi-view triangulation reconstruct the swarm well enough to produce a usable
adjacency matrix, and at what view count and composition?

"Usable" is answered by the headline metric `mAP` (average over tau in {0.5, 1, 2, 5} m of
AP@tau on Hungarian-matched predictions) and by median position error on matched pairs.
The adjacency matrix is thresholded at `d_max` (comms range) from estimated pairwise
distances; the same `mAP`/median-error machinery feeds the downstream adjacency
construction. The sweep asks how both quantities depend on (a) how many views and (b)
where those views are placed (composition), over a sized grid of scenes and densities, for
two operating cells (primary R=50 m, secondary R=100 m).

---

## 2. Method

Frozen pipeline, wiring only — every number is produced by the frozen path
(`eval_sweep_report.md` intro):

```
detect_blobs -> solve_correspondence (EPIPOLAR_THRESHOLD_PX=3.0) -> triangulate_dlt -> ml.metrics.evaluate
```

Detection of all 24 views runs ONCE per scene and is cached; correspondence +
triangulation + metrics are re-run per (composition, view count) cell on the cached subset
(a factorization of `ml.baseline_adapter.process_scene`). Metrics are computed exclusively
through the frozen `ml/metrics.py` implementation.

Sized grid (`eval_sweep_report.md` §"Sized grid"; `eval_sweep_summary.json` `config`):

- **Scenes:** 500 test scenes per cell, 2 cells (primary, secondary) = 1000 scenes total
  (`config.n_scenes: 1000`).
- **Density bins:** low [5, 15] (~10), mid [28, 42] (~35), high [48, 60] (~55, straddling
  the ~50-drone false-track threshold); target mix `{"low": 166, "mid": 168, "high": 166}`
  per cell per bin.
- **View counts:** 2, 3, 4, 5, 6, 7, 8.
- **Compositions:** ground, level, aerial, mixed (ordered camera lists per `config.composition_views`;
  mixed interleaves one camera from each tier).
- **Cells:** primary R=50 m, a_max 9.6 px/drone, standoff 139 m; secondary R=100 m,
  a_max 4.8 px/drone, standoff 278 m (both W=1920, f=2666.67) — `config.cell_radius_m`,
  `config.cell_a_max_px`.
- **Wall-clock:** 5513.6 s (91.9 min) for 1000 scenes at 14 jobs (`report.md` §"Sized grid";
  `config.wall_clock_s: 5513.65`), 29000 rows total (`config.n_rows: 29000`), well under the
  2 h escalation guard.
- **Sanity cell:** mixed_v24 at V=24 (direct check vs the measured V=24 anchor), enabled
  (`config.sanity_v24: true`).

---

## 3. Controls and what they rule out

- **Sanity anchor.** The mixed_v24 cell at V=24 gives primary mAP 0.9890, median_err 0.0302 m,
  count_err +0.5700, detector recall 0.9903 (n=500) (`report.md` §"Sanity anchor check" table;
  `summary.json` `sanity_anchor_v24.primary.mixed_v24`). This is consistent within noise with
  the measured V=24 reference from the T5 acceptance run (20 test scenes, seeds 0–19): mAP 0.9865,
  median_err 0.0303 m, count_err +0.7, detector recall 0.9873 (`docs/PROGRESS.md` Session 15b).
  The high-view-count curves (V=8) must also be consistent with this anchor; a large deviation
  would indicate a bug — none was observed (V=8 mAP 0.9821–0.9859 across compositions,
  `report.md` mAP tables). Secondary V=24 mAP 0.9923, median_err 0.0602 m, count_err +0.4160,
  recall 0.9997 (`summary.json` `sanity_anchor_v24.secondary.mixed_v24`).
- **Frozen-metrics path.** Both tracks score through one frozen `ml/metrics.py` implementation,
  built before either track produced numbers. Nothing was re-fitted to the sweep output.
- **Intersection-set handling.** Any metric compared across configurations is computed on the
  intersection set (seeds present in BOTH configurations). Every scene runs every configuration,
  so the intersection is the shared scene set; per-config `n_empty` / `n_no_match` are reported.
  mAP of an empty reconstruction is 0.0 (a real result); `median_err_m` is NaN for a scene with
  no matched pairs (excluded from the mean, counted in `n_no_match`) (`report.md` §"Intersection-set
  handling"; `summary.json` `intersection_note`). In this sweep `n_empty = 0` and `n_no_match = 0`
  for every cell (e.g. `per_cell_composition_view.*.n_empty`).
- **Detector recall is scoring-only.** Recall is computed by projecting ground truth through the
  view K + w2c and counting detections within radius 5.0 px (`config.recall_radius_px: 5.0`). It is
  never fed into the reconstruction pipeline (`report.md` §"Detector recall per density bin").
- **Detector input is the same pixels the ML model will see.** Detection runs on the rendered
  PNGs; there is no ground-truth 2D projection into the pipeline (spec §T5).

---

## 4. mAP vs view count per composition

Per-composition mAP per cell (mean over 500 scenes/cell, all densities pooled).
Source: `report.md` §"mAP and median error at each view count per composition" — **mAP**
tables; `summary.json` `per_cell_composition_view["<cell>|<comp>|<V>"].mAP_mean` for every cell.

### primary (R=50 m, a_max=9.6 px/drone) — mAP

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 | n_empty@V=8 |
|---|---|---|---|---|---|---|---|---|
| ground | 0.8833 | 0.9374 | 0.9643 | 0.9757 | 0.9823 | 0.9838 | 0.9850 | 0 |
| level | 0.9020 | 0.9316 | 0.9500 | 0.9701 | 0.9801 | 0.9828 | 0.9833 | 0 |
| aerial | 0.8881 | 0.9339 | 0.9609 | 0.9740 | 0.9797 | 0.9804 | 0.9821 | 0 |
| mixed | 0.9439 | 0.9709 | 0.9765 | 0.9803 | 0.9842 | 0.9847 | 0.9859 | 0 |

### secondary (R=100 m, a_max=4.8 px/drone) — mAP

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 | n_empty@V=8 |
|---|---|---|---|---|---|---|---|---|
| ground | 0.8837 | 0.9341 | 0.9634 | 0.9768 | 0.9822 | 0.9848 | 0.9863 | 0 |
| level | 0.8947 | 0.9205 | 0.9459 | 0.9751 | 0.9808 | 0.9829 | 0.9851 | 0 |
| aerial | 0.8834 | 0.9265 | 0.9564 | 0.9761 | 0.9800 | 0.9828 | 0.9845 | 0 |
| mixed | 0.9331 | 0.9650 | 0.9737 | 0.9788 | 0.9815 | 0.9842 | 0.9863 | 0 |

Plot: `logs/ml_sweep/plot_mAP_vs_views.png` (one curve per composition per cell; error bars
are std over the cell's scenes at that composition and view count).

---

## 5. Median position error vs view count

Per-composition median position error in metres (mean of per-scene matched-pair median errors).
Source: `report.md` §"mAP and median error at each view count per composition" — **median
position error (m)** tables; `summary.json` `per_cell_composition_view["<cell>|<comp>|<V>"].median_err_m_mean`.

### primary — median position error (m)

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 |
|---|---|---|---|---|---|---|---|
| ground | 0.0565 | 0.0421 | 0.0412 | 0.0386 | 0.0347 | 0.0322 | 0.0315 |
| level | 0.0427 | 0.0428 | 0.0431 | 0.0406 | 0.0346 | 0.0303 | 0.0295 |
| aerial | 0.0424 | 0.0418 | 0.0410 | 0.0384 | 0.0345 | 0.0319 | 0.0312 |
| mixed | 0.0493 | 0.0421 | 0.0398 | 0.0382 | 0.0376 | 0.0368 | 0.0361 |

### secondary — median position error (m)

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 |
|---|---|---|---|---|---|---|---|
| ground | 0.0863 | 0.0840 | 0.0820 | 0.0765 | 0.0689 | 0.0641 | 0.0630 |
| level | 0.1080 | 0.0846 | 0.0850 | 0.0796 | 0.0674 | 0.0597 | 0.0582 |
| aerial | 0.0865 | 0.0841 | 0.0819 | 0.0768 | 0.0687 | 0.0639 | 0.0627 |
| mixed | 0.0967 | 0.0837 | 0.0800 | 0.0759 | 0.0751 | 0.0739 | 0.0717 |

Plot: `logs/ml_sweep/plot_median_err_vs_views.png`.

---

## 6. Detector recall per density bin

Recall within radius 5.0 px (GT projected via view K + w2c; scoring only, never feeds the
pipeline). Source: `report.md` §"Detector recall per density bin" table;
`summary.json` `detector_recall_per_density_bin`.

| cell | low | mid | high |
|---|---|---|---|
| primary | 0.9978 | 0.9897 | 0.9836 |
| secondary | 0.9998 | 0.9997 | 0.9996 |

Near-perfect and flat across density. This is a stated condition of the comparison: the
geometric baseline's accuracy is not limited by detection on these renders.

---

## 7. Count error vs density

Mean count error (predicted N minus true N) per density bin, meaned over scenes and over
compositions/view counts at the intersection set (same scene set everywhere). Source:
`report.md` §P5 and §"Count error vs density" tables; `summary.json`
`count_err_vs_density` (per-cell `count_err_per_bin`).

| cell | low | mid | high | monotonic? |
|---|---|---|---|---|
| primary | -0.0564 | 0.2602 | 1.2393 | True |
| secondary | -0.0591 | 0.3112 | 1.5005 | True |

Count error is strictly monotonic in density in both cells. Ratio (count_err(high) /
count_err(low)): primary −21.96, secondary −25.41 (`report.md` §P5 verdict). Per-composition
count-error tables at each view count are in `report.md` §"mAP and median error at each view
count per composition" (count-error tables; e.g. primary ground V=2: −2.1460, primary ground
V=8: +0.7480; secondary mixed V=8: +0.7400) and in `summary.json`
`per_cell_composition_view.*.count_err_mean`.

---

## 8. The knee at V=3

The knee (largest single-step mAP increase) is at V=3 for 7 of 8 composition/cell cells and at
V=5 for secondary/level. Source: `report.md` §P2 table (knee column);
`summary.json` `p2_mAP_vs_views["<cell>"]["<comp>"].knee_view_count` (e.g. primary/ground
deltas: V3 +0.0541, V4 +0.0269, V5 +0.0114, V6 +0.0066, V7 +0.0015 — largest step at V=3).

| cell | ground | level | aerial | mixed |
|---|---|---|---|---|
| primary | V=3 | V=3 | V=3 | V=3 |
| secondary | V=3 | **V=5** | V=3 | V=3 |

---

## 9. Composition matters at V=2–3 and washes out by V=8

At low view counts the mixed-tier composition beats all-ground; the gap is largest at low
view counts and high density. Source: `report.md` §P1 table (delta mAP, mixed − ground, on
the intersection set); `summary.json` `p1_mixed_vs_ground`.

- V=2 primary high density: delta mAP = +0.0829 (n=115).
- V=2 primary mid density: delta mAP = +0.0640 (n=134).
- V=2 secondary high density: delta mAP = +0.0729 (n=128).
- V=2 secondary mid density: delta mAP = +0.0577 (n=118).
- V=2 primary low density: +0.0390; V=2 secondary low: +0.0255.

The gap narrows monotonically as views are added (e.g. primary high density: 0.0829 @V2 →
0.0659 @V3 → 0.0328 @V4 → 0.0140 @V5 → 0.0036 @V6 → 0.0004 @V7 → 0.0003 @V8), and all
compositions saturate near ~0.98 mAP by V=8 (`report.md` mAP tables). Mixed's median position
error is, however, consistently *higher* than the pure tiers at low view counts (e.g. primary
mixed V=2 median_err 0.0493 m vs ground 0.0565 m) — the mixed mAP advantage comes from
recall/well-conditioned geometry, not always from lower position error.

---

## 10. Results against a_max in pixels-per-drone (PATCH 6)

Per spec §6, results are expressed against a_max in pixels per drone, not swarm radius in
metres. a_max depends only on drone size, d/R and W; absolute metres never enter the
comparison axis. Operating cells (`report.md` §"Sized grid"; `config.cell_radius_m`,
`config.cell_a_max_px`):

| cell | R | a_max | standoff | W | f |
|---|---|---|---|---|---|
| primary | 50 m | 9.6 px/drone | 139 m | 1920 | 2666.67 px |
| secondary | 100 m | 4.8 px/drone | 278 m | 1920 | 2666.67 px |

So the study covers two points on the apparent-size axis: 9.6 px/drone (primary) and
4.8 px/drone (secondary). At 4.8 px/drone, mAP reaches 0.9845–0.9863 by V=8 and median
position error 0.0582–0.0717 m (`report.md` secondary tables) — reconstruction works down to
~5 px/drone with enough views.

---

## 11. Prediction log — P1 PARTIAL, P2 MATCH, P5 MATCH

Verdicts are stated plainly; partial predictions are **not** rounded up. Source:
`report.md` §P1/§P2/§P5 verdict lines; `summary.json` `p1_mixed_vs_ground`, `p2_mAP_vs_views`,
`p5_count_err_vs_density`. Numbers below are from the full test set (1000 scenes).

| ID | Prediction | Verdict | Observed |
|---|---|---|---|
| P1 | Mixed-tier beats all-ground at equal view count; gap widens with density | **PARTIAL** | Gap confirmed at low view counts, widest at high density (V=2 primary high delta mAP +0.0829; V=2 primary mid +0.0640; V=2 secondary high +0.0729). But at high view counts all compositions saturate near ~0.98 and the strict "mixed ≥ ground at all V" fails (e.g. V=8 primary high delta −0.0003 — noise; V=8 secondary mid −0.0022). Gap-widens-with-density holds at low V but not uniformly across the grid. |
| P2 | mAP rises with view count; knee between 3 and 5 views | **MATCH** | mAP does rise V=2→8 for every composition/cell (monotonic). Knee (largest step) at V=3 for 7/8 cells, V=5 for secondary/level — inside the predicted 3–5 window. Strict monotonicity holds at V=7→8 on the full test set (all 8 cells show V=7→8 positive delta: +0.0005 to +0.0022); the ~0.001 dip observed on the 200-scene sample has washed out as noise. |
| P5 | Count error grows monotonically with density | **MATCH** | Strictly monotonic in both cells: primary −0.0564 → +0.2602 → +1.2393; secondary −0.0591 → +0.3112 → +1.5005 (`report.md` §P5 table; `summary.json` `p5_count_err_vs_density`, `monotonic: true`). |

---

## 12. Scope limits

Stated here, not buried. Source: spec §2 (in/out of scope) and spec open item 7.

- **Plain backgrounds are the easiest possible detection case.** All numbers here are an
  upper bound on performance, not an estimate of field performance. Detector recall is
  near-perfect (0.9836–0.9998 across cells/densities, §6), and near-perfect recall
  compresses the composition differences this study set out to measure — by V=8 every
  composition is near ceiling, so the composition signal is visible only at low view counts.
- **Camera pose is given, not estimated.** Real deployment would need to solve pose
  estimation; this work measures reconstruction error conditional on known pose.
- **Synthetic-to-real gap is unmeasured and unbounded by this work.** Real imagery, real
  drones, and real deployment are explicitly out of scope (spec §2).
- **Static single timestep only; 5–60 drones; no temporal input.**
- **This is a feasibility proof, not a product** (spec §2). The bar is a defensible finding.

---

## ML track — NOT ATTEMPTED IN V1 (documented negative result, no numbers)

**NOT ATTEMPTED IN V1 (ruling, Session 15e).** The ML track is not attempted in V1 and no
ML numbers are reported here. This is a documented negative result, not an untried track.

- **T6 G2 overfit gate FAILS** (Sessions 15d–15e). Best checkpoint (600 steps, weighted
  MSE `pos_weight=500`, `target_sigma=2.0`): aggregate median_err 1.0196 m, but count
  error in range +241 to +354 (gate requires count error within ±1 per scene). A diffuse
  low-contrast volume (pred_max ~0.16, background ~0.04, peak-to-background ~2.4:1 at
  true drone locations) with ~300–400 spurious local maxima.
- **Structural cause (Session 15e ruling), measured by Agent H's diagnostic
  (`ml/extract_debug.py`):** at encoder stride 8 with a_max ~10 px/drone, each drone
  occupies ~1.2 feature pixels, so the encoder cannot produce a peak narrower than its
  receptive field. Spec PATCH 5 set stride ≤ 8 as a floor; the model is at the floor.
  The diagnostics show the model **learned position but not sharpness**: an absolute
  threshold sweep up to 0.7×pred_max removes zero peaks (count error stays +241..+354
  through 0.5×), true-drone-location voxels sit at 2.4:1 over background, but 1,012
  local maxima share the same value band as 231 drones — no threshold separates them.
- **Named path forward (not run in V1):** higher feature resolution (stride ≤ 4,
  dilated convolutions, or an FPN-style upsampling head) plus multi-hour training runs.
- G2 was redefined (Session 15c) to be computed through frozen `ml/metrics.py` on the 8
  overfit scenes: passes when median position error < 1.0 m AND count error within ±1.
- **ML-vs-geometric comparison and G4 are NOT ATTEMPTED IN V1.** G4 condition:
  `ml_median_error >= 0.9 * geometric_median_error` at density ≤ 20 and views ≥ 4. G4
  is inapplicable because the ML track is not attempted; its G4 gate row is marked
  NOT ATTEMPTED IN V1 in the spec gate table.

Status history and gate state: `docs/PROGRESS.md` Sessions 15b (G2 false PASS found),
15c (G2 redefined; F and G dispatched), 15d (F accepted; G escalated, G2 FAIL),
15e (H and I accepted; G2 STAYS FAILED, ML track paused; documented negative result).
