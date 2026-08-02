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

**Source of every number below:** Agent F's T7 evaluation sweep (accepted, commit `6a0e3d3`).
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

- **Scenes:** 100 test scenes per cell, 2 cells (primary, secondary) = 200 scenes total
  (`config.n_scenes: 200`).
- **Density bins:** low [5, 15] (~10), mid [28, 42] (~35), high [48, 60] (~55, straddling
  the ~50-drone false-track threshold); balanced ~33/34/33 scenes per cell per bin
  (`summary.json` `per_cell_density_bin_*`: n_scenes 33 / 34 / 33).
- **View counts:** 2, 3, 4, 5, 6, 7, 8.
- **Compositions:** ground, level, aerial, mixed (ordered camera lists per `config.composition_views`;
  mixed interleaves one camera from each tier).
- **Cells:** primary R=50 m, a_max 9.6 px/drone, standoff 139 m; secondary R=100 m,
  a_max 4.8 px/drone, standoff 278 m (both W=1920, f=2666.67) — `config.cell_radius_m`,
  `config.cell_a_max_px`.
- **Wall-clock:** 1448.4 s (24.1 min) for 200 scenes at 14 jobs (`report.md` §"Sized grid";
  `config.wall_clock_s: 1448.44`), 5800 rows total (`config.n_rows: 5800`), well under the
  2 h escalation guard.
- **Sanity cell:** mixed_v24 at V=24 (direct check vs the measured V=24 anchor), enabled
  (`config.sanity_v24: true`).

---

## 3. Controls and what they rule out

- **Sanity anchor.** The mixed_v24 cell at V=24 gives primary mAP 0.9874, median_err 0.0302 m,
  count_err +0.6400, detector recall 0.9904 (n=100) (`report.md` §"Sanity anchor check" table;
  `summary.json` `sanity_anchor_v24.primary.mixed_v24`). This is consistent within noise with
  the measured V=24 reference from the T5 acceptance run (20 test scenes, seeds 0–19): mAP 0.9865,
  median_err 0.0303 m, count_err +0.7, detector recall 0.9873 (`report.md` §"Sanity anchor check"
  heading; `docs/PROGRESS.md` Session 15b). The high-view-count curves (V=8) must also be
  consistent with this anchor; a large deviation would indicate a bug — none was observed
  (V=8 mAP 0.9805–0.9851 across compositions, `report.md` mAP tables). Secondary V=24 mAP 0.9932,
  median_err 0.0603 m, count_err +0.3600 (`summary.json` `sanity_anchor_v24.secondary.mixed_v24`).
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

Per-composition mAP per cell (mean over 100 scenes/cell, all densities pooled).
Source: `report.md` §"mAP and median error at each view count per composition" — **mAP**
tables; `summary.json` `per_cell_composition_view["<cell>|<comp>|<V>"].mAP_mean` for every cell.

### primary (R=50 m, a_max=9.6 px/drone) — mAP

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 | n_empty@V=8 |
|---|---|---|---|---|---|---|---|---|
| ground | 0.8802 | 0.9396 | 0.9596 | 0.9748 | 0.9785 | 0.9814 | 0.9805 | 0 |
| level | 0.9036 | 0.9287 | 0.9520 | 0.9690 | 0.9810 | 0.9845 | 0.9837 | 0 |
| aerial | 0.8881 | 0.9392 | 0.9592 | 0.9750 | 0.9768 | 0.9790 | 0.9815 | 0 |
| mixed | 0.9489 | 0.9703 | 0.9768 | 0.9791 | 0.9812 | 0.9822 | 0.9848 | 0 |

### secondary (R=100 m, a_max=4.8 px/drone) — mAP

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 | n_empty@V=8 |
|---|---|---|---|---|---|---|---|---|
| ground | 0.8924 | 0.9409 | 0.9605 | 0.9753 | 0.9820 | 0.9813 | 0.9836 | 0 |
| level | 0.9110 | 0.9310 | 0.9468 | 0.9765 | 0.9803 | 0.9841 | 0.9829 | 0 |
| aerial | 0.8744 | 0.9194 | 0.9509 | 0.9736 | 0.9793 | 0.9825 | 0.9827 | 0 |
| mixed | 0.9375 | 0.9667 | 0.9745 | 0.9792 | 0.9789 | 0.9840 | 0.9851 | 0 |

Plot: `logs/ml_sweep/plot_mAP_vs_views.png` (one curve per composition per cell).

---

## 5. Median position error vs view count

Per-composition median position error in metres (mean of per-scene matched-pair median errors).
Source: `report.md` §"mAP and median error at each view count per composition" — **median
position error (m)** tables; `summary.json` `per_cell_composition_view["<cell>|<comp>|<V>"].median_err_m_mean`.

### primary — median position error (m)

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 |
|---|---|---|---|---|---|---|---|
| ground | 0.0429 | 0.0420 | 0.0413 | 0.0390 | 0.0349 | 0.0324 | 0.0316 |
| level | 0.0427 | 0.0427 | 0.0428 | 0.0405 | 0.0344 | 0.0304 | 0.0295 |
| aerial | 0.0429 | 0.0418 | 0.0413 | 0.0388 | 0.0348 | 0.0320 | 0.0313 |
| mixed | 0.0498 | 0.0432 | 0.0404 | 0.0386 | 0.0381 | 0.0373 | 0.0366 |

### secondary — median position error (m)

| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 |
|---|---|---|---|---|---|---|---|
| ground | 0.0862 | 0.0842 | 0.0828 | 0.0771 | 0.0690 | 0.0641 | 0.0630 |
| level | 0.0844 | 0.0850 | 0.0861 | 0.0809 | 0.0681 | 0.0597 | 0.0584 |
| aerial | 0.0857 | 0.0836 | 0.0815 | 0.0763 | 0.0684 | 0.0639 | 0.0627 |
| mixed | 0.0964 | 0.0842 | 0.0808 | 0.0770 | 0.0757 | 0.0747 | 0.0724 |

Plot: `logs/ml_sweep/plot_median_err_vs_views.png`.

---

## 6. Detector recall per density bin

Recall within radius 5.0 px (GT projected via view K + w2c; scoring only, never feeds the
pipeline). Source: `report.md` §"Detector recall per density bin" table;
`summary.json` `detector_recall_per_density_bin`.

| cell | low | mid | high |
|---|---|---|---|
| primary | 0.9979 | 0.9882 | 0.9852 |
| secondary | 0.9999 | 0.9997 | 0.9996 |

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
| primary | -0.0554 | 0.1846 | 1.2121 | True |
| secondary | -0.0543 | 0.2961 | 1.5455 | True |

Count error is strictly monotonic in density in both cells. Ratio (count_err(high) /
count_err(low)): primary −21.89, secondary −28.44 (`report.md` §P5 verdict). Per-composition
count-error tables at each view count are in `report.md` §"mAP and median error at each view
count per composition" (count-error tables; e.g. primary ground V=2: −2.2900, primary ground
V=8: +1.0400; secondary mixed V=8: +0.7900) and in `summary.json`
`per_cell_composition_view.*.count_err_mean`.

---

## 8. The knee at V=3

The knee (largest single-step mAP increase) is at V=3 for 7 of 8 composition/cell cells and at
V=5 for secondary/level. Source: `report.md` §P2 table (knee column);
`summary.json` `p2_mAP_vs_views["<cell>"]["<comp>"].knee_view_count` (e.g. primary/ground
deltas: V3 +0.0594, V4 +0.0200, V5 +0.0152, V6 +0.0036, V7 +0.0030 — largest step at V=3).

| cell | ground | level | aerial | mixed |
|---|---|---|---|---|
| primary | V=3 | V=3 | V=3 | V=3 |
| secondary | V=3 | **V=5** | V=3 | V=3 |

---

## 9. Composition matters at V=2–3 and washes out by V=8

At low view counts the mixed-tier composition beats all-ground; the gap is largest at low
view counts and high density. Source: `report.md` §P1 table (delta mAP, mixed − ground, on
the intersection set); `summary.json` `p1_mixed_vs_ground`.

- V=2 primary high density: delta mAP = +0.1149 (n=33).
- V=2 primary mid density: delta mAP = +0.0619 (n=34).
- V=2 secondary high density: delta mAP = +0.0701 (n=33).
- V=2 secondary mid density: delta mAP = +0.0488 (n=34).
- V=2 primary low density: +0.0296; V=2 secondary low: +0.0162.

The gap narrows monotonically as views are added (e.g. primary high density: 0.1149 @V2 →
0.0779 @V3 → 0.0350 @V4 → 0.0084 @V5 → 0.0049 @V6 → −0.0003 @V7 → 0.0052 @V8), and all
compositions saturate near ~0.98 mAP by V=8 (`report.md` mAP tables). Mixed's median position
error is, however, consistently *higher* than the pure tiers at low view counts (e.g. primary
mixed V=2 median_err 0.0498 m vs ground 0.0429 m) — the mixed mAP advantage comes from
recall/well-conditioned geometry, not from lower position error.

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
4.8 px/drone (secondary). At 4.8 px/drone, mAP reaches 0.9827–0.9851 by V=8 and median
position error 0.0584–0.0724 m (`report.md` secondary tables) — reconstruction works down to
~5 px/drone with enough views.

---

## 11. Prediction log — P1/P2 PARTIAL, P5 MATCH

Verdicts are stated plainly; partial predictions are **not** rounded up. Source:
`report.md` §P1/§P2/§P5 verdict lines; `summary.json` `p1_mixed_vs_ground`, `p2_mAP_vs_views`,
`p5_count_err_vs_density`.

| ID | Prediction | Verdict | Observed |
|---|---|---|---|
| P1 | Mixed-tier beats all-ground at equal view count; gap widens with density | **PARTIAL** | Gap confirmed at low view counts, widest at high density (V=2 primary high delta mAP +0.1149; V=2 primary mid +0.0619; V=2 secondary high +0.0701). But at high view counts all compositions saturate near ~0.98 and the strict "mixed ≥ ground at all V" fails (e.g. V=7 primary high delta −0.0003 — noise; V=8 secondary mid −0.0034). Gap-widens-with-density holds at low V but not uniformly across the grid. |
| P2 | mAP rises with view count; knee between 3 and 5 views | **PARTIAL** | mAP does rise V=2→8 for every composition/cell. Knee (largest step) at V=3 for 7/8 cells, V=5 for secondary/level — inside the predicted 3–5 window. Strict monotonicity fails at V=7→8 with ~0.001 dips (e.g. primary ground 0.9814→0.9805; secondary level 0.9841→0.9829) — noise, not a real decline. |
| P5 | Count error grows monotonically with density | **MATCH** | Strictly monotonic in both cells: primary −0.0554 → +0.1846 → +1.2121; secondary −0.0543 → +0.2961 → +1.5455 (`report.md` §P5 table; `summary.json` `p5_count_err_vs_density`, `monotonic: true`). |

---

## 12. Scope limits

Stated here, not buried. Source: spec §2 (in/out of scope) and spec open item 7.

- **Plain backgrounds are the easiest possible detection case.** All numbers here are an
  upper bound on performance, not an estimate of field performance. Detector recall is
  near-perfect (0.9852–0.9999 across cells/densities, §6), and near-perfect recall
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
