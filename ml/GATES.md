# Frozen Gate Definitions

**This file is FROZEN.** The overnight loop computes its checksum before every
iteration and aborts the entire run if it changes. Editing it to make code pass
is the failure mode this mechanism exists to prevent.

Changing a gate requires a human, in a supervised session, with the change
recorded in `docs/PROGRESS.md` and a new checksum baseline written by hand.

---

## G2 — overfit gate (the loop's target)

The model must reconstruct 8 training scenes it has been trained on directly.

```
G2 PASSES when, on the 8 overfit scenes, computed through frozen ml/metrics.py:

    median position error  <  1.0 m
AND count error            within [-1, +1]   on EVERY scene, not on average
```

Fixed parameters, not tunable by any agent:

| Parameter | Value |
|---|---|
| Overfit scene seeds | 2000-2007 |
| Eval view indices | (0, 3, 6, 8, 11, 14, 17, 21) |
| Metrics implementation | `ml.metrics.evaluate` (frozen) |
| tau set | `ml.metrics.DEFAULT_TAUS` (frozen) |
| Position extraction | `ml.model.extract_positions` |

**Why count error is the load-bearing half:** a flat or diffuse volume produces
hundreds of spurious local maxima, so count error explodes. Position error alone
can look acceptable while the model outputs garbage, which is exactly what
happened on 2026-08-02 (0.985 m position error with +286 to +364 count error).
Position error alone is not a pass.

## G4 — leak check (applies later, not to the loop)

```
HALT if ml_median_error < 0.9 * geometric_median_error
        at density <= 20 and views >= 4
```

A learned model beating well-conditioned triangulation under near-perfect
detection is a bug signal, not a win.

---

## Baseline measurements (2026-08-02, for comparison, not targets)

| Quantity | Value |
|---|---|
| All-zeros volume MSE on 8 scenes | 0.0020 |
| Best achieved loss | 0.002008 (indistinguishable from all-zeros) |
| Position error at that point | 0.985 m |
| Count error at that point | +286 to +364 |
| Spurious local maxima | ~1,012 vs 231 true drones |
| Peak-to-background ratio | 2.4 : 1 |
| True-loc voxel mean | 0.0987 |
| Volume mean | 0.0499 |
| Background median | 0.0416 |
| True-loc voxels above volume p99 | 51.5% |
| pred_max vs true-loc mean | 1.66x (strongest voxels NOT at drones) |

**Loss thresholds are forbidden as gate criteria.** The original gate used
recent-mean loss < 0.05, which was 25x the trivial all-zeros baseline and passed
a model that had learned nothing. Any gate proposed in terms of loss is
rejected.

---

## Rules that survive unattended operation

1. No agent edits this file.
2. No agent edits `ml/metrics.py`, `tests/test_predictions_ml.py`,
   `ml/splits.json`, `calib.json`, or anything under `stage1_geometry/`.
3. No agent changes the overfit seeds, the eval view indices, or the tau set.
4. A gate failure is a normal, expected outcome. Log it with measured numbers
   and move to the next queued fix. Do not reinterpret it as a pass.
5. Suspiciously clean results are bugs until proven otherwise. If a fix produces
   a sudden perfect score, record it and flag it as SUSPICIOUS rather than
   writing the sentinel.
