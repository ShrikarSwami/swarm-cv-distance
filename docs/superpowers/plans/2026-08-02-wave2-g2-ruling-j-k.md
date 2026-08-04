# Wave 2 continuation — G2 ruling housekeeping + Agent J (sweep scale) + Agent K (T9 UI)

**Date:** 2026-08-02
**Source:** human orchestrator directive (pasted 2026-08-02). Standing rules unchanged.

---

## Ruling (binding)

**G2 STAYS FAILED. ML TRACK PAUSED, NOT DELETED.**

- Cause is structural, not a bug: at stride 8 with a_max ~10px each drone occupies
  ~1.2 feature pixels, so the encoder cannot produce a peak narrower than its
  receptive field. Spec PATCH 5 set stride <= 8 as a floor; we are at the floor.
- Do not attempt further fixes now. Cost is unbounded (76 min/epoch, five loss
  variants already tried without a clean long run), and the comparison it would
  enable is saturated: the baseline sits at mAP 0.98 with 0.985-1.000 detector
  recall. No headroom; G4 says a win at low density is a leak signal anyway.
- Record in the spec as a documented negative result with Agent H's measured
  evidence, the specific structural cause, and the named path forward (higher
  feature resolution: stride <= 4, dilated convs, or an FPN-style upsampling
  head; plus multi-hour training runs).
- Mark the ML sections of RESULTS_ML.md and G4 as **NOT ATTEMPTED IN V1** with
  that reason. Do not present it as untried.

---

## Task 1 — Housekeeping (orchestrator-owned)

1. Commit Agent H (`ml/extract_debug.py`) and Agent I (`docs/RESULTS_ML.md`
   draft) with their acceptances.
2. Update `docs/PROGRESS.md` with both acceptances and this ruling.
3. Record the G2 negative result in the design spec
   (`docs/superpowers/specs/2026-07-31-ml-swarm-reconstruction-design.md`):
   H's measured evidence, structural cause, named path forward. Mark ML
   sections of RESULTS_ML.md and G4 as NOT ATTEMPTED IN V1.
4. Implement run-scoped checkpoint paths in `ml/train.py`
   (`checkpoints/<tag>/run_<N>/`, never a shared `latest.pt`) before the repo
   is touched again. The 0.985m run is unrecoverable and must not repeat.

Note (orchestrator finding): `ml/baseline_adapter.py` (T5, accepted Session
15b) is untracked in git and the real 5000-seed `ml/splits.json` (Session 15)
is uncommitted. The committed `ml/eval_sweep.py` imports
`ml.baseline_adapter`. These must be committed as part of housekeeping so the
repo is consistent before Agent J/K run.

## Task 2 — Agent J owns `ml/eval_sweep.py` — SCALE THE SWEEP

- Re-run at the FULL test split: **500 primary + 500 secondary**. Same 29
  cells, same frozen `ml/metrics.py`, same sized budget otherwise.
- F took 24.1 min for 200 scenes at 14 jobs, so estimate before launching; if
  the estimate exceeds 2 hours, escalate rather than launching in-session.
- Report: whether P1's gap-widening strengthens or weakens with 5x the scenes;
  whether P2's V=7->8 dip (~0.001) survives or washes out as noise; per-cell
  error bars on every curve.
- Regenerate both plots and update RESULTS_ML.md's numbers to the full-test-set
  run. Every number retraceable to the new run; no numbers carried forward.
- Acceptance: sweep completes, curves regenerated, P1/P2/P5 re-recorded as
  predicted/observed/ratio/match against the larger sample.

## Task 3 — Agent K owns `ml/recon_app.py` — T9 UI

- Baseline-only. The ML model is paused, so the UI reconstructs via the frozen
  geometric path.
- Select a scene from the manifest; then either pick exact angles, pick a
  random N, or pick a random number of random angles.
- Runs the baseline, displays predicted vs ground-truth 3D overlay plus that
  run's metrics via frozen `ml/metrics.py`.
- Acceptance: end-to-end from scene selection to overlaid reconstruction with
  metrics, on a real scene from `~/swarm_ml`.

## STOP after both

Escalate with J's full-test-set verdicts and K's acceptance.
