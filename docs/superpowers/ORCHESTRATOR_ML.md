# Orchestrator: ML Swarm Reconstruction Track

Sibling to `ORCHESTRATOR.md`. Same principles, different specifics.

**Source of truth:** `docs/superpowers/specs/2026-07-31-ml-swarm-reconstruction-design.md`

---

## What carries over unchanged

- A component is done when its acceptance command exits 0, run by the
  orchestrator, never because a subagent said so
- Acceptance is an exit code, not a judgment
- Prediction tests run after every merge, violations halt the loop
- Exclusive file ownership, subagents read anything, write only what they own
- Never loosen an acceptance criterion or prediction test to make code pass
- Suspiciously clean results are bugs until proven otherwise

## What is new for this track

Three failure modes exist here that did not exist in the geometric track.

**1. Long runs must not execute inside an agent session.** The rendered sweep hung
when run inside an agent session, and the fix was detached execution. The full
render campaign here is days, and training on MPS is hours. Subagents **build and
smoke-test** these. The human launches the full run detached. An orchestrator that
starts a multi-day render inside a subagent has already failed.

**2. The control is editable.** The geometric track is what ML is compared
against. If an agent "improves" the baseline mid-comparison, the control is gone
and no exit code catches it. The entire geometric track is frozen.

**3. Metrics are shared across two tracks.** If each track computes its own
metrics, the comparison is meaningless regardless of what the numbers say. One
frozen metrics implementation, both tracks consume it.

---

## Skills

- **findskills** at session start, to surface anything relevant not listed here
- **superpowers** for subagent fan-out, one agent per component, ownership enforced
- **claude-mem** to persist cross-session context, since this campaign spans days.
  `docs/PROGRESS.md` remains the authority for gate state. Memory is convenience,
  not evidence. Never resolve a gate from memory; re-run the command.

---

## File ownership map

```
tools/calibrate.py              owner: calibration agent
ml/scene_gen.py                 owner: scene agent
ml/render_harness.py            owner: harness agent
ml/control.py                   owner: harness agent
ml/status_app.py                owner: status agent
ml/pack_dataset.py              owner: packing agent
ml/baseline_adapter.py          owner: baseline agent
ml/model.py                     owner: model agent
ml/train.py                     owner: model agent
ml/eval_sweep.py                owner: eval agent
ml/recon_app.py                 owner: ui agent

docs/RESULTS_ML.md              owner: HUMAN ONLY. No agent writes findings.
```

## Frozen

```
stage1_geometry/**              ENTIRE geometric track. It is the control.
bundle_schema.py                inherited
data_contract.py, b1-b5         inherited
tests/test_predictions*.py      both suites
ml/metrics.py                   frozen the moment its acceptance passes
calib.json                      frozen once written by T0
ml/splits.json                  frozen once written by T4
```

Any diff touching a frozen path is an automatic halt.

---

## tests/test_predictions_ml.py

Write this **before building anything**, escalate the list for approval. These
encode geometry and invariants as assertions, so architecture bugs fail loudly
rather than surfacing as mediocre accuracy after hours of training.

```
back-projection round trip   project known 3D point into all views, back-project,
                             recover within 1 voxel
tier label integrity         elevation recomputed from extrinsics matches the
                             declared tier, every camera, every scene
permutation invariance       shuffle input view order -> identical output within
                             float tolerance. Non-negotiable for a set predictor.
heatmap target fidelity      soft-argmax of a constructed target recovers the
                             ground-truth position within voxel resolution
scrambled extrinsics         wrong extrinsics -> error explodes vs correct
known offset                 shift all ground truth by delta -> reported error
                             shifts by delta
single view                  degenerate, must not return confident 3D
a_max formula                d*W/(2R), both symmetries (inherited)
metric path identity         same input arrays -> geometric and ML produce
                             identical metric values from the same code
split disjointness           zero scene-seed overlap across train/val/test
```

Permutation invariance and back-projection round trip are the highest value. They
catch the two bugs most likely to masquerade as "the model just needs more data."

---

## Escalation triggers

Inherited, plus track-specific. Halt and surface when:

1. Any diff touching a frozen path
2. Any prediction test failing
3. Any acceptance command that cannot exit 0 after two attempts
4. Any suspiciously clean result: exact zeros, perfect scores, a metric unchanged
   after a parameter change
5. Any spec ambiguity requiring interpretation
6. Any subagent reporting success without its acceptance command having run
7. Any proposal to change an acceptance criterion, tau set, or match threshold
8. **T0 detectability gate, either outcome.** Pass or fail, the human sees the
   measured pixel size before any dataset generation begins.
9. **Any run expected to exceed ~10 minutes.** Build it, smoke-test it, hand it
   over. Do not execute it in-session.
10. **Model fails to overfit 8 scenes.** Architecture or target construction is
    wrong. More data will not fix it.
11. **ML beats geometric at low density.** Leak signal, not a win. See G4.

---

## Build order

```
0.  T0 calibration + detectability gate      -> ESCALATE either outcome
1.  tests/test_predictions_ml.py             -> ESCALATE list for approval
2.  ml/metrics.py                            -> freeze on acceptance
3.  T1 scene + camera generator
4.  T2 render harness (build + smoke only)
5.  T3 status page
    --- HUMAN LAUNCHES FULL RENDER, DETACHED, DAYS ---
6.  T4 packing + splits                      -> G1 leak check
7.  T5 geometric baseline on identical scenes
8.  T6 model + training                      -> G2 overfit gate FIRST
9.  T7 evaluation sweep                      -> G4 leak check
10. T8 held-out confirmation of winners
11. T9 reconstruction UI
12. T10 RESULTS_ML.md                        -> human writes
```

Metrics before either track produces a number. Otherwise the comparison is
retrofitted.

---

## The prompt

```
You are the orchestrator for the ML swarm reconstruction build.

SOURCE OF TRUTH:
docs/superpowers/specs/2026-07-31-ml-swarm-reconstruction-design.md
Read it fully before acting. It overrides your assumptions. If ambiguous,
escalate rather than interpret.

OPERATING RULES: docs/superpowers/ORCHESTRATOR_ML.md. Read this too.
TRACKER: docs/PROGRESS.md. Update after every component, not at the end.

SETUP:
- Run findskills to surface relevant skills before starting.
- Use superpowers subagents for fan-out. One agent per component, exclusive
  file ownership per the ownership map.
- Use claude-mem for cross-session context. PROGRESS.md remains the authority
  for gate state. Never resolve a gate from memory, re-run the command.

LOOP:
1. Read spec and PROGRESS.md. Pick the next incomplete component.
2. Spawn a subagent with: the relevant spec section verbatim, its exclusive file
   ownership list, and its acceptance command.
3. Require incremental commits and PROGRESS.md updates as it goes.
4. When it reports done, RUN THE ACCEPTANCE COMMAND YOURSELF. Its report is
   not evidence.
5. Run tests/test_predictions_ml.py.
6. If both pass: commit, update the gate table with the commit hash, append to
   the prediction-vs-observed log, continue.
7. If either fails: return the actual failure output to the subagent. Two
   attempts, then escalate.

BEFORE BUILDING ANYTHING:
- Complete T0 and escalate the measured detectability number, pass or fail.
- Write tests/test_predictions_ml.py and escalate the list for approval.
- Build ml/metrics.py and freeze it before either track produces a number.

RULES:
- A component is done when its acceptance command exits 0. Never because an
  agent said so.
- Never edit a frozen path. The entire geometric track is frozen; it is the
  control this work is compared against.
- Never modify an acceptance criterion, tau set, or match threshold to make
  code pass. Escalate.
- Never let a subagent write outside its ownership list.
- Never start a run expected to exceed ~10 minutes inside an agent session.
  Build it, smoke-test it, hand it to the human for detached launch.
- Failed attempts go in the PROGRESS.md session log. Do not hide them.
- Suspiciously clean results are bugs until proven otherwise.

ESCALATE, HALTING THE LOOP, WHEN:
- T0 detectability gate resolves, either outcome
- a frozen path needs changing
- a prediction test fails
- an acceptance command cannot pass after two attempts
- a result is suspiciously clean
- the spec is ambiguous
- a subagent claims success without its acceptance command having run
- the model fails to overfit 8 scenes
- ML accuracy exceeds the geometric baseline at low density

BUILD ORDER: per ORCHESTRATOR_ML.md. Do not reorder.

Start with T0 only. Escalate the result before proceeding.
```

---

## Spec patches required before launch

Three gaps, all needing a human decision. Do not launch until resolved.

**1. How does the geometric baseline get its detections?**
The spec says the ML track needs no Object Index pass, which is true, but T5 runs
the geometric baseline on the same scenes and its detector needs 2D detections
from somewhere. Three options, and they are not equivalent:
  - Run `detect_blobs.py` on the same plain-background PNGs. Honest, both tracks
    consume identical pixels. Recommended.
  - Render an Object Index pass for these scenes. Reintroduces the emission
    shader blocker for no benefit.
  - Hand the baseline ground-truth 2D projections. Unfair advantage to the
    control, invalidates the comparison.
Recommend option one, and measure detector recall on plain backgrounds rather
than assuming it is perfect. If it is near-perfect, that is a reportable
condition of the comparison, not a background assumption.

**2. Pure-tier compositions cannot reach 8 views at 6 cameras per tier.**
The spec renders 6 per tier but sweeps view counts 2 to 8. An 8-camera all-ground
composition is unrenderable. Either cap pure-tier comparisons at 6 views, or
render 8 per tier (24 angles, 33% more renders). Decide after T0 gives the
per-frame time, since the cost is now measurable rather than guessed.

**3. G4 is not executable as written.**
"ML mAP not implausibly above geometric" is a judgment, and this whole design
exists because judgments pass things. Needs a number in the same form as the
geometric track's `phase2 >= phase1 * 0.9`. Proposed:
```
HALT if ml_median_error < 0.9 * geometric_median_error
        at density <= 20 and views >= 4
```
