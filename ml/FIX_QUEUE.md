# ML Fix Queue

One task per loop iteration. Work strictly top to bottom, first `PENDING` task.

**Agents may:** change a task's status, fill in its result block, append a new
task at the END with a written rationale.

**Agents may not:** reorder tasks, delete tasks, edit any task's acceptance
criterion, or edit the diagnosis below.

Status values: `PENDING`, `IN_PROGRESS`, `PASSED`, `FAILED`, `SKIPPED`, `SUSPICIOUS`.

---

## The diagnosis this queue is built on

Measured 2026-08-02, not guessed:

The model learns **position** but not **sharpness**. It produces a broad
low-contrast plateau (peak-to-background 2.4:1) whose strongest voxels are not
at drone locations (pred_max is 1.66x the true-loc mean). Hungarian matching
still localises to ~1 m, so positional information is present.

Root cause: at encoder stride 8 with a_max ~10 px, each drone occupies ~1.2
feature pixels. **The encoder cannot produce a peak narrower than its own
receptive field.**

Already refuted, do not retry:
- Extraction threshold tuning. Raising it 300x removed zero peaks. At
  0.7 x pred_max, count collapses but median error explodes to 36.8 m because
  true-drone voxels (~0.099) fall below the threshold (0.115). Signal and
  spurious occupy the same value band.
- Weighted MSE at pos_weight=500 alone. pred_max rose 0.06 to 0.15; the diffuse
  field persisted.
- Count/mass term alone. Overwhelmed by the diffuse background.
- Sigma bump 1.5 to 2.0 cells alone.
- Focal loss alone.
- Background-floor crush. Crushed peaks via cross-scene interference.

Every one of those was a **loss-side** fix. The diagnosis is **architectural**.
This queue is ordered accordingly.

---

## FIX-01 — Stride-4 encoder

**Status:** IN_PROGRESS

Reduce encoder downsampling from 3 stages to 2. Each drone then occupies ~2.4
feature pixels instead of ~1.2.

Keep everything else at the current best configuration (weighted MSE
pos_weight=500, count term enabled, sigma 2.0 cells).

**Memory warning:** halving stride quadruples feature-map area. Peak was
3,090 MiB at V=8 stride 8, so expect roughly 12 GB. On a 24 GB machine this
should fit but is not guaranteed. If it OOMs, reduce the *feature channel
count* or use gradient checkpointing. Do NOT reduce the eval view count, the
voxel resolution, or anything named in GATES.md.

**Predict before running.** Expected: peak-to-background ratio rises above the
2.4:1 baseline; count error falls well below the +286 to +364 baseline.

**Result:**
```
predicted:
observed:
median_err_m:
count_err per scene:
peak_to_background:
verdict:
```

---

## FIX-02 — FPN-style upsampling head

**Status:** PENDING

Keep the stride-8 encoder for its receptive field, then upsample back toward
stride 2 with lateral skip connections from the earlier, higher-resolution
stages before back-projection.

This is the standard answer to "needs both context and resolution" and is
cheaper in memory than FIX-01 because the deep layers stay small.

Apply on top of FIX-01 if FIX-01 helped but did not pass; apply standalone if
FIX-01 OOMed.

**Predict before running.**

**Result:**
```
predicted:
observed:
median_err_m:
count_err per scene:
peak_to_background:
verdict:
```

---

## FIX-03 — Tighter Gaussian target

**Status:** PENDING

Reduce target sigma from 2.0 cells to 1.0, on top of whichever structural fix
performed best so far.

A tighter target cannot help a model that physically cannot render a narrow
peak, which is why this sits below the architectural fixes rather than above
them. It should only be tried once feature resolution has improved.

**Predict before running.**

**Result:**
```
predicted:
observed:
median_err_m:
count_err per scene:
peak_to_background:
verdict:
```

---

## FIX-04 — Longer training at the best configuration

**Status:** PENDING

The best previous attempt was 600 steps on 8 scenes, which is only ~75 epochs.
Run the best-performing configuration from FIX-01 to FIX-03 for as many steps
as fit inside the unit timeout, checkpointing so a subsequent iteration can
resume.

At ~1.3 s/step a 60-minute unit allows roughly 2,500 steps. Leave headroom for
setup and evaluation: target 2,000 steps.

**Predict before running.**

**Result:**
```
predicted:
observed:
median_err_m:
count_err per scene:
peak_to_background:
verdict:
```

---

## FIX-05 — Focal loss on the best structural configuration

**Status:** PENDING

CenterNet-style penalty-reduced focal loss, applied on top of the best
structural result. Focal alone was already refuted at stride 8; the question
here is whether it helps once the encoder can actually represent a peak.

**Predict before running.**

**Result:**
```
predicted:
observed:
median_err_m:
count_err per scene:
peak_to_background:
verdict:
```

---

## Queue exhausted

If every task above reaches a terminal status without G2 passing, write a final
summary to `docs/PROGRESS.md` containing:

- the measured result of every fix attempted
- the best peak-to-background ratio achieved and by which configuration
- whether the trend across fixes suggests the diagnosis is right and the fixes
  were insufficient, or the diagnosis itself is wrong
- a recommendation for the next human-supervised session

Then write `OVERNIGHT_COMPLETE` on its own line in `docs/PROGRESS.md` and stop.

**A queue-exhausted run with five honestly-logged failures is a successful
run.** It bounds the problem and rules out five hypotheses with numbers. Do not
manufacture a pass.
