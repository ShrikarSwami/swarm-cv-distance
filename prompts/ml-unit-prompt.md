# Unit Prompt — ML Model Fix Loop

You are one iteration of an unattended overnight loop. You do **one** task, log
it honestly, commit, and exit. A fresh process handles the next task.

There is **no human available**. Nobody will answer a question, approve an
exception, or catch a mistake tonight. Everything below is written on the
assumption that the only thing protecting the result is your own discipline plus
the loop's checksums.

---

## Read first, in this order

1. `ml/GATES.md` — the frozen gate. Never edit it.
2. `ml/FIX_QUEUE.md` — the task list and the diagnosis behind it.
3. `docs/PROGRESS.md` — what previous iterations found.

---

## Your single unit of work

1. Find the first task in `ml/FIX_QUEUE.md` with status `PENDING`.
   - If none, go to **Queue exhausted** below.
2. Mark it `IN_PROGRESS` and commit that single change immediately, so a crash
   leaves a visible trace.
3. **Write your prediction into the task's result block, and commit it, BEFORE
   running anything.** Predicted median position error, predicted count error,
   predicted peak-to-background ratio. This is not optional and it is not a
   formality: a prediction written after the fact is worthless, and the
   prediction-versus-observed log is the project's primary defence against
   fooling itself.
4. Implement the change. You may write to **`ml/model.py` and `ml/train.py`
   only**.
5. Run the G2 gate exactly as `ml/GATES.md` defines it. Seeds 2000-2007, eval
   views (0,3,6,8,11,14,17,21), metrics through frozen `ml.metrics.evaluate`.
6. Record observed numbers next to your prediction. Per-scene count error, not
   just the aggregate.
7. Set the task status:
   - `PASSED` — both G2 conditions met on every scene
   - `FAILED` — either condition missed. This is a normal outcome.
   - `SKIPPED` — could not be attempted; say exactly why
   - `SUSPICIOUS` — passed, but in a way that looks too clean
8. Append a session entry to `docs/PROGRESS.md`: task, predicted, observed,
   ratio, match, and what the result implies for the remaining queue.
9. Commit everything. Exit.

**One task per invocation.** Do not start a second task even if time remains.
The loop will invoke you again.

---

## Hard constraints

**Files you may write:** `ml/model.py`, `ml/train.py`, `ml/FIX_QUEUE.md`
(status and result blocks only), `docs/PROGRESS.md`, and files under
`checkpoints/`.

**Files you may never write:** `ml/GATES.md`, `ml/metrics.py`, `ml/splits.json`,
`calib.json`, `tests/test_predictions_ml.py`, anything under
`stage1_geometry/`, anything under `ml/webapp/`, `ml/baseline_adapter.py`,
`ml/eval_sweep.py`, `ml/adjacency_eval.py`, `ml/recon_app.py`.

The loop verifies checksums of the frozen set before every iteration and aborts
the entire run if any changed. If you believe a frozen file must change, do not
change it: write the argument into `docs/PROGRESS.md` and mark the task
`SKIPPED`.

**Data:** `~/swarm_ml` and `~/swarm_ml_packed` are read-only to you. Never
delete or modify a rendered scene or a packed shard.

**`/Volumes/My Passport` is never touched.** It is a read-only NTFS volume
holding unrelated data that must survive. Do not read from it, write to it, or
attempt to remount it.

**Environment:** export `PYTORCH_ENABLE_MPS_FALLBACK=1`.
`aten::grid_sampler_2d_backward` has no MPS implementation in torch 2.12.1.
Batch size stays 1: batch growth is superlinear (1.3 / 2.7 / 7.0 / 21.9 / 87 s
for B=1/2/4/6/8).

**Checkpoints are run-scoped:** `checkpoints/<task_id>/run_<N>/`. Never write a
shared `latest.pt`. A previous session reported a G2 number from a checkpoint
that a later run had already overwritten, making the number unreproducible.
Every number you report must be reproducible from a named checkpoint file that
still exists when you exit.

**Timeout:** you have 60 minutes. At ~1.3 s/step that is roughly 2,500 training
steps. Budget for setup and evaluation; target 2,000 or fewer. If you are
approaching the limit, checkpoint, log honestly that the task was truncated, and
exit cleanly rather than being killed mid-write.

---

## Things that will feel reasonable tonight and are forbidden

Ranked by how likely they are to seem justified at 3am.

1. **Weakening the gate.** Loosening the count-error bound, switching to an
   aggregate instead of per-scene, reporting position error alone, or replacing
   the criterion with a loss threshold. The original loss-threshold gate passed
   a model that had learned nothing. If the gate seems wrong, the finding is
   that the gate is wrong, and that finding goes in `PROGRESS.md` for a human.
2. **Reducing the eval view set, the voxel resolution, or the overfit seeds to
   fit memory.** Fix memory by reducing channel count or using gradient
   checkpointing, both of which leave the gate intact.
3. **Re-running until a favourable seed appears.** One configuration, one
   evaluation, reported.
4. **Reporting a number you cannot reproduce.** If the checkpoint is gone, the
   number is gone. Say so.
5. **Retrying a refuted fix** because the queue is running out. The refuted list
   in `FIX_QUEUE.md` is refuted with measurements.
6. **Softening a failure in the log.** Five honestly-logged failures bound the
   problem and rule out five hypotheses. A manufactured pass destroys the whole
   run's value and wastes a human's time discovering it later.

---

## If G2 passes

Do not immediately write the sentinel. First check whether the pass is real:

- Is per-scene count error genuinely within [-1, +1] on all 8 scenes?
- Is the peak-to-background ratio meaningfully above the 2.4:1 baseline?
- Does the checkpoint reload from disk and reproduce the same numbers?
- Could ground truth have leaked into the prediction path?

If all four check out, mark `PASSED`, write a full summary to
`docs/PROGRESS.md`, and write `OVERNIGHT_COMPLETE` on its own line in
`docs/PROGRESS.md`.

If anything looks off, mark the task `SUSPICIOUS`, write down exactly what looks
off, and **do not** write the sentinel. Overfitting 8 scenes should be easy for
a working architecture, so a pass is plausible, but a jump from +300 count error
to 0 deserves one minute of scepticism.

---

## Queue exhausted

Every task terminal, no pass. Write the summary described at the bottom of
`ml/FIX_QUEUE.md`, then write `OVERNIGHT_COMPLETE` on its own line in
`docs/PROGRESS.md`, then exit.

This is a legitimate outcome, not a failure of the run.
