"""One-shot diagnostic: untrained vs FIX-01 trained model on the 8 overfit scenes.

Evaluates a RANDOMLY INITIALIZED model (no training, fresh init, same architecture)
on the same 8 scenes, same eval views, same frozen ml.metrics.evaluate as the
FIX-01 G2 gate. Also reports the training loss curve for the FIX-01 run.

Usage (repo root):
    PYTORCH_ENABLE_MPS_FALLBACK=1 venv/bin/python -m ml.diag_untrained
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# MUST precede torch import (grid_sampler_2d_backward MPS fallback).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from ml.train import (
    EVAL_VIEWS,
    EVAL_VIEWS_DOC,
    N_VIEWS,
    OVERFIT_DEFAULT_STEPS,
    POS_WEIGHT_DEFAULT,
    TARGET_SIGMA_CELLS,
    COUNT_MASK_T,
    COUNT_FLOOR,
    COUNT_BG_WEIGHT,
    SceneDataset,
    ViewSubsetSampler,
    _load_shard,
    _decode_png_bytes,
    _build_target,
    _peak_support_mask,
    _identity_collate,
    _make_train_step,
    build_shard_map,
    load_splits,
    resolve_scenes,
)
from ml.model import VoxelFusionModel, extract_positions
from ml import metrics


def evaluate_with_background(model, scenes, device, view_idxs, tag):
    """Same as evaluate_overfit but also reports background mean and peak-to-bg.

    Returns (rows, bg_means, peak_to_bg_ratios).
    """
    model.eval()
    rows = []
    bg_means = []
    ptb_ratios = []
    for (sid, shard_path, row) in scenes:
        d = _load_shard(shard_path)
        png = d["png_bytes"][row]
        views = np.stack([_decode_png_bytes(png[j]) for j in view_idxs])
        cameras = [d["cameras"][row][j] for j in view_idxs]
        grid = {
            "center": np.asarray(d["swarm_center"][row], dtype=np.float32),
            "radius_m": float(d["radius_m"][row]),
        }
        positions = np.asarray(d["positions"][row], dtype=np.float64)
        with torch.no_grad():
            vol = model.forward_volume(
                torch.from_numpy(views), cameras, grid)[0, 0].cpu().numpy()
        pred = extract_positions(vol, grid)
        met = metrics.evaluate(pred, positions)

        # Background mean: mean of voxels NOT in the peak-support mask.
        target = _build_target(
            positions, d["swarm_center"][row], float(d["radius_m"][row]),
            TARGET_SIGMA_CELLS)
        peak_mask = _peak_support_mask(target)
        bg_voxels = vol[peak_mask < 0.5]
        bg_mean = float(bg_voxels.mean()) if len(bg_voxels) > 0 else 0.0

        pred_max = float(vol.max())
        ptb = pred_max / bg_mean if bg_mean > 0 else float("inf")

        rows.append({
            "tag": tag,
            "scene_id": int(sid),
            "n_true": int(met["n_true"]),
            "n_pred": int(met["n_pred"]),
            "count_err": int(met["count_err"]),
            "median_err_m": float(met["median_err_m"]),
            "pred_max": pred_max,
            "bg_mean": bg_mean,
            "ptb": ptb,
        })
        bg_means.append(bg_mean)
        ptb_ratios.append(ptb)
    return rows


def compute_step0_loss(model, scenes, device):
    """Compute the training loss at step 0 (fresh random init, one forward pass).

    Uses the first scene, a random view subset of EVAL_VIEWS size, and the same
    loss as training (weighted MSE + count term, no optimizer step).
    """
    model.train()
    # Use the first overfit scene with EVAL_VIEWS to match training conditions.
    sid, shard_path, row = scenes[0]
    d = _load_shard(shard_path)

    # Use EVAL_VIEWS (8 views) as a representative view subset.
    view_indices = list(EVAL_VIEWS)
    png = d["png_bytes"][row]
    views_np = np.stack([_decode_png_bytes(png[j]) for j in view_indices])

    target = _build_target(
        np.asarray(d["positions"][row], dtype=np.float64),
        d["swarm_center"][row],
        float(d["radius_m"][row]),
        TARGET_SIGMA_CELLS)
    peak_mask = _peak_support_mask(target)
    n_drones = int(d["n_drones"][row])
    vpp = float((target * peak_mask).sum() / n_drones) if n_drones > 0 else 1.0

    batch = {
        "views": torch.from_numpy(views_np),
        "cameras": [d["cameras"][row][j] for j in view_indices],
        "grid": {
            "center": np.asarray(d["swarm_center"][row], dtype=np.float32),
            "radius_m": float(d["radius_m"][row]),
        },
        "target": torch.from_numpy(target.copy()),
        "peak_mask": torch.from_numpy(peak_mask),
        "n_drones": n_drones,
        "vpp": vpp,
    }

    # Run forward pass and compute loss (no backward, no optimizer step).
    views_t = batch["views"].to(device)
    target_t = batch["target"].to(device)
    peak_mask_t = batch["peak_mask"].to(device)

    with torch.no_grad():
        pred = model.forward_volume(views_t, batch["cameras"], batch["grid"])[0, 0]
        w = 1.0 + POS_WEIGHT_DEFAULT * target_t
        mse = ((pred - target_t).square() * w).mean()
        in_mass = (pred * peak_mask_t).sum() / batch["vpp"]
        bg_excess = torch.relu(pred - COUNT_FLOOR) * (1.0 - peak_mask_t)
        bg_drones = bg_excess.sum() / batch["vpp"]
        count = ((in_mass - batch["n_drones"]).square()
                 + COUNT_BG_WEIGHT * bg_drones.square())
        loss = mse + 0.1 * count

    return float(loss.detach().cpu()), float(mse.detach().cpu()), float(count.detach().cpu())


def compute_checkpoint_loss(model, scenes, device):
    """Compute loss for a trained model on the first overfit scene with EVAL_VIEWS."""
    model.eval()
    sid, shard_path, row = scenes[0]
    d = _load_shard(shard_path)
    view_indices = list(EVAL_VIEWS)
    png = d["png_bytes"][row]
    views_np = np.stack([_decode_png_bytes(png[j]) for j in view_indices])

    target = _build_target(
        np.asarray(d["positions"][row], dtype=np.float64),
        d["swarm_center"][row],
        float(d["radius_m"][row]),
        TARGET_SIGMA_CELLS)
    peak_mask = _peak_support_mask(target)
    n_drones = int(d["n_drones"][row])
    vpp = float((target * peak_mask).sum() / n_drones) if n_drones > 0 else 1.0

    views_t = torch.from_numpy(views_np).to(device)
    target_t = torch.from_numpy(target.copy()).to(device)
    peak_mask_t = torch.from_numpy(peak_mask).to(device)

    with torch.no_grad():
        pred = model.forward_volume(
            views_t,
            [d["cameras"][row][j] for j in view_indices],
            {"center": np.asarray(d["swarm_center"][row], dtype=np.float32),
             "radius_m": float(d["radius_m"][row])})[0, 0]
        w = 1.0 + POS_WEIGHT_DEFAULT * target_t
        mse = ((pred - target_t).square() * w).mean()
        in_mass = (pred * peak_mask_t).sum() / vpp
        bg_excess = torch.relu(pred - COUNT_FLOOR) * (1.0 - peak_mask_t)
        bg_drones = bg_excess.sum() / vpp
        count = ((in_mass - n_drones).square()
                 + COUNT_BG_WEIGHT * bg_drones.square())
        loss = mse + 0.1 * count

    return float(loss.detach().cpu()), float(mse.detach().cpu()), float(count.detach().cpu())


def main():
    import argparse
    p = argparse.ArgumentParser(description="Untrained-model diagnostic")
    p.add_argument("--root", default=os.path.expanduser("~/swarm_ml_packed"))
    p.add_argument("--device", default="cpu",
                   help="cpu (safe, no MPS fallback needed)")
    args = p.parse_args()

    device = torch.device(args.device)
    print("Device:", device)
    print()

    # Resolve the 8 overfit scenes (same as FIX-01 G2 gate).
    splits = load_splits(args.root)
    overfit_seeds = list(range(2000, 2008))
    train_set = set(splits["train"])
    missing = [s for s in overfit_seeds if s not in train_set]
    if missing:
        print("ERROR: overfit seeds not in train split: %r" % missing)
        return 1

    shard_map = build_shard_map(args.root)
    missing = [s for s in overfit_seeds if s not in shard_map]
    if missing:
        print("ERROR: scenes missing from packed shards: %r" % missing[:10])
        return 1

    scenes = [(sid, shard_map[sid][0], shard_map[sid][1]) for sid in overfit_seeds]
    print("Overfit scenes:", overfit_seeds)
    print()

    # ── Diagnostic 1: Randomly initialized (untrained) model ──────────────
    print("=" * 72)
    print("DIAGNOSTIC 1: RANDOMLY INITIALIZED MODEL (no training, seed=0)")
    print("=" * 72)
    torch.manual_seed(0)
    np.random.seed(0)
    untrained = VoxelFusionModel(feat_channels=64).to(device)
    untrained.eval()

    t0 = time.perf_counter()
    rows_untrained = evaluate_with_background(
        untrained, scenes, device, EVAL_VIEWS, "untrained_mixed8")
    wall = time.perf_counter() - t0
    print("  Eval wall-clock: %.1f s" % wall)
    print()
    print("  %-7s %-6s %-6s %-9s %-13s %-8s %-10s %-6s"
          % ("scene", "n_true", "n_pred", "count_err", "median_err_m",
             "pred_max", "bg_mean", "PTB"))
    for r in rows_untrained:
        print("  %-7d %-6d %-6d %-9d %-13.4f %-8.4f %-10.6f %-5.1f"
              % (r["scene_id"], r["n_true"], r["n_pred"], r["count_err"],
                 r["median_err_m"], r["pred_max"], r["bg_mean"], r["ptb"]))
    untrained_n_pred = [r["n_pred"] for r in rows_untrained]
    untrained_count_err = [r["count_err"] for r in rows_untrained]
    untrained_med = float(np.median([r["median_err_m"] for r in rows_untrained]))
    untrained_ptb = float(np.median([r["ptb"] for r in rows_untrained]))
    untrained_bg = float(np.median([r["bg_mean"] for r in rows_untrained]))
    untrained_pmax = float(np.median([r["pred_max"] for r in rows_untrained]))

    print()
    print("  Summary (untrained, mixed8):")
    print("    n_pred range:          %d to %d" % (min(untrained_n_pred), max(untrained_n_pred)))
    print("    count_err range:       %+d to %+d" % (min(untrained_count_err), max(untrained_count_err)))
    print("    median position error: %.4f m" % untrained_med)
    print("    pred_max (median):     %.4f" % untrained_pmax)
    print("    background mean (med): %.6f" % untrained_bg)
    print("    peak-to-background:    %.1f : 1" % untrained_ptb)
    print()

    # ── Diagnostic 2: FIX-01 trained model ─────────────────────────────────
    print("=" * 72)
    print("DIAGNOSTIC 2: FIX-01 TRAINED MODEL (checkpoint)")
    print("=" * 72)
    ckpt_path = os.path.join(
        _REPO_ROOT, "checkpoints", "overfit_mse_s2.0_v2-8", "run_2", "latest.pt")
    if not os.path.isfile(ckpt_path):
        print("  Checkpoint not found: %s" % ckpt_path)
        print("  (skipping trained-model eval — already in PROGRESS.md)")
    else:
        trained = VoxelFusionModel(feat_channels=64).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trained.load_state_dict(ckpt["model"])
        trained.eval()

        t0 = time.perf_counter()
        rows_trained = evaluate_with_background(
            trained, scenes, device, EVAL_VIEWS, "fix01_mixed8")
        wall = time.perf_counter() - t0
        print("  Eval wall-clock: %.1f s" % wall)
        print()
        print("  %-7s %-6s %-6s %-9s %-13s %-8s %-10s %-6s"
              % ("scene", "n_true", "n_pred", "count_err", "median_err_m",
                 "pred_max", "bg_mean", "PTB"))
        for r in rows_trained:
            print("  %-7d %-6d %-6d %-9d %-13.4f %-8.4f %-10.6f %-5.1f"
                  % (r["scene_id"], r["n_true"], r["n_pred"], r["count_err"],
                     r["median_err_m"], r["pred_max"], r["bg_mean"], r["ptb"]))
        trained_n_pred = [r["n_pred"] for r in rows_trained]
        trained_count_err = [r["count_err"] for r in rows_trained]
        trained_med = float(np.median([r["median_err_m"] for r in rows_trained]))
        trained_ptb = float(np.median([r["ptb"] for r in rows_trained]))
        trained_bg = float(np.median([r["bg_mean"] for r in rows_trained]))
        trained_pmax = float(np.median([r["pred_max"] for r in rows_trained]))

        print()
        print("  Summary (FIX-01 trained, mixed8):")
        print("    n_pred range:          %d to %d" % (min(trained_n_pred), max(trained_n_pred)))
        print("    count_err range:       %+d to %+d" % (min(trained_count_err), max(trained_count_err)))
        print("    median position error: %.4f m" % trained_med)
        print("    pred_max (median):     %.4f" % trained_pmax)
        print("    background mean (med): %.6f" % trained_bg)
        print("    peak-to-background:    %.1f : 1" % trained_ptb)
        print()

    # ── Loss curve points ─────────────────────────────────────────────────
    print("=" * 72)
    print("TRAINING LOSS CURVE (FIX-01 run)")
    print("=" * 72)
    print("  All-zeros baseline (from GATES.md): 0.0020")
    print()

    # Step 0 loss: fresh model with seed=0 (deterministic).
    torch.manual_seed(0)
    np.random.seed(0)
    step0_model = VoxelFusionModel(feat_channels=64).to(device)
    loss0, mse0, count0 = compute_step0_loss(step0_model, scenes, device)
    print("  Step 0 loss (fresh init, seed=0, scene 2000, V=8):")
    print("    loss = %.6f  (mse=%.6f  count=%.6f)" % (loss0, mse0, count0))
    print()

    # Step 600 loss: from the FIX-01 checkpoint.
    if os.path.isfile(ckpt_path):
        loss600, mse600, count600 = compute_checkpoint_loss(trained, scenes, device)
        print("  Step 600 loss (FIX-01 checkpoint, scene 2000, V=8):")
        print("    loss = %.6f  (mse=%.6f  count=%.6f)" % (loss600, mse600, count600))
        print()
    else:
        print("  Step 600 loss: checkpoint not found, cannot compute")
        print()

    print("  Step 100 loss: NOT CAPTURED in logs (supervised session, no")
    print("    intermediate checkpoint saved at step 100; ckpt_every=250.")
    print("    Cannot recover without re-running training.")
    print()

    # ── Interpretation ────────────────────────────────────────────────────
    print("=" * 72)
    print("INTERPRETATION (pre-registered criteria)")
    print("=" * 72)
    print()
    print("  Pre-registered decision rule:")
    print("    - If untrained model produces ~300 predictions and ~2.4:1 contrast,")
    print("      training is doing nothing meaningful. Problem is optimization/loss,")
    print("      NOT architecture.")
    print("    - If untrained model is clearly different (far more/fewer predictions,")
    print("      flat output), training IS doing something and the problem is")
    print("      elsewhere.")
    print()

    # Compare.
    untrained_range = (min(untrained_n_pred), max(untrained_n_pred))
    trained_range_known = (280, 356)  # from PROGRESS.md FIX-01 observed

    print("  Untrained model n_pred range: %d to %d" % untrained_range)
    print("  FIX-01 trained n_pred range:  %d to %d (from PROGRESS.md)" % trained_range_known)
    print("  Untrained PTB: %.1f : 1" % untrained_ptb)
    print("  FIX-01 trained PTB: ~3.8 : 1 (from PROGRESS.md)")
    print()

    if untrained_range[0] >= 200 and untrained_range[1] <= 500 and untrained_ptb < 3.0:
        print("  >>> VERDICT: Untrained model produces ~%d-%d predictions at %.1f:1")
        print("      contrast — comparable to FIX-01's ~%d-%d at ~3.8:1. Training is")
        print("      doing NEARLY NOTHING. The problem is optimization or the loss,")
        print("      NOT architecture." % (untrained_range[0], untrained_range[1],
                untrained_ptb, trained_range_known[0], trained_range_known[1]))
    else:
        print("  >>> VERDICT: Untrained model is CLEARLY DIFFERENT from trained.")
        print("      Training IS doing something; the problem is elsewhere.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
