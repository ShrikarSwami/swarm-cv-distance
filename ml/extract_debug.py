"""AGENT H diagnostic: is the G2 overfit failure an EXTRACTION defect or an
ARCHITECTURE defect?

Owned file: `ml/extract_debug.py`. Read-only on every other file. No
retraining, no edits to ml/model.py / ml/train.py / ml/metrics.py / splits.json.

The G2 gate FAILED on a model trained on the 8 overfit scenes (seeds 2000-2007):
pred_max ~0.14-0.15, background ~0.04, ~300-400 spurious local maxima. The frozen
`ml.model.extract_positions` threshold is RELATIVE (h.max()*1e-3), so the diffuse
field qualifies en masse. G reported aggregate median_err_m = 0.9853 m even WITH
~350 spurious peaks, so the hypothesis under test is: the model IS learning; the
defect is the extraction threshold, not the architecture.

The three diagnostics:
  STEP 0        — checkpoint reproducibility (precondition). Load each of the 4
                  checkpoints, run the frozen gate eval path (EVAL_VIEWS mixed8),
                  report {aggregate median_err_m, count_err range, n_pred range,
                  pred_max range}. Is run-2 (the 0.9853 m checkpoint) recoverable?
  DIAGNOSTIC 1  — absolute-threshold sweep on the best reproducible checkpoint:
                  frac * pred_max for frac in {0.3, 0.5, 0.7}, plus the default
                  1e-3 relative threshold for reference. Does count error
                  collapse and does the G2 condition hold?
  DIAGNOSTIC 2  — voxel value at true drone locations (max over +/-1-voxel
                  neighbourhood) vs volume mean and vs background level.
  DIAGNOSTIC 3  — histogram shape: bimodal (flat field + peak mass) vs
                  unimodal (genuinely diffuse).

Run:
    python3 ml/extract_debug.py [--root ~/swarm_ml_packed] [--device mps]
"""

from __future__ import annotations

import argparse
import os
import sys

# MUST precede torch import (grid_sampler_2d_backward has no MPS kernel in
# torch 2.12.1; forward is native MPS, backward needs the CPU fallback).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml import metrics  # noqa: E402
from ml.model import (  # noqa: E402
    VOXEL_GRID_RES,
    VoxelFusionModel,
    extract_positions,
    _local_maxima,
)
from ml.train import (  # noqa: E402  (read-only reuse of the frozen eval path)
    EVAL_VIEWS,
    N_VIEWS,
    _decode_png_bytes,
    _load_shard,
    build_shard_map,
    load_splits,
)

SEEDS = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]

# name -> checkpoint path (relative to repo root)
CHECKPOINTS = [
    ("pre-fix  checkpoints/latest.pt          (00:43, cw=0.0, 50 st)", "checkpoints/latest.pt"),
    ("run3     checkpoints/g2_fix/latest.pt   (12:38, pw=2000, 1500 st)", "checkpoints/g2_fix/latest.pt"),
    ("focal    checkpoints/g2_fix_focal/latest.pt (12:51, focal, 600 st)", "checkpoints/g2_fix_focal/latest.pt"),
    ("sigma2   checkpoints/g2_fix_sigma2/latest.pt (13:08, sigma=2, 600 st)", "checkpoints/g2_fix_sigma2/latest.pt"),
]

FRACS = (0.3, 0.5, 0.7)


# ---------------------------------------------------------------------------
# Parameterized extraction (clone of frozen extract_positions; threshold only)
# ---------------------------------------------------------------------------

def extract_positions_thresh(heatmap, grid, thresh):
    """Clone of `ml.model.extract_positions` with an ABSOLUTE threshold.

    Everything else is byte-identical to the frozen path: 3x3x3 local maxima
    (`_local_maxima` imported verbatim), value-descending order, 1.5-voxel
    clustering, 512-peak cap, per-peak 5x5x5 soft-argmax. The only difference
    is `thresh = max(float(h.max()) * 1e-3, 1e-6)` -> the caller-supplied
    absolute `thresh`.
    """
    h = np.asarray(heatmap, dtype=np.float32)
    if h.ndim == 4:
        h = h[0]
    if h.shape != (VOXEL_GRID_RES, VOXEL_GRID_RES, VOXEL_GRID_RES):
        raise ValueError("heatmap must be (64, 64, 64), got %s" % (h.shape,))
    center = np.asarray(grid["center"], dtype=np.float64)
    radius = float(grid["radius_m"])
    res = VOXEL_GRID_RES
    cell = 2.0 * radius / res

    idx = np.arange(res, dtype=np.float64)
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    centers = center + (np.stack([xx, yy, zz], axis=-1) + 0.5) * cell - radius

    maxima = _local_maxima(h)
    peaks = np.argwhere(maxima & (h >= thresh))
    if len(peaks) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    vals = h[maxima & (h >= thresh)]
    order = np.argsort(-vals, kind="stable")
    peaks = peaks[order]
    if len(peaks) > 512:  # hard cap bounds the (quadratic) clustering
        peaks = peaks[:512]
    kept = []
    for p in peaks:
        if all(np.linalg.norm(p - q) > 1.5 for q in kept):
            kept.append(p)
    kept = np.asarray(kept, dtype=np.int64)

    out = np.zeros((len(kept), 3), dtype=np.float32)
    for a, p in enumerate(kept):
        lo = np.maximum(p - 2, 0)
        hi = np.minimum(p + 3, res)
        win = centers[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].reshape(-1, 3)
        w = h[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].ravel().astype(np.float64)
        s = w.sum()
        if s > 0:
            out[a] = (w[:, None] * win).sum(axis=0) / s
        else:
            out[a] = centers[tuple(p)]
    return out


# ---------------------------------------------------------------------------
# Frozen-path heatmap computation (mirrors train.evaluate_overfit decode+forward)
# ---------------------------------------------------------------------------

def compute_heatmaps(model, scenes, device, view_idxs=EVAL_VIEWS):
    """Per-scene heatmaps exactly as the G2 gate computes them (EVAL_VIEWS)."""
    model.eval()
    out = []
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
        out.append({
            "scene_id": int(sid),
            "n_true": int(len(positions)),
            "heatmap": vol.astype(np.float32),
            "grid": grid,
            "positions": positions,
        })
    return out


def rows_from_heatmaps(heatmaps, pred_fn):
    """Score heatmaps with an arbitrary pred_fn(heatmap, grid) -> (K, 3).

    pred_fn = frozen `extract_positions` reproduces the G2 gate rows exactly.
    """
    rows = []
    for sc in heatmaps:
        pred = pred_fn(sc["heatmap"], sc["grid"])
        met = metrics.evaluate(pred, sc["positions"])
        rows.append({
            "scene_id": sc["scene_id"],
            "n_true": int(met["n_true"]),
            "n_pred": int(met["n_pred"]),
            "count_err": int(met["count_err"]),
            "median_err_m": float(met["median_err_m"]),
            "pred_max": float(sc["heatmap"].max()),
        })
    return rows


def aggregate(rows):
    """Aggregate exactly as train._print_eval_table does."""
    med = float(np.median([r["median_err_m"] for r in rows])) \
        if len(rows) else float("nan")
    count_ok = all(-1 <= r["count_err"] <= 1 for r in rows)
    return med, count_ok


# ---------------------------------------------------------------------------
# True-location voxel values (DIAGNOSTIC 2)
# ---------------------------------------------------------------------------

def true_voxel_values(heatmap, grid, positions):
    """Voxel value at each true drone position (max over +/-1-voxel nbhd).

    SS3.3 affine map: cell centre of index i is center + (i+0.5)*cell - radius,
    cell = 2*radius_m/64. Continuous voxel coord of world p:
        x = (p - center + radius)/cell - 0.5   ->  voxel index = floor(x).
    The max over the +/-1 neighbourhood absorbs quantization error.
    """
    center = np.asarray(grid["center"], dtype=np.float64)
    radius = float(grid["radius_m"])
    cell = 2.0 * radius / VOXEL_GRID_RES
    p = np.asarray(positions, dtype=np.float64)
    x = (p - center + radius) / cell - 0.5
    idx = np.floor(x).astype(np.int64)
    idx = np.clip(idx, 0, VOXEL_GRID_RES - 1)
    vals = []
    for (i, j, k) in idx:
        lo = (max(i - 1, 0), max(j - 1, 0), max(k - 1, 0))
        hi = (min(i + 2, VOXEL_GRID_RES), min(j + 2, VOXEL_GRID_RES),
              min(k + 2, VOXEL_GRID_RES))
        vals.append(float(heatmap[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].max()))
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# STEP 0 / DIAGNOSTIC 1
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = VoxelFusionModel(feat_channels=64).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def step0(scenes, device):
    print("=" * 78)
    print("STEP 0 — CHECKPOINT REPRODUCIBILITY (frozen gate path, EVAL_VIEWS"
          " mixed8)")
    print("=" * 78)
    hdr = "  %-52s %-9s %-14s %-9s %-8s" % (
        "checkpoint", "agg_med_m", "count_err", "n_pred", "pred_max")
    print(hdr)
    results = []
    for name, path in CHECKPOINTS:
        full = os.path.join(_REPO_ROOT, path)
        try:
            model, ck = load_model(full, device)
            cfg = ck.get("cfg", {})
            print("\n  %s" % name)
            print("    cfg: steps=%s max_views=%s count_weight=%s pos_weight=%s"
                  " loss=%s target_sigma=%s"
                  % (cfg.get("total_steps"), cfg.get("max_views"),
                     cfg.get("count_weight"), cfg.get("pos_weight"),
                     cfg.get("loss", "mse"), cfg.get("target_sigma_cells")))
        except Exception as e:  # noqa: BLE001
            print("\n  %s\n    LOAD FAILED: %r — skipping" % (name, e))
            results.append((name, "LOAD-FAILED", None))
            continue

        heatmaps = compute_heatmaps(model, scenes, device)
        rows = rows_from_heatmaps(heatmaps, extract_positions)
        agg_med, count_ok = aggregate(rows)
        ce = [r["count_err"] for r in rows]
        np_ = [r["n_pred"] for r in rows]
        pm = [r["pred_max"] for r in rows]
        print("    rows:")
        for r in rows:
            print("      scene %d  n_true=%d n_pred=%d count_err=%+d "
                  "median_err_m=%.4f pred_max=%.4f"
                  % (r["scene_id"], r["n_true"], r["n_pred"], r["count_err"],
                     r["median_err_m"], r["pred_max"]))
        print("    aggregate median_err_m=%.4f  count_err range [%+d, %+d]  "
              "n_pred range [%d, %d]  pred_max range [%.4f, %.4f]"
              % (agg_med, min(ce), max(ce), min(np_), max(np_), min(pm), max(pm)))
        results.append((name, agg_med, rows, heatmaps, model))
    return results


def diag1(scenes, device, best):
    name, agg_med, rows, heatmaps, model = best
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 1 — ABSOLUTE THRESHOLD SWEEP on %s" % name)
    print("=" * 78)
    print("  pred_max per scene = the heatmap's own max (frac is relative to it)")
    print("  default reference = frozen extract_positions (thresh = "
          "h.max()*1e-3):")
    d_rows = rows_from_heatmaps(heatmaps, extract_positions)
    d_med, d_ok = aggregate(d_rows)
    print("    aggregate median_err_m=%.4f  count_ok=%s" % (d_med, d_ok))
    for r in d_rows:
        print("      scene %d  count_err=%+d median_err_m=%.4f pred_max=%.4f"
              % (r["scene_id"], r["count_err"], r["median_err_m"],
                 r["pred_max"]))

    for frac in FRACS:
        pred_fn = lambda h, g: extract_positions_thresh(h, g, frac * float(h.max()))  # noqa: E731
        rows_f = rows_from_heatmaps(heatmaps, pred_fn)
        med_f, ok_f = aggregate(rows_f)
        print("\n  frac=%s (thresh = %.3g * pred_max):" % (frac, frac))
        print("    aggregate median_err_m=%.4f  count_err all in [-1,+1]: %s"
              % (med_f, ok_f))
        for r in rows_f:
            print("      scene %d  n_true=%d n_pred=%d count_err=%+d "
                  "median_err_m=%.4f"
                  % (r["scene_id"], r["n_true"], r["n_pred"], r["count_err"],
                     r["median_err_m"]))
        g2 = (med_f < 1.0 and ok_f)
        print("    G2 condition (agg median < 1.0 AND every count_err in "
              "[-1,+1]): %s" % ("PASS" if g2 else "FAIL"))


# ---------------------------------------------------------------------------
# DIAGNOSTIC 2 / 3
# ---------------------------------------------------------------------------

def diag2_3(scenes, device, best):
    name, agg_med, rows, heatmaps, model = best
    print("\n" + "=" * 78)
    print("DIAGNOSTIC 2 — SIGNAL AT TRUE DRONE LOCATIONS (%s)" % name)
    print("=" * 78)
    pooled_true = []
    pooled_vol = []
    pooled_peak = []   # voxels above 0.5*pred_max (peak-mass mask)
    per_scene = []
    for sc in heatmaps:
        h = sc["heatmap"]
        tv = true_voxel_values(h, sc["grid"], sc["positions"])
        vmean = float(h.mean())
        vmed = float(np.median(h))
        p99 = float(np.percentile(h, 99))
        pmax = float(h.max())
        # background: median of the non-peak volume (exclude +/-1-voxel nbhd
        # of every true location)
        nbhd_mask = _true_nbhd_mask(sc["grid"], sc["positions"])
        bg = float(np.median(h[~nbhd_mask])) if (~nbhd_mask).any() else float("nan")
        per_scene.append({
            "scene_id": sc["scene_id"],
            "n_true": sc["n_true"],
            "true_mean": float(tv.mean()),
            "true_median": float(np.median(tv)),
            "true_min": float(tv.min()),
            "vol_mean": vmean,
            "vol_median": vmed,
            "bg_median": bg,
            "p99": p99,
            "pred_max": pmax,
        })
        pooled_true.append(tv)
        pooled_vol.append(h)
        pooled_peak.append(h[h > 0.5 * pmax])
    true_all = np.concatenate(pooled_true)
    vol_all = np.concatenate([v.ravel() for v in pooled_vol])
    peak_all = np.concatenate(pooled_peak)
    print("  %-6s %-7s %-10s %-10s %-10s %-10s %-10s %-10s %-8s"
          % ("scene", "n_true", "true_mean", "true_med", "vol_mean",
             "vol_med", "bg_med", "p99", "pred_max"))
    for p in per_scene:
        print("  %-6d %-7d %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f "
              "%-8.4f"
              % (p["scene_id"], p["n_true"], p["true_mean"], p["true_median"],
                 p["vol_mean"], p["vol_median"], p["bg_median"], p["p99"],
                 p["pred_max"]))
    print("  POOLED: true-loc mean=%.5f median=%.5f (min=%.5f) | vol mean=%.5f "
          "vol median=%.5f | bg median=%.5f | p99=%.5f"
          % (true_all.mean(), np.median(true_all), true_all.min(),
             vol_all.mean(), np.median(vol_all),
             np.median([p["bg_median"] for p in per_scene]),
             float(np.percentile(vol_all, 99))))
    print("  true-loc mean / bg-mean ratio: %.1f:1" %
          (true_all.mean() / max(np.median([p["bg_median"] for p in per_scene]), 1e-12)))
    print("  true-loc values > p99 of volume: %.1f%%"
          % (100.0 * (true_all > np.percentile(vol_all, 99)).mean()))
    print("  true-loc values > 0.5*pred_max: %.1f%%"
          % (100.0 * (true_all > 0.5 * vol_all.max()).mean()))

    print("\n" + "=" * 78)
    print("DIAGNOSTIC 3 — HISTOGRAM SHAPE (%s)" % name)
    print("=" * 78)
    pmax = float(vol_all.max())
    print("  pooled volume: %d voxels, pred_max=%.4f, mean=%.5f, median=%.5f"
          % (len(vol_all), pmax, vol_all.mean(), np.median(vol_all)))
    for mult, label in ((0.1, "0.1x"), (0.5, "0.5x"), (0.9, "0.9x")):
        th = mult * pmax
        n = int((vol_all > th).sum())
        print("  voxels > %.2f (%.3g x pred_max): %d (%.5f%%)"
              % (th, mult, n, 100.0 * n / len(vol_all)))
    # flat background level: mode-ish (median) and the bulk 50th-90th pct band
    print("  flat background: vol median=%.5f, p10=%.5f, p50=%.5f, p90=%.5f"
          % (np.median(vol_all), np.percentile(vol_all, 10),
             np.percentile(vol_all, 50), np.percentile(vol_all, 90)))
    # histogram at coarse bands (value = band lower edge)
    print("  histogram (coarse bins, pooled):")
    bins = np.linspace(0.0, pmax, 41)  # 40 bins
    counts, _ = np.histogram(vol_all, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], counts):
        bar = "#" * int(round(80.0 * c / counts.max())) if counts.max() else ""
        print("    [%.5f, %.5f) %7d  %s" % (lo, hi, c, bar))
    print("  peak-mass occupancy (voxels > 0.5*pred_max): %d (%.5f%%)"
          % (len(peak_all), 100.0 * len(peak_all) / len(vol_all)))

    # Local-maxima survival curve: how many 3x3x3 local maxima clear each
    # ABSOLUTE threshold? Pooled over the 8 scenes. Shows whether any
    # threshold cleanly separates drone peaks from background maxima.
    lmax_vals = np.concatenate([h[_local_maxima(h)] for h in
                                (sc["heatmap"] for sc in heatmaps)])
    print("  local-maxima survival curve (pooled, n_local_max=%d):"
          % len(lmax_vals))
    abs_threshs = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16)
    for t in abs_threshs:
        n = int((lmax_vals >= t).sum())
        print("    >= %.3f: %d" % (t, n))
    # number of true drones pooled, for reference
    n_true_pooled = int(sum(sc["n_true"] for sc in heatmaps))
    print("    (pooled n_true = %d drones)" % n_true_pooled)


def _true_nbhd_mask(grid, positions, width=1):
    """Boolean mask over the 64^3 volume of the +/-width nbhd of true voxels."""
    center = np.asarray(grid["center"], dtype=np.float64)
    radius = float(grid["radius_m"])
    cell = 2.0 * radius / VOXEL_GRID_RES
    p = np.asarray(positions, dtype=np.float64)
    x = (p - center + radius) / cell - 0.5
    idx = np.floor(x).astype(np.int64)
    mask = np.zeros((VOXEL_GRID_RES,) * 3, dtype=bool)
    for (i, j, k) in idx:
        lo = (max(i - width, 0), max(j - width, 0), max(k - width, 0))
        hi = (min(i + width + 1, VOXEL_GRID_RES),
              min(j + width + 1, VOXEL_GRID_RES),
              min(k + width + 1, VOXEL_GRID_RES))
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return mask


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/swarm_ml_packed"))
    ap.add_argument("--device", default="mps", choices=("mps", "cpu"))
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")

    splits = load_splits(args.root)
    train_set = set(splits["train"])
    missing = [s for s in SEEDS if s not in train_set]
    if missing:
        raise ValueError("overfit seeds not in train split: %r" % (missing,))
    shard_map = build_shard_map(args.root)
    scenes = [(sid, shard_map[sid][0], shard_map[sid][1]) for sid in SEEDS]
    print("scenes:", [(s, os.path.basename(shard_map[s][0]), shard_map[s][1])
                      for s in SEEDS])
    print("EVAL_VIEWS (mixed8):", EVAL_VIEWS, " N_VIEWS:", N_VIEWS)

    t0 = time.perf_counter()
    results = step0(scenes, device)
    print("\n[step0 wall %.1fs]" % (time.perf_counter() - t0))

    # Choose the best reproducible checkpoint for the diagnostics.
    viable = [r for r in results if len(r) >= 5 and r[1] is not None]
    if not viable:
        print("\nNO checkpoint loaded or yielded a stable reproducible eval. "
              "STOPPING — diagnostics not run.")
        return 1
    # best = lowest aggregate median_err_m among checkpoints that produced rows
    best = min(viable, key=lambda r: r[1])
    print("\nBEST AVAILABLE REPRODUCIBLE CHECKPOINT for diagnostics: %s"
          % best[0])
    print("  (aggregate median_err_m=%.4f)" % best[1])

    diag1(scenes, device, best)
    diag2_3(scenes, device, best)
    print("\n[total wall %.1fs]" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    import time
    sys.exit(main())
