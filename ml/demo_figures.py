#!/usr/bin/env python3
"""ml/demo_figures.py — generate demonstration figures from the frozen geometric baseline.

Produces four PNGs in logs/figures/:
  hero_seed63_v8mixed.png            — HERO (3 panels: renders, blob overlay, 3D overlay)
  view_progression_seed63_mixed.png  — VIEW-COUNT PROGRESSION (V=2,3,5,8)
  density_progression_v8mixed.png    — DENSITY PROGRESSION (~10/~35/~55 drones)
  tier_comparison_seed243_v2.png     — TIER COMPARISON (all-ground vs mixed)

ALL metrics printed on figures come from the run that made each figure.
Logs every pipeline run to logs/figures/demo_runs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap — mirrors ml/recon_app.py
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE1 = os.path.join(_REPO_ROOT, "stage1_geometry")
for _p in (_STAGE1, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Headless backend (Agg) before pyplot
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers "3d")

# ---------------------------------------------------------------------------
# Frozen imports (read / call / never modify)
# ---------------------------------------------------------------------------
from ml.baseline_adapter import (  # noqa: E402
    process_scene,
    _detect_views,
    _build_rig,
    scene_dir_for_seed,
    _read_rgb,
    IMAGE_SIZE,
)
from ml.recon_app import (  # noqa: E402
    load_manifest,
    _load_scene_data,
    _overlay_pred_positions,
    _match_pred_to_true,
    camera_positions,
    N_VIEWS_TOTAL,
    DEFAULT_ROOT,
)
from ml.eval_sweep import COMPOSITIONS  # noqa: E402  frozen ground/level/aerial/mixed

# ---------------------------------------------------------------------------
# Palette — same as recon_app.py (validated: CVD-OK, contrast-OK; grays are
# intentionally muted secondary markers with direct labels, per dataviz rule).
# ---------------------------------------------------------------------------
C_GT = "#1db954"       # ground truth — green
C_PRED = "#4488ff"     # predictions — blue
C_ERR = "#d62728"      # error vectors — red
C_GHOST = "#ff7f0e"    # ghost predictions — orange
C_MISS = "#666666"     # missed ground truth — gray (hollow)
C_CAM = "#888888"      # camera positions — gray (triangle)

# ---------------------------------------------------------------------------
# Default seeds (picked for their visible phenomena; overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_HERO_SEED = 63          # 30 drones, primary — clean hero, V=3 knee
DEFAULT_DENSITY_LOW = 148       # 10 drones — count_err=0
DEFAULT_DENSITY_MID = 103       # 35 drones — count_err=+1, one ghost
DEFAULT_DENSITY_HIGH = 18       # 55 drones — count_err=+4, phantom tracks
DEFAULT_TIER_SEED = 243         # 55 drones, primary — V=2 mixed − ground ≈ +0.102 mAP
OUT_DIR = os.path.join(_REPO_ROOT, "logs", "figures")
LOG_PATH = os.path.join(OUT_DIR, "demo_runs.jsonl")

# 3D viewing angle (fixed for visual consistency across subplots)
ELEV, AZIM = 25, -60

# ---------------------------------------------------------------------------
# Pipeline runner + logging
# ---------------------------------------------------------------------------

def run_and_log(root: str, seed: int, comp: str, v: int, figure: str,
                log_path: str) -> dict:
    """Run the frozen pipeline on (seed, composition, view count).

    Returns (result_dict, pred_positions_3d). Logs the run to log_path.
    """
    views = COMPOSITIONS[comp][:v]
    t0 = time.time()
    result = process_scene(root, seed, views, 5.0)  # recall_radius_px=5.0
    elapsed = time.time() - t0

    # Recover predicted 3D positions for the overlay (frozen replay).
    gt, cam = _load_scene_data(root, seed)
    pred_pos = _overlay_pred_positions(root, seed, views, cam=cam)
    cam_pos = camera_positions(cam, views)

    m = result["metrics"]
    log_entry = {
        "figure": figure,
        "seed": seed,
        "cell": result["cell"],
        "n_drones_gt": m["n_true"],
        "composition": comp,
        "view_count": v,
        "views": views,
        "n_pred": m["n_pred"],
        "count_err": m["count_err"],
        "median_err_m": m["median_err_m"],
        "mAP": m["mAP"],
        "chamfer_m": m["chamfer_m"],
        "n_tracks": result.get("n_tracks"),
        "detector_recall": result.get("detector_recall"),
        "wall_clock_s": elapsed,
        "metrics_full": m,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print("  [%.1fs] seed=%d comp=%s V=%d  median=%.4f mAP=%.4f count=%+d"
          % (elapsed, seed, comp, v, m["median_err_m"], m["mAP"], m["count_err"]),
          flush=True)
    return {"result": result, "pred": pred_pos, "cam_pos": cam_pos, "gt_dict": gt}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_3d_overlay(ax, pred: np.ndarray, true: np.ndarray,
                    cam_pos: np.ndarray | None = None,
                    elev: float = ELEV, azim: float = AZIM):
    """Draw the canonical GT/pred/ghost/missed overlay on a 3D axes.

    Replicates recon_app.plot_overlay's legend and colors on an existing ax.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    cam_pos = np.asarray(cam_pos, dtype=np.float64) if cam_pos is not None else np.empty((0, 3))

    rows, cols, ghost_idx, missed_idx = _match_pred_to_true(pred, true)

    # Ground truth (green).
    if true.shape[0]:
        ax.scatter(true[:, 0], true[:, 1], true[:, 2], c=C_GT, s=42,
                   depthshade=False, label="GT (%d)" % true.shape[0])
    # Predictions (blue).
    if pred.shape[0]:
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=C_PRED, s=30,
                   depthshade=False, label="pred (%d)" % pred.shape[0])
    # Error vectors (red).
    for r, c in zip(rows, cols):
        ax.plot([true[c, 0], pred[r, 0]], [true[c, 1], pred[r, 1]],
                [true[c, 2], pred[r, 2]], c=C_ERR, lw=0.8, alpha=0.7)
    if len(rows):
        ax.plot([], [], [], c=C_ERR, lw=0.8, label="error vec (%d)" % len(rows))
    # Ghosts (orange x).
    if len(ghost_idx):
        g = pred[ghost_idx]
        ax.scatter(g[:, 0], g[:, 1], g[:, 2], c=C_GHOST, marker="x", s=80,
                   depthshade=False, label="ghost (%d)" % len(ghost_idx))
    # Missed (gray hollow circles).
    if len(missed_idx):
        m = true[missed_idx]
        ax.scatter(m[:, 0], m[:, 1], m[:, 2], facecolors="none",
                   edgecolors=C_MISS, s=110, depthshade=False,
                   label="missed (%d)" % len(missed_idx))
    # Cameras (gray triangles).
    if cam_pos.shape[0]:
        ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], c=C_CAM,
                   marker="^", s=40, depthshade=False,
                   label="cameras (%d)" % cam_pos.shape[0])

    ax.set_xlabel("East (m)", fontsize=7, labelpad=3)
    ax.set_ylabel("North (m)", fontsize=7, labelpad=3)
    ax.set_zlabel("Up (m)", fontsize=7, labelpad=3)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=elev, azim=azim)

    # Square-ish aspect from the data bounds.
    all_pts = np.vstack([p for p in (true, pred, cam_pos) if p.shape[0]])
    if all_pts.shape[0]:
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        ax.set_box_aspect([float(s) for s in span])

    ax.legend(loc="upper left", fontsize=6, framealpha=0.6)


def draw_detection_overlay(ax, render_path: str, det_points: np.ndarray | None,
                           tier_label: str = "", angle_idx: int = -1):
    """Show a render PNG with detected blobs overlaid as bright markers.

    Used for HERO panel (b).
    """
    rgb = _read_rgb(render_path)  # (H, W, 3) uint8
    ax.imshow(rgb)
    if det_points is not None and det_points.shape[0] > 0:
        ax.scatter(det_points[:, 0], det_points[:, 1],
                   c=C_PRED, s=18, edgecolors="white", linewidths=0.6,
                   alpha=0.95, zorder=5, label="detected (%d)" % det_points.shape[0])
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    else:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    if tier_label:
        ax.text(0.02, 0.95, "%s (angle %d)" % (tier_label, angle_idx),
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Figure 1: HERO — 3 panels
# ---------------------------------------------------------------------------

def make_hero(seed: int, comp: str, v: int, root: str, log_path: str,
              out_dir: str) -> str:
    """Generate the HERO figure: (a) 4 renders by tier, (b) blob overlay, (c) 3D overlay."""
    print("=== HERO: seed=%d, %s, V=%d ===" % (seed, comp, v))
    r = run_and_log(root, seed, comp, v, "hero", log_path)
    m = r["result"]["metrics"]

    # --- Panel (a): 4 input renders labelled by tier ---
    # Pick one render from each tier + one extra (ground with different az).
    # ground angles 0-7, level 8-15, aerial 16-23.
    hero_angles = [
        (0, "ground"), (8, "level"), (16, "aerial"), (23, "aerial"),
    ]
    sd = scene_dir_for_seed(root, seed)
    render_paths = [os.path.join(sd, "angle_%02d.png" % a) for a, _ in hero_angles]

    # --- Panel (b): detected blobs on one render ---
    cam = r["gt_dict"]  # unused; we need cameras.json
    _, cam_dict = _load_scene_data(root, seed)
    rig = _build_rig(cam_dict)
    standoff = float(cam_dict["standoff_m"])
    # Detections on angle 0 (ground) for panel (b).
    b_angle = 0
    b_dets = _detect_views(sd, cam_dict, rig, standoff, [b_angle])
    b_det_pts = b_dets.points_per_view[0] if b_dets.points_per_view else np.empty((0, 2))

    # --- Build figure with manual axes for clean panel layout ---
    fig = plt.figure(figsize=(18, 13))

    # Panel (a): 2x2 renders in top-left quadrant
    gs_a = fig.add_gridspec(2, 2, left=0.02, right=0.35, top=0.95,
                            bottom=0.53, wspace=0.02, hspace=0.06)
    for i, (angle, tier) in enumerate(hero_angles):
        ax_a = fig.add_subplot(gs_a[i // 2, i % 2])
        img = _read_rgb(render_paths[i])
        ax_a.imshow(img)
        ax_a.set_title("angle %02d — %s" % (angle, tier), fontsize=9, pad=3)
        ax_a.set_axis_off()
    fig.text(0.185, 0.97, "(a) Input renders", fontsize=11, fontweight="bold",
             ha="center", va="top")

    # Panel (b): blob overlay in top-right quadrant
    ax_b = fig.add_axes([0.40, 0.53, 0.58, 0.42])
    draw_detection_overlay(ax_b, render_paths[0], b_det_pts,
                           tier_label="ground", angle_idx=b_angle)
    fig.text(0.69, 0.97, "(b) Detected blobs — angle 0 (ground)",
             fontsize=11, fontweight="bold", ha="center", va="top")

    # Panel (c): 3D overlay spanning bottom full width
    ax_c = fig.add_axes([0.12, 0.04, 0.76, 0.44], projection="3d")
    draw_3d_overlay(ax_c, r["pred"],
                    np.asarray(r["gt_dict"]["positions"], dtype=np.float64),
                    cam_pos=r["cam_pos"])
    fig.text(0.50, 0.49, "(c) 3D reconstruction — predicted vs ground truth",
             fontsize=11, fontweight="bold", ha="center", va="top")
    # Annotation: all numbers from the run that made this figure.
    note_lines = [
        "V=%d %s  |  %d drones" % (v, comp, m["n_true"]),
        "median err: %.4f m  |  mAP: %.4f" % (m["median_err_m"], m["mAP"]),
        "count err: %+d  |  chamfer: %.4f m" % (m["count_err"], m["chamfer_m"]),
    ]
    fig.text(0.50, 0.01, "\n".join(note_lines), fontsize=10, ha="center",
             va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))

    # Figure-level title.
    fig.suptitle("HERO: seed %d — %d drones, primary cell, geometric baseline"
                 % (seed, m["n_true"]), fontsize=14, y=0.99, fontweight="bold")

    out_path = os.path.join(out_dir, "hero_seed%d_v%d%s.png" % (seed, v, comp))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved: %s" % out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: VIEW-COUNT PROGRESSION — 2x2 subplots (V=2,3,5,8)
# ---------------------------------------------------------------------------

def make_view_progression(seed: int, comp: str,
                          view_counts: list[int],
                          root: str, log_path: str,
                          out_dir: str) -> str:
    """Generate VIEW-COUNT PROGRESSION: 4 overlays at V=2,3,5,8."""
    print("=== VIEW-COUNT PROGRESSION: seed=%d, %s, V=%s ==="
          % (seed, comp, view_counts))

    runs = []
    for vc in view_counts:
        runs.append(run_and_log(root, seed, comp, vc, "view_progression", log_path))

    # Load GT positions (same for all runs of same seed).
    gt_arr = np.asarray(runs[0]["gt_dict"]["positions"], dtype=np.float64)

    # Compute global 3D bounds (same scene, same axes for fair comparison).
    all_pts = [gt_arr]
    for r in runs:
        if r["pred"].shape[0]:
            all_pts.append(r["pred"])
        if r["cam_pos"].shape[0]:
            all_pts.append(r["cam_pos"])
    bounds = np.vstack(all_pts)
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                             subplot_kw={"projection": "3d"})
    axes_flat = axes.ravel()

    for i, (ax, run, vc) in enumerate(zip(axes_flat, runs, view_counts)):
        m = run["result"]["metrics"]
        draw_3d_overlay(ax, run["pred"], gt_arr, cam_pos=run["cam_pos"])
        # Fix axis limits for fair comparison.
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        # Metric annotation on the subplot.
        label = "V=%d  median=%.4f m\nmAP=%.4f  count err=%+d" % (
            vc, m["median_err_m"], m["mAP"], m["count_err"])
        ax.set_title(label, fontsize=9, pad=4)
        # Highlight the V=3 knee panel.
        if vc == 3:
            ax.text2D(0.03, 0.95, "KNEE", transform=ax.transAxes,
                      fontsize=11, fontweight="bold", color=C_ERR,
                      ha="left", va="top",
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3cd",
                                edgecolor=C_ERR, alpha=0.9))

    fig.suptitle("View-count progression: seed %d, %d drones, primary, mixed"
                 % (seed, runs[0]["result"]["metrics"]["n_true"]),
                 fontsize=13, y=1.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = os.path.join(out_dir, "view_progression_seed%d_%s.png"
                            % (seed, comp))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved: %s" % out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: DENSITY PROGRESSION — 1x3 (low / mid / high at V=8)
# ---------------------------------------------------------------------------

def make_density_progression(seeds_info: list[dict], comp: str, v: int,
                             root: str, log_path: str,
                             out_dir: str) -> str:
    """Generate DENSITY PROGRESSION: 3 overlays at different drone counts."""
    labels = [s["label"] for s in seeds_info]
    print("=== DENSITY PROGRESSION: %s, %s, V=%d ===" % (labels, comp, v))

    runs = []
    for s in seeds_info:
        runs.append(run_and_log(root, s["seed"], comp, v,
                                "density_progression", log_path))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw={"projection": "3d"})
    for i, (ax, run, si) in enumerate(zip(axes, runs, seeds_info)):
        m = run["result"]["metrics"]
        gt_arr = np.asarray(run["gt_dict"]["positions"], dtype=np.float64)
        draw_3d_overlay(ax, run["pred"], gt_arr, cam_pos=run["cam_pos"])
        # Subplot label.
        title = ("%d drones (seed %d)\n"
                 "count err=%+d  median=%.4f m  mAP=%.4f"
                 % (m["n_true"], si["seed"], m["count_err"],
                    m["median_err_m"], m["mAP"]))
        ax.set_title(title, fontsize=9, pad=6)

    fig.suptitle("Density progression: V=%d %s, primary cell — phantom tracks at high density"
                 % (v, comp), fontsize=13, y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = os.path.join(out_dir, "density_progression_v%d_%s.png"
                            % (v, comp))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved: %s" % out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: TIER COMPARISON — 1x2 (all-ground vs mixed at V=2)
# ---------------------------------------------------------------------------

def make_tier_comparison(seed: int, v: int, root: str, log_path: str,
                         out_dir: str) -> str:
    """Generate TIER COMPARISON: V=2 all-ground vs mixed."""
    print("=== TIER COMPARISON: seed=%d, V=%d ===" % (seed, v))

    r_ground = run_and_log(root, seed, "ground", v, "tier_comparison", log_path)
    r_mixed = run_and_log(root, seed, "mixed", v, "tier_comparison", log_path)

    m_g = r_ground["result"]["metrics"]
    m_m = r_mixed["result"]["metrics"]
    gt_arr = np.asarray(r_ground["gt_dict"]["positions"], dtype=np.float64)

    fig, (ax_g, ax_m) = plt.subplots(1, 2, figsize=(14, 6),
                                      subplot_kw={"projection": "3d"})
    draw_3d_overlay(ax_g, r_ground["pred"], gt_arr, cam_pos=r_ground["cam_pos"])
    draw_3d_overlay(ax_m, r_mixed["pred"], gt_arr, cam_pos=r_mixed["cam_pos"])

    ax_g.set_title("All-ground V=%d\nmAP=%.4f  median=%.4f m  count=%+d"
                   % (v, m_g["mAP"], m_g["median_err_m"], m_g["count_err"]),
                   fontsize=9, pad=6)
    ax_m.set_title("Mixed V=%d\nmAP=%.4f  median=%.4f m  count=%+d"
                   % (v, m_m["mAP"], m_m["median_err_m"], m_m["count_err"]),
                   fontsize=9, pad=6)

    # Delta annotation.
    delta_mAP = m_m["mAP"] - m_g["mAP"]
    delta_med = m_m["median_err_m"] - m_g["median_err_m"]
    fig.text(0.5, 0.02,
             "Mixed − ground:  ΔmAP = %+.4f    Δmedian = %+.4f m"
             % (delta_mAP, delta_med),
             ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4edda",
                       edgecolor="#28a745", alpha=0.9))

    fig.suptitle("Tier composition at V=%d: seed %d, %d drones, primary"
                 % (v, seed, m_g["n_true"]),
                 fontsize=13, y=1.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out_path = os.path.join(out_dir, "tier_comparison_seed%d_v%d.png"
                            % (seed, v))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved: %s" % out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate demonstration figures from the frozen geometric "
                    "baseline (ml/demo_figures.py). Outputs to logs/figures/.")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="data root (default %s)" % DEFAULT_ROOT)
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="output directory for figures (default %s)" % OUT_DIR)

    # Seed overrides (defaults produce the figures described in the spec).
    p.add_argument("--hero-seed", type=int, default=DEFAULT_HERO_SEED,
                   help="seed for HERO and VIEW-COUNT (default %d)"
                        % DEFAULT_HERO_SEED)
    p.add_argument("--density-low", type=int, default=DEFAULT_DENSITY_LOW,
                   help="low-density seed for DENSITY PROGRESSION (default %d)"
                        % DEFAULT_DENSITY_LOW)
    p.add_argument("--density-mid", type=int, default=DEFAULT_DENSITY_MID,
                   help="mid-density seed for DENSITY PROGRESSION (default %d)"
                        % DEFAULT_DENSITY_MID)
    p.add_argument("--density-high", type=int, default=DEFAULT_DENSITY_HIGH,
                   help="high-density seed for DENSITY PROGRESSION (default %d)"
                        % DEFAULT_DENSITY_HIGH)
    p.add_argument("--tier-seed", type=int, default=DEFAULT_TIER_SEED,
                   help="high-density seed for TIER COMPARISON (default %d)"
                        % DEFAULT_TIER_SEED)
    p.add_argument("--skip-fig", nargs="*", default=None,
                   help="skip specific figures: hero view_density tier")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    # Clear the log at the start of each run (append within run_and_log).
    if os.path.isfile(LOG_PATH):
        os.remove(LOG_PATH)

    root = os.path.expanduser(args.root)
    skip = set(args.skip_fig or [])
    produced = []

    print("Root: %s" % root)
    print("Output: %s" % args.out_dir)
    print("Log: %s" % LOG_PATH)
    print("")

    # ---- Figure 1: HERO ----
    if "hero" not in skip:
        path = make_hero(args.hero_seed, "mixed", 8, root, LOG_PATH, args.out_dir)
        produced.append(path)
        print("")

    # ---- Figure 2: VIEW-COUNT PROGRESSION ----
    if "view" not in skip:
        path = make_view_progression(
            args.hero_seed, "mixed", [2, 3, 5, 8],
            root, LOG_PATH, args.out_dir)
        produced.append(path)
        print("")

    # ---- Figure 3: DENSITY PROGRESSION ----
    if "density" not in skip:
        density_seeds = [
            {"seed": args.density_low, "label": "low (~10 drones)"},
            {"seed": args.density_mid, "label": "mid (~35 drones)"},
            {"seed": args.density_high, "label": "high (~55 drones)"},
        ]
        path = make_density_progression(
            density_seeds, "mixed", 8, root, LOG_PATH, args.out_dir)
        produced.append(path)
        print("")

    # ---- Figure 4: TIER COMPARISON ----
    if "tier" not in skip:
        path = make_tier_comparison(
            args.tier_seed, 2, root, LOG_PATH, args.out_dir)
        produced.append(path)
        print("")

    # ---- Summary ----
    print("=" * 60)
    print("Produced %d figures:" % len(produced))
    for p in produced:
        print("  %s" % os.path.basename(p))
    print("")
    print("Seeds used:")
    print("  HERO + VIEW-COUNT : seed %d (30 drones, primary)" % args.hero_seed)
    print("  DENSITY low       : seed %d (10 drones)" % args.density_low)
    print("  DENSITY mid       : seed %d (35 drones)" % args.density_mid)
    print("  DENSITY high      : seed %d (55 drones)" % args.density_high)
    print("  TIER comparison   : seed %d (55 drones, primary)" % args.tier_seed)
    print("")
    print("Log: %s" % LOG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
