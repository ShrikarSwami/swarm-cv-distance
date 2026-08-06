"""3D visualisation: ground truth vs. reconstruction overlay, adjacency graph.

Uses matplotlib (Agg backend — no display required).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers "3d"


def plot_reconstruction_overlay(
    pred: np.ndarray,
    true: np.ndarray,
    cam_pos: np.ndarray | None,
    view_idxs: list[int],
    out_path: str,
    title: str = "Reconstruction",
    metrics: dict | None = None,
) -> str:
    """Save a 3D overlay PNG.

    - Ground truth: green spheres
    - Predictions: blue spheres
    - Hungarian-matched error vectors: red lines
    - Ghost predictions (unmatched): orange ×
    - Missed drones (unmatched truth): hollow gray circles
    - Camera positions: gray triangles

    Returns the saved path.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    rows, cols, ghost_idx, missed_idx = _match_pred_to_true(pred, true)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Ground truth (green)
    if true.shape[0]:
        ax.scatter(true[:, 0], true[:, 1], true[:, 2], c="#1db954", s=42,
                   depthshade=False, label="ground truth (%d)" % true.shape[0])

    # Predictions (blue)
    if pred.shape[0]:
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="#4488ff", s=30,
                   depthshade=False, label="predicted (%d)" % pred.shape[0])

    # Error vectors between matched pairs (red lines)
    for r, c in zip(rows, cols):
        ax.plot([true[c, 0], pred[r, 0]], [true[c, 1], pred[r, 1]],
                [true[c, 2], pred[r, 2]], c="#d62728", lw=1.0, alpha=0.8)
    if len(rows):
        ax.plot([], [], [], c="#d62728", lw=1.0,
                label="error vectors (%d)" % len(rows))

    # Ghosts: unmatched predictions (orange x)
    if len(ghost_idx):
        g = pred[ghost_idx]
        ax.scatter(g[:, 0], g[:, 1], g[:, 2], c="#ff7f0e", marker="x", s=80,
                   depthshade=False, label="ghosts (%d)" % len(ghost_idx))

    # Missed: unmatched ground truth (hollow gray)
    if len(missed_idx):
        m = true[missed_idx]
        ax.scatter(m[:, 0], m[:, 1], m[:, 2], facecolors="none",
                   edgecolors="#666666", s=110, depthshade=False,
                   label="missed (%d)" % len(missed_idx))

    # Camera positions (gray triangles)
    if cam_pos is not None and cam_pos.shape[0]:
        ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], c="#888888",
                   marker="^", s=60, depthshade=False,
                   label="cameras (%d)" % cam_pos.shape[0])

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title(title)

    # Square-ish aspect from data bounds
    all_pts = []
    for p in (true, pred):
        if p.shape[0]:
            all_pts.append(p)
    if cam_pos is not None and cam_pos.shape[0]:
        all_pts.append(cam_pos)
    if all_pts:
        pts = np.vstack(all_pts)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))

    ax.legend(loc="upper left", fontsize=9)

    # Metrics annotation
    if metrics:
        lines = [
            "Metrics (ml.metrics.evaluate, frozen):",
            "  n_true=%d  n_pred=%d  count_err=%+d"
            % (metrics.get("n_true", 0), metrics.get("n_pred", 0),
               metrics.get("count_err", 0)),
            "  median_err_m=%.4f  chamfer_m=%.4f  mAP=%.4f"
            % (metrics.get("median_err_m", float("nan")),
               metrics.get("chamfer_m", float("nan")),
               metrics.get("mAP", float("nan"))),
        ]
        note = "\n".join(lines)
        fig.text(0.02, 0.02, note, fontsize=8, color="#333333",
                 family="monospace")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_adjacency_graph(A: np.ndarray, positions: np.ndarray,
                         out_path: str, title: str = "Adjacency Graph (d_max)") -> str:
    """Save a 2D top-down view of the adjacency graph.

    Nodes at (East, North) positions, edges drawn between connected pairs.
    Isolated nodes are highlighted.

    Returns the saved path.
    """
    A = np.asarray(A, dtype=np.int8)
    positions = np.asarray(positions, dtype=np.float64)
    N = positions.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    if N == 0:
        ax.set_title(title + " (empty)")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path

    degrees = np.sum(A, axis=1)

    # Draw edges
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                ax.plot([positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        c="#aaaaaa", lw=0.5, alpha=0.6, zorder=1)

    # Draw nodes, colouring isolated ones differently
    isolated = degrees == 0
    if np.any(~isolated):
        ax.scatter(positions[~isolated, 0], positions[~isolated, 1],
                   c="#4488ff", s=30, zorder=2,
                   label="connected (%d)" % np.sum(~isolated))
    if np.any(isolated):
        ax.scatter(positions[isolated, 0], positions[isolated, 1],
                   c="#ff4444", s=30, marker="x", zorder=2,
                   label="isolated (%d)" % np.sum(isolated))

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Hungarian matching (visualisation only — metrics come from ml.metrics.evaluate)
# ---------------------------------------------------------------------------

def _match_pred_to_true(pred: np.ndarray, true: np.ndarray):
    """Unrestricted Hungarian match for visualisation.

    Returns (rows, cols, ghost_idx, missed_idx).
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    finite = np.isfinite(pred).all(axis=1)
    pred = pred[finite]
    K, N = pred.shape[0], true.shape[0]
    if K == 0 or N == 0:
        return (np.empty((0,), dtype=int), np.empty((0,), dtype=int),
                np.arange(K), np.arange(N))
    D = cdist(pred, true)
    rows, cols = linear_sum_assignment(D)
    pred_used = set(int(r) for r in rows)
    true_used = set(int(c) for c in cols)
    ghost = np.array([i for i in range(K) if i not in pred_used], dtype=int)
    missed = np.array([i for i in range(N) if i not in true_used], dtype=int)
    return rows, cols, ghost, missed
