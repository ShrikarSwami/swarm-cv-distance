"""Export module: adjacency matrix, positions, metrics for GA/PSO consumption.

The GA/PSO critical-node search in the drone-swarm-splitting repo expects an
adjacency matrix as a JSON edge list:
    {
        "n_nodes": N,
        "adjacency": [[i, j], ...]
    }

This module produces exactly that format, plus optional position and metric
dumps for verification.
"""

from __future__ import annotations

import json
import os
import numpy as np


def export_results(
    positions: np.ndarray,
    adjacency: np.ndarray,
    true_positions: np.ndarray | None,
    metrics: dict,
    out_dir: str,
    scene_id: str,
    d_max: float,
    backend: str,
) -> dict[str, str]:
    """Export all results to a directory.

    Produces:
        adjacency.json     — GA/PSO-consumable edge list
        adjacency.npy      — dense adjacency matrix (NumPy)
        adjacency.csv      — dense adjacency matrix (CSV)
        positions.csv      — reconstructed positions (x,y,z)
        ground_truth.csv   — ground truth positions (if available)
        metrics.json       — full metric dict
        summary.txt        — human-readable summary

    Returns dict of filename -> path.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved: dict[str, str] = {}

    # Adjacency matrix — edge list for GA/PSO
    edges = []
    N = adjacency.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j]:
                edges.append([int(i), int(j)])

    adj_doc = {
        "n_nodes": int(N),
        "n_edges": len(edges),
        "d_max_m": float(d_max),
        "adjacency": edges,
    }
    path = os.path.join(out_dir, "adjacency.json")
    with open(path, "w") as f:
        json.dump(adj_doc, f, indent=2)
    saved["adjacency_json"] = path

    # Dense adjacency (npy)
    path = os.path.join(out_dir, "adjacency.npy")
    np.save(path, adjacency)
    saved["adjacency_npy"] = path

    # Positions CSV
    path = os.path.join(out_dir, "positions.csv")
    np.savetxt(path, positions, fmt="%.6f", delimiter=",",
               header="east_m,north_m,up_m", comments="")
    saved["positions_csv"] = path

    # Ground truth CSV
    if true_positions is not None and true_positions.shape[0]:
        path = os.path.join(out_dir, "ground_truth.csv")
        np.savetxt(path, true_positions, fmt="%.6f", delimiter=",",
                   header="east_m,north_m,up_m", comments="")
        saved["ground_truth_csv"] = path

    # Metrics JSON
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    saved["metrics_json"] = path

    # Human-readable summary
    lines = [
        "Swarm Reconstruction Results",
        "============================",
        "",
        "Scene:        %s" % scene_id,
        "Backend:      %s" % backend,
        "d_max:        %.2f m" % d_max,
        "",
        "Reconstruction:",
        "  n_reconstructed: %d" % N,
        "  n_ground_truth:  %d" % (true_positions.shape[0] if true_positions is not None else 0),
        "",
        "Metrics:",
        "  n_true:      %d" % metrics.get("n_true", 0),
        "  n_pred:      %d" % metrics.get("n_pred", 0),
        "  count_err:   %+d" % metrics.get("count_err", 0),
        "  median_err:  %.4f m" % metrics.get("median_err_m", float("nan")),
        "  chamfer:     %.4f m" % metrics.get("chamfer_m", float("nan")),
        "  mAP:         %.4f" % metrics.get("mAP", float("nan")),
        "",
        "Adjacency (d_max=%.2f m):" % d_max,
        "  n_nodes:     %d" % N,
        "  n_edges:     %d" % len(edges),
    ]
    if N > 1:
        pct = 100.0 * 2 * len(edges) / (N * (N - 1))
        lines.append("  density:     %.2f %%" % pct)
    lines.append("")
    lines.append("GA/PSO input: adjacency.json")
    lines.append("  -> drop-in replacement for simulation-derived adjacency matrix")
    lines.append("  -> format: {\"n_nodes\": N, \"adjacency\": [[i,j], ...]}")

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    saved["summary_txt"] = path

    return saved


def _json_default(o):
    """Fallback for non-JSON-serializable types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    raise TypeError("not JSON serializable: %r" % (o,))
