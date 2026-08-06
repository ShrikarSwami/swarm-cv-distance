"""Adjacency matrix computation and export for GA/PSO consumption.

Given reconstructed 3D positions, computes pairwise Euclidean distances and
thresholds at `d_max` (the assumed comms range) to produce an inferred
adjacency matrix: A[i, j] = 1 if distance(positions[i], positions[j]) <= d_max.

The adjacency matrix format is a drop-in replacement for the simulation-derived
matrix the existing GA/PSO critical-node search consumes.
"""

from __future__ import annotations

import json
import numpy as np
from scipy.spatial.distance import cdist


def compute_adjacency(positions: np.ndarray, d_max: float) -> np.ndarray:
    """Compute the adjacency matrix from 3D positions.

    Args:
        positions: (N, 3) float64 — world ENU metres.
        d_max: comms range threshold in metres.

    Returns:
        (N, N) int8 adjacency matrix (0/1, symmetric, zero diagonal).
    """
    positions = np.asarray(positions, dtype=np.float64)
    N = positions.shape[0]
    if N == 0:
        return np.zeros((0, 0), dtype=np.int8)
    D = cdist(positions, positions)  # (N, N)
    A = (D <= d_max).astype(np.int8)
    np.fill_diagonal(A, 0)  # no self-loops
    return A


def adjacency_stats(A: np.ndarray) -> dict:
    """Return summary statistics for an adjacency matrix.

    Returns dict with keys:
        n_nodes, n_edges, edge_density, mean_degree, min_degree, max_degree,
        n_isolated (nodes with degree 0), n_components (approximate via
        connected components).
    """
    N = A.shape[0]
    if N == 0:
        return {"n_nodes": 0, "n_edges": 0, "edge_density": 0.0,
                "mean_degree": 0.0, "min_degree": 0, "max_degree": 0,
                "n_isolated": 0, "n_components": 0}

    degrees = np.sum(A, axis=1)
    n_edges = int(np.sum(A)) // 2  # undirected, count each pair once

    # Connected components via simple BFS
    visited = np.zeros(N, dtype=bool)
    n_components = 0
    for start in range(N):
        if visited[start]:
            continue
        n_components += 1
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

    return {
        "n_nodes": N,
        "n_edges": n_edges,
        "edge_density": float(2 * n_edges / (N * (N - 1))) if N > 1 else 0.0,
        "mean_degree": float(np.mean(degrees)),
        "min_degree": int(np.min(degrees)),
        "max_degree": int(np.max(degrees)),
        "n_isolated": int(np.sum(degrees == 0)),
        "n_components": n_components,
    }


# ---------------------------------------------------------------------------
# GA/PSO-compatible export formats
# ---------------------------------------------------------------------------

def export_adjacency_json(A: np.ndarray, path: str,
                          metadata: dict | None = None) -> str:
    """Export adjacency matrix as JSON for GA/PSO consumption.

    Format:
        {
            "n_nodes": N,
            "d_max": <threshold>,
            "adjacency": [[i, j], ...]   — edge list (undirected, i < j)
            ...metadata...
        }

    This is the GA/PSO track's expected input format (edge list for sparse
    efficiency at scale). Returns the saved path.
    """
    N = A.shape[0]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                edges.append([int(i), int(j)])

    doc = {
        "n_nodes": int(N),
        "n_edges": len(edges),
        "adjacency": edges,
    }
    if metadata:
        doc.update(metadata)

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


def export_adjacency_npy(A: np.ndarray, path: str) -> str:
    """Export adjacency matrix as a dense .npy file (for NumPy consumers)."""
    np.save(path, A)
    return path


def export_adjacency_csv(A: np.ndarray, path: str) -> str:
    """Export adjacency matrix as CSV (N×N, comma-separated, no header)."""
    np.savetxt(path, A, fmt="%d", delimiter=",")
    return path


# ---------------------------------------------------------------------------
# Pairwise distance matrix (for downstream analysis)
# ---------------------------------------------------------------------------

def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Full (N, N) pairwise Euclidean distance matrix."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return cdist(positions, positions)
