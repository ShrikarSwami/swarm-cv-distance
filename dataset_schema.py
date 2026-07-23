"""
Phase 3 dataset schema: loader, saver, and on-demand distance/adjacency.

clip.npz contents:
    K           float64  (n_views, 3, 3)      camera intrinsics
    extrinsics  float64  (n_views, 4, 4)      camera-to-world transforms
    positions   float32  (n_frames, n_drones, 3)  world-space XYZ
    meta        dict     scene config, environment, weather, etc.

Distances and adjacency are NOT stored — computed on demand with tunable D_MAX.
"""

import json
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults (from scene_config.py, kept in sync)
# ---------------------------------------------------------------------------

DEFAULT_D_MAX = 3949.0   # 85% target, provisional for 5km scene
DEFAULT_DRONE_SIZE = 0.5  # meters, assumption not confirmed spec
DEFAULT_AREA_KM = 5.0
DEFAULT_HEIGHT_RANGE_M = 1000.0


# ---------------------------------------------------------------------------
# Save a clip
# ---------------------------------------------------------------------------

def save_clip(path, K, extrinsics, positions, meta):
    """Save a clip to .npz format.

    Args:
        path: str or Path — output .npz file
        K: (n_views, 3, 3) float64 camera intrinsics
        extrinsics: (n_views, 4, 4) float64 camera-to-world
        positions: (n_frames, n_drones, 3) float32 world positions
        meta: dict with environment, weather, seed, etc.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        K=np.asarray(K, dtype=np.float64),
        extrinsics=np.asarray(extrinsics, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float32),
        meta=np.array(meta, dtype=object),
    )


def load_clip(path, d_max=DEFAULT_D_MAX):
    """Load a clip and compute derived quantities on demand.

    Returns dict with:
        K, extrinsics, positions, meta — as stored
        distances — (F, N, N) float64 pairwise distances
        adjacency — (F, N, N) bool thresholded at d_max
    """
    data = np.load(path, allow_pickle=True)
    positions = data["positions"]  # (F, N, 3)
    distances = np.linalg.norm(
        positions[:, :, None, :] - positions[:, None, :, :], axis=-1
    )
    adjacency = distances <= d_max
    return {
        "K": data["K"],
        "extrinsics": data["extrinsics"],
        "positions": positions,
        "meta": data["meta"].item(),
        "distances": distances,
        "adjacency": adjacency,
    }


# ---------------------------------------------------------------------------
# Projection utility
# ---------------------------------------------------------------------------

def project_positions(positions_3d, K, extrinsics):
    """Project 3D world positions to 2D pixel coordinates.

    Args:
        positions_3d: (N, 3) or (F, N, 3) world positions
        K: (3, 3) or (n_views, 3, 3) intrinsics
        extrinsics: (4, 4) or (n_views, 4, 4) camera-to-world

    Returns:
        pixel_coords: same shape but last dim is 2 (u, v)
        depths: same shape but last dim is 1 (z-depth in camera frame)
    """
    single_view = positions_3d.ndim == 2
    if single_view:
        positions_3d = positions_3d[np.newaxis]  # (1, N, 3)
    K = np.asarray(K)
    extrinsics = np.asarray(extrinsics)
    if K.ndim == 2:
        K = K[np.newaxis]  # (1, 3, 3)
    if extrinsics.ndim == 2:
        extrinsics = extrinsics[np.newaxis]  # (1, 4, 4)

    n_views = K.shape[0]
    all_pixels = []
    all_depths = []

    for v in range(n_views):
        # World to camera: inv(extrinsics) @ world_pos
        R = extrinsics[v, :3, :3]
        t = extrinsics[v, :3, 3]
        cam_pos = R.T @ (positions_3d - t).T  # (3, N)
        cam_pos = cam_pos.T  # (N, 3)

        # Project with intrinsics
        projected = (K[v] @ cam_pos.T).T  # (N, 3)
        pixels = projected[:, :2] / projected[:, 2:3]
        depths = projected[:, 2:3]

        all_pixels.append(pixels)
        all_depths.append(depths)

    result_pixels = np.stack(all_pixels, axis=0)  # (V, N, 2)
    result_depths = np.stack(all_depths, axis=0)  # (V, N, 1)

    if single_view:
        return result_pixels[0], result_depths[0]
    return result_pixels, result_depths


# ---------------------------------------------------------------------------
# Dataset directory helpers
# ---------------------------------------------------------------------------

def init_dataset(dataset_root, scene_config=None):
    """Create the dataset directory structure with metadata."""
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "clips").mkdir(exist_ok=True)

    meta = scene_config or {
        "d_max": DEFAULT_D_MAX,
        "drone_size_m": DEFAULT_DRONE_SIZE,
        "area_km": DEFAULT_AREA_KM,
        "height_range_m": DEFAULT_HEIGHT_RANGE_M,
        "n_drones": 20,
    }
    with open(root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    splits = {"train": [], "val": [], "test": []}
    with open(root / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    return root


def add_clip_to_split(dataset_root, clip_name, split="train"):
    """Add a clip to a dataset split."""
    splits_path = Path(dataset_root) / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    if clip_name not in splits[split]:
        splits[split].append(clip_name)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
