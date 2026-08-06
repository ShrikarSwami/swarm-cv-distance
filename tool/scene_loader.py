"""Scene data loader — reads bundled scene directories.

Each scene directory (tool/scenes/NNNNN/) contains:
    - angle_00.png .. angle_23.png  (24 rendered views, 1080×1920 RGB)
    - cameras.json                   (camera rig + per-view K, c2w, etc.)
    - ground_truth.json              (true drone positions for evaluation)

The scene index (tool/scenes/index.json) maps scene IDs to metadata.
"""

from __future__ import annotations

import json
import os
import numpy as np

_SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")

# Cached scene index
_index: dict | None = None


def _load_index() -> dict:
    """Load scene index, cached on first call."""
    global _index
    if _index is None:
        idx_path = os.path.join(_SCENES_DIR, "index.json")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError("scene index not found: %s" % idx_path)
        with open(idx_path) as f:
            _index = json.load(f)
    return _index


def list_scenes(split: str | None = None) -> list[dict]:
    """Return metadata for all bundled scenes, optionally filtered by split."""
    idx = _load_index()
    scenes = []
    for sid, meta in sorted(idx.items()):
        if split and meta.get("split") != split:
            continue
        scenes.append({"id": sid, **meta})
    return scenes


def get_scene_dir(scene_id: str) -> str:
    """Return the absolute path to a scene directory."""
    d = os.path.join(_SCENES_DIR, scene_id)
    if not os.path.isdir(d):
        raise FileNotFoundError("scene %s not found in %s" % (scene_id, _SCENES_DIR))
    return d


def load_images(scene_id: str, view_indices: list[int]) -> list[np.ndarray]:
    """Load specified view images from a scene as uint8 (H, W, 3) arrays.

    Args:
        scene_id: e.g. "00000"
        view_indices: list of angle indices 0..23

    Returns:
        list of uint8 (1080, 1920, 3) numpy arrays.
    """
    from PIL import Image

    scene_dir = get_scene_dir(scene_id)
    images = []
    for v in view_indices:
        path = os.path.join(scene_dir, "angle_%02d.png" % v)
        if not os.path.isfile(path):
            raise FileNotFoundError("image not found: %s" % path)
        images.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    return images


def load_cameras(scene_id: str, view_indices: list[int]) -> list[dict]:
    """Load camera parameters for the specified views.

    Each dict has: K, w2c_R, w2c_t, c2w, focal_px, standoff_m, angle_idx,
    tier, elevation_deg, azimuth_deg.

    Args:
        scene_id: e.g. "00000"
        view_indices: list of angle indices 0..23

    Returns:
        list of camera dicts for the requested views.
    """
    scene_dir = get_scene_dir(scene_id)
    cam_path = os.path.join(scene_dir, "cameras.json")
    with open(cam_path) as f:
        cam_data = json.load(f)

    cameras = []
    for v in view_indices:
        vdata = cam_data["views"][v]
        cameras.append({
            "K": np.asarray(vdata["K"], dtype=np.float64),
            "c2w": np.asarray(vdata["c2w"], dtype=np.float64),
            "w2c_R": np.asarray(vdata.get("w2c_R",
                _derive_w2c(np.asarray(vdata["c2w"], dtype=np.float64))[0]),
            dtype=np.float64),
            "w2c_t": np.asarray(vdata.get("w2c_t",
                _derive_w2c(np.asarray(vdata["c2w"], dtype=np.float64))[1]),
            dtype=np.float64),
            "focal_px": float(cam_data.get("focal_px", 2666.67)),
            "standoff_m": float(cam_data.get("standoff_m", 139.0)),
            "angle_idx": int(vdata["angle_idx"]),
            "tier": vdata.get("tier", "?"),
            "elevation_deg": float(vdata.get("elevation_deg", 0)),
            "azimuth_deg": float(vdata.get("azimuth_deg", 0)),
        })
    return cameras


def load_ground_truth(scene_id: str) -> tuple[np.ndarray, int]:
    """Load ground-truth drone positions.

    Returns:
        (positions (N, 3) float64 ENU metres, n_drones).
    """
    scene_dir = get_scene_dir(scene_id)
    gt_path = os.path.join(scene_dir, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)
    return np.asarray(gt["positions"], dtype=np.float64), int(gt["n_drones"])


def _derive_w2c(c2w: np.ndarray):
    """Derive w2c_R, w2c_t from a c2w 4x4 matrix (Blender convention)."""
    from data_contract import blender_c2w_to_opencv_w2c
    return blender_c2w_to_opencv_w2c(c2w)


def camera_positions(cameras: list[dict]) -> np.ndarray:
    """Extract (V, 3) world ENU camera positions from camera dicts (c2w[:3, 3])."""
    pos = []
    for c in cameras:
        c2w = c["c2w"]
        pos.append(c2w[:3, 3])
    return np.asarray(pos, dtype=np.float64)
