"""Geometric reconstruction backend: blob detection → epipolar correspondence → DLT triangulation.

Imports from the frozen stage1_geometry pipeline (never modified).
Pure geometry — no torch, no ML model, no Blender.

reconstruct(images, cameras) -> (positions Nx3, confidences None)
"""

from __future__ import annotations

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap — make the frozen stage-1 modules and ml package importable
# from the repo root (this tool lives in a worktree of the same repo).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE1 = os.path.join(_REPO_ROOT, "stage1_geometry")
for _p in (_STAGE1, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Frozen imports (read / call / never edit) ---
from data_contract import (  # noqa: E402
    CameraRig,
    CONVENTION_TAG,
    IMAGE_SIZE,
    Detections,
    blender_c2w_to_opencv_w2c,
)
from detect_blobs import detect_blobs  # noqa: E402
from b3_correspondence import solve_correspondence  # noqa: E402
from b5_triangulation import triangulate_dlt  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (mirror ml/baseline_adapter.py)
# ---------------------------------------------------------------------------
DRONE_SIZE_M = 0.5
EPIPOLAR_THRESHOLD_PX = 3.0
_IMAGE_W, _IMAGE_H = IMAGE_SIZE  # 1920, 1080


def _build_rig(camera_dicts: list[dict]) -> CameraRig:
    """Build a frozen CameraRig from a list of camera dicts.

    Each dict must have: K (3,3), w2c_R (3,3), w2c_t (3,), c2w (4,4), focal_px.
    """
    K_list, R_list, t_list, c2w_list = [], [], [], []
    for c in camera_dicts:
        K_list.append(np.asarray(c["K"], dtype=np.float64))
        c2w = np.asarray(c["c2w"], dtype=np.float64)
        c2w_list.append(c2w)
        # Derive w2c from c2w if not already present in the dict
        if "w2c_R" in c and "w2c_t" in c:
            R_list.append(np.asarray(c["w2c_R"], dtype=np.float64))
            t_list.append(np.asarray(c["w2c_t"], dtype=np.float64))
        else:
            R_w2c, t_w2c = blender_c2w_to_opencv_w2c(c2w)
            R_list.append(R_w2c)
            t_list.append(t_w2c)
    return CameraRig(
        K=np.stack(K_list),
        w2c_R=np.stack(R_list),
        w2c_t=np.stack(t_list),
        c2w=np.stack(c2w_list),
        focal_px=float(camera_dicts[0].get("focal_px", 2666.67)),
        convention=CONVENTION_TAG,
        geometry_class="mixed",
    )


def _detect_views(images: list[np.ndarray], rig: CameraRig,
                  standoff_m: float) -> Detections:
    """Run the frozen blob detector on each image."""
    pts: list[np.ndarray] = []
    for rgb in images:
        dets = detect_blobs(
            rgb=rgb,
            drone_size_m=DRONE_SIZE_M,
            focal_px=rig.focal_px,
            standoff_m=standoff_m,
            image_width_px=_IMAGE_W,
        )
        pts.append(dets.points_per_view[0] if dets.points_per_view
                   else np.empty((0, 2), dtype=np.float64))
    return Detections(points_per_view=pts, image_size=IMAGE_SIZE)


def reconstruct(images: list[np.ndarray],
                cameras: list[dict],
                ) -> tuple[np.ndarray, np.ndarray | None]:
    """Geometric reconstruction via blob detection → epipolar → DLT.

    Args:
        images: list of V uint8 (H, W, 3) RGB arrays.
        cameras: list of V camera dicts with K, w2c_R, w2c_t, c2w, focal_px,
                 standoff_m.

    Returns:
        (positions (N, 3) float64 ENU metres, confidences None).
        confidences is always None for the geometric backend (no per-point
        confidence estimate).
    """
    V = len(images)
    if V < 2:
        return np.empty((0, 3), dtype=np.float64), None

    rig = _build_rig(cameras)
    standoff_m = float(cameras[0].get("standoff_m", 139.0))

    # Detect blobs
    detections = _detect_views(images, rig, standoff_m)

    # Epipolar correspondence
    tracks = solve_correspondence(
        detections=detections,
        rig=rig,
        epipolar_threshold=EPIPOLAR_THRESHOLD_PX,
    )

    # DLT triangulation
    recon = triangulate_dlt(tracks, rig, detections)
    positions = np.asarray(recon.positions_3d, dtype=np.float64)

    # Geometric backend has no per-point confidence; return None.
    return positions, None
