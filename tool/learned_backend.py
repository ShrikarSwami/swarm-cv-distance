"""Learned reconstruction backend: T6 voxel-fusion model.

STATUS (2026-08-06): The learned model currently fails the G2 acceptance gate
(count error > 1 on all scenes). It is included as a stub so the pluggable
interface is wired from day one, but it will return empty results until the
model passes G2.

See MODEL_DETAILS.md and the main repo's ml/FIX_QUEUE.md for current status.

reconstruct(images, cameras) -> (positions Nx3, confidences N)
"""

from __future__ import annotations

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def reconstruct(images: list[np.ndarray],
                cameras: list[dict],
                ) -> tuple[np.ndarray, np.ndarray | None]:
    """Learned-model reconstruction (T6 voxel-fusion).

    Currently returns empty results — the model has not yet passed G2.
    See ml/FIX_QUEUE.md and MODEL_DETAILS.md.

    When the model passes G2, wire the trained checkpoint here:
        1. Load VoxelFusionModel from ml.model
        2. Load checkpoint weights
        3. Run forward_volume -> extract_positions
        4. Return (positions, confidences) where confidences come from
           the peak heatmap values.

    Args:
        images: list of V uint8 (H, W, 3) RGB arrays.
        cameras: list of V camera dicts.

    Returns:
        (positions, confidences) — currently (empty, None).
    """
    # Stub: model not yet passing G2.
    # When ready, uncomment and wire:
    #
    #   import torch
    #   from ml.model import VoxelFusionModel, extract_positions
    #
    #   CKPT = os.path.join(_REPO_ROOT, "checkpoints", "best.pt")
    #   model = VoxelFusionModel(feat_channels=64)
    #   ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    #   model.load_state_dict(ckpt["model"])
    #   model.eval()
    #
    #   grid = {
    #       "center": ...,  # swarm centre from cameras
    #       "radius_m": ...,
    #   }
    #   with torch.no_grad():
    #       vol = model.forward_volume(images, cameras, grid)[0, 0].cpu().numpy()
    #   positions = extract_positions(vol, grid)
    #   # confidences from peak voxel values
    #   ...
    #
    # See ml/recon_app.py and ml/model.py for the full implementation.

    return np.empty((0, 3), dtype=np.float64), None
