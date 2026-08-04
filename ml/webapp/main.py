"""Web wrapper exposing ml/recon_app.py as a REST API (no reimplementation).

All reconstruction, matching, and metrics math is imported and called from the
frozen modules (ml/recon_app.py, ml/baseline_adapter.py, ml/metrics.py) —
nothing is rewritten here. Previously this file inlined copies of
`_scene_dir`, `_load_scene_data`, `camera_positions`, `selected_view_angles`,
and `_overlay_pred_positions`; those have been deleted in favour of the
frozen implementations.
"""

from fastapi import FastAPI
import numpy as np
import os
import json

from ml.recon_app import (
    process_scene,
    _load_scene_data,
    _overlay_pred_positions,
    camera_positions,
    selected_view_angles,
)

app = FastAPI()

# Load manifest
MANIFEST_PATH = os.path.join(os.path.expanduser("~"), "swarm_ml", "manifest.jsonl")
with open(MANIFEST_PATH, "r") as f:
    SCENES = [json.loads(line) for line in f if line.strip()]


@app.get("/scenes")
async def list_scenes():
    return SCENES


@app.get("/angles")
async def list_angles():
    # Example: return angles for a scene (simplified)
    return {"angles": list(range(24))}


@app.post("/reconstruct")
async def reconstruct(scene_seed: int, view_idxs: list[int], recall_radius_px: float = 5.0):
    result = process_scene(os.path.expanduser("~"), scene_seed, view_idxs, recall_radius_px)
    # Replay pipeline to get 3D positions for the overlay (as in recon_app.py)
    gt, cam = _load_scene_data(os.path.expanduser("~"), scene_seed)
    true = np.asarray(gt["positions"], dtype=np.float64)
    pred_positions = _overlay_pred_positions(os.path.expanduser("~"), scene_seed, view_idxs, cam=cam)
    cam_pos = camera_positions(cam, view_idxs)
    # Prepare response
    response = {
        "metrics": result["metrics"],
        "true_positions": true.tolist(),
        "pred_positions": pred_positions.tolist(),
        "camera_positions": cam_pos.tolist(),
        "view_angles": selected_view_angles(cam, view_idxs)
    }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
