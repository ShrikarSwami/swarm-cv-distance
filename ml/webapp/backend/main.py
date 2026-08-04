"""FastAPI application for the web frontend reconstruction UI.

This wraps the existing ml/recon_app.py geometric reconstruction pipeline
as a REST API for browser-based interaction. No metric reimplementation
in JavaScript - all computation happens on the backend.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import json
import numpy as np

# Add the repo root to sys.path to import stage1_geometry modules
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STAGE1 = os.path.join(REPO_ROOT, "stage1_geometry")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, STAGE1)

# Import from ml.recon_app
from ml.recon_app import (
    load_manifest, find_scene, choose_angles, camera_positions,
    selected_view_angles, _load_scene_data, _overlay_pred_positions,
    process_scene, scene_metadata_line, N_VIEWS_TOTAL, DEFAULT_ROOT,
    DEFAULT_RECALL_RADIUS_PX
)

from ml.metrics import DEFAULT_TAUS

# Import models
from .models import (
    SceneInfo, SceneMetadata, ReconstructionRequest, ReconstructionResult,
    DronePrediction, Point3D, MetricDict, AngleMode
)
from .scene_loader import load_scenes_from_manifest, get_scene_info

# Initialize FastAPI app
app = FastAPI(
    title="Swarm CV Distance - Reconstruction API",
    description="Geometric reconstruction pipeline wrapper for browser-based visualization",
    version="1.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API root endpoint with basic info."""
    return {
        "name": "Swarm CV Distance - Reconstruction API",
        "version": "1.0.0",
        "description": "Geometric reconstruction pipeline for drone swarm analysis",
        "endpoints": [
            "/api/scenes - List available scenes",
            "/api/scene/{seed} - Get scene details",
            "/api/reconstruct - Run reconstruction",
            "/api/cameras/{seed} - Get camera poses"
        ]
    }


@app.get("/api/scenes")
async def list_scenes(
    split: str = None,
    cell: str = None,
    limit: int = 40
) -> List[SceneMetadata]:
    """List available scenes from the manifest.

    Args:
        split: Filter by split (test/train/val)
        cell: Filter by cell (primary/secondary)
        limit: Maximum number of scenes to return

    Returns:
        List of scene metadata
    """
    try:
        scenes = load_scenes_from_manifest(
            root=DEFAULT_ROOT,
            split=split,
            cell=cell,
            limit=limit
        )

        # Convert to SceneMetadata objects
        result = []
        for scene in scenes:
            metadata = SceneMetadata(
                seed=int(scene["seed"]),
                split=scene.get("split", "?"),
                cell=scene.get("cell", "?"),
                n_drones=int(scene.get("n_drones", 0)),
                n_views=int(scene.get("n_views", N_VIEWS_TOTAL)),
                a_max=float(scene.get("a_max", 0.0)),
                description=scene.get("description", None)
            )
            result.append(metadata)

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading scenes: {str(e)}")


@app.get("/api/scene/{seed}")
async def get_scene_details(seed: int) -> SceneInfo:
    """Get complete scene information for a seed.

    Args:
        seed: Scene seed number

    Returns:
        Complete scene information including metadata, cameras, etc.
    """
    try:
        scene_info = get_scene_info(seed, DEFAULT_ROOT)

        # Convert to SceneInfo object
        cameras = []
        for cam in scene_info["cameras"]:
            cameras.append(SceneInfo.CameraView(
                angle_idx=cam["angle_idx"],
                tier=cam["tier"],
                elevation_deg=float(cam["elevation_deg"]),
                azimuth_deg=float(cam["azimuth_deg"])
            ))

        metadata = SceneInfo.SceneMetadata(
            seed=scene_info["seed"],
            split=scene_info["metadata"].get("split", "?"),
            cell=scene_info["metadata"].get("cell", "?"),
            n_drones=scene_info["n_drones"],
            n_views=scene_info["metadata"].get("n_views", N_VIEWS_TOTAL),
            a_max=float(scene_info["a_max"])
        )

        scene = SceneInfo(
            seed=scene_info["seed"],
            metadata=metadata,
            cameras=cameras
        )

        return scene

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scene details: {str(e)}")


@app.get("/api/cameras/{seed}")
async def get_scene_cameras(seed: int):
    """Get camera poses for a scene.

    Args:
        seed: Scene seed number

    Returns:
        Camera poses and metadata for all 24 views
    """
    try:
        scene_info = get_scene_info(seed, DEFAULT_ROOT)

        # Return cameras with positions
        cameras_data = []
        for i, cam in enumerate(scene_info["cameras"]):
            camera_data = {
                "angle_idx": cam["angle_idx"],
                "tier": cam["tier"],
                "elevation_deg": cam["elevation_deg"],
                "azimuth_deg": cam["azimuth_deg"],
                "position": {
                    "x": float(scene_info["camera_positions"][i][0]),
                    "y": float(scene_info["camera_positions"][i][1]),
                    "z": float(scene_info["camera_positions"][i][2])
                }
            }
            cameras_data.append(camera_data)

        return {
            "seed": seed,
            "cameras": cameras_data,
            "tier_composition": scene_info["tier_composition"]
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cameras: {str(e)}")


@app.post("/api/reconstruct")
async def reconstruct_scene(request: ReconstructionRequest):
    """Run reconstruction for a scene with selected views.

    Args:
        request: Reconstruction request with scene seed and angle selection

    Returns:
        Reconstruction results including metrics and 3D positions
    """
    try:
        # Validate scene exists
        scenes = load_manifest(DEFAULT_ROOT)
        scene = find_scene(scenes, request.scene_seed)

        # Load scene data
        gt, cam = _load_scene_data(DEFAULT_ROOT, request.scene_seed)
        true = np.asarray(gt["positions"], dtype=np.float64)

        # Determine view indices based on mode
        import random
        rng = random.Random(0)  # Deterministic for testing

        if request.angle_selection.mode == AngleMode.ALL:
            view_idxs = list(range(N_VIEWS_TOTAL))
        elif request.angle_selection.mode == AngleMode.EXACT:
            view_idxs = request.angle_selection.exact_indices or []
        elif request.angle_selection.mode == AngleMode.RANDOM_N:
            n = request.angle_selection.n or 6
            view_idxs = sorted(rng.sample(range(N_VIEWS_TOTAL), n))
        elif request.angle_selection.mode == AngleMode.RANDOM_RANDOM:
            maxv = request.angle_selection.max_views or 8
            k = rng.randint(1, maxv)
            view_idxs = sorted(rng.sample(range(N_VIEWS_TOTAL), k))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown angle mode: {request.angle_selection.mode}")

        # Run reconstruction using the frozen pipeline
        result = process_scene(
            DEFAULT_ROOT,
            request.scene_seed,
            view_idxs,
            request.recall_radius_px
        )
        result["recall_radius_px"] = request.recall_radius_px

        # Get predicted positions for overlay
        pred_positions = _overlay_pred_positions(
            DEFAULT_ROOT,
            request.scene_seed,
            view_idxs,
            cam=cam
        )

        # Get camera positions
        cam_pos = camera_positions(cam, view_idxs)

        # Compute tier composition
        cameras_meta = selected_view_angles(cam, view_idxs)
        tier_counts = {"ground": 0, "level": 0, "aerial": 0}
        for cam_meta in cameras_meta:
            tier = cam_meta["tier"]
            if tier in tier_counts:
                tier_counts[tier] += 1

        # Match predictions to truths for 3D visualization
        from ml.recon_app import _match_pred_to_true
        rows, cols, ghost_idx, missed_idx = _match_pred_to_true(pred_positions, true)

        # Prepare results
        true_positions = [Point3D.from_array(p) for p in true]
        pred_positions_result = [Point3D.from_array(p) for p in pred_positions]

        # Build matched pairs
        matched_pairs = list(zip(rows, cols))

        # Convert metrics to MetricDict format
        metric_dict = MetricDict(
            n_true=result["metrics"]["n_true"],
            n_pred=result["metrics"]["n_pred"],
            per_tau=result["metrics"]["per_tau"],
            mAP=result["metrics"]["mAP"],
            median_err_m=result["metrics"]["median_err_m"],
            chamfer_m=result["metrics"]["chamfer_m"],
            count_err=result["metrics"]["count_err"]
        )

        # Build reconstruction result
        reconstruction_result = ReconstructionResult(
            scene_seed=request.scene_seed,
            metrics=metric_dict,
            true_positions=true_positions,
            pred_positions=pred_positions_result,
            matched_pairs=matched_pairs,
            ghost_indices=list(ghost_idx),
            missed_indices=list(missed_idx),
            camera_positions=[Point3D.from_array(p) for p in cam_pos],
            view_indices=view_idxs,
            wall_clock_s=result.get("wall_clock_s", 0.0),
            tier_composition=tier_counts,
            detector_recall=result.get("detector_recall"),
            adjacency_f1=None
        )

        return reconstruction_result.dict()

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reconstruction: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if ~/swarm_ml exists
        if not os.path.exists(DEFAULT_ROOT):
            return {
                "status": "error",
                "message": f"Data root not found: {DEFAULT_ROOT}"
            }

        # Check if manifest exists
        manifest_path = os.path.join(DEFAULT_ROOT, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            return {
                "status": "error",
                "message": f"Manifest not found: {manifest_path}"
            }

        # Try to load first few scenes
        scenes = load_manifest(DEFAULT_ROOT)
        return {
            "status": "ok",
            "message": "API is healthy",
            "scenes_count": len(scenes),
            "data_root": str(DEFAULT_ROOT)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)