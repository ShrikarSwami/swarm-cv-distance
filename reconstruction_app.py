#!/usr/bin/env python3
"""
Reconstruction App — Section 5 of the Stage 1 geometry pipeline.

Full-stack FastAPI backend + Three.js single-page frontend.
Run: python reconstruction_app.py [<bundle_path>]

Starts FastAPI on localhost:8820. Opens browser to the Three.js single-page app.
Without <bundle_path>, the page shows an upload drop zone.

Endpoints:
  GET  /              — Single HTML page (Three.js from CDN)
  POST /api/upload    — Upload a .zip bundle
  POST /api/run       — Run the reconstruction pipeline
  POST /api/export    — Export result debug JSON
  GET  /api/status    — Return current preloaded / uploaded bundle info
"""

# ============================================================================
# Path setup — stage1_geometry modules use bare imports (no package prefix)
# ============================================================================
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), 'stage1_geometry'))

# ============================================================================
# Imports
# ============================================================================
import io
import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from data_contract import (
    SwarmTruth,
    CameraRig,
    Detections,
    Tracks,
    Reconstruction,
    DEFAULT_FOCAL_PX,
    IMAGE_SIZE,
    make_K,
    CONVENTION_TAG,
    project_point,
    blender_c2w_to_opencv_w2c,
)
from detect_blobs import detect_blobs, apparent_px, read_object_index_exr
from b3_correspondence import solve_correspondence
from b4_scoring import score_full, associate_tracks_to_truth
from b5_triangulation import triangulate_dlt as b5_triangulate_dlt
from bundle_schema import BundleManifest, BundlePoses, BundleGroundTruth

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def bundle_poses_to_rig(
    poses: BundlePoses,
    manifest: BundleManifest,
    view_indices: Optional[list[int]] = None,
) -> CameraRig:
    """Convert BundlePoses (filtered to selected views) into a CameraRig.

    The returned rig has views in ascending view_idx order (re-indexed 0..V-1).
    """
    views = sorted(poses.views, key=lambda v: v.view_idx)
    if view_indices is not None and len(view_indices) > 0:
        view_set = set(view_indices)
        views = [v for v in views if v.view_idx in view_set]

    if len(views) == 0:
        raise ValueError("No views selected — cannot build CameraRig")

    V = len(views)
    K = np.array([v.K for v in views], dtype=np.float64)

    w2c_R_list: list[np.ndarray] = []
    w2c_t_list: list[np.ndarray] = []
    c2w_list: list[np.ndarray] = []

    flip = np.diag([1.0, -1.0, -1.0, 1.0])

    for v in views:
        c2w_bl = np.array(v.c2w, dtype=np.float64)
        c2w_list.append(c2w_bl)
        # Blender c2w -> OpenCV c2w -> invert -> OpenCV w2c
        M_cv = c2w_bl @ flip
        w2c = np.linalg.inv(M_cv)
        w2c_R_list.append(w2c[:3, :3])
        w2c_t_list.append(w2c[:3, 3])

    w2c_R = np.array(w2c_R_list, dtype=np.float64)
    w2c_t = np.array(w2c_t_list, dtype=np.float64)
    c2w = np.array(c2w_list, dtype=np.float64)

    gen = manifest.generated_by or {}
    geometry_class = gen.get("geometry_class", "all_ground")

    rig = CameraRig(
        K=K,
        w2c_R=w2c_R,
        w2c_t=w2c_t,
        c2w=c2w,
        focal_px=manifest.focal_px,
        convention=CONVENTION_TAG,
        geometry_class=geometry_class,
    )
    return rig


def load_ground_truth(bundle_path: str, manifest: BundleManifest) -> Optional[SwarmTruth]:
    """Load ground-truth SwarmTruth from ground_truth.json (if present)."""
    if not manifest.has_ground_truth:
        return None
    gt_path = _os.path.join(bundle_path, "ground_truth.json")
    if not _os.path.exists(gt_path):
        return None
    try:
        gt = BundleGroundTruth.validate_file(gt_path)
        positions = np.array(gt.positions, dtype=np.float64)
        drone_ids = np.array(gt.drone_ids, dtype=np.int32)
        return SwarmTruth(positions=positions, drone_ids=drone_ids)
    except Exception as exc:
        log.warning("Failed to load ground truth: %s", exc)
        return None


def _read_view_frame(bundle_path: str, view_idx: int, frame_idx: int = 0) -> np.ndarray:
    """Read a single view frame as an (H, W, 3) uint8 RGB array."""
    candidates = [
        _os.path.join(bundle_path, "views", f"cam_{view_idx:04d}", f"frame_{frame_idx:04d}.png"),
        _os.path.join(bundle_path, "views", f"cam_{view_idx:04d}", f"frame_{frame_idx:04d}.jpg"),
        _os.path.join(bundle_path, "views", f"cam_{view_idx:02d}", f"frame_{frame_idx:04d}.png"),
        _os.path.join(bundle_path, "views", f"cam_{view_idx}", f"frame_{frame_idx:04d}.png"),
    ]
    for path in candidates:
        if _os.path.exists(path):
            with Image.open(path) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
    raise FileNotFoundError(
        f"Frame not found for view {view_idx}, frame {frame_idx} (tried {candidates[0]})"
    )


def detect_from_bundle_views(
    bundle_path: str,
    manifest: BundleManifest,
    view_indices: list[int],
    frame_idx: int = 0,
) -> Detections:
    """Run the pixel detector on every selected view and return Detections.

    ``points_per_view`` matches the order of ``view_indices`` (sorted).
    """
    # Sort view indices for deterministic ordering
    sorted_idxs = sorted(view_indices)

    gen = manifest.generated_by or {}
    drone_size_m = gen.get("drone_size_m", 0.5)
    standoff_m = gen.get("standoff_m", 2000.0)
    focal_px = manifest.focal_px
    image_w = manifest.image_size_px[0]

    points_per_view: list[np.ndarray] = []
    for idx in sorted_idxs:
        try:
            rgb = _read_view_frame(bundle_path, idx, frame_idx)
            dets = detect_blobs(
                rgb=rgb,
                drone_size_m=drone_size_m,
                focal_px=focal_px,
                standoff_m=standoff_m,
                image_width_px=image_w,
            )
            points_per_view.append(dets.points_per_view[0])
        except FileNotFoundError:
            log.warning("View %d: frame not found, using empty detections", idx)
            points_per_view.append(np.empty((0, 2), dtype=np.float64))

    return Detections(
        points_per_view=points_per_view,
        image_size=tuple(manifest.image_size_px),
    )


def compute_detection_quality(
    bundle_path: str,
    view_indices: list[int],
    detections: Detections,
    frame_idx: int = 0,
) -> dict:
    """Compare pixel detections against Object Index EXR passes (if available).

    Returns a dict keyed by ``view_N`` or an empty dict when no EXR files exist.
    """
    obj_dir = _os.path.join(bundle_path, "object_index")
    if not _os.path.isdir(obj_dir):
        return {}

    sorted_idxs = sorted(view_indices)
    results: dict = {}

    for vi, view_idx in enumerate(sorted_idxs):
        exr_path = _os.path.join(obj_dir, f"cam_{view_idx:04d}_id_.exr")
        if not _os.path.exists(exr_path):
            continue

        id_pass = read_object_index_exr(exr_path)
        if not id_pass:
            continue

        id_centroids = np.array([[cx, cy] for cx, cy, _ in id_pass], dtype=np.float64)
        n_id = len(id_centroids)
        det_pts = detections.points_per_view[vi] if vi < len(detections.points_per_view) else np.empty((0, 2))
        n_det = len(det_pts)

        if n_det == 0 or n_id == 0:
            results[f"view_{view_idx}"] = {
                "detector_recall": 0.0,
                "fp": n_det,
                "centroid_error_px": 0.0,
                "merged_detections": 0,
            }
            continue

        # Greedy nearest-neighbour matching with 10 px threshold
        matched = 0
        centroid_error_sum = 0.0
        used_dets: set[int] = set()

        for ic in range(n_id):
            best_dist = float("inf")
            best_det = -1
            for idet in range(n_det):
                if idet in used_dets:
                    continue
                d = float(np.linalg.norm(det_pts[idet] - id_centroids[ic]))
                if d < best_dist and d < 10.0:
                    best_dist = d
                    best_det = idet
            if best_det >= 0:
                matched += 1
                centroid_error_sum += best_dist
                used_dets.add(best_det)

        recall = matched / n_id if n_id > 0 else 0.0
        centroid_err = centroid_error_sum / matched if matched > 0 else 0.0
        fp = n_det - matched
        merged = max(0, n_det - n_id)

        results[f"view_{view_idx}"] = {
            "detector_recall": recall,
            "fp": fp,
            "centroid_error_px": centroid_err,
            "merged_detections": merged,
        }

    return results


def _to_serializable(obj):
    """Convert numpy types to built-in Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Swarm CV Distance — Reconstruction Viewer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory stores -------------------------------------------------------
bundles: dict[str, dict] = {}          # bundle_id -> bundle info
results_store: dict[str, dict] = {}    # result_id -> full result data
latest_bundle_id: Optional[str] = None  # most recently loaded/uploaded bundle


# --- Root ----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# --- Status --------------------------------------------------------------

@app.get("/api/status")
async def api_status():
    """Return current active bundle info or ``{"loaded": false}``."""
    if latest_bundle_id is None or latest_bundle_id not in bundles:
        return {"loaded": False, "bundle_id": None}
    info = bundles[latest_bundle_id]
    manifest = info["manifest"]
    gen = manifest.generated_by or {}
    return {
        "loaded": True,
        "bundle_id": latest_bundle_id,
        "manifest": manifest.model_dump(),
        "n_views": manifest.n_views,
        "n_frames": manifest.n_frames,
        "view_indices": [v.view_idx for v in info["poses"].views],
        "standoff_m": gen.get("standoff_m"),
        "drone_size_m": gen.get("drone_size_m"),
    }


# --- Upload --------------------------------------------------------------

@app.post("/api/upload")
async def api_upload(file: UploadFile):
    """Upload a ``.zip`` bundle, extract, validate, and store in memory."""
    global latest_bundle_id
    if file.filename is None or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip files are accepted")
    bundle_id = str(uuid.uuid4())[:8]
    extract_dir = tempfile.mkdtemp(prefix="bundle_")
    try:
        with zipfile.ZipFile(file.file) as zf:
            zf.extractall(extract_dir)
        manifest_path = _os.path.join(extract_dir, "manifest.json")
        if not _os.path.exists(manifest_path):
            shutil.rmtree(extract_dir)
            raise HTTPException(400, "Bundle zip must contain a manifest.json")
        manifest = BundleManifest.validate_file(manifest_path)
        poses = BundlePaces.validate_file(_os.path.join(extract_dir, "poses.json"))
        bundles[bundle_id] = {"path": extract_dir, "manifest": manifest, "poses": poses}
        latest_bundle_id = bundle_id
        gen = manifest.generated_by or {}
        log.info("Uploaded bundle %s: %s (%d views, %d frames)", bundle_id, manifest.scene_id, manifest.n_views, manifest.n_frames)
        return {
            "bundle_id": bundle_id,
            "manifest": manifest.model_dump(),
            "n_views": manifest.n_views,
            "n_frames": manifest.n_frames,
            "view_indices": [v.view_idx for v in poses.views],
            "standoff_m": gen.get("standoff_m"),
            "drone_size_m": gen.get("drone_size_m"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise HTTPException(400, f"Failed to parse bundle: {exc}")


# --- Run Pipeline --------------------------------------------------------

@app.post("/api/run")
async def api_run(data: dict):
    """Run the full reconstruction pipeline on a loaded bundle."""
    bundle_id = data.get("bundle_id") or latest_bundle_id
    if bundle_id is None or bundle_id not in bundles:
        raise HTTPException(404, f"Bundle {bundle_id!r} not found — upload one first")

    info = bundles[bundle_id]
    bundle_path = info["path"]
    manifest = info["manifest"]
    poses = info["poses"]

    view_indices: list[int] = data.get("view_indices") or []
    epipolar_threshold: float = data.get("epipolar_threshold", 3.0)
    match_threshold_m: float = data.get("match_threshold_m", 1.5)
    frame_idx: int = data.get("frame_idx", 0)

    # Build rig (filtered to selected views, sorted by view_idx)
    rig = bundle_poses_to_rig(poses, manifest, view_indices if view_indices else None)

    # Determine which actual view indices are used (sorted)
    all_views = sorted(poses.views, key=lambda v: v.view_idx)
    used_idxs = [v.view_idx for v in all_views
                 if not view_indices or v.view_idx in view_indices]
    if not used_idxs:
        raise HTTPException(400, "No valid views selected")

    # Load ground truth
    truth = load_ground_truth(bundle_path, manifest)

    # Pixel detection on selected views
    detections = detect_from_bundle_views(bundle_path, manifest, used_idxs, frame_idx)

    # Detection quality (computed against EXR passes if available)
    detection_quality = compute_detection_quality(bundle_path, used_idxs, detections, frame_idx)

    # Correspondence solving
    tracks = solve_correspondence(
        detections, rig,
        epipolar_threshold=epipolar_threshold,
        min_views=2,
        max_reproj_error=max(10.0, epipolar_threshold * 3),
        seed=42,
    )

    # Triangulation
    recon = b5_triangulate_dlt(tracks, rig, detections)

    # Grading & error vectors
    grading: dict = {}
    ghost_positions: list[list[float]] = []
    error_vectors: list[dict] = []
    truth_positions: Optional[list] = None

    if truth is not None and truth.n_drones > 0:
        fidx = min(frame_idx, truth.n_frames - 1)
        truth_pos = truth.positions[fidx]
        truth_positions = truth_pos.tolist()

        score = score_full(tracks, recon, truth, rig, position_threshold_m=match_threshold_m)

        grading = {
            "precision": score.correspondence.precision,
            "recall": score.correspondence.recall,
            "f1": score.correspondence.f1,
            "n_matched": score.correspondence.n_matched,
            "n_ghost": score.correspondence.n_ghost,
            "n_missed": score.correspondence.n_missed,
            "n_tracks": score.correspondence.n_tracks,
            "n_drones": score.correspondence.n_drones,
            "median_error_m": score.triangulation.median_error_m,
            "p95_error_m": score.triangulation.p95_error_m,
            "max_error_m": score.triangulation.max_error_m,
            "mean_reproj_error_px": score.triangulation.mean_reprojection_error_px,
        }

        # Build error vectors and ghost/missed lists
        track_truth = associate_tracks_to_truth(
            tracks, recon, truth, rig, position_threshold_m=match_threshold_m,
        )

        for di, track_idx in enumerate(track_truth.drone_to_track):
            if track_idx >= 0 and track_idx < len(recon.positions_3d):
                tp = truth_pos[di].tolist()
                rp = recon.positions_3d[track_idx].tolist()
                dist = float(np.linalg.norm(truth_pos[di] - recon.positions_3d[track_idx]))
                error_vectors.append({"from": tp, "to": rp, "distance_m": dist})

        for ti, drone_id in enumerate(track_truth.track_to_drone):
            if drone_id < 0 and ti < len(recon.positions_3d):
                ghost_positions.append(recon.positions_3d[ti].tolist())

    # Build camera-view data for frustum rendering
    camera_views = []
    for v in all_views:
        if v.view_idx in used_idxs:
            camera_views.append({
                "view_idx": v.view_idx,
                "K": v.K,
                "c2w": v.c2w,
            })

    # Assemble result
    result_id = str(uuid.uuid4())[:8]
    result = _to_serializable({
        "bundle_id": bundle_id,
        "result_id": result_id,
        "parameters": {
            "view_indices": used_idxs,
            "epipolar_threshold": epipolar_threshold,
            "match_threshold_m": match_threshold_m,
            "frame_idx": frame_idx,
        },
        "n_reconstructed": len(recon.positions_3d),
        "reconstruction": {
            "positions_3d": recon.positions_3d,
            "reprojection_errors": recon.reprojection_errors,
        },
        "grading": grading,
        "detection_quality": detection_quality if detection_quality else None,
        "has_ground_truth": truth is not None,
        "ground_truth_positions": truth_positions,
        "ghost_positions": ghost_positions,
        "error_vectors": error_vectors,
        "camera_views": camera_views,
        "manifest": manifest.model_dump(),
    })

    results_store[result_id] = result
    log.info(
        "Run %s: %d tracks -> %d reconstructed, %d matched, %d ghosts",
        result_id,
        len(tracks.tracks),
        len(recon.positions_3d),
        grading.get("n_matched", 0),
        grading.get("n_ghost", 0),
    )
    return result


# --- Export --------------------------------------------------------------

@app.post("/api/export")
async def api_export(data: dict):
    """Export a previous run result as debug JSON."""
    result_id = data.get("result_id")
    bundle_id = data.get("bundle_id")

    if result_id and result_id in results_store:
        return results_store[result_id]

    if bundle_id:
        # Return the latest result for this bundle
        for rid in reversed(list(results_store.keys())):
            if results_store[rid].get("bundle_id") == bundle_id:
                return results_store[rid]

    raise HTTPException(404, "No result found for the given bundle_id or result_id")


# ============================================================================
# CLI Entry Point
# ============================================================================

PRELOADED_BUNDLE_ID: Optional[str] = None  # set by preload_bundle for --test check


def preload_bundle(bundle_path: str) -> str:
    """Load a bundle directory into the in-memory store. Returns the bundle_id."""
    global latest_bundle_id, PRELOADED_BUNDLE_ID
    bundle_path = _os.path.abspath(bundle_path)
    manifest_path = _os.path.join(bundle_path, "manifest.json")
    if not _os.path.exists(manifest_path):
        raise FileNotFoundError(f"Not a valid bundle directory: {bundle_path} (no manifest.json)")
    manifest = BundleManifest.validate_file(manifest_path)
    poses = BundlePaces.validate_file(_os.path.join(bundle_path, "poses.json"))
    bundle_id = str(uuid.uuid4())[:8]
    bundles[bundle_id] = {"path": bundle_path, "manifest": manifest, "poses": poses}
    latest_bundle_id = bundle_id
    PRELOADED_BUNDLE_ID = bundle_id
    log.info("Loaded bundle %s: %s (%d views, %d frames)", bundle_id, manifest.scene_id, manifest.n_views, manifest.n_frames)
    return bundle_id


# ============================================================================
# HTML Frontend (inline Three.js single-page app)
# ============================================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swarm CV Distance — Reconstruction Viewer</title>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; color: #ccc; background: #1a1a2e; }

#app { display: flex; height: 100vh; }
#viewport { flex: 1; position: relative; background: #0d0d1a; }
#viewport canvas { display: block; width: 100%; height: 100%; }
#sidebar { width: 340px; min-width: 340px; overflow-y: auto; background: #16213e; border-left: 1px solid #0f3460; padding: 12px; display: flex; flex-direction: column; gap: 10px; }

h2 { font-size: 14px; font-weight: 600; color: #e0e0e0; margin-bottom: 4px; border-bottom: 1px solid #0f3460; padding-bottom: 4px; }
h3 { font-size: 12px; font-weight: 600; color: #aaa; margin-bottom: 4px; }
label { font-size: 12px; color: #999; display: block; margin: 4px 0 2px; }
input[type=number], input[type=text] { width: 100%; padding: 4px 6px; border: 1px solid #0f3460; border-radius: 4px; background: #1a1a2e; color: #ccc; font-size: 12px; }
input[type=range] { width: 100%; accent-color: #e94560; }

.info-grid { display: grid; grid-template-columns: auto 1fr; gap: 2px 8px; font-size: 11px; }
.info-grid dt { color: #888; }
.info-grid dd { color: #ccc; text-align: right; }

.checkbox-group { display: flex; flex-wrap: wrap; gap: 4px; max-height: 120px; overflow-y: auto; padding: 4px; }
.checkbox-group label { display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 11px; margin: 0; }
.checkbox-group input { accent-color: #e94560; }

.btn { padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600; }
.btn-primary { background: #e94560; color: #fff; }
.btn-primary:hover { background: #d63850; }
.btn-primary:disabled { background: #5a2a32; cursor: not-allowed; }
.btn-secondary { background: #0f3460; color: #ccc; }
.btn-secondary:hover { background: #1a4a80; }

#results-panel { display: none; }

.metric { display: flex; justify-content: space-between; padding: 2px 0; font-size: 11px; border-bottom: 1px solid #0f3460; }
.metric .label { color: #999; }
.metric .value { font-weight: 600; color: #e0e0e0; }
.metric .value.good { color: #4caf50; }
.metric .value.warn { color: #ff9800; }
.metric .value.bad { color: #f44336; }

#upload-overlay { position: fixed; inset: 0; background: rgba(13,13,26,0.92); display: none; justify-content: center; align-items: center; z-index: 100; }
#upload-overlay.active { display: flex; }
.drop-zone { border: 2px dashed #0f3460; border-radius: 16px; padding: 48px; text-align: center; cursor: pointer; background: rgba(22,33,62,0.8); transition: border-color 0.2s; }
.drop-zone:hover, .drop-zone.drag-over { border-color: #e94560; }
.drop-zone h2 { font-size: 18px; margin: 12px 0; border: none; }
.drop-zone p { color: #666; font-size: 13px; }

#status-bar { font-size: 11px; color: #666; text-align: center; padding: 4px 0; }

#spinner { display: none; width: 20px; height: 20px; border: 2px solid #0f3460; border-top-color: #e94560; border-radius: 50%; animation: spin 0.6s linear infinite; margin: 8px auto; }
@keyframes spin { to { transform: rotate(360deg); } }

.error-text { color: #f44336; font-size: 11px; padding: 4px; }
.hidden { display: none !important; }

.detection-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.detection-table th { text-align: left; color: #888; padding: 2px 4px; border-bottom: 1px solid #0f3460; }
.detection-table td { padding: 2px 4px; color: #ccc; }

#timeline { display: none; align-items: center; gap: 6px; }
#timeline.active { display: flex; }
#timeline input { flex: 1; }
#timeline span { font-size: 11px; min-width: 60px; text-align: center; }
</style>
</head>
<body>
<div id="app">
  <!-- 3D Viewport -->
  <div id="viewport"></div>

  <!-- Sidebar -->
  <div id="sidebar">
    <h2>Swarm CV Distance</h2>
    <div id="status-bar">Loading...</div>

    <!-- Bundle Info -->
    <div id="bundle-info" class="hidden">
      <h3>Bundle Info</h3>
      <dl class="info-grid">
        <dt>Scene</dt><dd id="info-scene">-</dd>
        <dt>Views</dt><dd id="info-views">-</dd>
        <dt>Frames</dt><dd id="info-frames">-</dd>
        <dt>Focal</dt><dd id="info-focal">-</dd>
        <dt>Standoff</dt><dd id="info-standoff">-</dd>
      </dl>
    </div>

    <!-- View Selector -->
    <div id="view-selector" class="hidden">
      <h3>Views</h3>
      <div class="checkbox-group" id="view-checkboxes"></div>
      <label><input type="checkbox" id="toggle-all-views" checked> Select all</label>
    </div>

    <!-- Parameters -->
    <div id="params">
      <h3>Parameters</h3>
      <label>Epipolar threshold (px):
        <input type="range" id="epipolar" min="0.5" max="20" step="0.1" value="3.0">
        <span id="epipolar-value">3.0</span>
      </label>
      <label>Match threshold (m):
        <input type="number" id="match-threshold" min="0.1" max="100" step="0.1" value="1.5">
      </label>
    </div>

    <button class="btn btn-primary" id="run-btn" onclick="runReconstruction()">Run</button>
    <div id="spinner"></div>
    <div id="error-box" class="error-text hidden"></div>

    <!-- Results -->
    <div id="results-panel">
      <h3>Grading</h3>
      <div id="grade-metrics"></div>

      <div id="detection-panel" class="hidden">
        <h3>Detection Quality</h3>
        <div id="detection-metrics"></div>
      </div>
    </div>

    <!-- Timeline -->
    <div id="timeline">
      <span>Frame</span>
      <input type="range" id="frame-slider" min="0" max="0" value="0" step="1">
      <span id="frame-label">0 / 0</span>
    </div>
  </div>

  <!-- Upload Overlay -->
  <div id="upload-overlay" class="active">
    <div class="drop-zone" id="drop-zone">
      <h2>Drop a bundle ZIP here</h2>
      <p>or click to browse &nbsp;|&nbsp; .zip only</p>
      <input type="file" id="file-input" accept=".zip" style="display:none">
    </div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

// ---- App state ------------------------------------------------------------
let currentBundleId = null;
let currentResult = null;

// ---- Three.js setup ------------------------------------------------------
const viewport = document.getElementById('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d1a);

const camera = new THREE.PerspectiveCamera(45, viewport.clientWidth / viewport.clientHeight, 1, 50000);
camera.position.set(500, 800, 1000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = false;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.target.set(0, 0, 0);
controls.update();

// ---- Lights ---------------------------------------------------------------
const ambient = new THREE.AmbientLight(0x404060, 1.0);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(500, 1000, 800);
scene.add(dirLight);
const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
dirLight2.position.set(-500, -200, -800);
scene.add(dirLight2);

// ---- Ground grid ----------------------------------------------------------
const grid = new THREE.GridHelper(2000, 40, 0x333355, 0x222244);
scene.add(grid);

// ---- Axes -----------------------------------------------------------------
const axes = new THREE.AxesHelper(200);
scene.add(axes);

// ---- Scene objects container (cleared between runs) -----------------------
const sceneObjects = new THREE.Group();
scene.add(sceneObjects);

// ---- Render loop ----------------------------------------------------------
function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

// ---- Resize handler -------------------------------------------------------
window.addEventListener('resize', () => {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// ---- Coordinate conversion (ENU -> Three.js Y-up) -------------------------
function enuToVec3(p) {
  // ENU: [E, N, U] -> Three.js: [x=E, y=U, z=N]
  return new THREE.Vector3(p[0], p[2], p[1]);
}

// ---- Scene rendering helpers ----------------------------------------------

function clearScene() {
  while (sceneObjects.children.length > 0) {
    const child = sceneObjects.children[0];
    sceneObjects.remove(child);
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
  }
}

function createSphere(pos, color, radius, opacity) {
  const geo = new THREE.SphereGeometry(radius, 16, 16);
  const mat = new THREE.MeshStandardMaterial({
    color: color,
    roughness: 0.4,
    metalness: 0.1,
    transparent: opacity !== undefined && opacity < 1,
    opacity: opacity !== undefined ? opacity : 1.0,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(enuToVec3(pos));
  return mesh;
}

function createErrorLine(fromPos, toPos, color) {
  const from = enuToVec3(fromPos);
  const to = enuToVec3(toPos);
  const geo = new THREE.BufferGeometry().setFromPoints([from, to]);
  const mat = new THREE.LineBasicMaterial({ color: color, linewidth: 1 });
  return new THREE.Line(geo, mat);
}

function renderCameraFrustum(c2w, K, color) {
  const group = new THREE.Group();
  // Extract camera position (last col of c2w)
  const pos = new THREE.Vector3(c2w[0][3], c2w[1][3], c2w[2][3]);
  // Extract rotation (upper-left 3x3 of c2w)
  const rotMatrix = new THREE.Matrix4();
  rotMatrix.set(
    c2w[0][0], c2w[0][1], c2w[0][2], 0,
    c2w[1][0], c2w[1][1], c2w[1][2], 0,
    c2w[2][0], c2w[2][1], c2w[2][2], 0,
    0, 0, 0, 1
  );
  const quat = new THREE.Quaternion().setFromRotationMatrix(rotMatrix);

  // Compute FOV from K
  const fx = K[0][0];
  const fy = K[1][1];
  const cx = K[0][2];
  const cy = K[1][2];
  const fovY = 2 * Math.atan(cy / fy);
  const fovX = 2 * Math.atan(cx / fx);

  // Frustum extends forward (in Blender camera -Z = forward)
  const dist = 100;
  const halfH = dist * Math.tan(fovY / 2);
  const halfW = dist * Math.tan(fovX / 2);

  // Corners in camera space (Blender: +X right, +Y up, -Z forward)
  const corners = [
    new THREE.Vector3(-halfW, -halfH, -dist),
    new THREE.Vector3( halfW, -halfH, -dist),
    new THREE.Vector3( halfW,  halfH, -dist),
    new THREE.Vector3(-halfW,  halfH, -dist),
  ];
  // Apply c2w rotation and translation
  corners.forEach(c => c.applyQuaternion(quat).add(pos));

  // Convert pos to Three.js ENU space
  const camPos = enuToVec3([pos.x, pos.z, pos.y]); // swap back from internal Three.js

  // But wait — the camera position in c2w is in ENU coordinates.
  // pos from c2w matrix: pos.x=E, pos.y=N, pos.z=U
  // We need to convert to Three.js: x=E, y=U, z=N
  const threePos = new THREE.Vector3(pos.x, pos.z, pos.y);

  // Draw lines from camera to each corner + connecting corners
  const mat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.4 });

  for (let i = 0; i < 4; i++) {
    const corner = corners[i];
    // Convert corner from ENU to Three.js
    const threeCorner = new THREE.Vector3(corner.x, corner.z, corner.y);
    const lineGeo = new THREE.BufferGeometry().setFromPoints([threePos, threeCorner]);
    group.add(new THREE.Line(lineGeo, mat));
  }
  // Connect corners (loop)
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    const a = corners[i];
    const b = corners[j];
    const p1 = new THREE.Vector3(a.x, a.z, a.y);
    const p2 = new THREE.Vector3(b.x, b.z, b.y);
    const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    group.add(new THREE.Line(lineGeo, mat));
  }

  // Small sphere at camera position
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(5, 8, 8),
    new THREE.MeshStandardMaterial({ color: color, emissive: color, emissiveIntensity: 0.3 })
  );
  sphere.position.copy(threePos);
  group.add(sphere);

  return group;
}

function renderResults(data) {
  clearScene();
  currentResult = data;

  // Auto-fit camera to scene bounds
  const allPoints = [];
  if (data.ground_truth_positions) data.ground_truth_positions.forEach(p => allPoints.push(p));
  if (data.reconstruction && data.reconstruction.positions_3d) data.reconstruction.positions_3d.forEach(p => allPoints.push(p));
  if (data.ghost_positions) data.ghost_positions.forEach(p => allPoints.push(p));

  if (allPoints.length > 0) {
    // Compute bounding box
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    allPoints.forEach(p => {
      const v = enuToVec3(p);
      minX = Math.min(minX, v.x); maxX = Math.max(maxX, v.x);
      minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y);
      minZ = Math.min(minZ, v.z); maxZ = Math.max(maxZ, v.z);
    });
    const center = new THREE.Vector3((minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2);
    const size = Math.max(maxX-minX, maxY-minY, maxZ-minZ, 100);

    controls.target.copy(center);
    camera.position.set(center.x + size*0.8, center.y + size*0.6, center.z + size);
    controls.update();

    // Adjust grid
    const gridSize = Math.pow(10, Math.ceil(Math.log10(size)));
    scene.remove(grid);
    const newGrid = new THREE.GridHelper(gridSize, Math.min(40, Math.round(gridSize / 10)), 0x333355, 0x222244);
    newGrid.position.y = Math.min(minY - 10, -10);
    scene.add(newGrid);
    scene.remove(grid); // remove the old one from scene
    // Actually the old grid is still in the scene, let's just replace it
  }

  // ---- Ground truth: green spheres ----
  if (data.ground_truth_positions && data.has_ground_truth) {
    data.ground_truth_positions.forEach((pos, idx) => {
      // Check if this drone is missed (faded gray)
      const isMissed = data.missed_drone_indices && data.missed_drone_indices.includes(idx);
      if (isMissed) {
        const sphere = createSphere(pos, 0x888888, 12, 0.35);
        sceneObjects.add(sphere);
      } else {
        const sphere = createSphere(pos, 0x00cc44, 14);
        sceneObjects.add(sphere);
      }
    });
  }

  // ---- Reconstructed: blue spheres (only matched ones, ghosts handled separately) ----
  if (data.reconstruction && data.reconstruction.positions_3d) {
    data.reconstruction.positions_3d.forEach((pos, idx) => {
      const sphere = createSphere(pos, 0x4488ff, 10);
      sceneObjects.add(sphere);
    });
  }

  // ---- Ghosts: red spheres ----
  if (data.ghost_positions) {
    data.ghost_positions.forEach(pos => {
      const sphere = createSphere(pos, 0xff2222, 11);
      // Add a faint glow
      const glow = createSphere(pos, 0xff4444, 15, 0.3);
      sceneObjects.add(sphere);
      sceneObjects.add(glow);
    });
  }

  // ---- Error vectors ----
  if (data.error_vectors) {
    data.error_vectors.forEach(ev => {
      let color;
      if (ev.distance_m < 1.0) color = 0x44cc44;
      else if (ev.distance_m < 3.0) color = 0xcccc44;
      else color = 0xcc4444;
      sceneObjects.add(createErrorLine(ev.from, ev.to, color));
    });
  }

  // ---- Camera frustums ----
  if (data.camera_views) {
    data.camera_views.forEach((view, idx) => {
      const hue = (idx * 60) % 360;
      const color = new THREE.Color(`hsl(${hue}, 80%, 60%)`);
      const frustumGroup = renderCameraFrustum(view.c2w, view.K, color);
      sceneObjects.add(frustumGroup);
    });
  }
}

// ---- Internal rendering helper for back-end calls -------------------------

function renderSceneData(data) {
  renderResults(data);
}

// ---- API helpers ----------------------------------------------------------

async function apiPost(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`API error ${resp.status}: ${err}`);
  }
  return resp.json();
}

async function apiUpload(file) {
  const form = new FormData();
  form.append('file', file);
  const resp = await fetch('/api/upload', { method: 'POST', body: form });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Upload error ${resp.status}: ${err}`);
  }
  return resp.json();
}

// ---- UI helpers -----------------------------------------------------------

function showError(msg) {
  const el = document.getElementById('error-box');
  el.textContent = msg;
  el.classList.remove('hidden');
}

function hideError() {
  document.getElementById('error-box').classList.add('hidden');
}

function showSpinner() {
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('run-btn').disabled = true;
}

function hideSpinner() {
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('run-btn').disabled = false;
}

function fmtNum(v, decimals) {
  if (typeof v !== 'number') return String(v);
  return v.toFixed(decimals !== undefined ? decimals : (v < 10 ? 2 : 1));
}

// ---- Bundle loading -------------------------------------------------------

function populateBundleInfo(data) {
  document.getElementById('bundle-info').classList.remove('hidden');
  document.getElementById('info-scene').textContent = data.manifest?.scene_id || '-';
  document.getElementById('info-views').textContent = data.n_views || '-';
  document.getElementById('info-frames').textContent = data.n_frames || '-';
  document.getElementById('info-focal').textContent = data.manifest?.focal_px ? fmtNum(data.manifest.focal_px, 0) + ' px' : '-';
  document.getElementById('info-standoff').textContent = data.standoff_m ? fmtNum(data.standoff_m, 0) + ' m' : '-';
}

function populateViewSelector(data) {
  const container = document.getElementById('view-checkboxes');
  container.innerHTML = '';
  document.getElementById('view-selector').classList.remove('hidden');

  data.view_indices.forEach(idx => {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = idx;
    cb.checked = true;
    cb.dataset.viewIdx = idx;
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` Camera ${idx}`));
    container.appendChild(label);
  });

  // Select all toggle
  const toggleAll = document.getElementById('toggle-all-views');
  toggleAll.checked = true;
  toggleAll.onchange = () => {
    container.querySelectorAll('input[type=checkbox]').forEach(cb => cb.checked = toggleAll.checked);
  };

  // Individual click updates "select all" state
  container.addEventListener('change', () => {
    const all = container.querySelectorAll('input[type=checkbox]');
    const checked = container.querySelectorAll('input[type=checkbox]:checked');
    toggleAll.checked = all.length === checked.length;
  });

  // Timeline
  if (data.n_frames > 1) {
    const timeline = document.getElementById('timeline');
    timeline.classList.add('active');
    const slider = document.getElementById('frame-slider');
    slider.max = data.n_frames - 1;
    slider.value = 0;
    slider.oninput = () => {
      document.getElementById('frame-label').textContent = `${parseInt(slider.value) + 1} / ${data.n_frames}`;
    };
  }
}

function loadBundle(bundleId, data) {
  currentBundleId = bundleId;
  document.getElementById('upload-overlay').classList.remove('active');
  document.getElementById('status-bar').textContent = `Bundle: ${data.manifest?.scene_id || bundleId}`;
  clearScene();
  document.getElementById('results-panel').style.display = 'none';
  document.getElementById('detection-panel').classList.add('hidden');

  populateBundleInfo(data);
  populateViewSelector(data);
}

// ---- Run pipeline ---------------------------------------------------------

async function runReconstruction() {
  hideError();
  document.getElementById('results-panel').style.display = 'none';
  document.getElementById('detection-panel').classList.add('hidden');

  if (!currentBundleId) {
    showError('No bundle loaded');
    return;
  }

  // Gather selected views
  const checkboxes = document.querySelectorAll('#view-checkboxes input[type=checkbox]');
  const selectedViews = [];
  checkboxes.forEach(cb => {
    if (cb.checked) selectedViews.push(parseInt(cb.value));
  });

  if (selectedViews.length < 2) {
    showError('Select at least 2 views');
    return;
  }

  showSpinner();

  try {
    const data = await apiPost('/api/run', {
      bundle_id: currentBundleId,
      view_indices: selectedViews,
      epipolar_threshold: parseFloat(document.getElementById('epipolar').value),
      match_threshold_m: parseFloat(document.getElementById('match-threshold').value),
      frame_idx: parseInt(document.getElementById('frame-slider').value) || 0,
    });

    renderResults(data);
    displayResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    hideSpinner();
  }
}

function displayResults(data) {
  const panel = document.getElementById('results-panel');
  panel.style.display = 'block';

  // Grading
  const g = data.grading || {};
  const metrics = document.getElementById('grade-metrics');
  metrics.innerHTML = '';

  const gradeItems = [
    { label: 'Precision', value: g.precision, fmt: 'pct' },
    { label: 'Recall', value: g.recall, fmt: 'pct' },
    { label: 'F1 Score', value: g.f1, fmt: 'pct' },
    { label: 'Matched', value: g.n_matched },
    { label: 'Missed', value: g.n_missed },
    { label: 'Ghosts', value: g.n_ghost },
    { label: 'Median error', value: g.median_error_m, unit: 'm', fmt: 'dist' },
    { label: 'P95 error', value: g.p95_error_m, unit: 'm', fmt: 'dist' },
    { label: 'Max error', value: g.max_error_m, unit: 'm', fmt: 'dist' },
  ];

  gradeItems.forEach(item => {
    const d = document.createElement('div');
    d.className = 'metric';
    let valStr = '--';
    if (item.value !== undefined && item.value !== null) {
      if (item.fmt === 'pct') valStr = (item.value * 100).toFixed(1) + '%';
      else if (item.fmt === 'dist') valStr = fmtNum(item.value, 2) + ' ' + (item.unit || 'm');
      else valStr = String(item.value) + (item.unit ? ' ' + item.unit : '');
    }
    d.innerHTML = `<span class="label">${item.label}</span><span class="value">${valStr}</span>`;
    metrics.appendChild(d);
  });

  // Detection quality
  const dq = data.detection_quality;
  const dqPanel = document.getElementById('detection-panel');
  const dqMetrics = document.getElementById('detection-metrics');
  dqMetrics.innerHTML = '';

  if (dq && Object.keys(dq).length > 0) {
    dqPanel.classList.remove('hidden');
    const table = document.createElement('table');
    table.className = 'detection-table';
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th>View</th><th>Recall</th><th>FP</th><th>Centroid Err</th><th>Merged</th></tr>';
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    Object.entries(dq).forEach(([key, val]) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${key}</td><td>${(val.detector_recall * 100).toFixed(0)}%</td><td>${val.fp}</td><td>${val.centroid_error_px.toFixed(2)}px</td><td>${val.merged_detections}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    dqMetrics.appendChild(table);
  } else {
    dqPanel.classList.add('hidden');
  }
}

// ---- Upload UI ------------------------------------------------------------

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
  if (fileInput.files.length > 0) {
    await handleUpload(fileInput.files[0]);
    fileInput.value = '';
  }
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    await handleUpload(e.dataTransfer.files[0]);
  }
});

async function handleUpload(file) {
  hideError();
  showSpinner();
  try {
    const data = await apiUpload(file);
    loadBundle(data.bundle_id, data);
  } catch (err) {
    showError(err.message);
    hideSpinner();
  } finally {
    hideSpinner();
  }
}

// ---- Epipolar threshold display -------------------------------------------

document.getElementById('epipolar').addEventListener('input', function() {
  document.getElementById('epipolar-value').textContent = parseFloat(this.value).toFixed(1);
});

// ---- Init: check for preloaded bundle -------------------------------------

async function init() {
  try {
    const resp = await fetch('/api/status');
    const status = await resp.json();
    if (status.loaded) {
      loadBundle(status.bundle_id, status);
      // Auto-run with defaults
      await runReconstruction();
    } else {
      document.getElementById('status-bar').textContent = 'No bundle loaded — drop a ZIP above';
    }
  } catch (err) {
    document.getElementById('status-bar').textContent = 'Connecting...';
  }
}

init();
</script>
</body>
</html>
"""


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Swarm CV Distance — Reconstruction Viewer",
    )
    parser.add_argument(
        "bundle_path", nargs="?",
        help="Path to a bundle directory (optional; uploaded otherwise)",
    )
    parser.add_argument(
        "--port", type=int, default=8820,
        help="Server port (default: 8820)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a quick self-test and exit (verifies imports & app creation)",
    )
    args = parser.parse_args()

    if args.test:
        print("=== Reconstruction App Self-Test ===")
        print(f"  App title: {app.title}")
        print(f"  App version: {app.version}")
        print("  Routes:")
        for route in app.routes:
            if hasattr(route, "methods") and route.methods:
                print(f"    {', '.join(route.methods)} {route.path}")
        print("  Imports: OK")
        print("  App creation: OK")
        print("=== SELF-TEST PASSED ===")
        _sys.exit(0)

    if args.bundle_path:
        try:
            preload_bundle(args.bundle_path)
        except Exception as exc:
            log.error("Failed to load bundle: %s", exc)
            _sys.exit(1)

    log.info("Starting server on http://localhost:%d", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


# ============================================================================
# End of reconstruction_app.py
# ============================================================================
