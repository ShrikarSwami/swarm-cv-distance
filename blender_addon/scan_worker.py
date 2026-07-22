"""
M4 scan worker: runs under the project venv (needs numpy + OpenEXR), not
Blender's bundled Python. Invoked as a subprocess by SWARM_OT_scan in
swarm_scanner/__init__.py, because Blender's own bpy.data.images.load()
cannot read back the custom-named "id_" multilayer pass this project's
compositor graph writes (confirmed by hand: it loads as a 0x0 TARGA-typed
image) -- OpenEXR direct reads (already used by validate_rig_report.py)
are the only reliable path found for this pass name/format combination.

Given a manifest of real per-camera ID-pass EXRs (real occlusion baked in
by Cycles, not frustum math) and camera poses/intrinsics taken straight
from the actual rendered Blender cameras, this:
  1. extracts each drone's detection pixel as the centroid of its ID-pass
     footprint, per camera it appears in at all (a camera that doesn't see
     a drone contributes no detection for it -- real occlusion/out-of-frame,
     not a synthetic drop probability)
  2. layers Stage 1's synthetic pixel-noise model on top of those centroids
     (stand-in for real detector localization error -- see PROGRESS.md's
     subpixel finding for why a real detector isn't run instead)
  3. triangulates via Stage 1's triangulate_point()/reconstruct_swarm(),
     unchanged
  4. reports overall + near-D_MAX-threshold adjacency accuracy, and an edge
     list (with correctness) for the addon's viewport overlay to draw
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
import OpenEXR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "stage1_geometry"))
import multiview_triangulation_test as mvt  # noqa: E402


class _PoseCamera:
    """Stand-in for mvt.Camera carrying only what triangulate_point() reads
    (cam.P) -- built from the real rendered camera's pose, not a look-at
    target, so it's correct even for hand-rotated manual-mode cameras.
    """

    def __init__(self, P):
        self.P = np.array(P)


def _drone_centroid(id_pass, drone_id):
    ys, xs = np.where(np.round(id_pass) == drone_id)
    if len(ys) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def _edge_accuracy(true_D, est_D, d_max, mask=None):
    valid = ~np.isnan(est_D)
    if mask is not None:
        valid = valid & mask
    true_adj = (true_D <= d_max) & ~np.isnan(true_D)
    est_adj = (est_D <= d_max) & ~np.isnan(est_D)
    agree = (true_adj == est_adj) & valid
    denom = valid.sum()
    if denom == 0:
        return None
    return float(agree.sum() / denom)


def run(manifest_path, result_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

    drone_ids = [d["id"] for d in manifest["drones"]]
    true_pos = {d["id"]: np.array(d["position"]) for d in manifest["drones"]}
    id_to_idx = {did: i for i, did in enumerate(drone_ids)}
    n = len(drone_ids)

    cameras = [_PoseCamera(c["P"]) for c in manifest["cameras"]]
    detections = [[None] * n for _ in cameras]
    for ci, c in enumerate(manifest["cameras"]):
        exr = OpenEXR.File(c["exr_path"])
        id_pass = np.array(exr.parts[0].channels["id_.V"].pixels)
        for did in drone_ids:
            centroid = _drone_centroid(id_pass, did)
            if centroid is not None:
                detections[ci][id_to_idx[did]] = centroid

    rng = np.random.default_rng(manifest["noise_seed"])
    noise_std = manifest["pixel_noise_std"]
    for ci in range(len(cameras)):
        for di in range(n):
            if detections[ci][di] is not None:
                detections[ci][di] = detections[ci][di] + rng.normal(0, noise_std, size=2)

    est_positions, n_views = mvt.reconstruct_swarm(cameras, detections, n)

    true_arr = np.array([true_pos[did] for did in drone_ids])
    true_D = mvt.pairwise_distances(true_arr)
    est_D = mvt.pairwise_distances(est_positions)

    d_max = manifest["d_max"]
    near_band = manifest["near_threshold_frac"] * d_max
    near_mask = np.abs(true_D - d_max) <= near_band

    overall_accuracy = _edge_accuracy(true_D, est_D, d_max)
    near_threshold_accuracy = _edge_accuracy(true_D, est_D, d_max, mask=near_mask)

    valid = ~np.isnan(est_D)
    dist_err = np.abs(true_D[valid] - est_D[valid])

    true_adj = (true_D <= d_max) & ~np.isnan(true_D)
    est_adj = (est_D <= d_max) & ~np.isnan(est_D)
    edges = []
    for i, j in combinations(range(n), 2):
        if not (valid[i, j] and (true_adj[i, j] or est_adj[i, j])):
            continue
        edges.append({
            "i": drone_ids[i],
            "j": drone_ids[j],
            "true_adj": bool(true_adj[i, j]),
            "est_adj": bool(est_adj[i, j]),
            "correct": bool(true_adj[i, j] == est_adj[i, j]),
        })

    n_triangulated = int(np.sum(~np.isnan(est_positions).any(axis=1)))
    result = {
        "drones": [
            {
                "id": did,
                "true_position": true_pos[did].tolist(),
                "est_position": (None if np.isnan(est_positions[id_to_idx[did]]).any()
                                  else est_positions[id_to_idx[did]].tolist()),
                "n_views": int(n_views[id_to_idx[did]]),
            }
            for did in drone_ids
        ],
        "edges": edges,
        "n_triangulated": n_triangulated,
        "n_total": n,
        "overall_accuracy": overall_accuracy,
        "near_threshold_accuracy": near_threshold_accuracy,
        "mean_dist_error_m": float(dist_err.mean()) if len(dist_err) else None,
        "d_max": d_max,
    }
    with open(result_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    parser.add_argument("result_path")
    args = parser.parse_args()
    run(args.manifest_path, args.result_path)
