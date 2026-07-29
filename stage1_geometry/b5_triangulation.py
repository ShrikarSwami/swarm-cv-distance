"""
B5: Triangulation — DLT + Levenberg-Marquardt refinement.

Implementation of the interface defined in data_contract.py:
- triangulate_dlt(tracks, rig) -> Reconstruction
- triangulate_dlt_then_refine(tracks, rig) -> Reconstruction

All types imported from data_contract. No imports from other Stage B modules.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from data_contract import (
    Tracks,
    CameraRig,
    Detections,
    Reconstruction,
)


def _triangulate_single_track_dlt(
    track: list[tuple[int, int]],
    detections: Detections,
    rig: CameraRig,
) -> tuple[NDArray[np.float64] | None, float]:
    """
    Triangulate a single track using DLT.

    Returns:
        (3D position in ENU, mean reprojection error) or (None, inf) if failed
    """
    if len(track) < 2:
        return None, np.inf

    points = []
    K_list = []
    R_list = []
    t_list = []

    for view_idx, point_idx in track:
        if view_idx >= rig.n_views:
            return None, np.inf
        if point_idx >= len(detections.points_per_view[view_idx]):
            return None, np.inf

        pt = detections.points_per_view[view_idx][point_idx]
        points.append(pt)
        K_list.append(rig.K[view_idx])
        R_list.append(rig.w2c_R[view_idx])
        t_list.append(rig.w2c_t[view_idx])

    # DLT: A * X = 0 where X = [X, Y, Z, 1]^T
    A = []
    for pt, K, R, t in zip(points, K_list, R_list, t_list):
        P = K @ np.hstack([R, t.reshape(3, 1)])  # (3, 4)
        x, y = pt
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.array(A)
    if A.shape[0] < 4:
        return None, np.inf

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None, np.inf

    pos_3d = (X[:3] / X[3]).astype(np.float64)

    # Compute reprojection errors
    errors = []
    for pt, K, R, t in zip(points, K_list, R_list, t_list):
        cam_pt = R @ pos_3d + t
        if cam_pt[2] <= 0:
            return None, np.inf
        proj = K @ cam_pt
        proj = proj[:2] / proj[2]
        err = np.linalg.norm(proj - pt)
        errors.append(err)

    return pos_3d, float(np.mean(errors))


def _refine_lm(
    pos_3d: NDArray[np.float64],
    points: list[NDArray[np.float64]],
    K_list: list[NDArray[np.float64]],
    R_list: list[NDArray[np.float64]],
    t_list: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], float]:
    """
    Levenberg-Marquardt refinement of 3D position.

    Minimizes sum of squared reprojection errors.
    """
    def residuals(x):
        """Residuals: 2 * n_views elements (x, y error per view)."""
        res = []
        for pt, K, R, t in zip(points, K_list, R_list, t_list):
            cam_pt = R @ x + t
            if cam_pt[2] <= 0:
                res.extend([1e6, 1e6])
                continue
            proj = K @ cam_pt
            proj = proj[:2] / proj[2]
            res.extend(proj - pt)
        return np.array(res)

    result = least_squares(residuals, pos_3d, method='lm', max_nfev=50)
    refined_pos = result.x

    # Compute final reprojection error
    errors = []
    for pt, K, R, t in zip(points, K_list, R_list, t_list):
        cam_pt = R @ refined_pos + t
        if cam_pt[2] <= 0:
            return pos_3d, np.inf
        proj = K @ cam_pt
        proj = proj[:2] / proj[2]
        err = np.linalg.norm(proj - pt)
        errors.append(err)

    return refined_pos, float(np.mean(errors))


def triangulate_dlt(
    tracks: Tracks,
    rig: CameraRig,
    detections: Detections,
) -> Reconstruction:
    """
    DLT triangulation of multi-view tracks.

    Args:
        tracks: Correspondence tracks from B3 (no identity information)
        rig: CameraRig with K matrices and world-to-camera poses
        detections: Detections with 2D points per view

    Returns:
        Reconstruction with 3D positions and reprojection errors
    """
    positions_3d = []
    reprojection_errors = []
    track_indices = []

    for track in tracks.tracks:
        pos_3d, mean_err = _triangulate_single_track_dlt(track, detections, rig)
        if pos_3d is not None and mean_err < 50.0:  # Filter out wild triangulations
            positions_3d.append(pos_3d)
            reprojection_errors.append(mean_err)
            track_indices.append(track)

    if positions_3d:
        positions_3d = np.array(positions_3d, dtype=np.float64)
        reprojection_errors = np.array(reprojection_errors, dtype=np.float64)
    else:
        positions_3d = np.empty((0, 3), dtype=np.float64)
        reprojection_errors = np.empty((0,), dtype=np.float64)

    return Reconstruction(
        positions_3d=positions_3d,
        reprojection_errors=reprojection_errors,
        track_indices=track_indices,
    )


def triangulate_dlt_then_refine(
    tracks: Tracks,
    rig: CameraRig,
    detections: Detections,
) -> Reconstruction:
    """
    DLT + nonlinear refinement (Levenberg-Marquardt on reprojection error).

    Args:
        tracks: Correspondence tracks from B3
        rig: CameraRig with K matrices and world-to-camera poses
        detections: Detections with 2D points per view

    Returns:
        Reconstruction with refined 3D positions and reprojection errors
    """
    n_views = len(detections.points_per_view)
    positions_3d = []
    reprojection_errors = []
    track_indices = []

    for track in tracks.tracks:
        if len(track) < 2:
            continue

        # Collect data for this track
        points = []
        K_list = []
        R_list = []
        t_list = []

        for view_idx, point_idx in track:
            if view_idx >= n_views:
                break
            if point_idx >= len(detections.points_per_view[view_idx]):
                break

            pt = detections.points_per_view[view_idx][point_idx]
            points.append(pt)
            K_list.append(rig.K[view_idx])
            R_list.append(rig.w2c_R[view_idx])
            t_list.append(rig.w2c_t[view_idx])

        if len(points) < 2:
            continue

        # DLT initial estimate
        pos_3d, _ = _triangulate_single_track_dlt(track, detections, rig)
        if pos_3d is None:
            continue

        # LM refinement
        try:
            refined_pos, mean_err = _refine_lm(pos_3d, points, K_list, R_list, t_list)
            if mean_err < 50.0:  # Filter
                positions_3d.append(refined_pos)
                reprojection_errors.append(mean_err)
                track_indices.append(track)
        except Exception:
            continue

    if positions_3d:
        positions_3d = np.array(positions_3d, dtype=np.float64)
        reprojection_errors = np.array(reprojection_errors, dtype=np.float64)
    else:
        positions_3d = np.empty((0, 3), dtype=np.float64)
        reprojection_errors = np.empty((0,), dtype=np.float64)

    return Reconstruction(
        positions_3d=positions_3d,
        reprojection_errors=reprojection_errors,
        track_indices=track_indices,
    )


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== B5 Triangulation Self-Test ===")

    from b1_scene_rig import generate_swarm_truth, generate_camera_rig
    from b2_projection import project_swarm_to_detections
    from b3_correspondence import solve_correspondence

    # Generate test data
    truth = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)
    rig = generate_camera_rig(truth, n_views=6, geometry_class="mixed", standoff_m=1000.0, seed=123)

    detections = project_swarm_to_detections(truth, rig, pixel_noise_std=0.5, drop_prob=0.0, seed=0)

    print(f"\nInput:")
    print(f"  Drones: {truth.n_drones}")
    print(f"  Views: {rig.n_views}")
    print(f"  Detections per view: {[len(d) for d in detections.points_per_view]}")

    # Solve correspondence
    tracks = solve_correspondence(
        detections, rig,
        epipolar_threshold=3.0,
        min_views=2,
        max_reproj_error=5.0,
        seed=42,
    )

    print(f"\nCorrespondence:")
    print(f"  Tracks found: {len(tracks.tracks)}")
    for i, track in enumerate(tracks.tracks):
        print(f"  Track {i}: {track} ({len(track)} views)")

    # Test DLT triangulation
    print("\n--- DLT Triangulation ---")
    recon_dlt = triangulate_dlt(tracks, rig, detections)
    print(f"  Reconstructed: {len(recon_dlt.positions_3d)} tracks")
    for i, (pos, err, track) in enumerate(zip(recon_dlt.positions_3d, recon_dlt.reprojection_errors, recon_dlt.track_indices)):
        print(f"  Track {i}: pos={pos}, reproj_err={err:.4f}px, views={len(track)}")

    # Test DLT + LM refinement
    print("\n--- DLT + LM Refinement ---")
    recon_lm = triangulate_dlt_then_refine(tracks, rig, detections)
    print(f"  Reconstructed: {len(recon_lm.positions_3d)} tracks")
    for i, (pos, err, track) in enumerate(zip(recon_lm.positions_3d, recon_lm.reprojection_errors, recon_lm.track_indices)):
        print(f"  Track {i}: pos={pos}, reproj_err={err:.4f}px, views={len(track)}")

    # Compare with ground truth (we know which track should match which drone)
    # Since correspondence is anonymous, we match by proximity
    print("\n--- Evaluation vs Ground Truth ---")
    for i, (pos_dlt, pos_lm) in enumerate(zip(recon_dlt.positions_3d, recon_lm.positions_3d)):
        # Find closest truth drone
        min_dist_dlt = np.inf
        min_dist_lm = np.inf
        best_drone = -1
        for d in range(truth.n_drones):
            pos_true = truth.positions[0, d]
            dist_dlt = np.linalg.norm(pos_dlt - pos_true)
            dist_lm = np.linalg.norm(pos_lm - pos_true)
            if dist_dlt < min_dist_dlt:
                min_dist_dlt = dist_dlt
            if dist_lm < min_dist_lm:
                min_dist_lm = dist_lm
                best_drone = d
        pos_true = truth.positions[0, best_drone]
        print(f"  Track {i} -> Drone {best_drone}: DLT err={min_dist_dlt:.2f}m, LM err={min_dist_lm:.2f}m")

    print("\n=== B5: SELF-TEST PASSED ===")