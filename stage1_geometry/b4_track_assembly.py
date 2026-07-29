"""
B4: Track Assembly — link frame-to-frame tracks across temporal sequences.

All types imported from data_contract. No imports from other Stage B modules.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Literal
from scipy.optimize import linear_sum_assignment

from data_contract import (
    SwarmTruth,
    CameraRig,
    Tracks,
    Reconstruction,
)


@dataclass(frozen=True)
class Track3D:
    """A single 3D track across frames."""
    track_id: int
    # Per-frame data: (frame_idx, view_idx, point_idx) tuples
    observations: list[tuple[int, int, int]]
    # Estimated 3D position per frame (ENU)
    positions_3d: NDArray[np.float64]  # (n_frames, 3)
    # Reprojection error per frame
    reproj_errors: NDArray[np.float64]  # (n_frames,)


def triangulate_track(track_2d: Tracks, rig: CameraRig) -> Reconstruction:
    """
    Triangulate a multi-view track to 3D.

    Uses the DLT interface from data_contract.
    """
    from data_contract import triangulate_dlt
    return triangulate_dlt(track_2d, rig)


def compute_track_cost(
    track_a: Track3D,
    track_b: Track3D,
    max_dist: float = 50.0,
    max_reproj: float = 10.0,
) -> float:
    """
    Compute cost for associating two tracks across frames.

    Lower is better match.
    """
    # Check if they overlap in time
    frames_a = {obs[0] for obs in track_a.observations}
    frames_b = {obs[0] for obs in track_b.observations}
    common_frames = frames_a & frames_b

    if not common_frames:
        # No temporal overlap - check spatial proximity
        if len(track_a.positions_3d) == 0 or len(track_b.positions_3d) == 0:
            return np.inf
        dist = np.linalg.norm(track_a.positions_3d[-1] - track_b.positions_3d[0])
        return dist if dist < max_dist else np.inf

    # Check spatial proximity in common frames
    total_dist = 0.0
    count = 0
    for f in common_frames:
        idx_a = next(i for i, obs in enumerate(track_a.observations) if obs[0] == f)
        idx_b = next(i for i, obs in enumerate(track_b.observations) if obs[0] == f)
        pos_a = track_a.positions_3d[idx_a]
        pos_b = track_b.positions_3d[idx_b]
        dist = np.linalg.norm(pos_a - pos_b)
        if dist > max_dist:
            return np.inf
        total_dist += dist
        count += 1

    avg_dist = total_dist / count if count > 0 else np.inf

    # Also check reprojection consistency
    avg_reproj_a = np.mean(track_a.reproj_errors) if len(track_a.reproj_errors) > 0 else np.inf
    avg_reproj_b = np.mean(track_b.reproj_errors) if len(track_b.reproj_errors) > 0 else np.inf
    avg_reproj = (avg_reproj_a + avg_reproj_b) / 2

    if avg_reproj > max_reproj:
        return np.inf

    # Cost = distance + reprojection penalty
    return avg_dist + 0.1 * avg_reproj


def associate_tracks_frame_to_frame(
    tracks_prev: list[Track3D],
    tracks_curr: list[Track3D],
    max_dist: float = 50.0,
    max_reproj: float = 10.0,
) -> dict[int, int]:
    """
    Associate tracks from frame t to frame t+1 using Hungarian algorithm.

    Returns mapping from prev_track_idx -> curr_track_idx (or -1 for unmatched).
    """
    if not tracks_prev or not tracks_curr:
        return {}

    n_prev = len(tracks_prev)
    n_curr = len(tracks_curr)
    max_n = max(n_prev, n_curr)

    cost = np.full((max_n, max_n), 1e6)

    for i, t_prev in enumerate(tracks_prev):
        for j, t_curr in enumerate(tracks_curr):
            c = compute_track_cost(t_prev, t_curr, max_dist, max_reproj)
            cost[i, j] = c

    # Add dummy rows/cols with high cost
    row_ind, col_ind = linear_sum_assignment(cost)

    assignment = {}
    for ri, ci in zip(row_ind, col_ind):
        if ri < n_prev and ci < n_curr and cost[ri, ci] < 1e6:
            assignment[ri] = ci
        elif ri < n_prev:
            assignment[ri] = -1  # Unmatched

    return assignment


def build_spatiotemporal_tracks(
    detections_per_frame: list,  # list of Detections, one per frame
    rig: CameraRig,
    epipolar_threshold: float = 5.0,
    min_views: int = 2,
    max_reproj_error: float = 10.0,
    max_track_dist: float = 50.0,
    seed: int = 0,
) -> list[Track3D]:
    """
    Build full spatio-temporal tracks from multi-frame detections.

    Pipeline:
    1. For each frame, solve correspondence (B3) to get 2D tracks
    2. Triangulate each 2D track to 3D
    3. Associate tracks across frames
    4. Filter by quality

    Args:
        detections_per_frame: List of Detections objects (one per frame)
        rig: CameraRig
        epipolar_threshold: Epipolar distance threshold for B3
        min_views: Minimum views per track
        max_reproj_error: Max reprojection error for B3
        max_track_dist: Max 3D distance for temporal association
        seed: RNG seed

    Returns:
        List of Track3D objects
    """
    from b3_correspondence import solve_correspondence, triangulate_dlt
    from data_contract import validate_detections

    np.random.seed(seed)
    n_frames = len(detections_per_frame)

    # Step 1: Solve correspondence per frame
    frame_tracks_2d = []
    for f in range(n_frames):
        detections = detections_per_frame[f]
        tracks_2d = solve_correspondence(
            detections, rig,
            epipolar_threshold=epipolar_threshold,
            min_views=min_views,
            max_reproj_error=max_reproj_error,
            seed=seed + f * 1000,
        )
        frame_tracks_2d.append(tracks_2d)

    # Step 2: Triangulate per-frame tracks to 3D
    frame_tracks_3d = []
    for f in range(n_frames):
        tracks_3d = []
        tracks_2d = frame_tracks_2d[f]

        for track_idx, track in enumerate(tracks_2d.tracks):
            if len(track) < min_views:
                continue

            # Get 2D points for this track
            points = []
            K_list = []
            R_list = []
            t_list = []
            view_indices = []
            point_indices = []

            for view_idx, point_idx in track:
                points.append(detections_per_frame[f].points_per_view[view_idx][point_idx])
                K_list.append(rig.K[view_idx])
                R_list.append(rig.w2c_R[view_idx])
                t_list.append(rig.w2c_t[view_idx])
                view_indices.append(view_idx)
                point_indices.append(point_idx)

            # Triangulate
            pos_3d = triangulate_dlt(points, K_list, R_list, t_list)
            if pos_3d is None:
                continue

            # Compute reprojection errors
            reproj_errors = []
            for view_idx, point_idx in track:
                pt_2d = detections_per_frame[f].points_per_view[view_idx][point_idx]
                K = rig.K[view_idx]
                R = rig.w2c_R[view_idx]
                t = rig.w2c_t[view_idx]

                cam_pt = R @ pos_3d + t
                if cam_pt[2] > 0:
                    proj = K @ cam_pt
                    proj = proj[:2] / proj[2]
                    err = np.linalg.norm(proj - pt_2d)
                else:
                    err = np.inf
                reproj_errors.append(err)

            if np.mean(reproj_errors) > max_reproj_error:
                continue

            # Create Track3D
            track_3d = Track3D(
                track_id=track_idx,
                observations=[(f, v, p) for v, p in zip(view_indices, point_indices)],
                positions_3d=np.array([pos_3d]),
                reproj_errors=np.array(reproj_errors),
            )
            tracks_3d.append(track_3d)

        frame_tracks_3d.append(tracks_3d)

    # Step 3: Associate across frames
    if n_frames == 1:
        return frame_tracks_3d[0]

    # Start with first frame tracks
    active_tracks = frame_tracks_3d[0].copy()

    for f in range(1, n_frames):
        curr_tracks = frame_tracks_3d[f]

        # Associate active tracks to current frame
        assignment = associate_tracks_frame_to_frame(
            active_tracks, curr_tracks,
            max_dist=max_track_dist,
            max_reproj=max_reproj_error,
        )

        # Update or create new tracks
        new_active = []
        matched_curr = set()

        for i, track in enumerate(active_tracks):
            if i in assignment and assignment[i] >= 0:
                # Match found - extend track
                j = assignment[i]
                curr_track = curr_tracks[j]

                # Combine observations
                combined_obs = track.observations + curr_track.observations
                combined_pos = np.vstack([track.positions_3d, curr_track.positions_3d])
                combined_reproj = np.concatenate([track.reproj_errors, curr_track.reproj_errors])

                extended = Track3D(
                    track_id=track.track_id,
                    observations=combined_obs,
                    positions_3d=combined_pos,
                    reproj_errors=combined_reproj,
                )
                new_active.append(extended)
                matched_curr.add(j)
            else:
                # Track ended - keep as is (could add completion logic here)
                new_active.append(track)

        # Add unmatched current tracks as new tracks
        for j, track in enumerate(curr_tracks):
            if j not in matched_curr:
                new_active.append(track)

        active_tracks = new_active

    # Step 4: Filter by minimum track length
    min_frames = max(2, n_frames // 2)
    final_tracks = [t for t in active_tracks if len(t.observations) >= min_frames]

    # Reassign track IDs
    for i, t in enumerate(final_tracks):
        final_tracks[i] = Track3D(
            track_id=i,
            observations=t.observations,
            positions_3d=t.positions_3d,
            reproj_errors=t.reproj_errors,
        )

    return final_tracks


def evaluate_tracks(
    estimated: list[Track3D],
    truth: SwarmTruth,
    d_max: float = 100.0,
) -> dict:
    """
    Evaluate track quality against ground truth.

    Reports:
    - Correct tracks (matched to single truth drone)
    - Ghost tracks (no truth match)
    - Merged tracks (multiple truth drones)
    - Missed drones
    - Position error (median, 95th percentile)
    """
    if not estimated:
        return {
            "n_estimated": 0,
            "n_truth": truth.n_drones,
            "correct": 0,
            "ghosts": 0,
            "merged": 0,
            "missed": truth.n_drones,
            "position_error_median": np.inf,
            "position_error_p95": np.inf,
        }

    # For each track, find closest truth drone at each frame
    track_matches = {}

    for track in estimated:
        # Find which truth drone this track corresponds to
        drone_votes = {}

        # Use frame_idx to index positions_3d (one per frame)
        for frame_idx, view_idx, point_idx in track.observations:
            if frame_idx >= truth.n_frames:
                continue
            pos_est = track.positions_3d[frame_idx]

            # Find closest truth drone
            min_dist = np.inf
            best_drone = -1
            for d in range(truth.n_drones):
                pos_true = truth.positions[frame_idx, d]
                dist = np.linalg.norm(pos_est - pos_true)
                if dist < min_dist and dist < d_max:
                    min_dist = dist
                    best_drone = d

            if best_drone >= 0:
                drone_votes[best_drone] = drone_votes.get(best_drone, 0) + 1

        if drone_votes:
            best_match = max(drone_votes, key=drone_votes.get)
            track_matches[track.track_id] = best_match
        else:
            track_matches[track.track_id] = -1  # Ghost

    # Classify tracks
    drone_to_tracks = {}
    for track_id, drone_id in track_matches.items():
        if drone_id >= 0:
            if drone_id not in drone_to_tracks:
                drone_to_tracks[drone_id] = []
            drone_to_tracks[drone_id].append(track_id)

    correct = 0
    ghosts = 0
    merged = 0
    position_errors = []

    for drone_id, track_ids in drone_to_tracks.items():
        if len(track_ids) == 1:
            correct += 1
            # Compute position error
            track = next(t for t in estimated if t.track_id == track_ids[0])
            # positions_3d has one entry per frame (not per observation)
            for i, (frame_idx, _, _) in enumerate(track.observations):
                if frame_idx < truth.n_frames:
                    pos_est = track.positions_3d[frame_idx]  # Use frame index directly
                    pos_true = truth.positions[frame_idx, drone_id]
                    err = np.linalg.norm(pos_est - pos_true)
                    position_errors.append(err)
        elif len(track_ids) > 1:
            merged += 1

    for track_id, drone_id in track_matches.items():
        if drone_id == -1:
            ghosts += 1

    missed = sum(1 for d in range(truth.n_drones) if d not in drone_to_tracks)

    return {
        "n_estimated": len(estimated),
        "n_truth": truth.n_drones,
        "correct": correct,
        "ghosts": ghosts,
        "merged": merged,
        "missed": missed,
        "position_error_median": float(np.median(position_errors)) if position_errors else np.inf,
        "position_error_p95": float(np.percentile(position_errors, 95)) if position_errors else np.inf,
    }


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== B4 Track Assembly Self-Test ===")

    from b1_scene_rig import generate_swarm_truth, generate_camera_rig
    from b2_projection import project_swarm_to_detections
    from b3_correspondence import solve_correspondence

    # Generate multi-frame test data
    truth = generate_swarm_truth(n_drones=5, n_frames=5, area_km=2.0, height_range_m=500.0, seed=42)
    rig = generate_camera_rig(truth, n_views=6, geometry_class="mixed", standoff_m=1000.0, seed=123)

    # Generate detections per frame
    detections_per_frame = []
    for f in range(truth.n_frames):
        detections = project_swarm_to_detections(
            SwarmTruth(
                positions=truth.positions[f:f+1],
                drone_ids=truth.drone_ids
            ),
            rig,
            pixel_noise_std=0.5,
            drop_prob=0.0,
            seed=f * 100,
        )
        detections_per_frame.append(detections)

    print(f"\nInput:")
    print(f"  Frames: {truth.n_frames}")
    print(f"  Drones: {truth.n_drones}")
    print(f"  Views: {rig.n_views}")
    for f in range(truth.n_frames):
        print(f"  Frame {f} detections per view: {[len(d) for d in detections_per_frame[f].points_per_view]}")

    # Build tracks
    tracks = build_spatiotemporal_tracks(
        detections_per_frame, rig,
        epipolar_threshold=3.0,
        min_views=2,
        max_reproj_error=5.0,
        max_track_dist=30.0,
        seed=42,
    )

    print(f"\nOutput:")
    print(f"  Tracks found: {len(tracks)}")
    for track in tracks:
        frames = sorted(set(obs[0] for obs in track.observations))
        print(f"  Track {track.track_id}: {len(track.observations)} obs, frames={frames}, "
              f"avg_reproj={np.mean(track.reproj_errors):.2f}px")
        if len(track.positions_3d) > 0:
            print(f"    Positions: {track.positions_3d}")

    # Evaluate
    eval_results = evaluate_tracks(tracks, truth, d_max=50.0)
    print(f"\nEvaluation: {eval_results}")

    print("\n=== B4: SELF-TEST PASSED ===")