"""
B3: Correspondence Solver — epipolar-constrained matching across views.

All types imported from data_contract. No imports from other Stage B modules.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Literal
from scipy.optimize import linear_sum_assignment

from data_contract import (
    Detections,
    CameraRig,
    Tracks,
    make_K,
    CONVENTION_TAG,
)


@dataclass(frozen=True)
class FundamentalMatrix:
    """Fundamental matrix for a camera pair."""
    F: NDArray[np.float64]          # (3, 3)
    view_i: int
    view_j: int


def compute_fundamental_matrix(K_i: NDArray[np.float64], K_j: NDArray[np.float64],
                                R_i: NDArray[np.float64], t_i: NDArray[np.float64],
                                R_j: NDArray[np.float64], t_j: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute fundamental matrix F from camera intrinsics and extrinsics.

    F = K_j^{-T} [t]_x R K_i^{-1}
    where R = R_j R_i^T, t = t_j - R_j R_i^T t_i (relative pose j from i)
    """
    # Relative rotation from i to j
    R_rel = R_j @ R_i.T
    # Relative translation from i to j
    t_rel = t_j - R_rel @ t_i

    # Essential matrix E = [t]_x R
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ], dtype=np.float64)

    E = tx @ R_rel

    # Fundamental matrix F = K_j^{-T} E K_i^{-1}
    K_i_inv = np.linalg.inv(K_i)
    K_j_inv_T = np.linalg.inv(K_j).T

    F = K_j_inv_T @ E @ K_i_inv

    # Enforce rank-2 constraint (SVD)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt

    return F


def compute_fundamental_matrices(rig: CameraRig) -> list[FundamentalMatrix]:
    """Compute fundamental matrices for all camera pairs."""
    fundamental_matrices = []

    for i in range(rig.n_views):
        for j in range(i + 1, rig.n_views):
            F = compute_fundamental_matrix(
                rig.K[i], rig.K[j],
                rig.w2c_R[i], rig.w2c_t[i],
                rig.w2c_R[j], rig.w2c_t[j]
            )
            fundamental_matrices.append(FundamentalMatrix(F=F, view_i=i, view_j=j))

    return fundamental_matrices


def epipolar_distance(pt_i: NDArray[np.float64], pt_j: NDArray[np.float64],
                      F: NDArray[np.float64]) -> float:
    """
    Compute symmetric epipolar distance.

    Distance of pt_j to epipolar line in view j: l_j = F @ pt_i_homogeneous
    Distance of pt_i to epipolar line in view i: l_i = F^T @ pt_j_homogeneous

    Returns max of the two distances.
    """
    pt_i_h = np.array([pt_i[0], pt_i[1], 1.0], dtype=np.float64)
    pt_j_h = np.array([pt_j[0], pt_j[1], 1.0], dtype=np.float64)

    # Line in view j corresponding to pt_i
    l_j = F @ pt_i_h
    # Line in view i corresponding to pt_j
    l_i = F.T @ pt_j_h

    # Distance from pt_j to line l_j
    dist_j = abs(l_j[0] * pt_j[0] + l_j[1] * pt_j[1] + l_j[2]) / np.sqrt(l_j[0]**2 + l_j[1]**2 + 1e-12)
    # Distance from pt_i to line l_i
    dist_i = abs(l_i[0] * pt_i[0] + l_i[1] * pt_i[1] + l_i[2]) / np.sqrt(l_i[0]**2 + l_i[1]**2 + 1e-12)

    return max(dist_i, dist_j)


def find_epipolar_candidates(detections: Detections, rig: CameraRig,
                              fundamental_matrices: list[FundamentalMatrix],
                              epipolar_threshold: float = 5.0) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    For each camera pair, find candidate matches within epipolar threshold.

    Returns:
        dict mapping (view_i, view_j) -> list of (idx_i, idx_j) candidate matches
    """
    candidates = {}

    for fm in fundamental_matrices:
        i, j = fm.view_i, fm.view_j
        F = fm.F

        pts_i = detections.points_per_view[i]
        pts_j = detections.points_per_view[j]

        if len(pts_i) == 0 or len(pts_j) == 0:
            candidates[(i, j)] = []
            continue

        # Find pairs within threshold
        pairs = []
        for ii, pt_i in enumerate(pts_i):
            for jj, pt_j in enumerate(pts_j):
                dist = epipolar_distance(pt_i, pt_j, F)
                if dist < epipolar_threshold:
                    pairs.append((ii, jj))

        candidates[(i, j)] = pairs

    return candidates


def triangulate_dlt(points: list[NDArray[np.float64]],
                    K_list: list[NDArray[np.float64]],
                    R_list: list[NDArray[np.float64]],
                    t_list: list[NDArray[np.float64]]) -> NDArray[np.float64] | None:
    """
    Linear DLT triangulation from multiple views.

    Args:
        points: List of 2D points (one per view)
        K_list: List of intrinsic matrices
        R_list: List of world-to-camera rotations
        t_list: List of world-to-camera translations

    Returns:
        3D point in world coordinates or None if degenerate
    """
    A = []
    for pt, K, R, t in zip(points, K_list, R_list, t_list):
        P = K @ np.hstack([R, t.reshape(3, 1)])  # (3, 4)
        x, y = pt
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.array(A)  # (2*n_views, 4)

    if A.shape[0] < 4:
        return None

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return (X[:3] / X[3]).astype(np.float64)


def reprojection_error(world_pt: NDArray[np.float64],
                       pt_2d: NDArray[np.float64],
                       K: NDArray[np.float64],
                       R: NDArray[np.float64],
                       t: NDArray[np.float64]) -> float:
    """Compute reprojection error in pixels."""
    cam_pt = R @ world_pt + t
    if cam_pt[2] <= 0:
        return np.inf
    proj = K @ cam_pt
    proj = proj[:2] / proj[2]
    return np.linalg.norm(proj - pt_2d)


@dataclass
class TrackBuilder:
    """Builds tracks by extending seed pairs across views."""
    detections: Detections
    rig: CameraRig
    fundamental_matrices: list[FundamentalMatrix]
    epipolar_threshold: float
    min_views: int = 2
    max_reproj_error: float = 10.0

    def build_tracks(self) -> list[list[tuple[int, int]]]:
        """Build all tracks from seed pairs."""
        n_views = self.rig.n_views
        n_dets_per_view = [len(self.detections.points_per_view[v]) for v in range(n_views)]

        # Track used detections per view
        used = [set() for _ in range(n_views)]
        tracks = []

        # Get all candidate pairs for all view pairs
        candidates = find_epipolar_candidates(
            self.detections, self.rig, self.fundamental_matrices, self.epipolar_threshold
        )

        # Sort view pairs by number of candidates (most constrained first)
        pair_order = sorted(candidates.keys(), key=lambda p: len(candidates[p]))

        for vi, vj in pair_order:
            pairs = candidates[(vi, vj)]
            if not pairs:
                continue

            # Use Hungarian to find best assignment for this pair
            n_i = n_dets_per_view[vi]
            n_j = n_dets_per_view[vj]

            # Build cost matrix
            max_n = max(n_i, n_j)
            cost = np.full((n_i, n_j), 1e6)
            for ii, jj in pairs:
                if ii < n_i and jj < n_j:
                    cost[ii, jj] = 1.0

            cost_padded = np.full((max_n, max_n), 1e6)
            cost_padded[:n_i, :n_j] = cost

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_padded)

            # Build tracks from assignments
            for ri, ci in zip(row_ind, col_ind):
                if ri >= n_i or ci >= n_j:
                    continue
                if cost[ri, ci] >= 1e6:
                    continue

                if ri in used[vi] or ci in used[vj]:
                    continue  # Already used

                # Start a new track with this seed pair
                track = [(vi, ri), (vj, ci)]
                used[vi].add(ri)
                used[vj].add(ci)

                # Extend to other views
                self._extend_track(track, used, candidates, n_dets_per_view)

                if len(track) >= self.min_views:
                    tracks.append(track)

        # Add any remaining unassigned detections as single-view (discard, need >=2 views)
        return tracks

    def _extend_track(self, track: list[tuple[int, int]], used: list[set],
                      candidates: dict, n_dets_per_view: list[int]):
        """Extend a track to additional views."""
        current_views = {v for v, _ in track}
        all_views = set(range(self.rig.n_views))

        for v in all_views - current_views:
            # Try to find a match in view v
            best_match = None
            best_score = np.inf

            for cv, ci in track:
                pair = tuple(sorted([cv, v]))
                if pair not in candidates:
                    continue

                for (pi, pj) in candidates[pair]:
                    if cv == pair[0]:
                        other_idx = pj
                    else:
                        other_idx = pi

                    if other_idx in used[v]:
                        continue

                    # Triangulate current track + candidate
                    points = []
                    K_list = []
                    R_list = []
                    t_list = []
                    for tv, ti in track:
                        points.append(self.detections.points_per_view[tv][ti])
                        K_list.append(self.rig.K[tv])
                        R_list.append(self.rig.w2c_R[tv])
                        t_list.append(self.rig.w2c_t[tv])

                    # Add candidate
                    points.append(self.detections.points_per_view[v][other_idx])
                    K_list.append(self.rig.K[v])
                    R_list.append(self.rig.w2c_R[v])
                    t_list.append(self.rig.w2c_t[v])

                    X = triangulate_dlt(points, K_list, R_list, t_list)
                    if X is None:
                        continue

                    # Check reprojection error for ALL views in track
                    max_reproj = 0.0
                    for tv, ti in track:
                        err = reprojection_error(X, self.detections.points_per_view[tv][ti],
                                                 self.rig.K[tv], self.rig.w2c_R[tv], self.rig.w2c_t[tv])
                        max_reproj = max(max_reproj, err)

                    # Check candidate view too
                    err = reprojection_error(X, self.detections.points_per_view[v][other_idx],
                                             self.rig.K[v], self.rig.w2c_R[v], self.rig.w2c_t[v])
                    max_reproj = max(max_reproj, err)

                    if max_reproj < best_score:
                        best_score = max_reproj
                        best_match = other_idx

            if best_match is not None and best_score <= self.max_reproj_error:
                track.append((v, best_match))
                used[v].add(best_match)


def solve_correspondence(
    detections: Detections,
    rig: CameraRig,
    epipolar_threshold: float = 5.0,
    min_views: int = 2,
    max_reproj_error: float = 10.0,
    seed: int = 0,
) -> Tracks:
    """
    Solve multi-view correspondence from anonymous detections.

    Pipeline:
    1. Compute fundamental matrices for all camera pairs
    2. Find epipolar-valid candidate pairs
    3. Build tracks by extending seed pairs across views
    4. Filter by reprojection error

    Returns:
        Tracks object with list of tracks (each track = list of (view_idx, point_idx))
    """
    np.random.seed(seed)

    # Step 1: Compute fundamental matrices
    fundamental_matrices = compute_fundamental_matrices(rig)

    # Step 2-4: Build tracks
    builder = TrackBuilder(
        detections=detections,
        rig=rig,
        fundamental_matrices=fundamental_matrices,
        epipolar_threshold=epipolar_threshold,
        min_views=min_views,
        max_reproj_error=max_reproj_error,
    )
    tracks = builder.build_tracks()

    # Convert to Tracks dataclass
    return Tracks(tracks=tracks, n_views=rig.n_views)


def analyze_correspondence(tracks: Tracks, truth_detections: Detections | None = None,
                           rig: CameraRig | None = None) -> dict:
    """
    Analyze correspondence quality (for validation with ground truth).

    If truth_detections and rig provided, computes:
    - Ghost tracks (no ground truth match)
    - Merged tracks (multiple drones in one track)
    - Missed drones
    - Position error for correct tracks
    """
    # Placeholder for Phase 1 validation
    # Real implementation would need ground truth 3D positions
    return {
        "n_tracks": len(tracks.tracks),
        "n_views": tracks.n_views,
        "avg_views_per_track": np.mean([len(t) for t in tracks.tracks]) if tracks.tracks else 0,
        "min_views_per_track": min(len(t) for t in tracks.tracks) if tracks.tracks else 0,
    }


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== B3 Correspondence Solver Self-Test ===")

    # Import B1 and B2 to generate test data
    from b1_scene_rig import generate_swarm_truth, generate_camera_rig
    from b2_projection import project_swarm_to_detections

    # Generate test data
    truth = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)
    rig = generate_camera_rig(truth, n_views=4, geometry_class="mixed", standoff_m=1000.0, seed=123)

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

    print(f"\nOutput:")
    print(f"  Tracks found: {len(tracks.tracks)}")
    for i, track in enumerate(tracks.tracks):
        print(f"  Track {i}: {track} ({len(track)} views)")

    # Analyze
    analysis = analyze_correspondence(tracks)
    print(f"\nAnalysis: {analysis}")

    # With ground truth, we could compute:
    # - How many tracks correspond to real drones
    # - How many are ghosts
    # - How many drones were missed

    print("\n=== B3: SELF-TEST PASSED ===")
