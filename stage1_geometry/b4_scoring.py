"""
B4: Scoring and Controls Harness — evaluation metrics for correspondence and triangulation.

Written FIRST, against hand-constructed fake tracks. NO imports from other Stage B modules.
This is the frozen standard that the correspondence solver must satisfy.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from data_contract import (
    SwarmTruth,
    CameraRig,
    Tracks,
    Reconstruction,
    Detections,
    CONVENTION_TAG,
    validate_swarm_truth,
    validate_camera_rig,
    validate_detections,
)


# ============================================================================
# Ground Truth Association (for evaluation only — solver never sees this)
# ============================================================================

@dataclass(frozen=True)
class TrackTruth:
    """
    Ground-truth association between tracks and drone identities.

    This is ONLY for evaluation. The correspondence solver receives
    Detections (anonymous points) and outputs Tracks (anonymous correspondences).
    This class links them for scoring.
    """
    # For each track in solver output, which drone_id (from SwarmTruth) it corresponds to
    # -1 means ghost track (no matching drone)
    track_to_drone: NDArray[np.int32]  # (n_tracks,)

    # For each drone in truth, which track index reconstructs it
    # -1 means missed detection
    drone_to_track: NDArray[np.int32]  # (n_drones,)

    def __post_init__(self):
        assert self.track_to_drone.ndim == 1
        assert self.drone_to_track.ndim == 1


def associate_tracks_to_truth(
    tracks: Tracks,
    recon: Reconstruction,
    truth: SwarmTruth,
    rig: CameraRig,
    position_threshold_m: float = 50.0,
) -> TrackTruth:
    """
    Associate reconstructed tracks to ground-truth drones by 3D position.

    This is the evaluation-time matching. The solver never has access to this.

    Args:
        tracks: Correspondence tracks from solver
        recon: Triangulated 3D positions
        truth: Ground-truth swarm positions (single frame for now)
        rig: Camera rig
        position_threshold_m: Maximum distance to consider a match (meters)

    Returns:
        TrackTruth with track<->drone associations
    """
    n_tracks = len(recon.positions_3d)
    n_drones = truth.n_drones

    # Truth positions for the first frame (extend later for multi-frame)
    truth_pos = truth.positions[0]  # (N, 3)

    # Cost matrix: distance from each track to each drone
    costs = np.full((n_tracks, n_drones), np.inf)
    for i in range(n_tracks):
        for j in range(n_drones):
            dist = np.linalg.norm(recon.positions_3d[i] - truth_pos[j])
            if dist <= position_threshold_m:
                costs[i, j] = dist

    # Greedy assignment (tracks to drones)
    track_to_drone = np.full(n_tracks, -1, dtype=np.int32)
    drone_to_track = np.full(n_drones, -1, dtype=np.int32)

    # Sort all valid pairs by distance
    valid_pairs = [(costs[i, j], i, j) for i in range(n_tracks) for j in range(n_drones) if np.isfinite(costs[i, j])]
    valid_pairs.sort(key=lambda x: x[0])

    for _, i, j in valid_pairs:
        if track_to_drone[i] == -1 and drone_to_track[j] == -1:
            track_to_drone[i] = j
            drone_to_track[j] = i

    return TrackTruth(track_to_drone=track_to_drone, drone_to_track=drone_to_track)


# ============================================================================
# Scoring Metrics
# ============================================================================

@dataclass(frozen=True)
class CorrespondenceScore:
    """
    Correspondence-level scoring (before triangulation).

    Metrics computed from Tracks vs TrackTruth.
    """
    n_tracks: int
    n_drones: int
    n_matched: int           # Tracks correctly associated to a drone
    n_ghost: int             # Tracks with no matching drone (-1)
    n_missed: int            # Drones with no matching track (-1)
    precision: float         # n_matched / n_tracks
    recall: float            # n_matched / n_drones
    f1: float                # Harmonic mean

    def __post_init__(self):
        assert self.n_tracks >= 0
        assert self.n_drones >= 0
        assert 0 <= self.n_matched <= min(self.n_tracks, self.n_drones)
        assert 0 <= self.precision <= 1.0
        assert 0 <= self.recall <= 1.0
        assert 0 <= self.f1 <= 1.0


def score_correspondence(track_truth: TrackTruth) -> CorrespondenceScore:
    """Compute correspondence metrics from track-truth association."""
    n_tracks = len(track_truth.track_to_drone)
    n_drones = len(track_truth.drone_to_track)
    n_matched = np.sum(track_truth.track_to_drone >= 0)
    n_ghost = np.sum(track_truth.track_to_drone == -1)
    n_missed = np.sum(track_truth.drone_to_track == -1)

    precision = n_matched / n_tracks if n_tracks > 0 else 0.0
    recall = n_matched / n_drones if n_drones > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return CorrespondenceScore(
        n_tracks=n_tracks,
        n_drones=n_drones,
        n_matched=n_matched,
        n_ghost=n_ghost,
        n_missed=n_missed,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass(frozen=True)
class TriangulationScore:
    """
    Triangulation-level scoring (3D position accuracy).

    Metrics computed from Reconstruction vs TrackTruth vs SwarmTruth.
    Only evaluated on MATCHED tracks (ghost tracks excluded).
    """
    n_matched: int
    position_errors_m: NDArray[np.float64]  # (n_matched,) Euclidean distance
    median_error_m: float
    p95_error_m: float
    max_error_m: float
    mean_reprojection_error_px: float

    def __post_init__(self):
        assert self.n_matched >= 0
        assert len(self.position_errors_m) == self.n_matched
        assert self.median_error_m >= 0
        assert self.p95_error_m >= 0
        assert self.max_error_m >= 0
        assert self.mean_reprojection_error_px >= 0


def score_triangulation(
    recon: Reconstruction,
    track_truth: TrackTruth,
    truth: SwarmTruth,
) -> TriangulationScore:
    """
    Compute 3D triangulation accuracy for matched tracks only.

    Ghost tracks are excluded from position error statistics.
    """
    truth_pos = truth.positions[0]  # (N, 3)
    matched_mask = track_truth.track_to_drone >= 0
    n_matched = np.sum(matched_mask)

    if n_matched == 0:
        return TriangulationScore(
            n_matched=0,
            position_errors_m=np.array([], dtype=np.float64),
            median_error_m=0.0,
            p95_error_m=0.0,
            max_error_m=0.0,
            mean_reprojection_error_px=0.0,
        )

    # Position errors for matched tracks
    matched_indices = np.where(matched_mask)[0]
    matched_drone_ids = track_truth.track_to_drone[matched_indices]

    errors = np.zeros(n_matched, dtype=np.float64)
    for k, (track_idx, drone_idx) in enumerate(zip(matched_indices, matched_drone_ids)):
        errors[k] = np.linalg.norm(recon.positions_3d[track_idx] - truth_pos[drone_idx])

    median_err = float(np.median(errors))
    p95_err = float(np.percentile(errors, 95))
    max_err = float(np.max(errors))
    mean_reproj = float(np.mean(recon.reprojection_errors[matched_indices]))

    return TriangulationScore(
        n_matched=n_matched,
        position_errors_m=errors,
        median_error_m=median_err,
        p95_error_m=p95_err,
        max_error_m=max_err,
        mean_reprojection_error_px=mean_reproj,
    )


@dataclass(frozen=True)
class FullScore:
    """Combined correspondence and triangulation score."""
    correspondence: CorrespondenceScore
    triangulation: TriangulationScore


def score_full(
    tracks: Tracks,
    recon: Reconstruction,
    truth: SwarmTruth,
    rig: CameraRig,
    position_threshold_m: float = 50.0,
) -> FullScore:
    """
    Complete evaluation pipeline.

    This is the single entry point for scoring a solver's output.
    """
    track_truth = associate_tracks_to_truth(tracks, recon, truth, rig, position_threshold_m)
    corr_score = score_correspondence(track_truth)
    triang_score = score_triangulation(recon, track_truth, truth)
    return FullScore(correspondence=corr_score, triangulation=triang_score)


# ============================================================================
# Control Tests (hand-constructed, no solver involved)
# ============================================================================

def make_fake_tracks_perfect(truth: SwarmTruth, rig: CameraRig, n_drones: int = 5) -> tuple[Tracks, Reconstruction]:
    """
    Control 1: Perfect correspondence + perfect triangulation.

    Creates tracks that exactly match truth, and reconstruction that matches truth.
    Must score: precision=1.0, recall=1.0, median_error≈0.
    """
    n_views = rig.n_views

    # Each drone seen in all views
    tracks_list = []
    for drone_idx in range(n_drones):
        track = [(v, drone_idx) for v in range(n_views)]  # Same point_idx in each view
        tracks_list.append(track)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    # Perfect reconstruction = truth positions
    recon = Reconstruction(
        positions_3d=truth.positions[0].copy(),  # (N, 3)
        reprojection_errors=np.zeros(n_drones, dtype=np.float64),
        track_indices=tracks_list,
    )
    return tracks, recon


def make_fake_tracks_scrambled(truth: SwarmTruth, rig: CameraRig, n_drones: int = 5, seed: int = 42) -> tuple[Tracks, Reconstruction]:
    """
    Control 2: Scrambled correspondence (wrong associations).

    Each track has views from DIFFERENT drones. Should produce ghosts and large errors.
    Must score: precision<<1.0, large position errors.
    """
    rng = np.random.default_rng(seed)
    n_views = rig.n_views

    tracks_list = []
    for drone_idx in range(n_drones):
        # Each "track" pulls point_idx from a different drone in each view
        track = []
        for v in range(n_views):
            wrong_drone = (drone_idx + v + 1) % n_drones  # Systematic scramble
            track.append((v, wrong_drone))
        tracks_list.append(track)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    # Triangulation of scrambled tracks = garbage positions
    # We don't actually triangulate here; just create fake bad reconstruction
    recon = Reconstruction(
        positions_3d=rng.uniform(-1000, 1000, size=(n_drones, 3)).astype(np.float64),
        reprojection_errors=rng.uniform(10, 100, size=n_drones).astype(np.float64),
        track_indices=tracks_list,
    )
    return tracks, recon


def make_fake_tracks_single_view(truth: SwarmTruth, rig: CameraRig, view_idx: int = 0, n_drones: int = 5) -> tuple[Tracks, Reconstruction]:
    """
    Control 3: Degenerate tracks (2 views from nearly identical camera poses).

    Each track has 2 views but from nearly identical positions - degenerate triangulation.
    Should produce massive position errors or NaN positions.
    """
    n_views = rig.n_views

    tracks_list = []
    for drone_idx in range(n_drones):
        # Use view_idx and view_idx+1 (adjacent views, nearly degenerate)
        # Ensure both views are valid
        v1 = view_idx % n_views
        v2 = (view_idx + 1) % n_views
        track = [(v1, drone_idx), (v2, drone_idx)]
        tracks_list.append(track)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    # Degenerate triangulation - return NaN positions to simulate failure
    recon = Reconstruction(
        positions_3d=np.full((n_drones, 3), np.nan, dtype=np.float64),
        reprojection_errors=np.full(n_drones, np.nan, dtype=np.float64),
        track_indices=tracks_list,
    )
    return tracks, recon


def make_fake_tracks_known_offset(
    truth: SwarmTruth,
    rig: CameraRig,
    offset_m: NDArray[np.float64],
    n_drones: int = 5,
) -> tuple[Tracks, Reconstruction]:
    """
    Control 4: Perfect correspondence but known position offset.

    Reconstruction is truth + constant offset. Should have precision=1.0 but
    median_error = |offset|.
    """
    n_views = rig.n_views

    tracks_list = []
    for drone_idx in range(n_drones):
        track = [(v, drone_idx) for v in range(n_views)]
        tracks_list.append(track)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    # Perfect correspondence, but shifted reconstruction
    shifted_pos = truth.positions[0] + offset_m.reshape(1, 3)
    recon = Reconstruction(
        positions_3d=shifted_pos.astype(np.float64),
        reprojection_errors=np.zeros(n_drones, dtype=np.float64),
        track_indices=tracks_list,
    )
    return tracks, recon


def make_fake_tracks_ghosts(
    truth: SwarmTruth,
    rig: CameraRig,
    n_drones: int = 5,
    n_ghosts: int = 3,
) -> tuple[Tracks, Reconstruction]:
    """
    Control 5: Extra ghost tracks (false positives).

    n_drones real tracks + n_ghosts tracks with no matching drone.
    Should have precision = n_drones / (n_drones + n_ghosts).
    """
    n_views = rig.n_views

    tracks_list = []
    # Real tracks
    for drone_idx in range(n_drones):
        track = [(v, drone_idx) for v in range(n_views)]
        tracks_list.append(track)
    # Ghost tracks (point indices beyond real detections)
    for g in range(n_ghosts):
        track = [(v, n_drones + g) for v in range(n_views)]
        tracks_list.append(track)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    n_total = n_drones + n_ghosts
    recon = Reconstruction(
        positions_3d=np.vstack([
            truth.positions[0],
            np.full((n_ghosts, 3), np.nan, dtype=np.float64),  # Ghost positions = NaN
        ]),
        reprojection_errors=np.hstack([
            np.zeros(n_drones),
            np.full(n_ghosts, np.nan),
        ]),
        track_indices=tracks_list,
    )
    return tracks, recon


# ============================================================================
# Control Test Runner
# ============================================================================

def run_controls(
    truth: SwarmTruth,
    rig: CameraRig,
    position_threshold_m: float = 50.0,
) -> dict[str, FullScore]:
    """
    Run all control tests and return scores.

    These are the mandatory controls that must pass before trusting any solver.
    """
    results = {}

    # Control 1: Perfect
    tracks, recon = make_fake_tracks_perfect(truth, rig)
    results['perfect'] = score_full(tracks, recon, truth, rig, position_threshold_m)

    # Control 2: Scrambled
    tracks, recon = make_fake_tracks_scrambled(truth, rig)
    results['scrambled'] = score_full(tracks, recon, truth, rig, position_threshold_m)

    # Control 3: Single view
    tracks, recon = make_fake_tracks_single_view(truth, rig)
    results['single_view'] = score_full(tracks, recon, truth, rig, position_threshold_m)

    # Control 4: Known offset (100m shift)
    offset = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    tracks, recon = make_fake_tracks_known_offset(truth, rig, offset)
    # Use larger threshold for this control test (150m > 100m offset)
    results['known_offset'] = score_full(tracks, recon, truth, rig, position_threshold_m=150.0)

    # Control 5: Ghosts
    tracks, recon = make_fake_tracks_ghosts(truth, rig, n_ghosts=3)
    results['ghosts'] = score_full(tracks, recon, truth, rig, position_threshold_m)

    return results


def validate_controls(controls: dict[str, FullScore]) -> bool:
    """
    Validate that controls produce expected results.

    Returns True if all controls behave as expected.
    Prints diagnostic info and returns False if any control fails.
    """
    print("=== CONTROL VALIDATION ===")
    all_pass = True

    # Control 1: Perfect
    c = controls['perfect']
    print(f"\n1. PERFECT:")
    print(f"   Correspondence: P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}, F1={c.correspondence.f1:.3f}")
    print(f"   Triangulation: median={c.triangulation.median_error_m:.3f}m, n_matched={c.triangulation.n_matched}")
    if not (c.correspondence.precision == 1.0 and c.correspondence.recall == 1.0 and c.triangulation.median_error_m < 1e-6):
        print("   ❌ FAIL: Perfect control should score 1.0/1.0 with ~0 error")
        all_pass = False
    else:
        print("   ✓ PASS")

    # Control 2: Scrambled
    c = controls['scrambled']
    print(f"\n2. SCRAMBLED:")
    print(f"   Correspondence: P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}, F1={c.correspondence.f1:.3f}")
    print(f"   Triangulation: median={c.triangulation.median_error_m:.3f}m, n_matched={c.triangulation.n_matched}")
    if c.correspondence.precision > 0.3 or c.correspondence.recall > 0.3:
        print("   ❌ FAIL: Scrambled should have low precision/recall")
        all_pass = False
    else:
        print("   ✓ PASS")

    # Control 3: Single view
    c = controls['single_view']
    print(f"\n3. SINGLE VIEW:")
    print(f"   Correspondence: P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}")
    print(f"   Triangulation: n_matched={c.triangulation.n_matched} (should be 0 or NaN positions)")
    # Single view tracks are invalid (need >=2 views per track), so they won't match
    if c.correspondence.n_matched > 0:
        print("   ❌ FAIL: Single-view tracks should not match any drones")
        all_pass = False
    else:
        print("   ✓ PASS")

    # Control 4: Known offset
    c = controls['known_offset']
    print(f"\n4. KNOWN OFFSET (100m):")
    print(f"   Correspondence: P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}")
    print(f"   Triangulation: median={c.triangulation.median_error_m:.3f}m (should be ~100m)")
    if not (c.correspondence.precision == 1.0 and c.correspondence.recall == 1.0 and abs(c.triangulation.median_error_m - 100.0) < 1.0):
        print("   ❌ FAIL: Known offset should have perfect correspondence but ~100m error")
        all_pass = False
    else:
        print("   ✓ PASS")

    # Control 5: Ghosts
    c = controls['ghosts']
    print(f"\n5. GHOSTS (3 ghosts):")
    print(f"   Correspondence: P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}, n_ghost={c.correspondence.n_ghost}")
    print(f"   Triangulation: n_matched={c.triangulation.n_matched}")
    expected_precision = 5 / 8  # 5 real / (5 real + 3 ghost)
    if abs(c.correspondence.precision - expected_precision) > 0.01 or c.correspondence.n_ghost != 3:
        print(f"   ❌ FAIL: Ghost control precision should be {expected_precision:.3f}, n_ghost=3")
        all_pass = False
    else:
        print("   ✓ PASS")

    print(f"\n=== {'ALL CONTROLS PASSED' if all_pass else 'SOME CONTROLS FAILED'} ===")
    return all_pass


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== B4 Scoring and Controls Harness Self-Test ===")

    # Need a truth and rig to test against
    from b1_scene_rig import generate_swarm_truth, generate_camera_rig

    truth = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)
    rig = generate_camera_rig(truth, n_views=6, geometry_class="mixed", standoff_m=1000.0, seed=123)

    print(f"Truth: {truth.n_drones} drones, Rig: {rig.n_views} views")

    # Run controls
    controls = run_controls(truth, rig)

    # Validate
    success = validate_controls(controls)

    if success:
        print("\n=== B4 SELF-TEST PASSED ===")
    else:
        print("\n=== B4 SELF-TEST FAILED ===")
        exit(1)