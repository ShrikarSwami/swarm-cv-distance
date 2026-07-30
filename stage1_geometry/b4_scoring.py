"""
B4: Scoring and Controls Harness — evaluation metrics for correspondence and triangulation.

Written FIRST, against hand-constructed fake tracks. NO imports from other Stage B modules.
This is the frozen standard that the correspondence solver must satisfy.

Classification (2026-07-30 unfreeze):
    Every track is classified into one of four categories:
    - matched:       all detections from one drone, position within threshold
    - imprecise:     all detections from one drone, position beyond threshold
    - false_track:   detections span two or more different drones
    - missed:        a true drone with no track

    Correspondence accuracy = (n_matched + n_imprecise) / n_tracks
    (fraction of tracks where every member detection belongs to the same true drone)
    Position error is computed over matched tracks only.
    n_ghost (backward compat) = n_imprecise + n_false_track.
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
    IMAGE_SIZE,
    CONVENTION_TAG,
    validate_swarm_truth,
    validate_camera_rig,
    validate_detections,
)


# ============================================================================
# Ground Truth Association (for evaluation only — solver never sees this)
# ============================================================================

def compute_detection_drone_ids(
    truth: SwarmTruth, rig: CameraRig
) -> list[NDArray[np.int32]]:
    """
    Compute ground-truth drone identity for each detection point_idx.

    Projects truth positions through the rig (no noise, no drop) and returns
    the drone_id for each detection in each view. The mapping is:
        detection_drone_ids[v][point_idx] = drone_id

    This is for evaluation only — the solver never has access to this.
    For rendered runs, the Object Index EXR pass provides equivalent information.
    """
    n_views = rig.n_views
    width, height = IMAGE_SIZE

    detection_drone_ids: list[NDArray[np.int32]] = []
    for v in range(n_views):
        K = rig.K[v]
        R_w2c = rig.w2c_R[v]
        t_w2c = rig.w2c_t[v]

        view_map: list[int] = []
        for f in range(truth.n_frames):
            world_pts = truth.positions[f]  # (N, 3)
            # Project: cam_pt = R @ world + t, pixel = K @ cam_pt / cam_pt[2]
            cam_pts = (R_w2c @ world_pts.T + t_w2c.reshape(3, 1))  # (3, N)
            depths = cam_pts[2, :]
            homogeneous = K @ cam_pts  # (3, N)
            pixels = homogeneous[:2, :] / homogeneous[2:3, :]  # (2, N)

            valid = (depths > 0) & \
                    (pixels[0, :] >= 0) & (pixels[0, :] < width) & \
                    (pixels[1, :] >= 0) & (pixels[1, :] < height)

            for di in np.where(valid)[0]:
                view_map.append(int(di))

        detection_drone_ids.append(np.array(view_map, dtype=np.int32))

    return detection_drone_ids


@dataclass(frozen=True)
class TrackTruth:
    """
    Ground-truth association between tracks and drone identities.

    This is ONLY for evaluation. The correspondence solver receives
    Detections (anonymous points) and outputs Tracks (anonymous correspondences).
    This class links them for scoring.

    Classification (four categories):
        matched:     all detections from one drone, position within threshold
        imprecise:   all detections from one drone, position beyond threshold
        false_track: detections span two or more different drones (or phantom)
        missed:      a true drone with no track
    """
    # For each track in solver output, which drone_id (from SwarmTruth) it corresponds to
    # -1 means no matching drone (false_track or unmatched)
    track_to_drone: NDArray[np.int32]  # (n_tracks,)

    # For each drone in truth, which track index reconstructs it
    # -1 means missed detection
    drone_to_track: NDArray[np.int32]  # (n_drones,)

    # Per-track classification flags (when detection_drone_ids provided)
    track_is_imprecise: NDArray[np.bool_]    # (n_tracks,) True = imprecise
    track_is_false_track: NDArray[np.bool_]  # (n_tracks,) True = false_track

    def __post_init__(self):
        assert self.track_to_drone.ndim == 1
        assert self.drone_to_track.ndim == 1
        assert self.track_is_imprecise.ndim == 1
        assert self.track_is_false_track.ndim == 1

    @property
    def n_matched(self) -> int:
        """Tracks with all detections from one drone, position within threshold."""
        return int(np.sum((self.track_to_drone >= 0) & ~self.track_is_imprecise & ~self.track_is_false_track))

    @property
    def n_imprecise(self) -> int:
        """Tracks with all detections from one drone, position beyond threshold."""
        return int(np.sum(self.track_is_imprecise))

    @property
    def n_false_track(self) -> int:
        """Tracks with detections from multiple drones, or phantom tracks."""
        return int(np.sum(self.track_is_false_track))

    @property
    def n_ghost(self) -> int:
        """Backward compat: imprecise + false_track, or unmatched tracks if no classification."""
        classified = np.any(self.track_is_imprecise | self.track_is_false_track)
        if classified:
            return self.n_imprecise + self.n_false_track
        else:
            # No classification provided — count unmatched tracks (old behavior)
            return int(np.sum(self.track_to_drone == -1))

    @property
    def n_missed(self) -> int:
        """Truth drones with no track."""
        return int(np.sum(self.drone_to_track == -1))


def associate_tracks_to_truth(
    tracks: Tracks,
    recon: Reconstruction,
    truth: SwarmTruth,
    rig: CameraRig,
    position_threshold_m: float = 50.0,
    detection_drone_ids: list[NDArray[np.int32]] | None = None,
) -> TrackTruth:
    """
    Associate reconstructed tracks to ground-truth drones and classify each track.

    When detection_drone_ids is provided, classification is by detection identity:
        all detections from same drone + position within threshold → matched
        all detections from same drone + position beyond threshold → imprecise
        detections from multiple drones → false_track

    When detection_drone_ids is None, falls back to position-based matching
    (old behavior). Classification flags are set to False (unknown).

    This is the evaluation-time matching. The solver never has access to this.

    Args:
        tracks: Correspondence tracks from solver
        recon: Triangulated 3D positions
        truth: Ground-truth swarm positions (single frame for now)
        rig: Camera rig
        position_threshold_m: Maximum distance to consider a match (meters)
        detection_drone_ids: Optional per-view drone identity mapping.
            detection_drone_ids[v][point_idx] = drone_id.
            Computed by compute_detection_drone_ids(truth, rig).

    Returns:
        TrackTruth with track<->drone associations and classification
    """
    n_tracks = len(recon.positions_3d)
    n_drones = truth.n_drones

    # Truth positions for the first frame (extend later for multi-frame)
    truth_pos = truth.positions[0]  # (N, 3)

    track_to_drone = np.full(n_tracks, -1, dtype=np.int32)
    drone_to_track = np.full(n_drones, -1, dtype=np.int32)
    track_is_imprecise = np.zeros(n_tracks, dtype=np.bool_)
    track_is_false_track = np.zeros(n_tracks, dtype=np.bool_)

    if detection_drone_ids is not None:
        # ---- Classification by detection identity ----
        # For each track, check if all detections map to the same drone.
        for t_idx, track in enumerate(tracks.tracks):
            drone_ids_in_track: set[int] = set()
            for view_idx, point_idx in track:
                if view_idx < len(detection_drone_ids) and \
                   point_idx < len(detection_drone_ids[view_idx]):
                    drone_id = int(detection_drone_ids[view_idx][point_idx])
                    drone_ids_in_track.add(drone_id)
                else:
                    # Out-of-range point_idx → phantom detection
                    drone_ids_in_track.add(-2)

            if len(drone_ids_in_track) == 1:
                drone_id = drone_ids_in_track.pop()
                if drone_id < 0:
                    # Phantom detection (point_idx out of range)
                    track_is_false_track[t_idx] = True
                else:
                    # All detections from same drone — check position
                    dist = np.linalg.norm(recon.positions_3d[t_idx] - truth_pos[drone_id])
                    if dist <= position_threshold_m:
                        # matched (default False for imprecise)
                        pass
                    else:
                        track_is_imprecise[t_idx] = True
                    # Assign track to drone (first-come or closer)
                    if drone_to_track[drone_id] == -1:
                        track_to_drone[t_idx] = drone_id
                        drone_to_track[drone_id] = t_idx
                    else:
                        # Drone already assigned — keep the closer one
                        existing = drone_to_track[drone_id]
                        existing_dist = np.linalg.norm(
                            recon.positions_3d[existing] - truth_pos[drone_id])
                        new_dist = dist
                        if new_dist < existing_dist:
                            # Replace existing assignment
                            track_to_drone[existing] = -1
                            track_is_imprecise[existing] = False
                            track_is_false_track[existing] = True
                            track_to_drone[t_idx] = drone_id
                            drone_to_track[drone_id] = t_idx
            else:
                # Multiple drones or phantom → false_track
                track_is_false_track[t_idx] = True
    else:
        # ---- Old behavior: position-based matching (no classification) ----
        costs = np.full((n_tracks, n_drones), np.inf)
        for i in range(n_tracks):
            for j in range(n_drones):
                dist = np.linalg.norm(recon.positions_3d[i] - truth_pos[j])
                if dist <= position_threshold_m:
                    costs[i, j] = dist

        valid_pairs = [(costs[i, j], i, j)
                       for i in range(n_tracks) for j in range(n_drones)
                       if np.isfinite(costs[i, j])]
        valid_pairs.sort(key=lambda x: x[0])

        for _, i, j in valid_pairs:
            if track_to_drone[i] == -1 and drone_to_track[j] == -1:
                track_to_drone[i] = j
                drone_to_track[j] = i

    return TrackTruth(
        track_to_drone=track_to_drone,
        drone_to_track=drone_to_track,
        track_is_imprecise=track_is_imprecise,
        track_is_false_track=track_is_false_track,
    )


# ============================================================================
# Scoring Metrics
# ============================================================================

@dataclass(frozen=True)
class CorrespondenceScore:
    """
    Correspondence-level scoring (before triangulation).

    Four categories:
        matched:     all detections from one drone, position within threshold
        imprecise:   all detections from one drone, position beyond threshold
        false_track: detections span two or more different drones (or phantom)
        missed:      a true drone with no track

    Correspondence accuracy = (n_matched + n_imprecise) / n_tracks
    (fraction of tracks where every member detection belongs to the same true drone)
    """
    n_tracks: int
    n_drones: int
    n_matched: int           # All detections from one drone, position within threshold
    n_imprecise: int         # All detections from one drone, position beyond threshold
    n_false_track: int       # Detections from multiple drones, or phantom
    n_ghost: int             # = n_imprecise + n_false_track (backward compat)
    n_missed: int            # Drones with no matching track (-1)
    precision: float         # (n_matched + n_imprecise) / n_tracks — correspondence accuracy
    recall: float            # (n_matched + n_imprecise) / n_drones
    f1: float                # Harmonic mean

    def __post_init__(self):
        assert self.n_tracks >= 0
        assert self.n_drones >= 0
        assert 0 <= self.n_matched <= self.n_tracks
        assert 0 <= self.n_imprecise <= self.n_tracks
        assert 0 <= self.n_false_track <= self.n_tracks
        # n_ghost may differ from imprecise+false_track when classification
        # is not provided (backward compat: n_ghost = unmatched tracks)
        assert self.n_ghost >= 0
        assert 0 <= self.precision <= 1.0
        assert 0 <= self.recall <= 1.0
        assert 0 <= self.f1 <= 1.0


def score_correspondence(track_truth: TrackTruth) -> CorrespondenceScore:
    """Compute correspondence metrics from track-truth association."""
    n_tracks = len(track_truth.track_to_drone)
    n_drones = len(track_truth.drone_to_track)
    n_matched = track_truth.n_matched
    n_imprecise = track_truth.n_imprecise
    n_false_track = track_truth.n_false_track
    n_ghost = track_truth.n_ghost
    n_missed = track_truth.n_missed

    # Correspondence accuracy: fraction of tracks where all detections
    # belong to the same true drone (regardless of position accuracy)
    precision = (n_matched + n_imprecise) / n_tracks if n_tracks > 0 else 0.0
    recall = (n_matched + n_imprecise) / n_drones if n_drones > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return CorrespondenceScore(
        n_tracks=n_tracks,
        n_drones=n_drones,
        n_matched=n_matched,
        n_imprecise=n_imprecise,
        n_false_track=n_false_track,
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

    # Filter out NaN errors (from NaN reconstruction positions)
    valid_errors = errors[np.isfinite(errors)]
    n_valid = len(valid_errors)

    if n_valid == 0:
        return TriangulationScore(
            n_matched=n_matched,
            position_errors_m=errors,
            median_error_m=0.0,
            p95_error_m=0.0,
            max_error_m=0.0,
            mean_reprojection_error_px=0.0,
        )

    median_err = float(np.median(valid_errors))
    p95_err = float(np.percentile(valid_errors, 95))
    max_err = float(np.max(valid_errors))
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
    detection_drone_ids: list[NDArray[np.int32]] | None = None,
) -> FullScore:
    """
    Complete evaluation pipeline.

    This is the single entry point for scoring a solver's output.

    Args:
        tracks: Correspondence tracks from solver
        recon: Triangulated 3D positions
        truth: Ground-truth swarm positions
        rig: Camera rig
        position_threshold_m: Match distance threshold (meters)
        detection_drone_ids: Optional per-view drone identity mapping for
            four-category classification. Compute via compute_detection_drone_ids(truth, rig).
            When None, falls back to position-only matching (old behavior).
    """
    track_truth = associate_tracks_to_truth(
        tracks, recon, truth, rig, position_threshold_m, detection_drone_ids)
    corr_score = score_correspondence(track_truth)
    triang_score = score_triangulation(recon, track_truth, truth)
    return FullScore(correspondence=corr_score, triangulation=triang_score)


# ============================================================================
# Control Tests (hand-constructed, no solver involved)
# ============================================================================

def _tracks_from_drone_mapping(
    detection_drone_ids: list[NDArray[np.int32]],
    n_drones: int,
    n_views: int,
) -> list[list[tuple[int, int]]]:
    """
    Build tracks from the detection-to-drone mapping.

    For each drone, finds the point_idx in each view where that drone appears,
    and returns the track as a list of (view_idx, point_idx) pairs.
    Drones not visible in a view are skipped for that view.
    """
    tracks_list: list[list[tuple[int, int]]] = []
    for drone_id in range(n_drones):
        track: list[tuple[int, int]] = []
        for v in range(n_views):
            ids = detection_drone_ids[v]
            matches = np.where(ids == drone_id)[0]
            if len(matches) > 0:
                track.append((v, int(matches[0])))
        if len(track) >= 2:  # Need at least 2 views for a valid track
            tracks_list.append(track)
    return tracks_list


def make_fake_tracks_perfect(truth: SwarmTruth, rig: CameraRig, n_drones: int = 5) -> tuple[Tracks, Reconstruction]:
    """
    Control 1: Perfect correspondence + perfect triangulation.

    Creates tracks that exactly match truth, and reconstruction that matches truth.
    Must score: all matched, precision=1.0, recall=1.0, median_error≈0.
    """
    n_views = rig.n_views
    det_ids = compute_detection_drone_ids(truth, rig)

    # Build tracks from actual detection mapping
    tracks_list = _tracks_from_drone_mapping(det_ids, n_drones, n_views)
    n_actual = len(tracks_list)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    # Perfect reconstruction = truth positions (only for drones with tracks)
    matched_drones = []
    for track in tracks_list:
        # Find which drone this track belongs to (first view's detection)
        v0, p0 = track[0]
        drone_id = int(det_ids[v0][p0])
        matched_drones.append(drone_id)

    recon = Reconstruction(
        positions_3d=truth.positions[0][matched_drones].copy(),
        reprojection_errors=np.zeros(n_actual, dtype=np.float64),
        track_indices=tracks_list,
    )
    return tracks, recon


def make_fake_tracks_scrambled(truth: SwarmTruth, rig: CameraRig, n_drones: int = 5, seed: int = 42) -> tuple[Tracks, Reconstruction]:
    """
    Control 2: Scrambled correspondence (wrong associations).

    Each track has detections from DIFFERENT drones across views.
    Should produce all false_track (detections span multiple drones).
    """
    rng = np.random.default_rng(seed)
    n_views = rig.n_views
    det_ids = compute_detection_drone_ids(truth, rig)

    tracks_list = []
    for drone_idx in range(n_drones):
        track = []
        for v in range(n_views):
            ids = det_ids[v]
            if len(ids) == 0:
                continue
            # Pick a drone that is NOT drone_idx (cycling through alternatives)
            other_drone = (drone_idx + v + 1) % n_drones
            other_positions = np.where(ids == other_drone)[0]
            if len(other_positions) == 0:
                # Fallback: any drone that isn't drone_idx
                other_positions = np.where(ids != drone_idx)[0]
            if len(other_positions) > 0:
                track.append((v, int(other_positions[0])))
        if len(track) >= 2:
            tracks_list.append(track)

    n_actual = len(tracks_list)
    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    rng2 = np.random.default_rng(seed + 1)
    recon = Reconstruction(
        positions_3d=rng2.uniform(-1000, 1000, size=(n_actual, 3)).astype(np.float64),
        reprojection_errors=rng2.uniform(10, 100, size=n_actual).astype(np.float64),
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
    det_ids = compute_detection_drone_ids(truth, rig)

    tracks_list = []
    for drone_id in range(n_drones):
        v1 = view_idx % n_views
        v2 = (view_idx + 1) % n_views
        # Find point_idx for this drone in each view
        p1_list = np.where(det_ids[v1] == drone_id)[0]
        p2_list = np.where(det_ids[v2] == drone_id)[0]
        if len(p1_list) > 0 and len(p2_list) > 0:
            track = [(v1, int(p1_list[0])), (v2, int(p2_list[0]))]
            tracks_list.append(track)

    n_actual = len(tracks_list)
    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    recon = Reconstruction(
        positions_3d=np.full((n_actual, 3), np.nan, dtype=np.float64),
        reprojection_errors=np.full(n_actual, np.nan, dtype=np.float64),
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

    Reconstruction is truth + constant offset. Should have all imprecise
    (correct identity, position beyond threshold).
    """
    n_views = rig.n_views
    det_ids = compute_detection_drone_ids(truth, rig)

    tracks_list = _tracks_from_drone_mapping(det_ids, n_drones, n_views)
    n_actual = len(tracks_list)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    matched_drones = []
    for track in tracks_list:
        v0, p0 = track[0]
        drone_id = int(det_ids[v0][p0])
        matched_drones.append(drone_id)

    # Perfect correspondence, but shifted reconstruction
    shifted_pos = truth.positions[0][matched_drones] + offset_m.reshape(1, 3)
    recon = Reconstruction(
        positions_3d=shifted_pos.astype(np.float64),
        reprojection_errors=np.zeros(n_actual, dtype=np.float64),
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
    With detection identity: real tracks → matched, ghost tracks → false_track
    (point indices beyond real detections map to no real drone).
    """
    n_views = rig.n_views
    det_ids = compute_detection_drone_ids(truth, rig)

    tracks_list = []
    # Real tracks (from detection mapping)
    real_tracks = _tracks_from_drone_mapping(det_ids, n_drones, n_views)
    tracks_list.extend(real_tracks)

    # Ghost tracks (point indices beyond real detections)
    n_real_dets = max(len(ids) for ids in det_ids) if det_ids else 0
    for g in range(n_ghosts):
        track = [(v, n_real_dets + g) for v in range(n_views)]
        tracks_list.append(track)

    n_actual = len(tracks_list)
    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    matched_drones = []
    for track in real_tracks:
        v0, p0 = track[0]
        drone_id = int(det_ids[v0][p0])
        matched_drones.append(drone_id)

    recon = Reconstruction(
        positions_3d=np.vstack([
            truth.positions[0][matched_drones],
            np.full((n_ghosts, 3), np.nan, dtype=np.float64),
        ]),
        reprojection_errors=np.hstack([
            np.zeros(len(real_tracks)),
            np.full(n_ghosts, np.nan),
        ]),
        track_indices=tracks_list,
    )
    return tracks, recon


def make_fake_tracks_imprecise(
    truth: SwarmTruth,
    rig: CameraRig,
    n_drones: int = 5,
    offset_m: float = 100.0,
) -> tuple[Tracks, Reconstruction]:
    """
    Control 6: Correct correspondence but imprecise triangulation.

    All tracks have correct drone identity, but reconstruction positions are
    shifted far from truth by a deterministic offset. Should classify as:
    n_imprecise = number of tracks.
    """
    n_views = rig.n_views
    det_ids = compute_detection_drone_ids(truth, rig)

    tracks_list = _tracks_from_drone_mapping(det_ids, n_drones, n_views)
    n_actual = len(tracks_list)

    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    matched_drones = []
    for track in tracks_list:
        v0, p0 = track[0]
        drone_id = int(det_ids[v0][p0])
        matched_drones.append(drone_id)

    # Correct correspondence, but reconstruction shifted deterministically
    # Use a fixed direction (x-axis) to guarantee offset > threshold
    offset_vec = np.array([offset_m, 0.0, 0.0], dtype=np.float64)
    shifted_pos = truth.positions[0][matched_drones] + offset_vec.reshape(1, 3)
    recon = Reconstruction(
        positions_3d=shifted_pos.astype(np.float64),
        reprojection_errors=np.full(n_actual, 50.0, dtype=np.float64),
        track_indices=tracks_list,
    )
    return tracks, recon


def make_fake_tracks_false_track(
    truth: SwarmTruth,
    rig: CameraRig,
    n_drones: int = 5,
) -> tuple[Tracks, Reconstruction]:
    """
    Control 7: False tracks — each track mixes detections from two drones.

    Track i pulls its first half of views from drone i and second half from
    drone (i+1) % n_drones. All tracks are false_track.
    Should classify as: n_false_track = number of tracks created.
    """
    n_views = rig.n_views
    mid = n_views // 2
    det_ids = compute_detection_drone_ids(truth, rig)

    tracks_list = []
    for drone_idx in range(n_drones):
        track = []
        for v in range(n_views):
            ids = det_ids[v]
            if len(ids) == 0:
                continue
            if v < mid:
                # Use drone_idx's detection
                matches = np.where(ids == drone_idx)[0]
                if len(matches) > 0:
                    track.append((v, int(matches[0])))
            else:
                # Use (drone_idx+1)'s detection
                alt_drone = (drone_idx + 1) % n_drones
                matches = np.where(ids == alt_drone)[0]
                if len(matches) > 0:
                    track.append((v, int(matches[0])))
        if len(track) >= 2:
            tracks_list.append(track)

    n_actual = len(tracks_list)
    tracks = Tracks(tracks=tracks_list, n_views=n_views)

    # Reconstruction at midpoint between the two mixed drones
    rng = np.random.default_rng(77)
    pos_a = truth.positions[0]
    pos_b = np.roll(truth.positions[0], -1, axis=0)
    mid_pos = 0.5 * (pos_a + pos_b)
    # Randomly pick which midpoint to use for each track
    recon_pos = np.array([mid_pos[i % n_drones] for i in range(n_actual)])
    recon = Reconstruction(
        positions_3d=recon_pos.astype(np.float64),
        reprojection_errors=np.full(n_actual, 20.0, dtype=np.float64),
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
    # Precompute detection-to-drone mapping for classification
    det_drone_ids = compute_detection_drone_ids(truth, rig)

    results = {}

    # Control 1: Perfect → all matched
    tracks, recon = make_fake_tracks_perfect(truth, rig)
    results['perfect'] = score_full(tracks, recon, truth, rig, position_threshold_m, det_drone_ids)

    # Control 2: Scrambled → all false_track
    tracks, recon = make_fake_tracks_scrambled(truth, rig)
    results['scrambled'] = score_full(tracks, recon, truth, rig, position_threshold_m, det_drone_ids)

    # Control 3: Single view → no matches (degenerate)
    tracks, recon = make_fake_tracks_single_view(truth, rig)
    results['single_view'] = score_full(tracks, recon, truth, rig, position_threshold_m, det_drone_ids)

    # Control 4: Known offset (100m shift) → all imprecise (100m > 50m threshold)
    offset = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    tracks, recon = make_fake_tracks_known_offset(truth, rig, offset)
    results['known_offset'] = score_full(tracks, recon, truth, rig, position_threshold_m, det_drone_ids)

    # Control 5: Ghosts → matched + false_track (phantom tracks)
    tracks, recon = make_fake_tracks_ghosts(truth, rig, n_ghosts=3)
    results['ghosts'] = score_full(tracks, recon, truth, rig, position_threshold_m, det_drone_ids)

    # Control 6: Imprecise → all imprecise (correct identity, bad position)
    tracks, recon = make_fake_tracks_imprecise(truth, rig)
    results['imprecise'] = score_full(tracks, recon, truth, rig, position_threshold_m, det_drone_ids)

    # Control 7: False track → all false_track (mixed detections)
    tracks, recon = make_fake_tracks_false_track(truth, rig)
    results['false_track'] = score_full(tracks, recon, truth, rig, position_threshold_m, det_drone_ids)

    return results


def validate_controls(controls: dict[str, FullScore]) -> bool:
    """
    Validate that controls produce expected results.

    Returns True if all controls behave as expected.
    Prints diagnostic info and returns False if any control fails.
    """
    print("=== CONTROL VALIDATION (4-category) ===")
    all_pass = True

    # Control 1: Perfect → all matched
    c = controls['perfect']
    print(f"\n1. PERFECT (all matched):")
    print(f"   matched={c.correspondence.n_matched}, imprecise={c.correspondence.n_imprecise}, "
          f"false_track={c.correspondence.n_false_track}, missed={c.correspondence.n_missed}")
    print(f"   P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}, "
          f"F1={c.correspondence.f1:.3f}")
    print(f"   Triangulation: median={c.triangulation.median_error_m:.6f}m")
    ok = (c.correspondence.n_matched == 5 and c.correspondence.n_imprecise == 0
          and c.correspondence.n_false_track == 0 and c.correspondence.n_missed == 0
          and c.triangulation.median_error_m < 1e-6)
    print(f"   {'✓ PASS' if ok else '❌ FAIL'}")
    all_pass = all_pass and ok

    # Control 2: Scrambled → all false_track
    c = controls['scrambled']
    print(f"\n2. SCRAMBLED (all false_track):")
    print(f"   matched={c.correspondence.n_matched}, imprecise={c.correspondence.n_imprecise}, "
          f"false_track={c.correspondence.n_false_track}, missed={c.correspondence.n_missed}")
    print(f"   P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}")
    ok = (c.correspondence.n_matched == 0 and c.correspondence.n_imprecise == 0
          and c.correspondence.n_false_track == 5)
    print(f"   {'✓ PASS' if ok else '❌ FAIL'}")
    all_pass = all_pass and ok

    # Control 3: Single view → no matches (degenerate)
    c = controls['single_view']
    print(f"\n3. SINGLE VIEW (degenerate):")
    print(f"   matched={c.correspondence.n_matched}, n_ghost={c.correspondence.n_ghost}")
    ok = (c.correspondence.n_matched == 0)
    print(f"   {'✓ PASS' if ok else '❌ FAIL'}")
    all_pass = all_pass and ok

    # Control 4: Known offset → all imprecise (100m shift)
    c = controls['known_offset']
    print(f"\n4. KNOWN OFFSET 100m (all imprecise):")
    print(f"   matched={c.correspondence.n_matched}, imprecise={c.correspondence.n_imprecise}, "
          f"false_track={c.correspondence.n_false_track}")
    print(f"   median error: {c.triangulation.median_error_m:.1f}m (should be ~100m)")
    ok = (c.correspondence.n_matched == 0 and c.correspondence.n_imprecise == 5
          and c.correspondence.n_false_track == 0
          and abs(c.triangulation.median_error_m - 100.0) < 1.0)
    print(f"   {'✓ PASS' if ok else '❌ FAIL'}")
    all_pass = all_pass and ok

    # Control 5: Ghosts → matched + false_track (phantom tracks)
    c = controls['ghosts']
    print(f"\n5. GHOSTS (5 real + 3 phantom → matched=5, false_track=3):")
    print(f"   matched={c.correspondence.n_matched}, imprecise={c.correspondence.n_imprecise}, "
          f"false_track={c.correspondence.n_false_track}")
    print(f"   n_ghost (compat)={c.correspondence.n_ghost}")
    expected_p = 5 / 8  # (matched + imprecise) / n_tracks = 5 / 8
    ok = (c.correspondence.n_matched == 5 and c.correspondence.n_false_track == 3
          and c.correspondence.n_ghost == 3
          and abs(c.correspondence.precision - expected_p) < 0.01)
    print(f"   {'✓ PASS' if ok else '❌ FAIL'}")
    all_pass = all_pass and ok

    # Control 6: Imprecise → all imprecise
    c = controls['imprecise']
    print(f"\n6. IMPRECISE (correct identity, 100m offset):")
    print(f"   matched={c.correspondence.n_matched}, imprecise={c.correspondence.n_imprecise}, "
          f"false_track={c.correspondence.n_false_track}")
    print(f"   P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}")
    ok = (c.correspondence.n_matched == 0 and c.correspondence.n_imprecise == 5
          and c.correspondence.n_false_track == 0
          and c.correspondence.precision == 1.0 and c.correspondence.recall == 1.0)
    print(f"   {'✓ PASS' if ok else '❌ FAIL'}")
    all_pass = all_pass and ok

    # Control 7: False track → all false_track
    c = controls['false_track']
    print(f"\n7. FALSE TRACK (mixed detections):")
    print(f"   matched={c.correspondence.n_matched}, imprecise={c.correspondence.n_imprecise}, "
          f"false_track={c.correspondence.n_false_track}")
    print(f"   P={c.correspondence.precision:.3f}, R={c.correspondence.recall:.3f}")
    ok = (c.correspondence.n_matched == 0 and c.correspondence.n_imprecise == 0
          and c.correspondence.n_false_track == 5)
    print(f"   {'✓ PASS' if ok else '❌ FAIL'}")
    all_pass = all_pass and ok

    print(f"\n=== {'ALL CONTROLS PASSED' if all_pass else 'SOME CONTROLS FAILED'} ===")
    return all_pass


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== B4 Scoring and Controls Harness Self-Test ===")

    # Need a truth and rig to test against
    # Use small area (0.3km) so all 5 drones are visible in all views
    from b1_scene_rig import generate_swarm_truth, generate_camera_rig

    truth = generate_swarm_truth(n_drones=5, n_frames=1, area_km=0.3, height_range_m=100.0, seed=42)
    rig = generate_camera_rig(truth, n_views=6, geometry_class="mixed", standoff_m=2000.0, seed=123)

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