"""
Stage A: Frozen Data Contract for Phase 1

This module defines the typed data structures that ALL Phase 1 components
(SwarmTruth, CameraRig, Detections, Tracks, Reconstruction) must use.
No module may redefine conventions locally. Import from here only.

CONVENTION TAG: "opencv_enu" — meaning:
- World frame: ENU (East, North, Up) in meters
- Camera frame: OpenCV (+X right, +Y down, +Z forward)
- Principal point: (width/2, height/2)
- Image origin: top-left (0,0)
- Extrinsics in SwarmTruth: world-to-camera (R, t) where R is 3x3, t is 3x1
- Extrinsics in CameraRig (for render pipeline): camera-to-world 4x4 matrix (Blender)
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Convention Tag — single source of truth
# ============================================================================

CONVENTION_TAG: Literal["opencv_enu"] = "opencv_enu"
IMAGE_SIZE = (1920, 1080)  # (width, height)
PRINCIPAL_POINT = (IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0)  # (960.0, 540.0)
FOCAL_PX = 1400.0  # Stage 1 focal length in pixels — UNIFIED VALUE


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass(frozen=True)
class SwarmTruth:
    """
    Ground-truth swarm state. Immutable once created.

    Attributes:
        positions: (n_frames, n_drones, 3) float64 — world positions in ENU meters
        drone_ids: (n_drones,) int — stable identity per drone across frames
        n_frames: int
        n_drones: int
    """
    positions: NDArray[np.float64]  # (F, N, 3)
    drone_ids: NDArray[np.int32]    # (N,)

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_drones(self) -> int:
        return self.positions.shape[1]

    def __post_init__(self):
        assert self.positions.ndim == 3, f"positions must be (F,N,3), got {self.positions.shape}"
        assert self.positions.shape[2] == 3, "last dim must be 3 (ENU)"
        assert self.drone_ids.ndim == 1, "drone_ids must be 1D"
        assert len(self.drone_ids) == self.n_drones
        assert np.all(np.diff(self.drone_ids) > 0), "drone_ids must be strictly increasing"


@dataclass(frozen=True)
class CameraRig:
    """
    Multi-camera rig specification. Immutable.

    Two extrinsics representations are stored for different pipelines:
    - w2c: World-to-camera (R, t) for Stage 1 analytic pipeline
    - c2w: Camera-to-world 4x4 matrices for Blender render pipeline

    Both are derived from a single source in the rig generator and kept consistent.

    Attributes:
        K: (n_views, 3, 3) float64 — intrinsic matrices
        w2c_R: (n_views, 3, 3) float64 — world-to-camera rotations
        w2c_t: (n_views, 3) float64 — world-to-camera translations
        c2w: (n_views, 4, 4) float64 — camera-to-world (Blender matrix_world)
        convention: str — must equal CONVENTION_TAG
        geometry_class: Literal["all_ground", "mixed", "surround"] — camera placement style
    """
    K: NDArray[np.float64]           # (V, 3, 3)
    w2c_R: NDArray[np.float64]       # (V, 3, 3)
    w2c_t: NDArray[np.float64]       # (V, 3)
    c2w: NDArray[np.float64]         # (V, 4, 4)
    convention: str
    geometry_class: Literal["all_ground", "mixed", "surround"]

    @property
    def n_views(self) -> int:
        return self.K.shape[0]

    def __post_init__(self):
        assert self.convention == CONVENTION_TAG, f"convention must be '{CONVENTION_TAG}', got '{self.convention}'"
        V = self.n_views
        assert self.K.shape == (V, 3, 3), f"K must be (V,3,3), got {self.K.shape}"
        assert self.w2c_R.shape == (V, 3, 3), f"w2c_R must be (V,3,3), got {self.w2c_R.shape}"
        assert self.w2c_t.shape == (V, 3), f"w2c_t must be (V,3), got {self.w2c_t.shape}"
        assert self.c2w.shape == (V, 4, 4), f"c2w must be (V,4,4), got {self.c2w.shape}"
        # Verify orthonormality of rotations
        for v in range(V):
            R = self.w2c_R[v]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), f"View {v}: w2c_R not orthonormal"
            assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6), f"View {v}: w2c_R det != 1"
        # Verify consistency between w2c and c2w
        for v in range(V):
            R_c2w = self.c2w[v, :3, :3]
            t_c2w = self.c2w[v, :3, 3]
            R_w2c = self.w2c_R[v]
            t_w2c = self.w2c_t[v]
            # c2w = [R_c2w, t_c2w; 0, 1], w2c = [R_w2c, t_w2c; 0, 1]
            # R_w2c = R_c2w.T, t_w2c = -R_c2w.T @ t_c2w
            assert np.allclose(R_w2c, R_c2w.T, atol=1e-6), f"View {v}: R_w2c != R_c2w.T"
            assert np.allclose(t_w2c, -R_c2w.T @ t_c2w, atol=1e-6), f"View {v}: t_w2c inconsistent"


@dataclass(frozen=True)
class Detections:
    """
    2D detections per camera view. NO IDENTITY INFORMATION.

    This is the critical contract: detections are ANONYMOUS point sets.
    The correspondence solver receives this and MUST NOT have access to
    which detection belongs to which drone.

    Attributes:
        points_per_view: list of length n_views, each element is (n_dets_v, 2) float64 array
        image_size: (width, height) — same for all views in this pipeline
    """
    points_per_view: list[NDArray[np.float64]]  # len=V, each (M_v, 2)
    image_size: tuple[int, int] = IMAGE_SIZE

    def __post_init__(self):
        assert len(self.points_per_view) > 0
        for i, pts in enumerate(self.points_per_view):
            assert pts.ndim == 2 and pts.shape[1] == 2, f"View {i}: points must be (M,2), got {pts.shape}"
            # Verify all points are within image bounds (with small epsilon for noise)
            w, h = self.image_size
            assert np.all(pts[:, 0] >= -1.0) and np.all(pts[:, 0] <= w + 1.0), f"View {i}: x out of bounds"
            assert np.all(pts[:, 1] >= -1.0) and np.all(pts[:, 1] <= h + 1.0), f"View {i}: y out of bounds"


@dataclass(frozen=True)
class Tracks:
    """
    Multi-view tracks output by correspondence solver.

    Each track is a set of (view_idx, point_idx) pairs identifying the
    same physical point across views. NO TRUTH REFERENCE.

    Attributes:
        tracks: list of tracks; each track is list of (view_idx, point_idx) tuples
        n_views: total number of views in the rig
    """
    tracks: list[list[tuple[int, int]]]
    n_views: int

    def __post_init__(self):
        assert self.n_views > 0
        for i, track in enumerate(self.tracks):
            assert len(track) >= 2, f"Track {i}: must have >= 2 views, got {len(track)}"
            view_indices = [v for v, _ in track]
            assert len(set(view_indices)) == len(view_indices), f"Track {i}: duplicate view indices"
            assert all(0 <= v < self.n_views for v in view_indices), f"Track {i}: view index out of range"


@dataclass(frozen=True)
class Reconstruction:
    """
    3D reconstruction from triangulated tracks.

    Attributes:
        positions_3d: (n_tracks, 3) float64 — estimated world positions (ENU)
        reprojection_errors: (n_tracks,) float64 — mean reprojection error in pixels
        track_indices: list of tracks that were successfully triangulated
    """
    positions_3d: NDArray[np.float64]      # (T, 3)
    reprojection_errors: NDArray[np.float64]  # (T,)
    track_indices: list[list[tuple[int, int]]]  # original tracks that produced each position

    def __post_init__(self):
        assert self.positions_3d.ndim == 2 and self.positions_3d.shape[1] == 3
        assert self.reprojection_errors.ndim == 1
        assert len(self.positions_3d) == len(self.reprojection_errors)
        assert len(self.track_indices) == len(self.positions_3d)


# ============================================================================
# Projection Utilities (using the contract's conventions)
# ============================================================================

def make_K(focal_px: float = FOCAL_PX, image_size: tuple[int, int] = IMAGE_SIZE) -> NDArray[np.float64]:
    """Create intrinsic matrix per convention."""
    cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
    return np.array([
        [focal_px, 0.0, cx],
        [0.0, focal_px, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def project_point(world_pt: NDArray[np.float64], K: NDArray[np.float64],
                  R_w2c: NDArray[np.float64], t_w2c: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """
    Project a single 3D world point to 2D pixel.

    Args:
        world_pt: (3,) ENU world coordinates
        K: (3, 3) intrinsic matrix
        R_w2c: (3, 3) world-to-camera rotation
        t_w2c: (3,) world-to-camera translation

    Returns:
        (2,) pixel coordinates (u, v) or None if behind camera
    """
    cam_pt = R_w2c @ world_pt + t_w2c  # world to camera
    if cam_pt[2] <= 0:
        return None
    pix = K @ cam_pt
    return np.array([pix[0] / pix[2], pix[1] / pix[2]], dtype=np.float64)


def project_points_batch(world_pts: NDArray[np.float64], K: NDArray[np.float64],
                         R_w2c: NDArray[np.float64], t_w2c: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Batch project multiple 3D points. Returns (N, 3) with third column = depth or -1 if behind.
    """
    cam_pts = (R_w2c @ world_pts.T + t_w2c.reshape(3, 1)).T  # (N, 3)
    depths = cam_pts[:, 2]
    valid = depths > 0
    pixels = np.full((len(world_pts), 3), -1.0, dtype=np.float64)
    if np.any(valid):
        proj = (K @ cam_pts[valid].T).T
        pixels[valid, 0] = proj[:, 0] / proj[:, 2]
        pixels[valid, 1] = proj[:, 1] / proj[:, 2]
        pixels[valid, 2] = depths[valid]
    return pixels


def blender_c2w_to_opencv_w2c(c2w: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert Blender camera-to-world 4x4 matrix to OpenCV world-to-camera (R, t).

    Blender convention: +X right, +Y up, -Z forward (camera looks down -Z)
    OpenCV convention:  +X right, +Y down, +Z forward (camera looks down +Z)

    Conversion: M_cv = M_bl @ diag(1, -1, -1, 1), then w2c = inv(M_cv)
    """
    # Blender to OpenCV camera frame
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    M_cv = c2w @ flip
    # World-to-camera is inverse
    w2c = np.linalg.inv(M_cv)
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    return R_w2c, t_w2c


def opencv_w2c_to_blender_c2w(R_w2c: NDArray[np.float64], t_w2c: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of blender_c2w_to_opencv_w2c."""
    # Build w2c matrix
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    # Invert to get c2w in OpenCV frame
    c2w_cv = np.linalg.inv(w2c)
    # Convert to Blender frame
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_bl = c2w_cv @ flip
    return c2w_bl


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_swarm_truth(truth: SwarmTruth) -> bool:
    """Validate SwarmTruth internal consistency."""
    try:
        assert truth.positions.shape[2] == 3
        assert len(truth.drone_ids) == truth.n_drones
        assert np.all(np.diff(truth.drone_ids) > 0)
        # No NaN positions
        assert not np.any(np.isnan(truth.positions))
        # Reasonable bounds check (5km x 5km x 1km volume)
        assert np.all(np.abs(truth.positions[:, :, 0]) <= 5000)  # East
        assert np.all(np.abs(truth.positions[:, :, 1]) <= 5000)  # North
        assert np.all(truth.positions[:, :, 2] >= 0) and np.all(truth.positions[:, :, 2] <= 1000)  # Up
        return True
    except AssertionError as e:
        print(f"SwarmTruth validation failed: {e}")
        return False


def validate_camera_rig(rig: CameraRig) -> bool:
    """Validate CameraRig internal consistency."""
    try:
        assert rig.convention == CONVENTION_TAG
        assert rig.K.shape[0] == rig.n_views
        # Check principal point consistency
        for v in range(rig.n_views):
            cx, cy = rig.K[v, 0, 2], rig.K[v, 1, 2]
            assert abs(cx - PRINCIPAL_POINT[0]) < 1.0, f"View {v}: cx={cx} != {PRINCIPAL_POINT[0]}"
            assert abs(cy - PRINCIPAL_POINT[1]) < 1.0, f"View {v}: cy={cy} != {PRINCIPAL_POINT[1]}"
        return True
    except AssertionError as e:
        print(f"CameraRig validation failed: {e}")
        return False


def validate_detections(dets: Detections) -> bool:
    """Validate Detections internal consistency."""
    try:
        assert len(dets.points_per_view) > 0
        for pts in dets.points_per_view:
            assert pts.shape[1] == 2
            assert not np.any(np.isnan(pts))
        return True
    except AssertionError as e:
        print(f"Detections validation failed: {e}")
        return False


# ============================================================================
# Round-trip Test (run at module import in test mode)
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=== Data Contract Self-Test ===")

    # Test 1: K matrix principal point
    K = make_K()
    assert np.allclose(K[0, 2], PRINCIPAL_POINT[0])
    assert np.allclose(K[1, 2], PRINCIPAL_POINT[1])
    print("✓ K matrix principal point correct")

    # Test 2: Blender ↔ OpenCV conversion round-trip
    # Create a known camera pose
    R_bl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)  # looking down -Z
    t_bl = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    c2w_bl = np.eye(4)
    c2w_bl[:3, :3] = R_bl
    c2w_bl[:3, 3] = t_bl

    R_w2c, t_w2c = blender_c2w_to_opencv_w2c(c2w_bl)
    c2w_bl_reconstructed = opencv_w2c_to_blender_c2w(R_w2c, t_w2c)
    assert np.allclose(c2w_bl, c2w_bl_reconstructed, atol=1e-10)
    print("✓ Blender↔OpenCV conversion round-trip exact")

    # Test 3: Projection round-trip at known depth
    world_pt = np.array([10.0, 20.0, 50.0])
    K_test = make_K(focal_px=1000.0, image_size=(1920, 1080))
    R_test = np.eye(3)
    t_test = np.zeros(3)
    pix = project_point(world_pt, K_test, R_test, t_test)
    assert pix is not None
    # Back-project at same depth
    cam_pt = np.array([pix[0], pix[1], 1.0]) * world_pt[2] / (K_test[0, 0] * R_test[0, 0] + K_test[1, 1] * R_test[1, 1])  # simplified
    # Better: unproject using known depth
    depth = world_pt[2]
    cam_pt = np.linalg.inv(K_test) @ np.array([pix[0] * depth, pix[1] * depth, depth])
    world_reconstructed = R_test.T @ (cam_pt - t_test)
    assert np.allclose(world_pt, world_reconstructed, atol=1e-6)
    print("✓ Projection→unprojection round-trip exact at known depth")

    # Test 4: SwarmTruth creation and validation
    truth = SwarmTruth(
        positions=np.random.rand(1, 5, 3).astype(np.float64) * 1000,
        drone_ids=np.arange(5, dtype=np.int32)
    )
    assert validate_swarm_truth(truth)
    print("✓ SwarmTruth creation and validation works")

    # Test 5: CameraRig creation and validation
    V = 4
    rig = CameraRig(
        K=np.stack([make_K() for _ in range(V)]),
        w2c_R=np.stack([np.eye(3) for _ in range(V)]),
        w2c_t=np.zeros((V, 3)),
        c2w=np.stack([np.eye(4) for _ in range(V)]),
        convention=CONVENTION_TAG,
        geometry_class="mixed"
    )
    assert validate_camera_rig(rig)
    print("✓ CameraRig creation and validation works")

    # Test 6: Detections creation
    dets = Detections(
        points_per_view=[np.random.rand(10, 2).astype(np.float64) * 1000 for _ in range(V)]
    )
    assert validate_detections(dets)
    print("✓ Detections creation and validation works")

    print("\n=== ALL SELF-TESTS PASSED ===")
    sys.exit(0)