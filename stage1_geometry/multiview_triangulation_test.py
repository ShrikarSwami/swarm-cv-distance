"""
Stage 1: Synthetic multi-view triangulation sanity check.

Purpose
-------
Before investing in Blender rendering + a trained CV detector, this script
answers the geometry question in isolation: given N drones at known 3D
positions and M observing cameras at known poses, how accurately can we
recover drone positions (and therefore pairwise distances) from 2D
detections alone -- including the effect of realistic pixel-level detection
noise?

No GPU, no rendering engine, no external assets required. Runs on any Mac
with numpy/scipy/matplotlib installed:

    pip install numpy scipy matplotlib

Once you're happy with the camera count / placement / noise tolerance found
here, Stage 2 swaps the synthetic projection step for real Blender renders
+ a real YOLO detector, and reuses triangulate_point() unchanged.
"""

import numpy as np
from itertools import combinations

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Scene setup: ground-truth drone positions
# ---------------------------------------------------------------------------

def make_swarm(n_drones=10, area_km=2.0, altitude_spread_m=50.0, seed=0):
    """Random drone positions in a horizontal area, with some altitude jitter.
    Returns an (n_drones, 3) array in meters, ENU-style local frame.
    """
    g = np.random.default_rng(seed)
    xy = g.uniform(-area_km * 500, area_km * 500, size=(n_drones, 2))  # meters
    z = 100.0 + g.uniform(-altitude_spread_m / 2, altitude_spread_m / 2, size=(n_drones, 1))
    return np.hstack([xy, z])


# ---------------------------------------------------------------------------
# 2. Camera model (simple pinhole)
# ---------------------------------------------------------------------------

class Camera:
    """Pinhole camera with a position, look-at target, and focal length in pixels."""

    def __init__(self, position, look_at, image_size=(1920, 1080), focal_px=1400.0):
        self.position = np.array(position, dtype=float)
        self.image_size = image_size
        self.focal_px = focal_px
        self.cx, self.cy = image_size[0] / 2, image_size[1] / 2
        self.R = self._look_at_rotation(self.position, np.array(look_at, dtype=float))
        self.K = np.array([
            [focal_px, 0, self.cx],
            [0, focal_px, self.cy],
            [0, 0, 1],
        ])
        # Camera projection matrix P = K [R | t], t = -R @ position
        t = -self.R @ self.position
        self.P = self.K @ np.hstack([self.R, t.reshape(3, 1)])

    @staticmethod
    def _look_at_rotation(eye, target, world_up=np.array([0, 0, 1.0])):
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        # Camera looks down +forward; standard CV convention: z-forward, x-right, y-down
        R = np.vstack([right, -up, forward])
        return R

    def project(self, point3d):
        """Project a 3D world point to 2D pixel coords. Returns None if behind camera."""
        p = np.append(point3d, 1.0)
        proj = self.P @ p
        if proj[2] <= 0:
            return None  # behind camera
        return proj[:2] / proj[2]

    def in_frame(self, pixel):
        if pixel is None:
            return False
        x, y = pixel
        return 0 <= x <= self.image_size[0] and 0 <= y <= self.image_size[1]


def place_ring_of_cameras(n_cameras, radius_m, height_m, look_at=(0, 0, 100)):
    """Place cameras evenly around the swarm, all looking at the swarm centroid.
    This mimics your own drones/observation posts ringing the hostile swarm.

    Note: a flat ring at a height close to the swarm's own altitude views the
    swarm nearly edge-on. Stage 1's point-projection math has no notion of one
    drone occluding another, so it can't reveal that this causes severe real
    self-occlusion -- confirmed against an actual Blender render of this exact
    rig (2026-07-22): Stage 1 predicted ~15/20 drones in-frame per camera, the
    real render showed only 2-5. See place_dome_of_cameras for the fix used in
    Stage 2's actual rig.
    """
    cams = []
    for i in range(n_cameras):
        angle = 2 * np.pi * i / n_cameras
        pos = (radius_m * np.cos(angle), radius_m * np.sin(angle), height_m)
        cams.append(Camera(pos, look_at))
    return cams


def place_dome_of_cameras(n_cameras, slant_range_m, elev_min_deg, elev_max_deg, look_at=(0, 0, 100)):
    """Place cameras on a dome around look_at: elevation varies across cameras
    (not a flat ring), so the rig views the swarm at a steeper angle and
    self-occlusion between drones is reduced. Slant range (camera-to-look_at
    distance) is held constant across all cameras so apparent object size in
    frame doesn't shrink with elevation -- only viewing angle changes.
    """
    cams = []
    for i in range(n_cameras):
        azimuth = 2 * np.pi * i / n_cameras
        elev_deg = elev_min_deg + (elev_max_deg - elev_min_deg) * i / max(n_cameras - 1, 1)
        elev = np.radians(elev_deg)
        horiz = slant_range_m * np.cos(elev)
        height_offset = slant_range_m * np.sin(elev)
        pos = (
            look_at[0] + horiz * np.cos(azimuth),
            look_at[1] + horiz * np.sin(azimuth),
            look_at[2] + height_offset,
        )
        cams.append(Camera(pos, look_at))
    return cams


# ---------------------------------------------------------------------------
# 3. Simulated "detection": project + add pixel noise (stand-in for a real
#    object detector's localization error before Stage 2 swaps in YOLO)
# ---------------------------------------------------------------------------

def simulate_detections(drones, cameras, pixel_noise_std=2.0, drop_prob=0.05):
    """Returns detections[cam_idx][drone_idx] = 2D pixel coord or None (missed/occluded)."""
    detections = []
    for cam in cameras:
        cam_dets = []
        for d in drones:
            px = cam.project(d)
            if px is None or not cam.in_frame(px) or rng.random() < drop_prob:
                cam_dets.append(None)
            else:
                noise = rng.normal(0, pixel_noise_std, size=2)
                cam_dets.append(px + noise)
        detections.append(cam_dets)
    return detections


# ---------------------------------------------------------------------------
# 4. Triangulation (linear least-squares / DLT over all camera pairs seeing
#    a given drone)
# ---------------------------------------------------------------------------

def triangulate_point(cams_subset, pixels_subset):
    """Multi-view DLT triangulation. cams_subset: list of Camera, pixels_subset:
    list of 2D pixel coords, same length, length >= 2.
    """
    A = []
    for cam, px in zip(cams_subset, pixels_subset):
        x, y = px
        P = cam.P
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reconstruct_swarm(cameras, detections, n_drones):
    """For each drone index, gather all cameras that detected it and triangulate."""
    est_positions = np.full((n_drones, 3), np.nan)
    n_views_used = np.zeros(n_drones, dtype=int)
    for di in range(n_drones):
        cams_seeing, pix_seeing = [], []
        for ci, cam in enumerate(cameras):
            px = detections[ci][di]
            if px is not None:
                cams_seeing.append(cam)
                pix_seeing.append(px)
        if len(cams_seeing) >= 2:
            est_positions[di] = triangulate_point(cams_seeing, pix_seeing)
            n_views_used[di] = len(cams_seeing)
    return est_positions, n_views_used


# ---------------------------------------------------------------------------
# 5. Evaluation: does the RECONSTRUCTED distance graph match the TRUE one?
#    This is the number that actually matters for the critical-node pipeline.
# ---------------------------------------------------------------------------

def pairwise_distances(positions):
    n = len(positions)
    D = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        if not (np.any(np.isnan(positions[i])) or np.any(np.isnan(positions[j]))):
            d = np.linalg.norm(positions[i] - positions[j])
            D[i, j] = D[j, i] = d
    return D


def evaluate(true_pos, est_pos, d_max):
    true_D = pairwise_distances(true_pos)
    est_D = pairwise_distances(est_pos)

    valid = ~np.isnan(est_D)
    n_pairs_valid = valid.sum() // 2
    dist_err = np.abs(true_D[valid] - est_D[valid])

    true_adj = (true_D <= d_max) & ~np.isnan(true_D)
    est_adj = (est_D <= d_max) & ~np.isnan(est_D)
    both_known = valid
    agree = (true_adj == est_adj) & both_known
    edge_accuracy = agree.sum() / max(both_known.sum(), 1)

    print(f"Drones reconstructed: {np.sum(~np.isnan(est_pos).any(axis=1))}/{len(true_pos)}")
    print(f"Pairwise distances comparable: {n_pairs_valid}")
    if len(dist_err):
        print(f"Distance error -- mean: {dist_err.mean():.2f} m, "
              f"median: {np.median(dist_err):.2f} m, max: {dist_err.max():.2f} m")
    print(f"Adjacency (edge present/absent) agreement vs ground truth: {edge_accuracy*100:.1f}%")
    return dist_err, edge_accuracy


# ---------------------------------------------------------------------------
# Run a scenario
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_DRONES = 20
    N_CAMERAS = 4          # try 2, 3, 4, 6 and compare
    RING_RADIUS_M = 1200.0
    RING_HEIGHT_M = 150.0
    PIXEL_NOISE_STD = 2.0  # try 0.5 (near-perfect detector) vs 5+ (sloppy detector)
    D_MAX = 800.0          # your comms-range cutoff, in meters, from the earlier design

    drones = make_swarm(n_drones=N_DRONES, seed=1)
    cameras = place_ring_of_cameras(N_CAMERAS, RING_RADIUS_M, RING_HEIGHT_M)
    detections = simulate_detections(drones, cameras, pixel_noise_std=PIXEL_NOISE_STD)
    est_positions, n_views = reconstruct_swarm(cameras, detections, N_DRONES)

    print(f"\n--- Scenario: {N_DRONES} drones, {N_CAMERAS} cameras, "
          f"pixel noise std={PIXEL_NOISE_STD}px ---")
    evaluate(drones, est_positions, D_MAX)
