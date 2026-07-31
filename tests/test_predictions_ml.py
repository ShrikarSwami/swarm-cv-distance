"""
Prediction-based acceptance tests for the ML swarm-reconstruction track.

Written BEFORE any ML component is built (build order step 1). These encode
geometry and invariants as assertions, so architecture bugs fail loudly instead
of surfacing as mediocre accuracy after hours of training.

Six tests are active now: they check pure geometry/math with self-contained
reference implementations, grounded in the frozen geometric-track conventions
(`data_contract`, `detect_blobs.apparent_px`). Four are PENDING: they require a
component that has not been built yet, and are skipped with an explicit reason
naming the activating component. Activation = remove the skip marker; the body
is already the real test.

The manifest below lists all 10 tests with status; a meta-test asserts the
count is 10 so none can be silently dropped, and enforces that pending tests
are never skipped silently.

Run:  python -m pytest tests/test_predictions_ml.py -v
Acceptance: pytest exits 0, manifest count is 10, no test silently skipped.
"""

import os
import sys

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "stage1_geometry"))

from data_contract import (  # noqa: E402  (frozen conventions)
    make_K,
    project_point,
    blender_c2w_to_opencv_w2c,
)
from detect_blobs import apparent_px  # noqa: E402  (frozen detector formula)

# ---------------------------------------------------------------------------
# Frozen operating constants (PATCH 4, calib.json, data_contract)
# ---------------------------------------------------------------------------

DRONE_SIZE_M = 0.5
IMAGE_W, IMAGE_H = 1920, 1080
FOCAL_PX = 2666.67  # repo default, ~40 deg HFOV
VOXEL_GRID_RES = 64  # voxels per swarm-diameter axis (spec Section 4, encoder work)


def _cell_standoff(radius_m):
    """Framing standoff = R * 2f/W = 2.778*R (PATCH 4)."""
    return 2.0 * radius_m * FOCAL_PX / IMAGE_W


def _cell_a_max(radius_m):
    """a_max = d*W/(2R) at the framing standoff (drone at range = standoff)."""
    return DRONE_SIZE_M * IMAGE_W / (2.0 * radius_m)


OPERATING_CELLS = {
    "primary": {
        "radius_m": 50.0,
        "standoff_m": _cell_standoff(50.0),
        "a_max_px": _cell_a_max(50.0),
    },
    "secondary": {
        "radius_m": 100.0,
        "standoff_m": _cell_standoff(100.0),
        "a_max_px": _cell_a_max(100.0),
    },
}

# ---------------------------------------------------------------------------
# Manifest — all 10 prediction tests with status. The count is asserted by
# test_manifest_count_is_ten below; nothing may be added or removed without
# that test and this list changing together.
# ---------------------------------------------------------------------------

MANIFEST = [
    {"id": "back_projection_round_trip", "title": "back-projection round trip",
     "status": "active", "component": None},
    {"id": "heatmap_target_fidelity", "title": "heatmap target fidelity",
     "status": "active", "component": None},
    {"id": "a_max_formula", "title": "a_max = d*W/(2R), both symmetries",
     "status": "active", "component": None},
    {"id": "known_offset", "title": "known offset shifts reported error by delta",
     "status": "active", "component": None},
    {"id": "scrambled_extrinsics", "title": "scrambled extrinsics explode error",
     "status": "active", "component": None},
    {"id": "single_view_degenerate", "title": "single view is degenerate",
     "status": "active", "component": None},
    {"id": "tier_label_integrity", "title": "tier label integrity",
     "status": "active", "component": None},  # activated 2026-07-31 by T1
    {"id": "permutation_invariance", "title": "permutation invariance",
     "status": "pending",
     "component": "T6 ml/model.py"},
    {"id": "metric_path_identity", "title": "metric path identity",
     "status": "pending",
     "component": "ml/metrics.py"},
    {"id": "split_disjointness", "title": "split disjointness (G1)",
     "status": "pending",
     "component": "T4 ml/pack_dataset.py"},
]


# ---------------------------------------------------------------------------
# Reference implementations (the invariants, written as code)
# ---------------------------------------------------------------------------

def _voxel_centers(center, radius, res=VOXEL_GRID_RES):
    """Voxel-cell centers over a [-radius, radius]^3 box around center."""
    edges = np.linspace(-radius, radius, res + 1)
    cs = (edges[:-1] + edges[1:]) / 2.0
    X, Y, Z = np.meshgrid(cs, cs, cs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts + np.asarray(center, dtype=float)


def _soft_argmax(heatmap, voxel_centers):
    """Differentiable position extraction: heatmap-weighted voxel mean."""
    w = np.asarray(heatmap, dtype=float).ravel()
    C = np.asarray(voxel_centers, dtype=float).reshape(-1, 3)
    Z = w.sum()
    assert Z > 0.0, "empty heatmap: soft-argmax is undefined"
    return (w[:, None] * C).sum(axis=0) / Z


def _look_at_w2c(eye, target):
    """OpenCV-convention world-to-camera (R, t) looking from eye to target."""
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    R = np.vstack([right, -up, forward]).astype(np.float64)
    t = (-R @ np.asarray(eye, dtype=float)).astype(np.float64)
    return R, t


def _dome_poses(center, standoff, n_views, seed=0):
    """n_views cameras on a dome at slant range `standoff`, looking at center."""
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n_views):
        az = 2.0 * np.pi * i / n_views + rng.uniform(-0.12, 0.12)
        elev = np.radians(-35.0 + 70.0 * i / max(n_views - 1, 1))
        horiz = standoff * np.cos(elev)
        eye = np.asarray(center, dtype=float) + np.array([
            horiz * np.cos(az), horiz * np.sin(az), standoff * np.sin(elev),
        ])
        poses.append(_look_at_w2c(eye, center))
    return poses


def _project(P, K, R, t):
    """Pinhole projection; None if behind camera."""
    cam = R @ np.asarray(P, dtype=float) + t
    if cam[2] <= 0:
        return None
    pix = K @ cam
    return pix[:2] / pix[2]


def _triangulate_dlt(pixels, K_list, R_list, t_list):
    """Standard DLT from >= 2 views. Raises ValueError for a single view."""
    n = len(pixels)
    if n < 2:
        raise ValueError("single view cannot triangulate (degenerate)")
    A = []
    for (u, v), K, R, t in zip(pixels, K_list, R_list, t_list):
        P = K @ np.hstack([R, np.asarray(t).reshape(3, 1)])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def _median_matched_error(pred, true):
    """Hungarian min-distance matching, median of matched pair distances."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    D = cdist(pred, true)
    r, c = linear_sum_assignment(D)
    return float(np.median(D[r, c]))


def _recompute_elevation_deg(c2w, centroid):
    """Elevation of the camera position relative to the centroid plane."""
    eye = np.asarray(c2w)[:3, 3]
    rel = eye - np.asarray(centroid, dtype=float)
    return float(np.degrees(np.arcsin(rel[2] / np.linalg.norm(rel))))


def _tier_for_elevation(elev_deg):
    """Spec Section 5 tier thresholds, from elevation (not the stored label)."""
    if elev_deg < -20.0:
        return "ground"
    if elev_deg > 20.0:
        return "aerial"
    return "level"


# ===========================================================================
# Active tests (no component needed)
# ===========================================================================

def test_back_projection_round_trip():
    """Project a known 3D point into all views, back-project, recover within 1 voxel.

    This is the operation the voxel-fusion encoder relies on (spec Section 4
    step 2): a correct back-projection from the observed pixels must recover the
    true 3D position. A convention bug (flipped rotation, wrong K, z<0 flip)
    blows the error far past a voxel.
    """
    center = np.array([0.0, 0.0, 50.0])
    radius = 100.0
    voxel = 2.0 * radius / VOXEL_GRID_RES
    K = make_K(FOCAL_PX, (IMAGE_W, IMAGE_H))
    poses = _dome_poses(center, _cell_standoff(radius), n_views=8, seed=3)

    rng = np.random.default_rng(7)
    dirn = rng.normal(size=3)
    dirn /= np.linalg.norm(dirn)
    P = center + 0.9 * radius * dirn  # inside the swarm sphere

    pixels = [_project(P, K, R, t) for R, t in poses]
    assert all(p is not None for p in pixels), "point behind a camera"

    P_rec = _triangulate_dlt(pixels, [K] * 8,
                             [R for R, _ in poses], [t for _, t in poses])
    err = float(np.linalg.norm(P_rec - P))
    assert err <= voxel, "back-projection round trip exceeds 1 voxel: %.3f m" % err

    # Every view's back-projected ray also passes within 1 voxel of P.
    for (u, v), R, t in zip(pixels, [R for R, _ in poses], [t for _, t in poses]):
        C = -R.T @ t  # camera centre (world)
        d_cam = np.linalg.solve(K, [u, v, 1.0])
        d_world = R.T @ d_cam
        d_world /= np.linalg.norm(d_world)
        ray_dist = float(np.linalg.norm(np.cross(d_world, P - C)))
        assert ray_dist <= voxel, "a back-projected ray misses the true point"


def test_heatmap_target_fidelity():
    """Soft-argmax of a constructed Gaussian heatmap recovers the true position.

    The training target is a 3D Gaussian heatmap centred on each ground-truth
    drone (spec Section 4); the model reads positions via soft-argmax. A bug in
    either the target construction or the soft-argmax loses the position.
    """
    center = np.array([0.0, 0.0, 0.0])
    radius = 50.0
    voxel = 2.0 * radius / VOXEL_GRID_RES
    centers = _voxel_centers(center, radius)
    sigma = 1.5 * voxel

    # On-grid target.
    P_grid = centers[13 * VOXEL_GRID_RES**2 + 29 * VOXEL_GRID_RES + 47]
    h = np.exp(-np.sum((centers - P_grid) ** 2, axis=1) / (2.0 * sigma ** 2))
    rec = _soft_argmax(h, centers)
    assert np.linalg.norm(rec - P_grid) <= voxel

    # Off-grid target (drone not on a voxel centre) — still within 1 voxel.
    rng = np.random.default_rng(5)
    P_off = np.array([rng.uniform(-radius, radius) for _ in range(3)])
    h2 = np.exp(-np.sum((centers - P_off) ** 2, axis=1) / (2.0 * sigma ** 2))
    rec2 = _soft_argmax(h2, centers)
    assert np.linalg.norm(rec2 - P_off) <= voxel

    # Shifting the target by delta shifts the recovered position by delta.
    delta = np.array([7.0, -3.0, 4.0])
    h3 = np.exp(-np.sum((centers - (P_grid + delta)) ** 2, axis=1) / (2.0 * sigma ** 2))
    rec3 = _soft_argmax(h3, centers)
    assert np.linalg.norm(rec3 - (P_grid + delta)) <= voxel


def test_a_max_formula():
    """a_max = d*W/(2R) with both symmetries (inherited from the geometric track).

    The geometric track validated `apparent_px = d*f/standoff` (doubling
    standoff halves apparent size; doubling focal doubles it). a_max is that
    formula evaluated at the framing standoff standoff = R*2f/W, so the two
    symmetries become: inverse in R, linear in W. The frozen calib.json table
    and PATCH 4 operating cells pin exact values.
    """
    # Exact frozen values (calib.json a_max_table).
    assert _cell_a_max(50.0) == pytest.approx(9.6, rel=1e-9)
    assert _cell_a_max(100.0) == pytest.approx(4.8, rel=1e-9)

    # Symmetry 1 — inverse in R: doubling R halves a_max.
    R, W = 100.0, IMAGE_W
    assert _cell_a_max(2.0 * R) == pytest.approx(_cell_a_max(R) / 2.0, rel=1e-9)

    # Symmetry 2 — linear in W: doubling W doubles a_max.
    for R in (50.0, 100.0):
        assert (DRONE_SIZE_M * 2.0 * W / (2.0 * R)
                == pytest.approx(2.0 * _cell_a_max(R), rel=1e-9))

    # a_max equals the frozen detector formula at the framing standoff
    # (drone at range = standoff): apparent_px(d, 2Rf/W, f) = d*W/(2R).
    for R in (50.0, 100.0):
        standoff = _cell_standoff(R)
        assert apparent_px(DRONE_SIZE_M, standoff, FOCAL_PX) \
            == pytest.approx(_cell_a_max(R), rel=1e-9)

    # Frozen apparent_px symmetries (geometric-track validation).
    assert apparent_px(DRONE_SIZE_M, 2.0 * 1000.0, FOCAL_PX) \
        == pytest.approx(apparent_px(DRONE_SIZE_M, 1000.0, FOCAL_PX) / 2.0, rel=1e-9)
    assert apparent_px(DRONE_SIZE_M, 1000.0, 2.0 * FOCAL_PX) \
        == pytest.approx(2.0 * apparent_px(DRONE_SIZE_M, 1000.0, FOCAL_PX), rel=1e-9)

    # PATCH 4 operating-cell standoffs (rounded 139 m / 278 m).
    assert _cell_standoff(50.0) == pytest.approx(138.89, abs=1.0)
    assert _cell_standoff(100.0) == pytest.approx(277.78, abs=1.0)


def test_known_offset():
    """Shifting all ground truth by delta shifts the reported error by delta.

    The evaluation path (whatever metrics.py later implements) must report
    position error in the world frame: a perfect reconstruction shifted by a
    constant delta must report exactly |delta|, never 0 and never anything
    that hides the offset.
    """
    rng = np.random.default_rng(11)
    true = rng.uniform(-40.0, 40.0, size=(20, 3))

    # Perfect reconstruction -> zero error.
    assert _median_matched_error(true, true) == pytest.approx(0.0, abs=1e-9)

    # Every drone shifted by the same delta -> matched error == |delta|.
    delta = np.array([3.0, -2.0, 1.0])
    pred = true + delta
    D = cdist(pred, true)
    r, c = linear_sum_assignment(D)
    # The optimal matching is identity (delta is small vs. point spacing).
    assert np.array_equal(r, c)
    assert np.allclose(D[r, c], np.linalg.norm(delta), atol=1e-9)
    assert _median_matched_error(pred, true) \
        == pytest.approx(np.linalg.norm(delta), rel=1e-9)

    # Doubling the offset doubles the reported median error.
    pred2 = true + 2.0 * delta
    assert _median_matched_error(pred2, true) \
        == pytest.approx(2.0 * np.linalg.norm(delta), rel=1e-9)


def test_scrambled_extrinsics():
    """Wrong extrinsics -> error explodes vs correct extrinsics.

    A correct multi-view pipeline is pose-sensitive: give it scrambled poses
    for the same pixels and the reconstruction must collapse. If scrambling
    leaves the error unchanged, the pipeline is ignoring the extrinsics — a
    leak/bug signal (the model would pass on pose-free shortcuts).
    """
    center = np.array([0.0, 0.0, 50.0])
    radius = 100.0
    voxel = 2.0 * radius / VOXEL_GRID_RES
    K = make_K(FOCAL_PX, (IMAGE_W, IMAGE_H))
    poses = _dome_poses(center, _cell_standoff(radius), n_views=8, seed=17)

    rng = np.random.default_rng(21)
    P = center + 0.6 * radius * rng.normal(size=3) / np.linalg.norm(rng.normal(size=3))
    pixels = [_project(P, K, R, t) for R, t in poses]

    P_correct = _triangulate_dlt(pixels, [K] * 8,
                                 [R for R, _ in poses], [t for _, t in poses])
    err_correct = float(np.linalg.norm(P_correct - P))
    assert err_correct <= voxel, "correct extrinsics must recover the point"

    # Scramble: reassign each observed pixel to a *different* camera's pose.
    perm = [5, 1, 7, 3, 0, 4, 2, 6]
    R_sc = [poses[i][0] for i in perm]
    t_sc = [poses[i][1] for i in perm]
    P_scrambled = _triangulate_dlt(pixels, [K] * 8, R_sc, t_sc)
    err_scrambled = float(np.linalg.norm(P_scrambled - P))
    assert err_scrambled > 10.0 * err_correct, \
        "scrambled extrinsics must explode the error (got %.3f m)" % err_scrambled
    assert err_scrambled > 5.0 * voxel, \
        "scrambled extrinsics still land within 5 voxels — pipeline ignoring poses?"


def test_single_view_degenerate():
    """A single view is degenerate: it must not return a confident 3D position.

    One pixel defines a ray, not a point; two 3D positions far apart along the
    same ray project to the identical pixel. A single view therefore cannot
    localise depth, and any pipeline claiming otherwise is hallucinating.
    """
    K = make_K(FOCAL_PX, (IMAGE_W, IMAGE_H))
    eye = np.array([0.0, -1000.0, 100.0])
    center = np.array([0.0, 0.0, 50.0])
    R, t = _look_at_w2c(eye, center)

    # Two points on the same ray at very different depths.
    P1 = np.array([0.0, 0.0, 50.0])          # the look-at target (pinhole centre)
    P2 = eye + 2.0 * (P1 - eye)              # 1005 m further along the same ray

    p1 = _project(P1, K, R, t)
    p2 = _project(P2, K, R, t)
    assert p1 is not None and p2 is not None
    assert np.allclose(p1, p2, atol=1e-9), \
        "points on the same ray must project to the same pixel"
    assert np.linalg.norm(P1 - P2) > 100.0, \
        "the two ambiguous positions must be far apart"

    # The reference triangulator refuses a single view outright.
    with pytest.raises(ValueError, match="single view"):
        _triangulate_dlt([p1], [K], [R], [t])


# ===========================================================================
# Pending tests — activated when their component lands (remove the skip
# marker; the body is the real test).
# ===========================================================================

def test_tier_label_integrity():
    """Elevation recomputed from extrinsics matches the declared tier, every camera, every scene."""
    sys.path.insert(0, REPO_ROOT)
    from ml import scene_gen

    for cell_name in scene_gen.OPERATING_CELLS:
        for seed in (2000, 2001, 2002):
            scene = scene_gen.generate_scene(seed=seed, cell=cell_name)
            centroid = np.asarray(scene["swarm_center"], dtype=float)
            assert len(scene["cameras"]) == scene_gen.N_VIEWS
            for cam in scene["cameras"]:
                elev = _recompute_elevation_deg(cam["c2w"], centroid)
                assert _tier_for_elevation(elev) == cam["tier"], \
                    "declared tier %r disagrees with recomputed elevation %.2f deg" \
                    % (cam["tier"], elev)


@pytest.mark.skip(reason="PENDING: activated by T6 ml/model.py — permutation "
                         "invariance is non-negotiable for a set predictor")
def test_permutation_invariance():
    """Shuffle input view order -> identical output within float tolerance."""
    sys.path.insert(0, REPO_ROOT)
    from ml import model

    rng = np.random.default_rng(31)
    views = [rng.normal(size=(3, IMAGE_H, IMAGE_W)) for _ in range(8)]
    views_shuffled = views[::-1]

    out = model.forward(views)
    out_shuffled = model.forward(views_shuffled)
    assert np.allclose(out, out_shuffled, atol=1e-6), \
        "reordering the input views changed the output — the model is not a set predictor"


@pytest.mark.skip(reason="PENDING: activated by ml/metrics.py — one frozen "
                         "implementation shared by both tracks")
def test_metric_path_identity():
    """Same input arrays -> geometric and ML produce identical metric values from the same code."""
    sys.path.insert(0, REPO_ROOT)
    from ml import metrics

    rng = np.random.default_rng(41)
    true = rng.uniform(-40.0, 40.0, size=(12, 3))
    pred = true + rng.normal(scale=2.0, size=(12, 3))
    taus = [0.5, 1.0, 2.0, 5.0]

    ml_values = metrics.ml_evaluate(pred, true, taus=taus)
    geometric_values = metrics.geometric_evaluate(pred, true, taus=taus)
    assert ml_values == geometric_values, \
        "the two tracks computed different metric values — they are not on the same code path"


@pytest.mark.skip(reason="PENDING: activated by T4 ml/pack_dataset.py — G1 "
                         "leak check against ml/splits.json")
def test_split_disjointness():
    """Zero scene-seed overlap across train/val/test; PATCH 7 ranges respected."""
    import json

    sys.path.insert(0, REPO_ROOT)
    splits_path = os.path.join(REPO_ROOT, "ml", "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert not (train & val), "train/val seed overlap"
    assert not (train & test), "train/test seed overlap"
    assert not (val & test), "val/test seed overlap"

    # PATCH 7: test 0-999, val 1000-1999, train 2000+.
    assert all(0 <= s <= 999 for s in test), "test seed outside 0-999"
    assert all(1000 <= s <= 1999 for s in val), "val seed outside 1000-1999"
    assert all(s >= 2000 for s in train), "train seed outside 2000+"


# ===========================================================================
# Meta-tests
# ===========================================================================

def test_manifest_count_is_ten():
    """The manifest must list exactly 10 tests — none quietly dropped."""
    assert len(MANIFEST) == 10, "a prediction test was dropped from the manifest"


def test_manifest_ids_match_module_functions():
    """Every manifest entry must resolve to a real test function in this module."""
    module = sys.modules[__name__]
    for entry in MANIFEST:
        fn_name = "test_" + entry["id"]
        assert hasattr(module, fn_name), \
            "%s is in the manifest but has no test function" % entry["id"]
        assert callable(getattr(module, fn_name))


def test_no_test_silently_skipped():
    """Active tests are never skipped; pending tests carry an explicit reason
    naming their activating component."""
    module = sys.modules[__name__]
    for entry in MANIFEST:
        fn = getattr(module, "test_" + entry["id"])
        markers = list(getattr(fn, "pytestmark", []))
        skips = [m for m in markers if m.name == "skip"]
        if entry["status"] == "active":
            assert not skips, "active test %s is marked skip" % entry["id"]
        else:
            assert len(skips) == 1, \
                "pending test %s must carry exactly one explicit skip" % entry["id"]
            reason = str(skips[0].kwargs.get("reason", ""))
            assert "pending" in reason.lower() and entry["component"] in reason, \
                "pending test %s skip reason must name its activating component" % entry["id"]
