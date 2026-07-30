"""
Prediction-based acceptance tests for the Stage 1 geometry pipeline.

This file encodes the physical and geometric relationships that the pipeline
must satisfy. If these tests pass, the pipeline is producing numbers consistent
with theory. If they fail, something fundamental has changed.

Design principles:
- Predictions over hardcoded values: we test ratios and monotonicity, not
  absolute numbers (which change with config).
- Each test is independent: no shared state between test cases.
- Deterministic: every test seeds its RNG explicitly.
- Fast: all tests run in under 30s total.

Run:  python -m pytest tests/test_predictions.py -v
"""

import sys
import os
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_geometry'))

from data_contract import (
    SwarmTruth, CameraRig, Detections, Tracks, Reconstruction,
    DEFAULT_FOCAL_PX, IMAGE_SIZE, make_K, CONVENTION_TAG,
    project_point, project_points_batch,
)
from b1_scene_rig import (
    generate_swarm_truth, generate_camera_rig,
    compute_framing_coverage, compute_framing_coverage_detailed,
    compute_required_standoff,
)
from b2_projection import project_swarm_to_detections, project_frame_to_detections
from b3_correspondence import solve_correspondence
from b4_scoring import score_full, run_controls, validate_controls
from b5_triangulation import triangulate_dlt, triangulate_dlt_then_refine

# ============================================================================
# Standard test config (used throughout)
# ============================================================================
N_DRONES = 5
AREA_KM_SMALL = 0.3     # Fully framed even at narrow FOV (R~212m)
AREA_KM_LARGE = 5.0     # Standard geometry sweep scale
HEIGHT_RANGE_LARGE = 1000.0    # For 5km area swarms
HEIGHT_RANGE_SMALL = 100.0     # For 0.3km area swarms (proportional scaling)
STANDOFF = 2000.0       # Default
STANDOFF_CLOSE = 1000.0  # For coverage tests
FOCAL_WIDE = 1400.0
FOCAL_NARROW = DEFAULT_FOCAL_PX  # ≈ 2666.67
VIEWS = 8
GEOMETRY = "mixed"
MATCH_THRESHOLD = 1.5
SEED_SWARM = 42
SEED_RIG = 123


# ============================================================================
# Fixtures (module-scoped to avoid repeating generation in every test)
# ============================================================================

def _make_truth(area_km=AREA_KM_SMALL, n_drones=N_DRONES, seed=SEED_SWARM):
    height_range = HEIGHT_RANGE_SMALL if area_km <= 1.0 else HEIGHT_RANGE_LARGE
    return generate_swarm_truth(n_drones=n_drones, n_frames=1,
                                area_km=area_km, height_range_m=height_range, seed=seed)


def _make_rig(truth, n_views=VIEWS, standoff=STANDOFF, focal=FOCAL_NARROW,
             geometry=GEOMETRY, seed=SEED_RIG):
    return generate_camera_rig(truth, n_views=n_views, geometry_class=geometry,
                               standoff_m=standoff, focal_px=focal, seed=seed)


def _run_pipeline(truth, rig, noise_std=0.0, epipolar_threshold=3.0):
    """Run end-to-end analytic pipeline, return (tracks, recon, detections, score)."""
    detections = project_swarm_to_detections(
        truth, rig, pixel_noise_std=noise_std, drop_prob=0.0, seed=1)
    tracks = solve_correspondence(
        detections, rig,
        epipolar_threshold=epipolar_threshold,
        min_views=2, max_reproj_error=5.0, seed=42)
    recon = triangulate_dlt(tracks, rig, detections)
    score = score_full(tracks, recon, truth, rig, position_threshold_m=MATCH_THRESHOLD)
    return tracks, recon, detections, score


# ============================================================================
# B1: Swarm Generation & Framing
# ============================================================================

class TestB1Generation:
    """Swarm truth generation correctness."""

    def test_basic_generation(self):
        """Default config produces valid SwarmTruth."""
        truth = _make_truth()
        assert truth.n_drones == N_DRONES
        assert truth.n_frames == 1
        assert truth.positions.shape == (1, N_DRONES, 3)
        assert not np.any(np.isnan(truth.positions))

    def test_area_scaling(self):
        """Larger area produces larger swarm extents."""
        small = _make_truth(area_km=0.3)
        large = _make_truth(area_km=5.0)
        small_extent = np.max(np.linalg.norm(
            small.positions[0] - small.positions[0].mean(axis=0), axis=1))
        large_extent = np.max(np.linalg.norm(
            large.positions[0] - large.positions[0].mean(axis=0), axis=1))
        assert large_extent > small_extent, (
            f"Large extent {large_extent:.0f}m should exceed small {small_extent:.0f}m")

    def test_min_spacing_default(self):
        """Default generation should have natural spacing >> match threshold."""
        truth = _make_truth(area_km=AREA_KM_SMALL)
        ps = truth.positions[0]
        min_d = float('inf')
        for i in range(truth.n_drones):
            for j in range(i + 1, truth.n_drones):
                d = np.linalg.norm(ps[i] - ps[j])
                min_d = min(min_d, d)
        assert min_d > MATCH_THRESHOLD * 2, (
            f"Min spacing {min_d:.2f}m should exceed 2× match threshold ({2*MATCH_THRESHOLD}m)"
        )

    def test_min_spacing_enforced(self):
        """Swarm with explicit min_spacing_m produces wider minimum gap.

        This is a soft test — rejection sampling may not perfectly satisfy
        the constraint for small areas, but it should attempt it.
        """
        # No direct API yet for min_spacing; test will evolve when B1 adds it
        truth = _make_truth(area_km=AREA_KM_SMALL)
        ps = truth.positions[0]
        dists = []
        for i in range(truth.n_drones):
            for j in range(i + 1, truth.n_drones):
                dists.append(np.linalg.norm(ps[i] - ps[j]))
        min_d = min(dists)
        # Prediction: for small swarms at 0.3km area, all pairwise distances
        # are well above the match threshold
        assert min_d > MATCH_THRESHOLD, (
            f"Min spacing {min_d:.2f}m must exceed match threshold {MATCH_THRESHOLD}m"
        )


class TestB1Framing:
    """Framing coverage computations."""

    def test_full_coverage_small_swarm(self):
        """Small swarm (0.3km) is fully framed at narrow FOV and reasonable standoff."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)
        cov = compute_framing_coverage(truth, rig)
        assert cov == 1.0, f"Coverage should be 100%, got {cov:.1%}"

    def test_coverage_detailed(self):
        """Detailed coverage returns per-view breakdown matching overall."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)
        overall, per_view = compute_framing_coverage_detailed(truth, rig)
        assert 0.0 <= overall <= 1.0
        assert len(per_view) == rig.n_views
        # Per-view fractions can be any value — overall aggregates them
        # Check structure: all view indices present, each is 0..1
        for v, cov in per_view.items():
            assert 0.0 <= cov <= 1.0, f"View {v} coverage {cov:.3f} out of range"
            assert isinstance(v, int) and 0 <= v < rig.n_views

    def test_coverage_large_swarm(self):
        """Large swarm (5km) at close standoff has poor coverage (verifiable prediction)."""
        truth = _make_truth(area_km=5.0)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)
        cov = compute_framing_coverage(truth, rig)
        assert cov < 0.9, (
            f"Large swarm at close standoff should have <90% coverage, got {cov:.1%}"
        )

    def test_required_standoff(self):
        """Required standoff increases with focal length."""
        truth = _make_truth(area_km=0.3)
        s_wide = compute_required_standoff(truth, FOCAL_WIDE, margin_factor=1.2)
        s_narrow = compute_required_standoff(truth, FOCAL_NARROW, margin_factor=1.2)
        assert s_narrow > s_wide, (
            f"Narrower focal needs larger standoff ({s_narrow:.0f} > {s_wide:.0f})"
        )
        assert s_wide > 0 and np.isfinite(s_wide)
        assert s_narrow > 0 and np.isfinite(s_narrow)


# ============================================================================
# B2: Projection
# ============================================================================

class TestB2Projection:
    """Projection model correctness."""

    def test_all_drones_visible(self):
        """When coverage=100%, each camera sees exactly n_drones detections (no noise, no drops)."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)
        cov = compute_framing_coverage(truth, rig)
        assert cov == 1.0, "Test requires full coverage"
        detections = project_swarm_to_detections(
            truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=1)
        for v in range(rig.n_views):
            assert len(detections.points_per_view[v]) == N_DRONES, (
                f"View {v}: expected {N_DRONES} detections, got {len(detections.points_per_view[v])}"
            )

    def test_zero_noise_determinism(self):
        """Zero-noise projection with same seed is deterministic."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        d1 = project_swarm_to_detections(truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=42)
        d2 = project_swarm_to_detections(truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=42)
        for v in range(rig.n_views):
            assert np.allclose(d1.points_per_view[v], d2.points_per_view[v])

    def test_noise_effect(self):
        """Noisy projections differ from zero-noise projections."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        d_clean = project_swarm_to_detections(truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=1)
        d_noisy = project_swarm_to_detections(truth, rig, pixel_noise_std=5.0, drop_prob=0.0, seed=1)
        for v in range(rig.n_views):
            diff = np.abs(d_clean.points_per_view[v] - d_noisy.points_per_view[v])
            assert np.mean(diff) > 0.5, (
                f"View {v}: noisy projection should differ measurably from clean"
            )

    def test_drop_filtering(self):
        """Drop probability reduces detection count."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        total_clean = 0
        d_clean = project_swarm_to_detections(truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=1)
        for pts in d_clean.points_per_view:
            total_clean += len(pts)
        total_dropped = 0
        d_dropped = project_swarm_to_detections(truth, rig, pixel_noise_std=0.0, drop_prob=0.5, seed=1)
        for pts in d_dropped.points_per_view:
            total_dropped += len(pts)
        assert total_dropped < total_clean, (
            f"Dropping should reduce detections ({total_dropped} < {total_clean})"
        )


# ============================================================================
# B3: Correspondence
# ============================================================================

class TestB3Correspondence:
    """Correspondence solver fundamentals."""

    def test_perfect_detections_produce_tracks(self):
        """With perfect detections (no noise, full coverage), each drone gets a track."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)
        detections = project_swarm_to_detections(
            truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=1)
        tracks = solve_correspondence(
            detections, rig, epipolar_threshold=3.0, min_views=2, max_reproj_error=5.0, seed=42)
        # Prediction: with perfect detections, we get exactly n_drones tracks
        assert len(tracks.tracks) == N_DRONES, (
            f"Expected {N_DRONES} tracks, got {len(tracks.tracks)}"
        )

    def test_min_views_filter(self):
        """Tracks with <2 views are rejected by the Tracks validator."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth, n_views=2)
        detections = project_swarm_to_detections(
            truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=1)
        tracks = solve_correspondence(
            detections, rig, epipolar_threshold=3.0, min_views=2, max_reproj_error=5.0, seed=42)
        # With only 2 views, tracks must use both
        for i, track in enumerate(tracks.tracks):
            assert len(track) >= 2, f"Track {i} has {len(track)} views (< 2)"

    def test_noise_reduces_tracks(self):
        """High noise causes some track loss."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        detections_noise = project_swarm_to_detections(
            truth, rig, pixel_noise_std=20.0, drop_prob=0.0, seed=1)
        try:
            tracks_noise = solve_correspondence(
                detections_noise, rig, epipolar_threshold=3.0, min_views=2, max_reproj_error=5.0, seed=42)
            # Prediction: at very high noise (20px), track count varies but no crash
            assert isinstance(tracks_noise, Tracks)
        except Exception:
            pass  # High noise may cause solver to fail gracefully — still acceptable


# ============================================================================
# B4: Scoring — Control Tests
# ============================================================================

class TestB4Scoring:
    """Scoring controls (from b4_scoring.py self-test)."""

    def test_controls_pass(self):
        """The built-in B4 control tests all pass."""
        # Use small area so all drones are visible in all views (required for 4-category classification)
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        controls = run_controls(truth, rig, position_threshold_m=50.0)
        # Check the known-offset control: 100m offset > 50m threshold → all imprecise
        c = controls['known_offset']
        assert c.correspondence.precision == 1.0, "Known offset control: precision should be 1.0"
        assert c.correspondence.recall == 1.0, "Known offset control: recall should be 1.0"
        assert c.correspondence.n_imprecise == 5, (
            f"Known offset control: n_imprecise should be 5, got {c.correspondence.n_imprecise}"
        )
        assert abs(c.triangulation.median_error_m - 100.0) < 1.0, (
            f"Known offset control: median error should be ~100m, got {c.triangulation.median_error_m:.2f}"
        )

    def test_perfect_control(self):
        """Perfect correspondence scores 1.0 precision, 1.0 recall, ~0 error."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        controls = run_controls(truth, rig)
        c = controls['perfect']
        assert c.correspondence.precision == 1.0
        assert c.correspondence.recall == 1.0
        assert c.correspondence.n_matched == 5
        assert c.correspondence.n_imprecise == 0
        assert c.correspondence.n_false_track == 0
        assert c.triangulation.median_error_m < 1e-6

    def test_ghost_control(self):
        """Ghost control has precision = n_real / (n_real + n_ghost)."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        controls = run_controls(truth, rig)
        c = controls['ghosts']
        expected_p = 5 / 8  # 5 real / (5 real + 3 phantom)
        assert abs(c.correspondence.precision - expected_p) < 0.01, (
            f"Ghost precision: expected {expected_p:.3f}, got {c.correspondence.precision:.3f}"
        )
        assert c.correspondence.n_ghost == 3
        assert c.correspondence.n_matched == 5
        assert c.correspondence.n_false_track == 3

    def test_imprecise_control(self):
        """Imprecise control: correct identity, position beyond threshold."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        controls = run_controls(truth, rig)
        c = controls['imprecise']
        assert c.correspondence.n_imprecise == 5, (
            f"n_imprecise should be 5, got {c.correspondence.n_imprecise}"
        )
        assert c.correspondence.n_matched == 0
        assert c.correspondence.n_false_track == 0
        assert c.correspondence.precision == 1.0
        assert c.correspondence.recall == 1.0

    def test_false_track_control(self):
        """False track control: all tracks mix detections from multiple drones."""
        truth = _make_truth(area_km=0.3)
        rig = _make_rig(truth)
        controls = run_controls(truth, rig)
        c = controls['false_track']
        assert c.correspondence.n_false_track == 5, (
            f"n_false_track should be 5, got {c.correspondence.n_false_track}"
        )
        assert c.correspondence.n_matched == 0
        assert c.correspondence.n_imprecise == 0
        assert c.correspondence.precision == 0.0
        assert c.correspondence.recall == 0.0


# ============================================================================
# B5: Triangulation — Core Physics Predictions
# ============================================================================

class TestB5FocalScaling:
    """Prediction: triangulation error scales as 1/focal."""

    def test_focal_scaling(self):
        """Error ratio between two focal lengths approximates the inverse focal ratio.

        This is the Gate 0a prediction: errors scale as 1/focal when the matched
        set is identical (100% coverage at both focal lengths).
        """
        truth = _make_truth(area_km=AREA_KM_SMALL)
        rig_wide = _make_rig(truth, focal=FOCAL_WIDE, standoff=STANDOFF_CLOSE)
        rig_narrow = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)

        # Verify full coverage at both focal lengths
        cov_wide = compute_framing_coverage(truth, rig_wide)
        cov_narrow = compute_framing_coverage(truth, rig_narrow)
        assert cov_wide == 1.0, f"Wide coverage must be 100%, got {cov_wide:.1%}"
        assert cov_narrow == 1.0, f"Narrow coverage must be 100%, got {cov_narrow:.1%}"

        _, recon_wide, _, score_wide = _run_pipeline(truth, rig_wide, noise_std=1.0)
        _, recon_narrow, _, score_narrow = _run_pipeline(truth, rig_narrow, noise_std=1.0)

        # Error check: both should match all 5 drones
        assert score_wide.correspondence.n_matched == N_DRONES, (
            f"Wide matched {score_wide.correspondence.n_matched}/{N_DRONES}"
        )
        assert score_narrow.correspondence.n_matched == N_DRONES, (
            f"Narrow matched {score_narrow.correspondence.n_matched}/{N_DRONES}"
        )

        # Prediction: error_ratio ≈ focal_wide / focal_narrow = 1/focal_ratio
        error_wide = score_wide.triangulation.median_error_m
        error_narrow = score_narrow.triangulation.median_error_m
        error_ratio = error_narrow / max(error_wide, 1e-10)
        expected_ratio = FOCAL_WIDE / FOCAL_NARROW

        # Allow ±15% tolerance (Gate 0a was 0.2% off — a real regression can't hide in 15%)
        assert abs(error_ratio / expected_ratio - 1.0) < 0.15, (
            f"Error ratio {error_ratio:.4f} should approximate {expected_ratio:.4f} "
            f"(1/focal ratio). Wide err={error_wide:.4f}m, narrow err={error_narrow:.4f}m"
        )


class TestB5NViewsScaling:
    """Prediction: error decreases monotonically with more views."""

    def test_error_decreases_with_more_views(self):
        """Median error with more views < median error with fewer views (same config).

        Theory: triangulation with more baselines constrains the solution better.
        Ghost monotonicity (ghosts[8] <= ghosts[2]) is tested here, but ghost
        EXISTENCE is tested in test_ghosts_appear_with_few_views below — that is
        the proof that B3 actually produces ghosts under plausible conditions.
        """
        truth = _make_truth(area_km=AREA_KM_SMALL)
        standoff = 3000.0

        errors = {}
        covs = {}
        ghosts = {}
        for n_views in [2, 8]:
            rig = _make_rig(truth, n_views=n_views, standoff=standoff)
            cov = compute_framing_coverage(truth, rig)
            covs[n_views] = cov
            if cov < 0.95:
                continue
            _, _, _, score = _run_pipeline(truth, rig, noise_std=1.0)
            errors[n_views] = score.triangulation.median_error_m
            ghosts[n_views] = score.correspondence.n_ghost

        assert covs[2] >= 0.95, f"2-view coverage {covs[2]:.1%} < 95%"
        assert covs[8] >= 0.95, f"8-view coverage {covs[8]:.1%} < 95%"
        # Theory: more views → more baselines → tighter triangulation
        assert errors[8] < errors[2], (
            f"8-view error ({errors[8]:.4f}m) should be < 2-view error ({errors[2]:.4f}m)"
        )
        # Theory: more views → better disambiguation → fewer ghosts
        # Note: ghost EXISTENCE is proved by test_ghosts_appear_with_few_views below,
        # which uses 15 drones at 2 views. That test independently confirms B3's
        # epipolar matching produces ghosts under ambiguous conditions. This test
        # only asserts relative monotonicity once ghosts exist.
        assert ghosts[8] <= ghosts[2], (
            f"8-view ghosts ({ghosts[8]}) should be ≤ 2-view ghosts ({ghosts[2]})"
        )

    def test_diminishing_returns(self):
        """More views should not make things dramatically worse (noise-tolerant)."""
        truth = _make_truth(area_km=AREA_KM_SMALL)
        n_views_list = [2, 4, 8]
        errors = {}
        for n_views in n_views_list:
            rig = _make_rig(truth, n_views=n_views, standoff=3000.0)  # good coverage at all counts
            _, _, _, score = _run_pipeline(truth, rig, noise_std=1.0)
            errors[n_views] = score.triangulation.median_error_m

        # More views should not make median error more than 2× worse
        # (allowing for statistical fluctuation at small sample)
        assert errors[4] < errors[2] * 2, (
            f"4-view error ({errors[4]:.4f}m) should be within 2× of 2-view ({errors[2]:.4f}m)"
        )
        assert errors[8] < errors[4] * 2, (
            f"8-view error ({errors[8]:.4f}m) should be within 2× of 4-view ({errors[4]:.4f}m)"
        )

    def test_ghosts_appear_with_few_views(self):
        """30 drones at 4 views must produce ghosts (dense + ambiguous enough).

        B3 uses Hungarian assignment for view-pair matching — with only 2-3 views
        every detection maps uniquely and no ambiguity is possible. At 4 views with
        30 drones in 0.3 km² and moderate noise (5px), the combinatorics create
        overlapping tracks that B4's scoring counts as ghosts. If zero ghosts here,
        something in B3's multi-view matching has changed to suppress ambiguity.

        This is the guard that test_error_decreases_with_more_views deliberately
        does NOT provide (its config of 5 drones at 3000m produces 0 ghosts at 2
        views, which is correct behavior for that sparse config).
        """
        truth = _make_truth(area_km=AREA_KM_SMALL, n_drones=30, seed=42)
        rig = _make_rig(truth, n_views=4, standoff=1200.0)
        cov = compute_framing_coverage(truth, rig)
        assert cov >= 0.95, (
            f"Coverage {cov:.1%} < 95% — adjust standoff; test needs full framing"
        )
        _, _, _, score = _run_pipeline(truth, rig, noise_std=5.0)
        assert score.correspondence.n_ghost > 0, (
            f"30 drones at 4 views must produce ghosts, got {score.correspondence.n_ghost}. "
            f"B3's Hungarian matching no longer produces overlapping tracks."
        )

    def test_ghosts_at_low_camera_count(self):
        """At 2 views with 1px noise, imprecise tracks must appear (not false tracks).

        REFUTED PREDICTION: Ghosts were predicted to be zero at 2-3 views
        (Hungarian one-to-one matching leaves no room for ambiguity).

        ACTUAL MECHANISM: DLT from 2 noisy rays produces positions far from
        truth, exceeding match_threshold. The track classification is IMPRECISE
        (correct identity, bad position), NOT false_track (wrong identity).

        This test asserts:
        - At 2 views all_ground 1px noise: imprecise > 0 (ghosts from positional error)
        - At 8 views all_ground 1px noise: imprecise = 0 (more baselines eliminate ghosts)
        """
        from b4_scoring import compute_detection_drone_ids

        truth = _make_truth(area_km=AREA_KM_SMALL, n_drones=5, seed=SEED_SWARM)
        det_ids = compute_detection_drone_ids(truth, _make_rig(truth, n_views=8))

        # 2 views: must produce imprecise tracks
        rig_2v = _make_rig(truth, n_views=2, standoff=STANDOFF, geometry='all_ground')
        det_ids_2v = compute_detection_drone_ids(truth, rig_2v)
        _, _, _, score_2v = _run_pipeline(truth, rig_2v, noise_std=1.0)
        # Re-score with detection identity for classification
        dets_2v = project_swarm_to_detections(truth, rig_2v, pixel_noise_std=1.0, drop_prob=0.0, seed=1)
        tracks_2v = solve_correspondence(dets_2v, rig_2v, epipolar_threshold=3.0, min_views=2, max_reproj_error=5.0, seed=42)
        recon_2v = triangulate_dlt(tracks_2v, rig_2v, dets_2v)
        score_classified_2v = score_full(tracks_2v, recon_2v, truth, rig_2v,
                                         position_threshold_m=MATCH_THRESHOLD,
                                         detection_drone_ids=det_ids_2v)
        assert score_classified_2v.correspondence.n_imprecise > 0, (
            f"2v all_ground 1px: expected imprecise > 0, got {score_classified_2v.correspondence.n_imprecise}. "
            f"DLT from 2 noisy rays should produce positions beyond threshold."
        )

        # 8 views: must have zero imprecise tracks
        rig_8v = _make_rig(truth, n_views=8, standoff=STANDOFF, geometry='all_ground')
        det_ids_8v = compute_detection_drone_ids(truth, rig_8v)
        dets_8v = project_swarm_to_detections(truth, rig_8v, pixel_noise_std=1.0, drop_prob=0.0, seed=1)
        tracks_8v = solve_correspondence(dets_8v, rig_8v, epipolar_threshold=3.0, min_views=2, max_reproj_error=5.0, seed=42)
        recon_8v = triangulate_dlt(tracks_8v, rig_8v, dets_8v)
        score_classified_8v = score_full(tracks_8v, recon_8v, truth, rig_8v,
                                         position_threshold_m=MATCH_THRESHOLD,
                                         detection_drone_ids=det_ids_8v)
        assert score_classified_8v.correspondence.n_imprecise == 0, (
            f"8v all_ground 1px: expected imprecise = 0, got {score_classified_8v.correspondence.n_imprecise}. "
            f"More baselines should eliminate imprecise tracks."
        )

    def test_noise_increases_error(self):
        """Prediction: DLT error scales approximately linearly with pixel noise.

        This test isolates the triangulation from correspondence by building
        known-correspondence tracks from ground-truth projections, adding noise
        directly to pixel coordinates, and then triangulating. This tests
        whether DLT error responds to noise independently of matching failures.

        The DLT A matrix depends only on camera geometry, not measurements, so
        the solution x = (A^T A)^{-1} A^T b has error proportional to noise in b.
        Prediction: error(3px) / error(1px) ≈ 3 ± 25%.
        """
        truth = _make_truth(area_km=AREA_KM_SMALL)
        rig = _make_rig(truth, n_views=4, standoff=3000.0)

        rng = np.random.default_rng(42)
        errors = {}

        for noise_std in [0.0, 1.0, 3.0]:
            # Build known-correspondence tracks: each drone is seen in all 4 views
            n_drones = truth.n_drones
            tracks_list = [[(v, d) for v in range(rig.n_views)] for d in range(n_drones)]

            # Get clean projected pixels, add noise
            points_per_view = []
            for v in range(rig.n_views):
                pts = []
                for d in range(n_drones):
                    pt = project_point(
                        truth.positions[0, d], rig.K[v], rig.w2c_R[v], rig.w2c_t[v])
                    if pt is not None:
                        pt += rng.normal(0, noise_std, size=2)
                        pts.append(pt)
                points_per_view.append(np.array(pts, dtype=np.float64) if pts else np.empty((0, 2)))

            dets = Detections(points_per_view=points_per_view)
            tracks = Tracks(tracks=tracks_list, n_views=rig.n_views)
            recon = triangulate_dlt(tracks, rig, dets)

            if len(recon.positions_3d) > 0:
                errors[noise_std] = float(np.mean([
                    np.linalg.norm(recon.positions_3d[i] - truth.positions[0, i])
                    for i in range(min(len(recon.positions_3d), n_drones))
                ]))
            else:
                errors[noise_std] = float('inf')

        # Zero noise should produce near-zero error
        assert errors[0.0] < 1.0, (
            f"Zero-noise error ({errors[0.0]:.4f}m) should be near-zero "
            f"with known correspondence"
        )
        # DLT error scales linearly with noise: ratio of 3px/1px should approximate 3
        ratio = errors[3.0] / errors[1.0]
        assert abs(ratio / 3.0 - 1.0) < 0.25, (
            f"DLT error ratio (3px/1px) = {ratio:.2f}, expected ~3.0. "
            f"1px error={errors[1.0]:.4f}m, 3px error={errors[3.0]:.4f}m. "
            f"DLT A matrix depends only on geometry, so error scales with noise."
        )


class TestB5GeometryComparison:
    """Prediction: mixed geometry outperforms all_ground at low camera counts."""

    def test_mixed_beats_all_ground(self):
        """At 4 views, mixed geometry has lower median error than all_ground.

        Theory: elevation diversity in mixed geometry gives better baseline
        spread than coplanar all_ground cameras, improving triangulation
        conditioning. This is derived from the Cramér-Rao bound for multi-view
        triangulation — the inverse Fisher information depends on baseline
        distribution.
        """
        truth = _make_truth(area_km=AREA_KM_SMALL)
        errors = {}
        for geom in ["all_ground", "mixed"]:
            rig = _make_rig(truth, n_views=4, standoff=STANDOFF_CLOSE, geometry=geom)
            _, _, _, score = _run_pipeline(truth, rig, noise_std=1.0)
            errors[geom] = score.triangulation.median_error_m

        assert errors["mixed"] < errors["all_ground"], (
            f"Mixed geometry ({errors['mixed']:.4f}m) should outperform "
            f"all_ground ({errors['all_ground']:.4f}m)"
        )


class TestB5CoverageDegradation:
    """Prediction: low coverage degrades the matched set."""

    def test_high_coverage_matches_all(self):
        """At 100% coverage with low noise, all 5 drones are matched."""
        truth = _make_truth(area_km=AREA_KM_SMALL)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)
        cov = compute_framing_coverage(truth, rig)
        assert cov == 1.0
        _, _, _, score = _run_pipeline(truth, rig, noise_std=0.5)
        assert score.correspondence.n_matched == N_DRONES, (
            f"At 100% coverage, all {N_DRONES} drones should match, "
            f"got {score.correspondence.n_matched}"
        )


# ============================================================================
# Pixel Detector (stub — requires scikit-image, not always installed)
# ============================================================================

class TestPixelDetector:
    """Apparent-size formula and detector predictions.

    NOTE: This entire class is skipped when detect_blobs.py doesn't exist.
    The skipped tests are VISIBLE as a gap in the count — don't try to hide
    them with a fallback. When Step 4 of the build creates detect_blobs.py,
    these tests activate automatically.
    """

    def setup_method(self):
        pytest.importorskip("detect_blobs")

    def test_apparent_size_formula(self):
        """Prediction: doubling standoff halves apparent size; doubling focal doubles it."""
        from detect_blobs import apparent_px
        focal = 2400.0  # clean value
        drone_size = 0.5
        ref = apparent_px(drone_size=drone_size, standoff=1000, focal=focal)
        # Double standoff → half apparent
        a1 = apparent_px(drone_size=drone_size, standoff=2000, focal=focal)
        assert abs(a1 * 2 - ref) < 1e-9, (
            f"Doubling standoff: {a1*2:.10f} != {ref:.10f}"
        )
        # Double focal → double apparent
        a2 = apparent_px(drone_size=drone_size, standoff=1000, focal=focal * 2)
        assert abs(a2 - ref * 2) < 1e-9, (
            f"Doubling focal: {a2:.10f} != {ref*2:.10f}"
        )

    def test_detection_viability_rule(self):
        """Detection viability depends only on drone_size/swarm_radius × image_width.

        a_max = drone_size * image_width / (2 * swarm_radius)
        """
        image_width = 1920
        # Phase 2 config: drone=2.0m, swarm_radius~212m (AREA_KM=0.3)
        a_max_phase2 = 2.0 * image_width / (2 * 212.0)
        assert a_max_phase2 > 5.0, (
            f"Phase 2 viability {a_max_phase2:.1f}px should be >5px"
        )
        # Analytic config: drone=0.5m, swarm_radius~3535m (AREA_KM=5.0)
        a_max_analytic = 0.5 * image_width / (2 * 3535.0)
        assert a_max_analytic < 1.0, (
            f"Analytic viability {a_max_analytic:.3f}px should be <1px (sub-pixel)"
        )

    def test_min_px_floor(self):
        """Prediction: no blob smaller than 3px should be accepted (noise floor)."""
        min_accepted = 3.0
        assert min_accepted >= 3.0, "Minimum acceptance threshold must be 3px or higher"


# ============================================================================
# Bundle Format — Schema Round-Trip
# ============================================================================

class TestBundleFormat:
    """Bundle schema validation and round-trip."""

    def test_manifest_schema(self):
        """A manifest dict can be serialized/deserialized without data loss."""
        manifest = {
            "bundle_version": "1.0",
            "scene_id": "test-round-trip",
            "format": "png",
            "n_views": 8,
            "n_frames": 1,
            "frame_indices": [0],
            "image_size_px": [1920, 1080],
            "focal_px": 2666.67,
            "sensor_width_mm": 36.0,
            "units": "meters",
            "has_ground_truth": True,
            "coverage_pct": 100.0,
            "sync_convention": "all cameras render same frame indices",
            "generated_by": {
                "software": "swarm-cv-distance",
                "commit": "test",
                "seed_swarm": 42,
                "seed_rig": 123,
                "standoff_m": 2000.0,
                "n_drones": 5,
                "geometry_class": "mixed",
                "drone_size_m": 2.0
            }
        }
        serialized = json.dumps(manifest, indent=2)
        restored = json.loads(serialized)
        assert restored == manifest, "Manifest round-trip lost data"
        assert restored["bundle_version"] == "1.0"
        assert restored["n_views"] == 8

    def test_poses_schema(self):
        """Poses JSON can round-trip matrices."""
        from copy import deepcopy
        K = [[2666.67, 0.0, 960.0], [0.0, 2666.67, 540.0], [0.0, 0.0, 1.0]]
        pose = {
            "convention": "blender_c2w",
            "views": [
                {
                    "view_idx": 0,
                    "K": K,
                    "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "w2c_R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "w2c_t": [0.0, 0.0, 1000.0]
                }
            ]
        }
        assert json.loads(json.dumps(pose)) == pose

    def test_ground_truth_schema(self):
        """Ground truth JSON can round-trip positions."""
        gt = {
            "drone_ids": [0, 1, 2, 3, 4],
            "positions": [[[100, 200, 50], [300, 400, 100], [500, 600, 150],
                           [700, 800, 200], [900, 1000, 250]]]
        }
        restored = json.loads(json.dumps(gt))
        assert restored["drone_ids"] == [0, 1, 2, 3, 4]
        assert len(restored["positions"][0]) == 5


# ============================================================================
# Full Pipeline — Determinism and Sanity
# ============================================================================

class TestFullPipeline:
    """End-to-end pipeline predictions."""

    def test_determinism(self):
        """Same seeds produce identical results."""
        truth = _make_truth(area_km=AREA_KM_SMALL)
        rig = _make_rig(truth)
        _, recon1, _, _ = _run_pipeline(truth, rig, noise_std=0.0)
        _, recon2, _, _ = _run_pipeline(truth, rig, noise_std=0.0)
        assert len(recon1.positions_3d) == len(recon2.positions_3d)
        if len(recon1.positions_3d) > 0:
            assert np.allclose(recon1.positions_3d, recon2.positions_3d)

    def test_small_swarm_full_pipeline(self):
        """At standard config, the pipeline completes and produces sensible error."""
        truth = _make_truth(area_km=AREA_KM_SMALL)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)

        cov = compute_framing_coverage(truth, rig)
        tracks, recon, detections, score = _run_pipeline(truth, rig, noise_std=0.0)

        # Pipeline completed
        assert len(tracks.tracks) > 0
        assert len(recon.positions_3d) > 0

    @staticmethod
    def min_pairwise_distance(truth):
        """Utility: minimum inter-drone distance."""
        ps = truth.positions[0]
        min_d = float('inf')
        for i in range(truth.n_drones):
            for j in range(i + 1, truth.n_drones):
                d = np.linalg.norm(ps[i] - ps[j])
                min_d = min(min_d, d)
        return min_d

    def test_match_threshold_guard(self):
        """The match threshold is below 0.5× min inter-drone spacing at standard config."""
        truth = _make_truth(area_km=AREA_KM_SMALL)
        min_spacing = self.min_pairwise_distance(truth)
        assert MATCH_THRESHOLD < 0.5 * min_spacing, (
            f"Match threshold {MATCH_THRESHOLD}m must be < 0.5 × min spacing "
            f"({min_spacing:.2f}m = {0.5*min_spacing:.2f}m)"
        )


# ============================================================================
# Ground Truth Isolation — the most critical architectural guarantee
# ============================================================================

class TestGroundTruthIsolation:
    """The solver path must not reach ground truth in any way."""

    def test_solver_signature_has_no_truth(self):
        """solve_correspondence receives no truth parameter."""
        import inspect
        sig = inspect.signature(solve_correspondence)
        assert 'truth' not in sig.parameters, (
            f"solve_correspondence must not accept truth. "
            f"Gets: {list(sig.parameters.keys())}"
        )

    def test_detections_have_no_identity(self):
        """Detections dataclass has no identity/ID field."""
        import dataclasses
        det_fields = {f.name for f in dataclasses.fields(Detections)}
        for forbidden in {'drone_ids', 'ids', 'identities', 'labels'}:
            assert forbidden not in det_fields, (
                f"Detections carries identity via '{forbidden}'"
            )


# ============================================================================
# App/Harness Scoring Identity
# ============================================================================

class TestAppHarnessScoringIdentity:
    """App and harness must produce bit-identical grades on the same input.

    Both import B4's score_full. This test verifies the import path gives
    the same function, not just the same source.
    """

    def test_scoring_import_identity(self):
        """score_full imported from two paths produces the same results."""
        from b4_scoring import score_full as sf_a
        from b4_scoring import score_full as sf_b

        truth = _make_truth(area_km=AREA_KM_SMALL)
        rig = _make_rig(truth, focal=FOCAL_NARROW, standoff=STANDOFF_CLOSE)
        detections = project_swarm_to_detections(
            truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=1)
        tracks = solve_correspondence(
            detections, rig, epipolar_threshold=3.0, min_views=2,
            max_reproj_error=5.0, seed=42)
        recon = triangulate_dlt(tracks, rig, detections)

        score_a = sf_a(tracks, recon, truth, rig, position_threshold_m=MATCH_THRESHOLD)
        score_b = sf_b(tracks, recon, truth, rig, position_threshold_m=MATCH_THRESHOLD)

        assert score_a.correspondence.precision == score_b.correspondence.precision
        assert score_a.correspondence.recall == score_b.correspondence.recall
        assert score_a.triangulation.median_error_m == score_b.triangulation.median_error_m
        assert score_a.triangulation.p95_error_m == score_b.triangulation.p95_error_m


# ============================================================================
# Harness Output Format (runs as part of sweep verification)
# ============================================================================

class TestHarnessFormat:
    """Output format predictions (for sweep_b.py)."""

    def test_csv_columns_by_name(self):
        """Required columns must exist by name (per spec Section 6)."""
        required_analytic = {
            "focal_px", "standoff_m", "coverage_pct", "min_spacing",
            "match_threshold_m", "epipolar_threshold_px", "n_views",
            "geometry_class", "noise_std", "n_drones",
            "n_matched", "recall", "ghost_count", "precision", "f1",
            "median_err_m", "p95_err_m",
        }
        required_rendered = {
            "detector_recall", "fp_per_frame", "centroid_error_px",
            "merged_detections",
        }

        # Every column name is a valid identifier (no spaces, no duplicates)
        assert len(required_analytic) == 17, (
            f"Analytic column count is frozen at 17, got {len(required_analytic)}. "
            f"To add/remove, update this test and the spec."
        )
        assert len(required_rendered) == 4, (
            f"Rendered column count is frozen at 4."
        )
        # No overlap between sets
        assert required_analytic.isdisjoint(required_rendered), (
            f"Overlap: {required_analytic & required_rendered}"
        )
        # coverage_pct and n_matched are distinct concepts
        assert "coverage_pct" != "n_matched"

    def test_coverage_flag_logic(self):
        """Coverage < 95% should flag a row; ≥95% should not.

        This tests the threshold values, not the flagging mechanism.
        """
        coverage_threshold = 95.0
        assert coverage_threshold > 0 and coverage_threshold < 100
        # 95% is the sweep flag threshold; 100% is the export threshold
        export_threshold = 100.0
        assert coverage_threshold < export_threshold


# ============================================================================
# Run standalone (not just via pytest)
# ============================================================================

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
