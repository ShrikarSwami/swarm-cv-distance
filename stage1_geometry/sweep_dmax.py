"""
Stage 1 D_MAX calibration + camera-count sweep.

D_MAX is chosen empirically (not from an RF link-budget) so that pairwise
reachability in the Stage 1 synthetic scene lands in the ~80-90% range
Chen et al. report for N=55 (Table I) -- the goal is a graph sparse enough
for the GA/PSO critical-node search to have something to find, not a
near-complete graph. Candidates are the 80th/85th/90th percentile of the
scene's true pairwise-distance distribution.

For each candidate D_MAX, sweeps camera count x pixel noise as before, but
additionally reports "near-threshold" edge accuracy: accuracy restricted to
pairs whose TRUE distance sits within +/-20% of D_MAX. Those are the only
pairs noise can plausibly flip; overall edge accuracy is dominated by pairs
far from the threshold and hides exactly the failure mode we care about.
"""

import numpy as np

import multiview_triangulation_test as mvt
from scene_config import N_DRONES, RING_RADIUS_M, RING_HEIGHT_M, SWARM_SEED

N_CAMERAS_VALUES = [2, 3, 4, 6]
PIXEL_NOISE_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
N_TRIALS = 20
NEAR_THRESHOLD_MARGIN_FRAC = 0.20  # +/- 20% of D_MAX counts as "near threshold"
TARGET_REACHABILITIES = [0.80, 0.85, 0.90]

drones = mvt.make_swarm(n_drones=N_DRONES, seed=SWARM_SEED)
true_D = mvt.pairwise_distances(drones)
pairs_upper = true_D[np.triu_indices(N_DRONES, k=1)]

print("Scene pairwise-distance stats (20 drones, 2km-wide area, ~100m altitude):")
print(f"  min={pairs_upper.min():.0f}m  median={np.median(pairs_upper):.0f}m  max={pairs_upper.max():.0f}m")
print(f"  total pairs: {len(pairs_upper)}")
print()

candidates = {t: float(np.percentile(pairs_upper, t * 100)) for t in TARGET_REACHABILITIES}
print("Candidate D_MAX values for target pairwise reachability:")
for t, d in candidates.items():
    actual_reach = (pairs_upper <= d).mean() * 100
    print(f"  target {t*100:.0f}% -> D_MAX = {d:.0f}m (actual reachability at this scene: {actual_reach:.1f}%)")
print()


def near_threshold_accuracy(true_D, est_D, d_max, margin):
    valid = ~np.isnan(est_D)
    near = valid & (np.abs(true_D - d_max) <= margin)
    n_near_pairs = int(near.sum() / 2)  # matrix is symmetric, halve the count
    if near.sum() == 0:
        return np.nan, n_near_pairs
    true_adj = true_D <= d_max
    est_adj = est_D <= d_max
    agree = (true_adj == est_adj) & near
    return agree.sum() / near.sum() * 100, n_near_pairs


for target, d_max in candidates.items():
    margin = d_max * NEAR_THRESHOLD_MARGIN_FRAC
    print(f"=== D_MAX = {d_max:.0f}m (target {target*100:.0f}% reachability, "
          f"near-threshold band = +/-{margin:.0f}m) ===")
    header = (f"{'cams':>4} {'noise':>6} {'avg_recon':>10} {'overall_edge_acc%':>18} "
              f"{'near_thresh_acc%':>17} {'avg_n_near_pairs':>16}")
    print(header)
    print("-" * len(header))
    for n_cams in N_CAMERAS_VALUES:
        cameras = mvt.place_ring_of_cameras(n_cams, RING_RADIUS_M, RING_HEIGHT_M)
        for noise_std in PIXEL_NOISE_VALUES:
            trial_recon, trial_overall_acc, trial_near_acc, trial_near_n = [], [], [], []
            for trial in range(N_TRIALS):
                mvt.rng = np.random.default_rng(hash((n_cams, noise_std, trial, target)) % (2**32))
                detections = mvt.simulate_detections(drones, cameras, pixel_noise_std=noise_std)
                est_positions, n_views = mvt.reconstruct_swarm(cameras, detections, N_DRONES)
                est_D = mvt.pairwise_distances(est_positions)

                valid = ~np.isnan(est_D)
                true_adj = true_D <= d_max
                est_adj = est_D <= d_max
                overall_acc = ((true_adj == est_adj) & valid).sum() / max(valid.sum(), 1) * 100
                near_acc, n_near = near_threshold_accuracy(true_D, est_D, d_max, margin)

                trial_recon.append(int(np.sum(~np.isnan(est_positions).any(axis=1))))
                trial_overall_acc.append(overall_acc)
                trial_near_acc.append(near_acc)
                trial_near_n.append(n_near)

            print(f"{n_cams:>4} {noise_std:>6.1f} {np.mean(trial_recon):>7.1f}/{N_DRONES} "
                  f"{np.mean(trial_overall_acc):>18.1f} {np.nanmean(trial_near_acc):>17.1f} "
                  f"{np.mean(trial_near_n):>16.1f}")
    print()
