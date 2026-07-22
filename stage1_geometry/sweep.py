"""
Stage 1 parameter sweep: camera count x pixel-noise tolerance.

Reuses multiview_triangulation_test.py's scene/camera/triangulation code
unchanged, varying N_CAMERAS and PIXEL_NOISE_STD to characterize the
tradeoff before committing to a Stage 2 Blender camera rig.
"""

import contextlib
import io

import numpy as np

import multiview_triangulation_test as mvt

N_DRONES = 20
RING_RADIUS_M = 1200.0
RING_HEIGHT_M = 150.0
D_MAX = 800.0  # placeholder, same as multiview_triangulation_test.py -- not yet validated

N_CAMERAS_VALUES = [2, 3, 4, 6]
PIXEL_NOISE_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
N_TRIALS = 20  # per (n_cams, noise_std) cell, to average out single-draw variance

drones = mvt.make_swarm(n_drones=N_DRONES, seed=1)

rows = []
for n_cams in N_CAMERAS_VALUES:
    cameras = mvt.place_ring_of_cameras(n_cams, RING_RADIUS_M, RING_HEIGHT_M)
    for noise_std in PIXEL_NOISE_VALUES:
        trial_recon, trial_mean_err, trial_edge_acc = [], [], []
        for trial in range(N_TRIALS):
            mvt.rng = np.random.default_rng(hash((n_cams, noise_std, trial)) % (2**32))
            detections = mvt.simulate_detections(drones, cameras, pixel_noise_std=noise_std)
            est_positions, n_views = mvt.reconstruct_swarm(cameras, detections, N_DRONES)

            with contextlib.redirect_stdout(io.StringIO()):
                dist_err, edge_accuracy = mvt.evaluate(drones, est_positions, D_MAX)

            trial_recon.append(int(np.sum(~np.isnan(est_positions).any(axis=1))))
            trial_mean_err.append(dist_err.mean() if len(dist_err) else np.nan)
            trial_edge_acc.append(edge_accuracy * 100)

        rows.append({
            "n_cams": n_cams,
            "noise_std": noise_std,
            "recon": np.mean(trial_recon),
            "mean_err": np.nanmean(trial_mean_err),
            "edge_accuracy": np.mean(trial_edge_acc),
        })

header = f"{'cams':>4} {'noise':>6} {'avg_recon':>10} {'avg_mean_err':>13} {'avg_edge_acc%':>14}"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r['n_cams']:>4} {r['noise_std']:>6.1f} {r['recon']:>7.1f}/{N_DRONES} "
          f"{r['mean_err']:>13.2f} {r['edge_accuracy']:>14.1f}")
