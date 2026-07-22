"""
Stage 1 D_MAX calibration against the 5km x 5km x 1km scene (2026-07-22
scale change -- see PROGRESS.md and scene_config.py).

D_MAX is chosen empirically (not from an RF link-budget) so that pairwise
reachability lands in the ~80-90% range Chen et al. report for N=55
(Table I) -- the goal is a graph sparse enough for the GA/PSO critical-node
search to have something to find, not a near-complete graph. Candidates are
the 80th/85th/90th percentile of the scene's true pairwise-distance
distribution.

Deliberately scoped to ONLY the distance-distribution calibration this
round -- the camera-count / near-threshold-accuracy sweep that used to
follow this in the same file needs the camera rig, which is a separate,
not-yet-done task now that the old 1200m-radius ring is sized for the
wrong (2km) scene. Re-add that sweep once the rig is re-derived for 5km.
"""

import numpy as np

import multiview_triangulation_test as mvt
from scene_config import N_DRONES, AREA_KM, HEIGHT_RANGE_M, SWARM_SEED

TARGET_REACHABILITIES = [0.80, 0.85, 0.90]

drones = mvt.make_swarm(n_drones=N_DRONES, area_km=AREA_KM,
                         height_range_m=HEIGHT_RANGE_M, seed=SWARM_SEED)
true_D = mvt.pairwise_distances(drones)
pairs_upper = true_D[np.triu_indices(N_DRONES, k=1)]

print(f"Scene pairwise-distance stats ({N_DRONES} drones, {AREA_KM}km x {AREA_KM}km x "
      f"{HEIGHT_RANGE_M/1000:.0f}km volume):")
print(f"  min={pairs_upper.min():.0f}m  median={np.median(pairs_upper):.0f}m  max={pairs_upper.max():.0f}m")
print(f"  total pairs: {len(pairs_upper)}")
print()

candidates = {t: float(np.percentile(pairs_upper, t * 100)) for t in TARGET_REACHABILITIES}
print("Candidate D_MAX values for target pairwise reachability:")
for t, d in candidates.items():
    actual_reach = (pairs_upper <= d).mean() * 100
    print(f"  target {t*100:.0f}% -> D_MAX = {d:.0f}m (actual reachability at this scene: {actual_reach:.1f}%)")
