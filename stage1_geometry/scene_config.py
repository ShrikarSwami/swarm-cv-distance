"""
Shared scene/camera/D_MAX constants for Stage 1 and Stage 2.

Single source of truth so Stage 2's Blender scene, camera rig, and
adjacency threshold stay in lockstep with the Stage 1 assumptions they're
meant to validate against. Import this instead of re-declaring these
values in new scripts.
"""

N_DRONES = 20
AREA_KM = 2.0
ALTITUDE_SPREAD_M = 50.0
SWARM_SEED = 1

N_CAMERAS = 6
LOOK_AT = (0, 0, 100)

# Stage 1's camera-count/noise sweep used a flat ring at this radius/height.
# Kept as-is -- those sweep results (locked-in 6-camera decision) stand.
RING_RADIUS_M = 1200.0
RING_HEIGHT_M = 150.0

# Stage 2's actual rig: a flat ring at RING_HEIGHT_M views the swarm at only
# ~2.4 deg elevation (nearly edge-on), which caused severe real self-occlusion
# in the first render attempt (2026-07-22) that Stage 1's idealized frustum
# math never modeled -- only 8/20 drones had >=2-camera overlap, vs Stage 1's
# ~19.6/20 prediction. Switched to a dome: elevation spread across cameras,
# slant range held equal to the original ring's slant distance
# (sqrt(1200^2+50^2) ~= 1201m) so apparent drone size in frame is unchanged.
DOME_SLANT_RANGE_M = 1201.0
DOME_ELEV_MIN_DEG = 25.0
DOME_ELEV_MAX_DEG = 45.0

IMAGE_SIZE = (1920, 1080)
FOCAL_PX = 1400.0

# Locked in 2026-07-22: empirically calibrated to ~85% pairwise reachability
# on this scene (Chen et al. Table I targets ~80-90% for N=55), not derived
# from an RF link budget. Recalibrate against scene_config's own values if
# N_DRONES/AREA_KM/ALTITUDE_SPREAD_M ever change -- see
# stage1_geometry/sweep_dmax.py.
D_MAX = 1574.0
