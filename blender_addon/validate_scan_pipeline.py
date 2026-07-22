"""
M4 scan-pipeline validation, rerunnable (same spirit as M3's
validate_rig_render.py / validate_rig_report.py): drives the addon's own
operators end-to-end (register -> generate swarm -> place rig -> scan) so
what's being checked is the real render + subprocess + triangulation path,
not a synthetic stand-in.

Run:
    "<blender binary>" --background --python blender_addon/validate_scan_pipeline.py
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swarm_scanner

N_DRONES = 20
N_CAMERAS = 6
DISPLAY_SCALE = 20.0  # same reasoning as M3: true-scale (0.5m) drones are
                      # subpixel at these standoffs -- see PROGRESS.md

swarm_scanner.register()
scene = bpy.context.scene
scene.swarm_scan.drone_count = N_DRONES
scene.swarm_scan.seed = 1
scene.swarm_scan.display_scale = DISPLAY_SCALE
scene.swarm_scan.camera_count = N_CAMERAS
scene.swarm_scan.camera_mode = "RANDOM"
scene.swarm_scan.scan_seed = 7

assert bpy.ops.swarm.generate_swarm() == {"FINISHED"}
assert bpy.ops.swarm.place_cameras() == {"FINISHED"}

result = bpy.ops.swarm.scan()
print("scan operator result:", result)
assert result == {"FINISHED"}, "scan operator did not finish cleanly"

r = swarm_scanner._LAST_SCAN
assert r is not None, "no scan result stored"

print(f"Triangulated: {r['n_triangulated']}/{r['n_total']}")
print(f"Overall adjacency accuracy: {r['overall_accuracy']*100:.1f}%")
if r["near_threshold_accuracy"] is not None:
    print(f"Near-D_MAX accuracy: {r['near_threshold_accuracy']*100:.1f}% "
          f"(D_MAX={r['d_max']:.0f}m)")
else:
    print(f"Near-D_MAX accuracy: n/a, no pairs in band (D_MAX={r['d_max']:.0f}m)")
print(f"Mean distance error: {r['mean_dist_error_m']:.2f} m")
print(f"Edges drawn (overlay): {len(r['edges'])}")

# Pass bar: matches M3's rig-coverage bar (>=18/20 triangulated) since a
# real rig achieving that coverage is a precondition for the scan itself;
# accuracy bar is generous (noise + real occlusion, not a tight tolerance).
MIN_TRIANGULATED = 18
MIN_ACCURACY = 0.85

ok = r["n_triangulated"] >= MIN_TRIANGULATED and r["overall_accuracy"] >= MIN_ACCURACY
print("SCAN PIPELINE VALIDATION:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
