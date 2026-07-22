"""
M3 rig-coverage validation, phase 2 of 2 (report side).

Reads the object-index EXRs produced by validate_rig_render.py and reports
REAL per-camera visible-drone counts and cross-camera overlap. Pass bar:
every rig must give >= 18/20 drones with >= 2 cameras (2-camera visibility
is the triangulation minimum; 18/20 matches the coverage the old validated
2km rig achieved).

Run from the project venv (needs OpenEXR):
    venv/bin/python blender_addon/validate_rig_report.py
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import OpenEXR

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rig_validation")
N_DRONES = 20
MIN_TRIANGULATABLE = 18

rig_dirs = sorted(d for d in glob.glob(os.path.join(OUT_ROOT, "*")) if os.path.isdir(d))
if not rig_dirs:
    sys.exit(f"no rig renders found under {os.path.normpath(OUT_ROOT)} -- "
             "run validate_rig_render.py first")

all_pass = True
for rig_dir in rig_dirs:
    name = os.path.basename(rig_dir)
    seen = defaultdict(set)
    exrs = sorted(glob.glob(os.path.join(rig_dir, "cam*_id_.exr")))
    per_cam = []
    for ci, path in enumerate(exrs):
        f = OpenEXR.File(path)
        arr = np.array(f.parts[0].channels["id_.V"].pixels)
        ids = {int(v) for v in np.unique(arr) if v > 0}
        per_cam.append(len(ids))
        for d in ids:
            seen[d].add(ci)
    triangulatable = sum(1 for cams in seen.values() if len(cams) >= 2)
    ok = triangulatable >= MIN_TRIANGULATABLE
    all_pass &= ok
    print(f"{name:>10}: per-camera visible {per_cam}, "
          f"ever-visible {len(seen)}/{N_DRONES}, "
          f">=2-camera {triangulatable}/{N_DRONES} "
          f"[{'PASS' if ok else 'FAIL'}]")

print("RIG COVERAGE:", "PASS" if all_pass else "FAIL")
sys.exit(0 if all_pass else 1)
