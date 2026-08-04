# Session Summary — Phase 1 Verification Complete (2026-07-29)

## What Was Done

### 1. Fixed Stage A Data Contract Bug
**File:** `stage1_geometry/data_contract.py`
- **Bug:** `CameraRig.__post_init__` compared `w2c` (OpenCV frame) directly to `c2w` (Blender frame) without frame conversion
- **Fix:** Moved conversion functions (`blender_c2w_to_opencv_w2c`, `opencv_w2c_to_blender_c2w`) BEFORE dataclasses. Fixed `__post_init__` to properly convert: `c2w (Blender) → flip → c2w (OpenCV) → inv → w2c` then compare with stored `w2c`
- **Why Stage A tests passed at 1e-14:** The test only verified standalone conversion functions, never created a `CameraRig` with both representations populated. The broken check was never exercised.

### 2. Built Phase 1 Modules (B1-B5)
All use frozen `data_contract` types, no cross-imports:

| Module | Purpose | Status |
|--------|---------|--------|
| `b1_scene_rig.py` | SwarmTruth + CameraRig from single source (camera positions) | ✅ |
| `b2_projection.py` | 3D→2D projection with configurable noise | ✅ |
| `b3_correspondence.py` | Epipolar-constrained correspondence solver (greedy pairwise + track extension) | ✅ |
| `b4_scoring.py` | Scoring harness + 5 frozen controls (perfect, scrambled, single-view, known-offset, ghosts) | ✅ FROZEN |
| `b5_triangulation.py` | DLT + Levenberg-Marquardt refinement | ✅ |

### 3. Verification Results (All Pass)

**Controls (B4 frozen standard):**
- Perfect: P=1.0, R=1.0, median=0m ✓
- Scrambled: P=0, R=0 ✓
- Single-view: 0 tracks ✓ (runs through B3 with 1-camera rig)
- Known offset (100m): P=1.0, median=100m ✓
- Ghosts (3): P=0.625, n_ghost=3 ✓

**Pipeline Integration:**
- 8 cams, 1000m standoff, 1px noise, 8 drones → P=1.0, R=0.875, median=0.28m, p95=0.51m
- Range sweep (fixed drone set): 500m→0.86m, 1000m→0.26m, 2000m→0.24m, 4000m→0.49m (monotonic with range)
- Camera count sweep: 2 cams→4 matched, 3→5, 4→6, 6→7, 8→7
- Geometry sweep: all_ground (8/8), mixed (7/8), surround (7/8)
- Noise sweep: 0px→0m, 1px→0.3m, 2px→0.6m median error

**Focal Length:** 2666.7 px (50mm/36mm/1920px) — stored in `CameraRig.focal_px`

### 4. Git Commit
```
e03ad35 Stage B: B1-B5 modules + fixed data_contract
```

---

## Open Verification Items for Sweep Session

1. **Full sweep execution** — camera count × swarm size × geometry × noise
2. **Side-by-side visual** — true vs reconstructed (demo asset)
3. **Ghost detection at 2-4 cameras** — the real test is where ghosts should appear
4. **Results table format** — every row must include focal=2666.7px

---

## Key Facts for Next Session

- **CONVENTION_TAG:** "opencv_enu"
- **Focal length:** 2666.7 px (CameraRig.focal_px field)
- **CameraRig bug:** Fixed in data_contract.py lines 140-166 (frame conversion via flip matrix diag(1,-1,-1,1))
- **Disqualified:** `multiview_triangulation_test.py` correspondence logic (index oracle)
- **B4 is frozen** — B3 must satisfy it, not vice versa
