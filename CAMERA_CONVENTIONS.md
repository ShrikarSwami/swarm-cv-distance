# Camera Conventions — Single Source of Truth

**Created:** 2026-07-29  
**Purpose:** Document the exact camera model conventions used in Stage 1 (analytic) and Stage 2 (render) so they agree exactly. Phase 2 is a drop-in, not a rewrite.

---

## 1. Coordinate Conventions

| Space | Axes | Description |
|-------|------|-------------|
| **World (ENU)** | X=East, Y=North, Z=Up | Stage 1 `make_swarm()` generates positions in this frame. Blender scene also uses this (Z up). |
| **OpenCV / CV Camera** | X=Right, Y=Down, Z=Forward | Standard computer vision convention. Used by `multiview_triangulation_test.py:Camera` class. |
| **Blender Camera** | X=Right, Y=Up, Z=**Back** (-Z forward) | Blender's native camera looks down its local -Z axis. |

### Blender → OpenCV Extrinsic Conversion

```python
# Blender camera-to-world matrix: M_bl (4x4), columns are [right, up, -forward, pos]
# OpenCV camera-to-world matrix: M_cv = M_bl @ diag(1, -1, -1, 1)
# Projection extrinsics (world-to-camera): R|t = M_cv^{-1}
```

**In practice:** Stage 1's `Camera` class builds `Camera._look_at_rotation()` (line 73-81) already produces the OpenCV convention:
```python
R = np.vstack([right, -up, forward])  # rows = X_right, Y_down, Z_forward
```
This is **world-to-camera** rotation. Combined with `t = -R @ position`, the projection matrix `P = K @ [R | t]` maps world points directly to pixels.

---

## 2. Intrinsics (K Matrix)

```python
K = [[focal_px, 0,        cx],
     [0,        focal_px, cy],
     [0,        0,        1  ]]
```

| Parameter | Value | Source |
|-----------|-------|--------|
| `focal_px` | **1400.0** pixels | `scene_config.py:FOCAL_PX` |
| `cx` | `width / 2` = **960.0** | Principal point at image center |
| `cy` | `height / 2` = **540.0** | Principal point at image center |
| Image size | **1920 × 1080** | `scene_config.py:IMAGE_SIZE` |
| Sensor width | **36.0 mm** (full-frame) | `render_clip.py` config, `rerender_sky.py` |
| Focal length (mm) | **50 mm** (configurable) | `render_clip.py` default, M1 sweep config |

**Note:** Stage 1 works in **focal length in pixels** (1400 px). Stage 2 Blender works in **focal length in mm** (50 mm) with `sensor_width=36mm`. The conversion is:
```
focal_px = focal_mm * (width_px / sensor_width_mm) = 50 * (1920 / 36) ≈ 2667 px
```
**Discrepancy:** Stage 1 uses 1400 px, Stage 2 uses 50mm/36mm → 2667 px. This is a **known mismatch** that must be resolved before Phase 2 drop-in.

---

## 3. Extrinsics (Camera Pose)

### Stage 1 (Analytic)
- **Format:** `Camera.P = K @ [R | t]` where `R` is **world-to-camera** (3×3), `t = -R @ position` (3×1)
- **Position:** World coordinates (ENU meters)
- **Rotation:** Built by `_look_at_rotation(eye, target)` → `R` rows = `[right, -up, forward]` (OpenCV convention)
- **Verification:** `P @ [X, Y, Z, 1]^T` → pixel coordinates directly

### Stage 2 (Blender Render)
- **Stored in `clip.npz`:** `extrinsics` = **camera-to-world** 4×4 matrix (`cam_obj.matrix_world`)
- **Columns:** `[right, up, -forward, position]` — Blender native
- **To use in Stage 1 pipeline:** Convert Blender→OpenCV via `M_cv = M_bl @ diag(1, -1, -1, 1)`, then invert for world-to-camera
- **Projection utility (`dataset_schema.project_positions`, `validate_ground_truth.project_to_pixels`):**
  ```python
  R = ext[:3, :3]           # camera-to-world rotation
  t = ext[:3, 3]            # camera position in world
  cam_coords = R.T @ (world_pos - t)  # world-to-camera
  pixels = (K @ cam_coords.T).T / cam_coords[:, 2:3]
  ```
  This correctly handles Blender's camera-to-world matrix.

---

## 4. Image Origin & Pixel Coordinates

| Convention | Value |
|------------|-------|
| **Origin** | **Top-left** (0, 0) |
| **X axis** | Right (column index) |
| **Y axis** | Down (row index) |
| **Principal point** | Center: `(width/2, height/2)` = (960, 540) |
| **Depth** | Positive Z = in front of camera (OpenCV) |

Both Stage 1 projection and Stage 2 validation use this convention consistently.

---

## 5. Sensor Fit / Aspect Ratio

| Setting | Value |
|---------|-------|
| Sensor width | 36.0 mm |
| Sensor height | Computed: `36.0 * (1080/1920) = 20.25 mm` |
| Sensor fit | **Horizontal** (width drives FOV, height follows aspect) |
| Pixel aspect | 1.0 (square pixels) |

Blender camera: `cam_data.sensor_width = 36.0`, `cam_data.lens = focal_mm`. Blender defaults to "Horizontal" sensor fit when sensor_width is set and resolution aspect matches.

---

## 6. Projection Pipeline Summary

### Stage 1 (Synthetic)
```python
cam = Camera(position, look_at, image_size=(1920,1080), focal_px=1400.0)
pixel = cam.project(world_point)  # Returns (u, v) or None if behind
```

### Stage 2 (Render → Validation)
```python
# From clip.npz
K = clip["K"][view]           # (3,3) focal_px = 50 * 1920/36 ≈ 2667
ext = clip["extrinsics"][view] # (4,4) camera-to-world (Blender)
pixels = project_to_pixels(world_pos, K, ext)  # Uses R.T @ (pos - t)
```

---

## 7. Known Discrepancies to Resolve Before Phase 2

| Issue | Stage 1 | Stage 2 | Resolution Needed |
|-------|---------|---------|-------------------|
| **Focal length (px)** | 1400 | ~2667 (50mm/36mm) | Pick one; update both configs |
| **Principal point** | (960, 540) | (960, 540) | ✅ Agrees |
| **Image origin** | Top-left | Top-left | ✅ Agrees |
| **Extrinsics storage** | World-to-camera R|t (implicit in P) | Camera-to-world 4×4 (Blender matrix_world) | Document conversion; use consistently |
| **Coordinate frame** | ENU world | ENU world (Blender Z-up) | ✅ Agrees |

---

## 8. Verification Checklist (Run Before Phase 2)

- [ ] Single drone at known position, single camera: hand-compute pixel, verify code matches
- [ ] Round-trip: world → pixel → ray → world (at known depth) returns original position
- [ ] Two cameras, known baseline: triangulate known point, error < 1mm (noiseless)
- [ ] Blender render → read EXR → project GT positions → match ID-pass centroids to < 1px
- [ ] Focal length unified across `scene_config.py`, `render_clip.py`, `rerender_sky.py`

---

## 9. Files to Update When Conventions Change

1. `stage1_geometry/scene_config.py` — `FOCAL_PX`, `IMAGE_SIZE`
2. `stage1_geometry/multiview_triangulation_test.py` — `Camera` class defaults
3. `render_clip.py` — default config `focal_mm`, `sensor_width_mm`
4. `blender_addon/render_config.py` — config schema
5. `blender_addon/rerender_sky.py` / `validate_real_render.py` — camera setup
6. `dataset_schema.py` — projection utilities
7. `validate_ground_truth.py` — validation logic
8. `CAMERA_CONVENTIONS.md` — this document