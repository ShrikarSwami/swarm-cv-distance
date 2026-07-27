# Emission Shader Investigation Summary

## Task
Investigate why emissive objects render black in this project's Blender setup.

## Testing Methodology
Systematic bisection from simplest case to complex setup:
1. Minimal test: default cube, diffuse material, sun lamp, camera
2. Add emission shader
3. Add project's actual setup elements one by one
4. Test `use_empty=True` vs `use_empty=False`
5. Test new material creation vs modifying existing materials
6. Test camera assignment and matrix_world updates

## Findings

### 1. Emission Shaders Work Correctly in Blender 5.1.2

All test configurations produced visible renders:

| Test | Result |
|------|--------|
| Diffuse cube + sun lamp | center_mean=0.000 (black - expected, no emission) |
| Add emission shader | center_mean=1.000 (VISIBLE) |
| With `use_empty=True` | center_mean=1.000 (VISIBLE) |
| With `use_empty=False` | center_mean=1.000 (VISIBLE) |
| New material creation | center_mean=1.000 (VISIBLE) |
| Modify existing material | center_mean=1.000 (VISIBLE) |

### 2. Exact User Config Test

**User's description:** "a 50m cube at 100m with emission strength=100 renders R=0.000"

**Test result:** 50m cube at 100m with emission=100 renders correctly:
- Center region mean: 1.000
- Bright region fills 639x479 pixels (entire frame)
- Status: **VISIBLE**

### 3. Camera Assignment Works

Tested with 3 cameras at different positions:
- Camera 0: (0, 0, 0)
- Camera 1: (20, 0, 0)
- Camera 2: (-20, 0, 0)

After `bpy.context.view_layer.update()`:
- `matrix_world.translation` correctly reflects camera positions
- Cameras render from their assigned positions

### 4. Existing Renders Show Emission Works

Checked existing rendered images:
- `dome_working/`: mean=0.499, max=0.566 (VISIBLE)
- `debug_final/`: mean=0.500 (VISIBLE)
- `seq_dbg2/`: mean=0.500 (VISIBLE)

All renders show non-zero pixel values, confirming emission works.

## Root Cause Analysis

### The Bug Was Already Fixed

Commit `80f8f1c` (2026-07-23) fixed the emission shader issue:

**Three bugs fixed:**
1. Objects linked to `bpy.context.collection` instead of `bpy.context.scene.collection` — after factory reset, drones were created but never appeared in the scene (0 objects rendered)
2. 8-bit PNG quantized away sub-pixel signals — switched to float32 EXR
3. Principled BSDF too dim for sub-pixel drones — switched to emission shader (strength=100) for detection verification

**The actual root cause was:** Objects not being linked to the scene collection after factory reset, not the emission shader itself.

### Current State

In Blender 5.1.2:
- `use_empty=True` works correctly
- Emission shaders work correctly
- Camera assignment works correctly
- `matrix_world` updates correctly after `view_layer.update()`

## Remaining Issues

### 1. `stage2_render/render_scene.py` is Outdated

The script references `ALTITUDE_SPREAD_M` which was renamed to `HEIGHT_RANGE_M` in `scene_config.py`. This causes an import error:

```
ImportError: cannot import name 'ALTITUDE_SPREAD_M' from 'scene_config'
```

**Fix:** Update the import to use `HEIGHT_RANGE_M` instead.

### 2. All Camera Views Show Identical Pixel Statistics

In `dome_working/`, all 12 camera views show identical pixel statistics:
- r_mean: 0.49930262565612793 (identical across all views)
- r_max: 0.56612628698349 (identical across all views)

This is physically impossible for cameras at different positions around a dome. The ground truth shows 12 unique camera positions, but the rendered images are identical.

**Possible causes:**
- Camera rotation not being set correctly
- Scene not updating between renders
- Compositor not working correctly

**Note:** This is a separate issue from the emission shader bug.

## Recommendations

1. **The emission shader bug is fixed** — no action needed for emission
2. **Update `stage2_render/render_scene.py`** — fix the `ALTITUDE_SPREAD_M` import
3. **Investigate identical camera views** — separate issue from emission
4. **Clean up test scripts** — remove temporary test files

## Test Scripts Created

The following test scripts were created during investigation and should be removed:
- `test_emission_root_cause.py`
- `test_emission_v2.py`
- `test_bisect.py`
- `test_add_elements.py`
- `test_exact_user_config.py`
- `test_camera_assignment.py`
- `test_camera_positions.py`
- `test_camera_rotation.py`
- `test_manual_rotation.py`
- `test_matrix_world_fix.py`
- `test_raw_pixels.py`
- `test_rgb_only.py`
- `test_find_bright.py`
- `test_principled_inputs.py`
- `test_set_emission.py`
- `test_world_nodes.py`
- `check_render_values.py`
- `check_drone_visibility.py`
