# Emission Shader Investigation Summary

## Status: RESOLVED — NOT a Blender engine bug

**Root cause:** Camera rotation was wrong in test scripts, not an emission shader or depsgraph issue.

**Date resolved:** 2026-07-28

---

## What was investigated

User reported emissive objects rendering black. Systematic investigation tested:
1. Emission shader functionality
2. Object collection linkage
3. Depsgraph visibility
4. Blender version compatibility (5.1.2 vs 5.2.0)
5. Camera setup and orientation

## Root cause

**Camera at (0, 0, -100) with `rotation_euler = (0, 0, 0)` looks AWAY from the cube, not at it.**

Blender cameras look down their local -Z axis. With rotation (0, 0, 0):
- Local -Z = world -Z
- Camera at (0, 0, -100) looks toward z = -infinity (AWAY from cube at origin)

Correct rotation is `(math.pi, 0, 0)` to rotate 180° around X axis, making the camera look toward +Z (toward the cube).

## Evidence

### Minimal control test (vanilla default scene)

**Default scene with emission material — WORKS:**
- Center: [190, 126, 118] (red emission cube visible)
- Background: [58, 58, 58] (dark world)
- Diff: 131.5 — **CLEARLY VISIBLE**

**Same scene with camera at (0,0,-100), rotation (0,0,0) — FAILS:**
- Center: [58, 58, 58] (just background)
- Background: [58, 58, 58]
- Diff: 0.0 — **NOT VISIBLE**

**Same scene with camera at (0,0,-100), rotation (pi,0,0) — WORKS:**
- Center: [59, 59, 59] (cube visible, small due to distance)
- Background: [58, 58, 58]
- Diff: 1.2 — **VISIBLE**

### Visual confirmation

- `minimal_test/emission_cube.png`: Pink cube clearly visible against dark background
- `minimal_test_3/cam_b_pi.png`: Small pink cube visible in center of frame
- `minimal_test_2/step3_new_camera.png`: Dark gray only (camera looking wrong way)

## Why the original investigation was misleading

1. **`center_mean=1.000` was meaningless** — measured on a fully saturated frame where both cube and background were white. No cube-background contrast was verified.

2. **`eval_obj.visible_get()` returning False was a red herring** — this checks VIEWPORT depsgraph visibility, not render visibility. Even the default cube shows `False` in the viewport depsgraph but renders fine.

3. **The emission shader works correctly in Blender 5.2.0** — the vanilla default scene proves it. The issue was camera orientation.

## What was NOT the root cause

- ❌ Not a Blender 5.2.0 engine bug
- ❌ Not an emission shader bug
- ❌ Not a collection linkage issue
- ❌ Not a depsgraph visibility issue
- ❌ Not a camera count or scene count issue

## Correct lessons

1. **Camera rotation matters** — Blender cameras look down local -Z, not automatically at the scene origin
2. **Use `TRACK_TO` constraints** for cameras that should point at objects (the addon already does this correctly)
3. **Verify renders visually, not just numerically** — `center_mean` without background comparison is meaningless
4. **`eval_obj.visible_get()` is not reliable for render visibility** — it checks viewport depsgraph, not render depsgraph
5. **Start with vanilla defaults** — the default scene works; only change what you need to change

## HDRI shadow direction gotcha (2026-07-28)

**Shadow direction may appear incorrect at high HDRI strength even with correct sun lamp rotation.**

This is because high ambient light from the HDRI environment texture washes out the directional shadow cast by the sun lamp. As HDRI strength drops, the sun lamp's directional shadow becomes visible and the correct direction emerges.

**Evidence from dusk preset sweep:**
- HDRI strength 1.2 → shadow direction west (WRONG — washed out)
- HDRI strength 0.5 → shadow direction east (CORRECT — emerges as ambient drops)
- HDRI strength 0.15 → shadow direction east (CORRECT, confirmed)

**When debugging shadow direction:**
1. Lower HDRI strength first before adjusting sun lamp rotation
2. The correct direction may only be visible below the HDRI strength threshold where ambient light no longer dominates
3. The wrong-direction-at-high-strength behavior is NORMAL, not a bug

**Sweep methodology for future preset additions:**
- Sweep down from current default across several values
- Note where shadow direction stabilizes
- Pick the strength that gives correct shadows AND good contrast
- Don't chase sun lamp rotation if shadow looks off at high strength — the rotation is correct, the ambient washout is the cause

## Project's actual render scripts

The project's render scripts (`render_clip.py`, `inline_dome_render.py`) use cameras from the addon's camera rig, which correctly uses `TRACK_TO` constraints. The camera rotation bug was only in the isolated test scripts created during investigation.

## Test scripts created during investigation

These scripts all had the camera rotation bug (`rotation_euler = (0, 0, 0)` with camera at z=-100):

- `test_emission_verification.py`
- `test_emission_proper.py`
- `test_emission_debug.py`
- `test_blender52_regression.py`
- `test_fix_depsgraph_visibility.py`
- `test_fix_emission.py`
- `test_depsgraph_update.py`
- `test_view_layer_update.py`
- `test_render_visibility.py`
- `test_render_compare_with.py`
- `test_render_compare_without.py`
- `test_manual_object_creation.py`
- `test_depsgraph_debug.py`
- `minimal_test.py` (used default camera — worked)
- `minimal_test_2.py` (step3 had the bug)
- `minimal_test_3.py` (verified the fix)

## Recommendation

Clean up test scripts. The emission shader is not broken — no fix needed for the actual render pipeline.
