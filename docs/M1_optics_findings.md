# M1 Optics Findings: Coverage vs. Resolution at True Scale

**Date:** 2026-07-23 (updated with temporal detection and real-render validation)
**Status:** Per-frame detection infeasible; temporal detection viable at true scale

---

## The question

Can a camera rig simultaneously achieve two constraints on a hostile drone swarm observed from external ground posts?

1. **Multi-view coverage:** Every drone in the swarm must be visible to at least 2 cameras (required for triangulation — single-view depth is unreliable).
2. **Detectable pixel size:** Each drone must occupy enough pixels in the image for a detector to find it.

Both constraints must be satisfied at **true scale** — drones rendered at their actual 0.5m physical size, with no display inflation. The swarm volume is 5 km × 5 km × 1 km, matching the operational scenario inherited from Chen et al. (IEEE TVT 2022), whose QualNet simulations use 3 km × 3 km and 6 km × 6 km maps.

## Method

Two analyses were cross-referenced:

**Analytical optics sweep** (`optics_sweep.py`): Computes angular resolution, apparent pixel size on a 0.5m target, and a 1D-linear-tiling estimate of cameras needed for coverage. Covers four sensor classes, seven focal lengths (24mm–1200mm), eight standoffs (500m–10km).

**2D dome coverage simulation** (`coverage_sim.py`): Places cameras in a dome arrangement, projects each drone into each camera's FOV, and finds the minimum camera count for ≥2-view coverage. The 1D tiling underestimates camera counts by 2–10×.

The cross-reference (`cross_reference.py`) checks every config against both constraints simultaneously.

**Temporal detection sweep** (`temporal_detection.py`): Synthetic 90-condition sweep testing frame differencing, background subtraction, and temporal accumulation across speeds, fluxes, and backgrounds. Later validated against real Cycles renders.

**Real render validation** (`validate_real_render.py`, `analyze_real_render.py`): Renders true-scale 0.5m drones at 24mm/2km with actual Cycles raytracing. Measures real background noise characteristics and detection thresholds with modeled sensor noise.

## Headline conclusion

**Per-frame detection is infeasible.** No configuration achieves ≥2-camera dome coverage *and* produces a detectable true-scale pixel size for 0.5m drones across a 5km×5km×1km volume.

**Temporal detection (frame differencing) is viable.** Motion across frames makes sub-pixel drones detectable against both sky and terrain backgrounds at true scale, with the same detection threshold on both.

## Supporting numbers

### Corrected apparent pixel size

The original analysis reported 0.58px apparent size at 24mm/2km. This used a stale `FOCAL_PX = 1400` from `scene_config.py`. The correct value for a 24mm full-frame lens at 1920×1080 is **0.32px** — the drone is even more sub-pixel than originally reported.

| Lens | Standoff | Cameras needed | True-scale px on 0.5m drone |
|---|---|---|---|
| 24mm FF (1920px) | 2 km | 12 | 0.32 px |
| 24mm FF (6000px) | 2 km | 9 | 1.00 px |
| 24mm FF (8192px) | 2 km | 9 | 1.37 px |
| 50mm FF (8192px) | 5 km | 12 | 1.14 px |

### Why per-frame detection fails

- **8 px at 24mm requires ~80 m standoff.** At 80 m, the ±2.5 km swarm subtends ~180° — coverage is geometrically impossible.
- **8 px at 5 km standoff requires a 12.5 m target.** Different aircraft class, not a quadcopter.
- The two constraints pull in opposite directions: coverage demands wide FOV; resolution demands narrow FOV or close range.

### Temporal detection: the real-render-validated result

Frame differencing was tested across 90 synthetic conditions (speed × flux × background) and validated against real Cycles renders.

**Key finding from real renders:** The frame-differencing noise ratio between sky and terrain is **1.1×** (not the 6× the synthetic model assumed). Static terrain texture cancels in frame differencing — only sensor noise remains, which is identical on both backgrounds. The synthetic model's 6× ratio compared spatial variation (terrain texture) to sensor noise — the wrong comparison for temporal detection.

**Detection thresholds (validated against real renders):**

| Background | Spatial σ | Diff noise σ (ISO 400) | Min flux for SNR≥3 | Min flux for SNR≥5 |
|---|---|---|---|---|
| Sky | 0.00 | 0.57 | ~2 | ~5 |
| Terrain | 5.07 | 0.52 | ~2 | ~5 |

A real 0.5m drone at 2km standoff produces flux≈16 against sky (well above threshold). Frame differencing works on **both** backgrounds with the same threshold.

### Viewing geometry

Ground camera arrangements at all elevation angles achieve ~47% sky fraction (mixed backgrounds). No arrangement achieves both full ≥2-view coverage AND sky-dominated backgrounds. This is now moot — frame differencing eliminates static texture regardless of background type.

## Focal length audit

The stale `FOCAL_PX = 1400` value in `scene_config.py` converts to `1400 × 36 / 1920 = 26.25mm` in the Blender addon. This means M3/M4 renders used a 26.25mm lens, not 24mm.

**Impact assessment:**
- M1 optics sweep: **Unaffected** — uses 24mm directly in calculations, not `FOCAL_PX`
- M1 coverage simulation: **Unaffected** — pure Python, no Blender dependency
- M3/M4 renders: **Internally consistent** — used 26.25mm throughout, results valid for that lens
- Addon camera placement: Uses `FOCAL_PX` for dome distance calculation — slightly different standoff than 24mm would give, but rigs were validated by real renders
- M1 tables in PROGRESS.md: Report 24mm results (correct for the analytical sweep)

The 26.25mm vs 24mm difference is ~9% in focal length, producing ~0.29px instead of 0.32px apparent size — within the margin of the sub-pixel regime. No prior conclusion changes.

## Scope decisions

- **Target class stays at 0.5 m.** Multirotors somewhat larger than Intel Shooting Star (~38cm). Rejected 5–12.5m targets as a different aircraft class.
- **Scenario scale stays at 5 km × 5 km.** Inherited from Chen et al.'s operational scenario.
- **Temporal detection is the viable path.** Frame differencing closes the sub-pixel gap at true scale, validated against real renders.

## Stated limitations

- The 8 px / 3 px / 1 px detector thresholds are heuristics, not measured constraints.
- The dome arrangement is one camera placement among many.
- Coverage simulation uses a single random seed (42).
- Temporal detection assumes a moving drone — perfectly hovering drones are detectable via background subtraction but not frame differencing. Both methods require sufficient contrast (≥5 flux units, achievable for realistic drone reflectances against sky).
