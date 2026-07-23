# M1 Optics Findings: Coverage vs. Resolution at True Scale

**Date:** 2026-07-23
**Status:** M1 closed. Per-frame detection infeasible; temporal detection (frame differencing) viable under realistic deployment conditions, contingent on flux≥8.

---

## The question

Can a camera rig simultaneously achieve two constraints on a hostile drone swarm observed from external ground posts?

1. **Multi-view coverage:** Every drone must be visible to ≥2 cameras (required for triangulation).
2. **Detectable pixel size:** Each drone must occupy enough pixels for a detector to find it.

Both at **true scale** — 0.5m drones, no display inflation. Swarm: 5 km × 5 km × 1 km (inherited from Chen et al., IEEE TVT 2022).

## The reasoning chain (with corrections)

### Step 1: Per-frame detection is impossible at true scale

The actual apparent size of a 0.5m drone at 24mm/2km is **0.32px** — not the 0.58px originally reported. The 0.58px figure used a stale `FOCAL_PX = 1400` from `scene_config.py`; the real 24mm full-frame optics give 0.32px. (The stale value converts to 26.25mm in the Blender addon, affecting M3/M4 render calibration but not the M1 analytical sweep, which uses 24mm directly. No prior conclusion changes.)

The M1 cross-reference (`cross_reference.py`) checks every config against both constraints using the dome coverage simulation (realistic 2D placement, not the 1D tiling that underestimates camera counts by 2–10×):

- **≥8px (bbox detector):** zero configs pass at any tier, any standoff, any sensor.
- **≥3px (centroid detector):** zero configs pass.
- **≥1px (sub-pixel):** three edge-case configs barely pass, all requiring 6000+ pixel sensors.

Coverage demands wide FOV (24mm). Resolution demands narrow FOV or close range. The geometry of a 5km swarm makes this irreconcilable at 0.5m target scale.

### Step 2: Temporal detection appeared background-dependent

A synthetic 90-condition sweep (`temporal_detection.py`) tested frame differencing, background subtraction, and temporal accumulation. The initial result: frame differencing worked well against sky (flux≥5, SNR≥5) but required 4× more flux on terrain (flux≥20). The synthetic model assumed sky σ≈2, terrain σ≈12 — a 6× ratio.

### Step 3: The sky/terrain split was a measurement artifact

**The sky background was a flat constant color** (0.5, 0.6, 0.9) — no gradient, no texture, no atmospheric model. The σ=0.000 was not a physical sky; it was a uniform blue rectangle. The terrain had a real noise texture (σ=5.07). The 6× ratio compared a flat constant to textured terrain — not a real sky to terrain.

Re-rendered with Blender's physical sky model (multiple-scattering): sky σ=18.24 (horizon-to-zenith gradient). The frame-differencing noise ratio between physical sky and terrain: **1.1×** — essentially identical.

**Why static texture cancels:** Frame differencing subtracts consecutive frames. Static features (terrain texture, sky gradient) appear in both frames and cancel. Only dynamic components remain: sensor noise, the moving drone, and scene perturbations. The detection threshold is set by perturbation noise, not static background complexity.

### Step 4: Texture cancellation holds under realistic perturbation

Perturbation test (`perturbation_test.py`) injected camera jitter, cloud shadows, atmospheric shimmer, and combined effects into real Cycles renders. Results — minimum flux for SNR≥3:

| Perturbation | Sky | Terrain |
|---|---|---|
| Baseline (no perturbation) | flux≥2 | flux≥5 |
| Camera jitter ≤1.5px | flux≥2 | flux≥5 |
| Cloud shadows | flux≥2 | flux≥5 |
| Atmospheric shimmer ≤2.0 | flux≥2 | flux≥5 |
| Combined mild (j=0.3px, s=1.0) | flux≥5 | flux≥5 |
| Combined moderate (j=1.0px, s=5.0) | flux≥8 | flux≥15 |

Terrain does NOT degrade faster than sky under perturbation. Jitter shifts the entire frame uniformly, so texture features shift by the same amount in both frames and still cancel.

### Step 5: The flux≈16 assumption

A real 0.5m drone at 2km standoff is estimated to produce flux≈16 — above threshold in all realistic perturbation scenarios. **This is the load-bearing assumption.** It depends on:

| Factor | Assumed value | Sensitivity |
|---|---|---|
| Drone reflectance (albedo) | 10% (dark drone) | Direct proportionality — halving albedo halves flux |
| Background radiance | Sky ~10,000 cd/m² | Depends on sun angle, atmosphere |
| Atmospheric transmission (2km path) | ~0.85 | Reduces flux by ~15% |
| Sensor quantum efficiency | ~50% | Direct proportionality |
| Integration time / gain | Normalized to flux=16 | Camera-specific |

**Sensitivity analysis — what happens at lower flux:**

At **flux=8** (half assumed — e.g., brighter drone or lower contrast):
- Individual perturbations (jitter, shadows, mild shimmer): **all pass** on both backgrounds
- Combined moderate (j=1.0px, s=5.0): **fails** on terrain (SNR=3.0, borderline)
- Combined moderate (j=0.5px, s=2.0): **fails** on both (SNR=2.6–3.7)
- Severe shimmer (≥5.0): **fails**

At **flux=4** (quarter assumed — low-contrast drone):
- Individual perturbations: **marginal** (SNR 3–6 on sky, 2–4 on terrain)
- Combined moderate: **fails** on most conditions
- Shimmer ≥2.0: **fails**

**The conclusion flips from "works under realistic perturbation" to "marginal" at flux≈8–10.** This represents a 2× uncertainty in the reflectance/atmosphere model — a reasonable margin for an order-of-magnitude estimate, but not a precision measurement.

### Step 6: Viewing geometry is moot

Ground camera arrangements achieve ~47% sky fraction (mixed backgrounds). No arrangement achieves both full ≥2-view coverage AND sky-dominated backgrounds. This doesn't matter — frame differencing eliminates static texture regardless of background type.

## Supporting numbers

### Coverage (dome simulation, practical camera budgets)

| Lens | Standoff | Cameras | True-scale px |
|---|---|---|---|
| 24mm FF (1920px) | 2 km | 12 | 0.32 px |
| 24mm FF (6000px) | 2 km | 9 | 1.00 px |
| 24mm FF (8192px) | 2 km | 9 | 1.37 px |
| 50mm FF (8192px) | 5 km | 12 | 1.14 px |

### Focal length audit

`FOCAL_PX = 1400` → `1400 × 36 / 1920 = 26.25mm` in Blender addon. M1 sweep uses 24mm (correct). M3/M4 renders used 26.25mm (internally consistent). The ~9% difference (0.29px vs 0.32px) is within the sub-pixel regime. No conclusion changes.

## Scope decisions

- **Target class stays at 0.5m.** Multirotors, not military UAS. Rejected 5–12.5m targets.
- **Scenario scale stays at 5km × 5km.** Inherited from Chen et al.
- **Temporal detection is the viable path.** Frame differencing at true scale, contingent on flux≥8.

## What this means

Per-frame object detection (YOLO, etc.) cannot work at this scene scale with these optics. Temporal detection via frame differencing closes the gap — but it is not a drop-in replacement for per-frame detection. It requires:

1. A moving drone (frame differencing needs inter-frame displacement)
2. Sufficient contrast (flux≥8, meaning the drone must be distinguishable from background at the sensor level)
3. A camera platform stable enough that jitter stays below ~1.5px (achievable for ground masts, challenging for airborne observers)

The approach is viable but contingent. The flux≈16 estimate needs empirical validation against real sensor data before becoming a design commitment.

## Stated limitations

- Detector thresholds (8px/3px/1px) are heuristics, not measured constraints.
- Dome arrangement is one camera placement among many.
- Coverage simulation uses a single random seed (42).
- Perturbation test uses synthetic jitter/shimmer models, not measured values from real platforms.
- Flux≈16 is derived from assumed reflectance, atmospheric transmission, and sensor characteristics — not measured. The conclusion holds at flux≥8 (2× margin below assumed), but degrades below that.
- Temporal detection requires moving drones — hovering drones need background subtraction instead.
