# M1 Optics Findings: Coverage vs. Resolution at True Scale

**Date:** 2026-07-23
**Status:** Resolved — negative result for per-frame detection; temporal detection approved as next path

---

## The question

Can a camera rig simultaneously achieve two constraints on a hostile drone swarm observed from external ground posts?

1. **Multi-view coverage:** Every drone in the swarm must be visible to at least 2 cameras (required for triangulation — single-view depth is unreliable).
2. **Detectable pixel size:** Each drone must occupy enough pixels in the image for a detector to find it.

Both constraints must be satisfied at **true scale** — drones rendered at their actual 0.5m physical size, with no display inflation. The swarm volume is 5 km × 5 km × 1 km, matching the operational scenario inherited from Chen et al. (IEEE TVT 2022), whose QualNet simulations use 3 km × 3 km and 6 km × 6 km maps.

This matters because the broader project estimates inter-drone distances via multi-camera triangulation, then thresholds those distances to infer a communication graph. If the cameras can't detect the drones, the pipeline breaks at its first stage.

## Method

Two analyses were cross-referenced:

**Analytical optics sweep** (`optics_sweep.py`): Computes angular resolution (µrad/px), apparent pixel size on a 0.5m target, horizontal FOV, and a simple 1D-linear-tiling estimate of cameras needed for full swarm coverage. Covers four sensor classes (full-frame, APS-C, 1-inch, 1/2.3-inch), seven focal lengths (24mm–1200mm), eight standoffs (500m–10km), and three detector-class thresholds: bounding-box (≥8 px, YOLO-scale heuristic), centroid with known size (≥3 px), and sub-pixel/temporal (≥1 px).

**2D dome coverage simulation** (`coverage_sim.py`): Places cameras in a dome arrangement around the swarm volume (randomized elevation 20–50°, evenly-spaced azimuth with jitter), projects each drone into each camera's FOV, and counts how many cameras see each drone. Finds the minimum camera count for every drone to have ≥2 views. This is the realistic check — the 1D tiling from the optics sweep underestimates camera counts by 2–10×.

The cross-reference (`cross_reference.py`) checks every config against *both* constraints simultaneously, using the dome simulation's camera counts (not the 1D tiling) and the optics sweep's pixel sizes.

## Headline conclusion

**No configuration satisfies both constraints simultaneously.** For a 5 km × 5 km × 1 km swarm of 0.5 m drones, there is no combination of sensor, focal length, standoff distance, and camera count that achieves ≥2-camera dome coverage *and* produces a detectable true-scale pixel size.

## Supporting numbers

### What passes coverage (dome simulation, within practical camera budgets)

Only wide-angle lenses (24mm, 50mm full-frame) achieve ≥2-view coverage at all:

| Lens | Standoff | Cameras needed | True-scale px on 0.5m drone | Detector class |
|---|---|---|---|---|
| 24mm FF (1920px) | 2 km | 12 | 0.58 px | Sub-pixel |
| 24mm FF (6000px) | 2 km | 9 | 1.00 px | Sub-pixel (floor) |
| 24mm FF (8192px) | 2 km | 9 | 1.37 px | Sub-pixel |
| 24mm FF (1920px) | 5 km | 3 | 0.23 px | Sub-pixel |
| 50mm FF (1920px) | 7.5 km | 4 | 0.18 px | Sub-pixel |
| 50mm FF (8192px) | 5 km | 12 | 1.14 px | Sub-pixel (floor) |

Every lens ≥100mm fails coverage at every standoff — the dome simulation shows 0 views for all drones even at 30 cameras.

### Why the two constraints can't coexist

- **8 px at 24mm requires ~80 m standoff.** At 80 m, the ±2.5 km swarm extent subtends ~180° — no practical number of cameras can surround and cover it.
- **8 px at 5 km standoff requires a 12.5 m target.** That's a different aircraft class (Group 3–4 military UAS), not a quadcopter.
- **Close standoffs (<500 m)** give better pixel sizes but the swarm's angular extent makes coverage geometrically impossible with ≤30 cameras.
- **Long lenses (≥100 mm)** give good pixel sizes but their narrow FOV can never tile the 5 km volume — even 30 cameras leave most drones unseen.

The two constraints pull in opposite directions. Coverage demands wide FOV; resolution demands narrow FOV or close range. The geometry of a 5 km swarm makes this irreconcilable at the 0.5 m target scale.

## Scope decisions

The team reviewed five possible changes to resolve the impasse and made the following decisions:

- **Target class stays at 0.5 m.** The threat model is multirotors somewhat larger than commercial light-show drones (Intel Shooting Star ≈ 38 cm). Proposing 5–12.5 m targets to close the pixel gap is a different aircraft class, not a bigger quadcopter. Rejected.
- **Scenario scale stays at 5 km × 5 km.** This inherits from Chen et al.'s operational scenario (their QualNet maps are 3 km × 3 km and 6 km × 6 km). Shrinking to 1 km × 1 km would be a departure from the paper, not an inheritance. Rejected.
- **Temporal/motion detection is approved for investigation.** If motion across frames makes sub-pixel drones detectable when per-frame object detection cannot, the approach survives at true scale with the real threat model and scenario. If it doesn't, that is a significant negative result about camera-based swarm mapping at these scales.

## Stated limitations

These are rules of thumb, not measured constraints:

- The 8 px / 3 px / 1 px detector thresholds are order-of-magnitude heuristics from common detector practices. Actual detection performance depends on background clutter, target contrast, detector architecture, and training data — none of which are modeled here.
- The dome arrangement is one plausible camera placement among many. A different geometry (e.g., line, arc, elevated platform) might shift the coverage boundary. The dome was chosen as a reasonable default for surrounding a volume.
- **Temporal integration is explicitly not modeled in M1.** The 1 px "subpixel/temporal" threshold is a placeholder noting that motion across frames *could* extend detectability, but no temporal analysis was performed. That is the subject of the current investigation.
- Coverage simulation uses a single random seed (42) for drone positions and camera placement jitter. Results are representative but not exhaustive over all possible swarm configurations.
