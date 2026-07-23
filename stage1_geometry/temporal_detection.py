"""
Temporal/motion detection for sub-pixel drones.

Tests whether motion across frames makes sub-pixel drones detectable when
per-frame object detection cannot. All analysis at true scale — no display
inflation.

Approach:
1. Synthetic smoke test (numpy only, no Blender): models exact optical physics
   with known target positions, backgrounds, and noise.
2. If promising, real Blender renders for validation.

The synthetic approach generates a 1920×1080 image sequence with a sub-pixel
Gaussian blob (modeling a 0.5m drone at 2km standoff through a 24mm lens)
moving at known speed across known backgrounds. Applies frame-differencing,
running-background subtraction, and temporal accumulation. Measures SNR to
determine detection boundaries.

Key parameters (from M1 optics sweep):
- 24mm FF @ 2km: 0.58px apparent size on 0.5m drone
- 1920×1080 resolution, ~1400px focal length
- Pixel angular resolution: 43.9 µrad/px

Physics of sub-pixel detection:
- A 0.58px Gaussian blob has peak amplitude ~0.33 of a full-pixel target
  (energy spreads across ~3 pixels via the PSF)
- Frame differencing: drone position shifts by v_px = speed_m / (fps * GSD)
  pixels per frame. If v_px > ~0.3, the drone moves enough between frames
  to create a detectable difference signal.
- Temporal accumulation: stacking N frames with proper alignment adds signal
  linearly while noise grows as sqrt(N), improving SNR by sqrt(N).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants derived from M1 optics sweep
# ---------------------------------------------------------------------------

IMAGE_W, IMAGE_H = 1920, 1080
SENSOR_W_MM = 36.0
FOCAL_MM = 24.0
STANDOFF_M = 2000.0
DRONE_SIZE_M = 0.5
FPS = 30.0

# Angular resolution: µrad per pixel
THETA_UM_RAD = (SENSOR_W_MM * 1e-3) / (FOCAL_MM * 1e-3 * IMAGE_W) * 1e6
# Ground sample distance at standoff (m per pixel)
GSD_M_PER_PX = STANDOFF_M * THETA_UM_RAD * 1e-6
# Apparent pixel size of drone
APPARENT_PX = DRONE_SIZE_M / GSD_M_PER_PX

print(f"Optics: θ = {THETA_UM_RAD:.2f} µrad/px, GSD = {GSD_M_PER_PX:.4f} m/px")
print(f"Apparent drone size: {APPARENT_PX:.2f} px")
print()


# ---------------------------------------------------------------------------
# Background models
# ---------------------------------------------------------------------------

def make_sky_background(shape=(IMAGE_H, IMAGE_W), seed=0):
    """Uniform clear sky — best case for detection (no clutter)."""
    rng = np.random.default_rng(seed)
    # Sky: bright, low variance (Rayleigh-scattered blue, rendered as grayscale)
    base = 180.0  # typical sky luminance (0-255 scale)
    noise = rng.normal(0, 2.0, shape)  # sensor noise only
    return np.clip(base + noise, 0, 255).astype(np.float64)


def make_terrain_background(shape=(IMAGE_H, IMAGE_W), seed=0):
    """Textured terrain — realistic clutter case."""
    rng = np.random.default_rng(seed)
    bg = np.zeros(shape, dtype=np.float64)
    for scale in [10, 30, 100, 300]:
        # +1 ensures upsampled field is >= shape after repeat
        small_h = max(2, shape[0] // scale + 1)
        small_w = max(2, shape[1] // scale + 1)
        field = rng.uniform(80, 160, (small_h, small_w))
        upsampled = np.repeat(np.repeat(field, scale, axis=0), scale, axis=1)
        bg += upsampled[:shape[0], :shape[1]] / 4.0
    bg += rng.normal(0, 3.0, shape)
    return np.clip(bg, 0, 255).astype(np.float64)


def make_mixed_background(shape=(IMAGE_H, IMAGE_W), seed=0):
    """Sky in upper half, terrain in lower half — horizon transition."""
    sky = make_sky_background(shape, seed)
    terrain = make_terrain_background(shape, seed + 1000)
    mixed = np.copy(sky)
    mixed[shape[0] // 3:, :] = terrain[shape[0] // 3:, :]
    # Smooth transition
    transition = shape[0] // 3
    ramp = np.linspace(0, 1, 60)[:, np.newaxis]
    mixed[transition - 30:transition + 30, :] = (
        sky[transition - 30:transition + 30, :] * (1 - ramp) +
        terrain[transition - 30:transition + 30, :] * ramp
    )
    return mixed


# ---------------------------------------------------------------------------
# Target model
# ---------------------------------------------------------------------------

def make_drone_blob(shape, center_y, center_x, sigma_px, peak_amplitude):
    """Gaussian blob modeling a sub-pixel drone.

    sigma_px: PSF width in pixels (typically ~0.5-1.0 px for a well-focused
    sub-pixel target — the optics spread the energy across a few pixels).
    peak_amplitude: how bright the drone is relative to background.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    blob = peak_amplitude * np.exp(-r2 / (2 * sigma_px ** 2))
    return blob


def render_frame(bg, drone_positions, sigma_px=0.7, drone_flux=8.0):
    """Render one frame: background + drone blobs.

    drone_positions: list of (y, x) pixel coordinates.
    sigma_px: PSF width.
    drone_flux: peak brightness added by drone (above background).
    """
    frame = bg.copy()
    for (dy, dx) in drone_positions:
        # Only add if at least partially in frame
        if -5 < dx < bg.shape[1] + 5 and -5 < dy < bg.shape[0] + 5:
            blob = make_drone_blob(bg.shape, dy, dx, sigma_px, drone_flux)
            frame += blob
    return np.clip(frame, 0, 255)


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def frame_difference(frames):
    """Absolute difference between consecutive frames."""
    diffs = []
    for i in range(1, len(frames)):
        d = np.abs(frames[i].astype(np.float64) - frames[i - 1].astype(np.float64))
        diffs.append(d)
    return diffs


def running_background_subtract(frames, alpha=0.05):
    """Exponential moving average background model, then subtract.

    alpha: learning rate (smaller = slower adaptation, better for fast targets).
    Returns: list of foreground frames (frame - background_model).
    """
    bg_model = frames[0].astype(np.float64).copy()
    fg_frames = []
    for f in frames:
        fg = f.astype(np.float64) - bg_model
        bg_model = (1 - alpha) * bg_model + alpha * f.astype(np.float64)
        fg_frames.append(fg)
    return fg_frames


def temporal_accumulation(frames, method="sum"):
    """Sum or average aligned frames.

    For a static camera with moving targets, simple summation works if the
    target moves slowly (< 1px/frame). For faster motion, need to shift
    frames to align the target trajectory before summing.
    """
    arr = np.stack([f.astype(np.float64) for f in frames], axis=0)
    if method == "sum":
        return arr.sum(axis=0)
    elif method == "mean":
        return arr.mean(axis=0)
    elif method == "max":
        return arr.max(axis=0)
    return arr.sum(axis=0)


def shifted_accumulation(frames, shifts_per_frame):
    """Accumulate frames after shifting each to align a moving target.

    shifts_per_frame: (dy, dx) pixel shift per frame.
    """
    result = frames[0].astype(np.float64).copy()
    for i in range(1, len(frames)):
        shift_y = int(round(i * shifts_per_frame[0]))
        shift_x = int(round(i * shifts_per_frame[1]))
        shifted = np.roll(frames[i].astype(np.float64), (-shift_y, -shift_x), axis=(0, 1))
        result += shifted
    return result


# ---------------------------------------------------------------------------
# SNR measurement
# ---------------------------------------------------------------------------

def measure_snr(detection_image, target_pos, bg_region_size=200, target_radius=3):
    """Measure SNR: signal at target location vs background noise.

    target_pos: (y, x) of expected target location in the detection image.
    Uses a region away from the target for noise estimation.
    Returns: (snr, signal_mean, noise_std, noise_mean).
    """
    y, x = int(round(target_pos[0])), int(round(target_pos[1]))
    h, w = detection_image.shape

    # Signal: max in small region around target (peak detector)
    y1 = max(0, y - target_radius)
    y2 = min(h, y + target_radius + 1)
    x1 = max(0, x - target_radius)
    x2 = min(w, x + target_radius + 1)
    signal_region = detection_image[y1:y2, x1:x2]
    signal_mean = signal_region.max()  # peak, not mean — matches matched filter

    # Noise: std in a distant region (far from target, same approximate row)
    # Pick region that's at least 200px away horizontally
    if x < w // 2:
        noise_x1 = min(w - bg_region_size, x + 300)
    else:
        noise_x1 = max(0, x - 300 - bg_region_size)
    noise_x2 = noise_x1 + bg_region_size
    noise_y1 = max(0, y - 20)
    noise_y2 = min(h, y + 20)
    noise_region = detection_image[noise_y1:noise_y2, noise_x1:noise_x2]
    noise_std = noise_region.std()
    noise_mean = noise_region.mean()

    if noise_std < 0.5:
        # Avoid division by tiny numbers; use a floor
        noise_std = 0.5
    snr = (signal_mean - noise_mean) / noise_std
    return snr, signal_mean, noise_std, noise_mean


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    """Quick 10-frame smoke test: one drone, one speed, one background."""
    print("=" * 70)
    print("SMOKE TEST: 10 frames, 1 drone, sky background")
    print("=" * 70)

    n_frames = 10
    drone_speed_px = 1.0  # pixels per frame (fast transit)
    drone_flux = 8.0  # peak brightness above background
    bg = make_sky_background(seed=42)

    # Drone starts at left edge, flies right
    start_y = IMAGE_H // 2
    start_x = 100
    drone_trajectory = [
        (start_y, start_x + i * drone_speed_px) for i in range(n_frames)
    ]

    print(f"Drone speed: {drone_speed_px} px/frame = "
          f"{drone_speed_px * GSD_M_PER_PX * FPS:.1f} m/s")
    print(f"Drone flux: {drone_flux:.1f} total above bg (bg ≈ {bg.mean():.0f})")
    print(f"Apparent size: {APPARENT_PX:.2f} px (σ_PSF ≈ 0.7px)")
    # Show actual peak amplitude after PSF spreading
    sigma_psf = 0.7
    peak_amp = drone_flux / (2 * np.pi * sigma_psf**2)
    print(f"  → PSF peak amplitude: {peak_amp:.2f} (noise σ ≈ {bg.std():.2f})")
    print(f"  → Single-frame peak SNR (theory): {peak_amp / bg.std():.2f}")
    print()

    # Render frames
    frames = [render_frame(bg, [pos], drone_flux=drone_flux)
              for pos in drone_trajectory]

    # Check: can we see the drone in a single frame?
    single_snr = measure_snr(frames[5], drone_trajectory[5])
    print(f"Single-frame SNR at target: {single_snr[0]:.2f} "
          f"(signal={single_snr[1]:.1f}, noise_μ={single_snr[3]:.1f}, "
          f"noise_σ={single_snr[2]:.2f})")

    # Frame differencing
    diffs = frame_difference(frames)
    diff_at_target = measure_snr(diffs[4], drone_trajectory[5])  # diff[4] = frame[5]-frame[4]
    print(f"Frame-diff SNR at target:   {diff_at_target[0]:.2f} "
          f"(signal={diff_at_target[1]:.1f})")

    # Temporal accumulation (all 10 frames, no shift — only works if target is slow)
    acc_sum = temporal_accumulation(frames, method="sum")
    acc_at_target = measure_snr(acc_sum, drone_trajectory[5])
    print(f"Temporal sum SNR (10fr):    {acc_at_target[0]:.2f}")

    # Shifted accumulation (align to target motion)
    shift_per_frame = (0, drone_speed_px)
    shifted_acc = shifted_accumulation(frames, shift_per_frame)
    shifted_snr = measure_snr(shifted_acc, drone_trajectory[5])
    print(f"Shifted accum SNR (10fr):   {shifted_snr[0]:.2f}")

    # Background subtraction (slow adaptation — target persists as foreground)
    fg_frames = running_background_subtract(frames, alpha=0.05)
    fg_at_target = measure_snr(fg_frames[5], drone_trajectory[5])
    print(f"Background-sub SNR (fr5):   {fg_at_target[0]:.2f}")

    # Peak detection in accumulated image
    acc_patch = shifted_acc[
        max(0, start_y - 20):min(IMAGE_H, start_y + 20),
        max(0, start_x):min(IMAGE_W, start_x + n_frames * int(drone_speed_px) + 20)
    ]
    peak_val = acc_patch.max()
    peak_loc = np.unravel_index(acc_patch.argmax(), acc_patch.shape)
    print(f"\nShifted accumulation peak: {peak_val:.1f} at relative {peak_loc}")
    print(f"  Expected signal region: ~frame 5 position, {n_frames} frames accumulated")

    # Detection threshold analysis
    print(f"\n--- Detection analysis ---")
    print(f"  Single frame: SNR={single_snr[0]:.2f} → {'DETECTABLE' if single_snr[0] > 3 else 'NOT DETECTABLE'} (threshold: 3σ)")
    print(f"  Frame diff:   SNR={diff_at_target[0]:.2f} → {'DETECTABLE' if diff_at_target[0] > 3 else 'NOT DETECTABLE'}")
    print(f"  Shifted accum: SNR={shifted_snr[0]:.2f} → {'DETECTABLE' if shifted_snr[0] > 3 else 'NOT DETECTABLE'}")
    print(f"  Bg subtract:  SNR={fg_at_target[0]:.2f} → {'DETECTABLE' if fg_at_target[0] > 3 else 'NOT DETECTABLE'}")
    print()

    return {
        "single_snr": single_snr[0],
        "diff_snr": diff_at_target[0],
        "shifted_snr": shifted_snr[0],
        "bg_sub_snr": fg_at_target[0],
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def sweep():
    """Sweep speed × background × flux to find detection boundary."""
    print("=" * 70)
    print("PARAMETER SWEEP: speed × background × flux")
    print("=" * 70)
    print()

    backgrounds = {
        "sky": make_sky_background(seed=42),
        "terrain": make_terrain_background(seed=42),
        "mixed": make_mixed_background(seed=42),
    }

    speeds_px = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]  # px/frame
    fluxes = [5.0, 10.0, 20.0, 40.0, 80.0]  # total flux added by drone
    n_frames = 20

    results = []

    for bg_name, bg in backgrounds.items():
        print(f"\n--- Background: {bg_name} (μ={bg.mean():.1f}, σ={bg.std():.2f}) ---")
        print(f"{'Speed':>8s} {'Flux':>6s} {'v_m/s':>7s} "
              f"{'Single':>7s} {'Diff':>7s} {'Shift10':>8s} {'Shift20':>8s} {'BgSub':>7s}")

        for speed in speeds_px:
            for flux in fluxes:
                # Generate frames
                start_y = IMAGE_H // 2
                start_x = IMAGE_W // 4
                trajectory = [
                    (start_y, start_x + i * speed) for i in range(n_frames)
                ]
                frames = [render_frame(bg, [pos], drone_flux=flux)
                          for pos in trajectory]

                # Single frame (mid-sequence)
                mid = n_frames // 2
                single_snr = measure_snr(frames[mid], trajectory[mid])[0]

                # Frame differencing
                diffs = frame_difference(frames)
                diff_snr = measure_snr(diffs[mid - 1], trajectory[mid])[0]

                # Shifted accumulation with 10 and 20 frames
                for n_acc in [10, 20]:
                    acc = shifted_accumulation(
                        frames[:n_acc], (0, speed))
                    acc_snr = measure_snr(acc, trajectory[n_acc // 2])[0]
                    if n_acc == 10:
                        shift10_snr = acc_snr
                    else:
                        shift20_snr = acc_snr

                # Background subtraction
                fg = running_background_subtract(frames, alpha=0.05)
                fg_snr = measure_snr(fg[mid], trajectory[mid])[0]

                v_ms = speed * GSD_M_PER_PX * FPS
                print(f"{speed:>7.1f}px {flux:>5.1f} {v_ms:>6.1f}m/s "
                      f"{single_snr:>7.2f} {diff_snr:>7.2f} "
                      f"{shift10_snr:>8.2f} {shift20_snr:>8.2f} {fg_snr:>7.2f}")

                results.append({
                    "background": bg_name,
                    "bg_mean": float(bg.mean()),
                    "bg_std": float(bg.std()),
                    "speed_px": speed,
                    "flux": flux,
                    "speed_ms": v_ms,
                    "single_snr": float(single_snr),
                    "diff_snr": float(diff_snr),
                    "shift10_snr": float(shift10_snr),
                    "shift20_snr": float(shift20_snr),
                    "bg_sub_snr": float(fg_snr),
                })

    return results


# ---------------------------------------------------------------------------
# Summary and boundary analysis
# ---------------------------------------------------------------------------

def analyze_boundary(results):
    """Find detection boundaries: what flux/speed/background combinations work."""
    print("\n" + "=" * 70)
    print("DETECTION BOUNDARY ANALYSIS")
    print("=" * 70)

    for method_name, key in [
        ("Single-frame (YOLO-scale)", "single_snr"),
        ("Frame differencing", "diff_snr"),
        ("Shifted accumulation (10fr)", "shift10_snr"),
        ("Shifted accumulation (20fr)", "shift20_snr"),
        ("Background subtraction", "bg_sub_snr"),
    ]:
        print(f"\n{method_name}:")
        for threshold_label, threshold in [("SNR≥3 (marginal)", 3.0), ("SNR≥5 (confident)", 5.0)]:
            passing = [r for r in results if r[key] >= threshold]
            if passing:
                min_flux = min(r["flux"] for r in passing)
                max_speed = max(r["speed_px"] for r in passing)
                min_speed = min(r["speed_px"] for r in passing)
                bg_types = set(r["background"] for r in passing)
                print(f"  {threshold_label}: flux≥{min_flux:.1f}, "
                      f"speed={min_speed:.1f}-{max_speed:.1f}px/frame "
                      f"({min_speed * GSD_M_PER_PX * FPS:.0f}-{max_speed * GSD_M_PER_PX * FPS:.0f} m/s), "
                      f"backgrounds: {', '.join(sorted(bg_types))}")
            else:
                print(f"  {threshold_label}: NO CONFIGURATIONS PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent.parent / "logs" / "temporal_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smoke test first
    t0 = time.time()
    smoke_result = smoke_test()
    t_smoke = time.time() - t0
    print(f"\nSmoke test completed in {t_smoke:.1f}s")

    # Full sweep
    t0 = time.time()
    results = sweep()
    t_sweep = time.time() - t0
    print(f"\nSweep completed in {t_sweep:.1f}s ({len(results)} conditions)")

    # Analyze boundaries
    analyze_boundary(results)

    # Save results
    out_path = out_dir / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "optics": {
                "focal_mm": FOCAL_MM,
                "sensor_width_mm": SENSOR_W_MM,
                "standoff_m": STANDOFF_M,
                "apparent_px": APPARENT_PX,
                "gsd_m_per_px": GSD_M_PER_PX,
                "theta_um_rad": THETA_UM_RAD,
            },
            "smoke_test": smoke_result,
            "sweep": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
