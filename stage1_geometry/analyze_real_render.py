"""
Analyze the real Cycles renders: measure background characteristics,
add realistic sensor noise, and recompute detection thresholds.

The Cycles renders are noise-free (deterministic, fixed seed). Real cameras
add sensor noise. This script:
1. Loads the rendered frames
2. Measures spatial variation (the "clutter" a drone must overcome)
3. Adds realistic sensor noise (photon shot + read noise)
4. Computes frame-differencing statistics with and without a synthetic drone
5. Reports actual detection thresholds per background type
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
logs_dir = project_root / "logs" / "temporal_detection"

# ---------------------------------------------------------------------------
# Load rendered frames
# ---------------------------------------------------------------------------

def load_image(path):
    """Load PNG as grayscale float64 (0-255)."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)


# ---------------------------------------------------------------------------
# Add realistic sensor noise
# ---------------------------------------------------------------------------

def add_sensor_noise(frame, iso=100, read_noise_electrons=2.0):
    """Add realistic camera sensor noise.

    Models photon shot noise (Poisson, signal-dependent) + read noise (Gaussian).
    frame: grayscale 0-255, treated as photon counts scaled to 8-bit.

    iso: higher ISO = more gain = more visible noise.
    read_noise_electrons: read noise at base ISO.
    """
    # Convert 0-255 to approximate photon counts
    # At ISO 100, assume ~100 photons per digital count
    gain = iso / 100.0
    photons = frame * gain * 100.0  # approximate photon counts

    # Photon shot noise (Poisson → Gaussian approximation for large counts)
    shot_noise = np.random.poisson(np.maximum(photons, 1).astype(np.int32)).astype(np.float64)
    # Read noise (Gaussian)
    read_noise = np.random.normal(0, read_noise_electrons * gain, frame.shape)

    # Combine and convert back to 0-255
    noisy = (shot_noise + read_noise) / (gain * 100.0)
    return np.clip(noisy, 0, 255)


# ---------------------------------------------------------------------------
# Insert synthetic drone into frame
# ---------------------------------------------------------------------------

def insert_drone(frame, center_y, center_x, flux, sigma_px=0.7):
    """Insert a Gaussian blob (drone) into the frame."""
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    blob = flux * np.exp(-r2 / (2 * sigma_px ** 2))
    return frame + blob


# ---------------------------------------------------------------------------
# Detection analysis
# ---------------------------------------------------------------------------

def measure_snr_at(detection_img, target_y, target_x, bg_radius=200, target_radius=3):
    """Measure SNR at a specific location."""
    h, w = detection_img.shape

    # Signal: max in small region around target
    y1 = max(0, target_y - target_radius)
    y2 = min(h, target_y + target_radius + 1)
    x1 = max(0, target_x - target_radius)
    x2 = min(w, target_x + target_radius + 1)
    signal = detection_img[y1:y2, x1:x2].max()

    # Noise: std in distant region
    if target_x < w // 2:
        nx1 = min(w - bg_radius, target_x + 300)
    else:
        nx1 = max(0, target_x - 300 - bg_radius)
    nx2 = nx1 + bg_radius
    ny1 = max(0, target_y - 50)
    ny2 = min(h, target_y + 50)
    noise_region = detection_img[ny1:ny2, nx1:nx2]
    noise_std = noise_region.std()
    noise_mean = noise_region.mean()

    snr = (signal - noise_mean) / max(noise_std, 0.1)
    return snr, signal, noise_mean, noise_std


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("REAL RENDER ANALYSIS: detection thresholds with sensor noise")
    print("=" * 70)

    # Load renders
    sky_path = logs_dir / "real_render_sky.png"
    terrain_path = logs_dir / "real_render_terrain.png"

    if not sky_path.exists() or not terrain_path.exists():
        print("ERROR: Render files not found. Run validate_real_render.py first.")
        sys.exit(1)

    sky = load_image(sky_path)
    terrain = load_image(terrain_path)

    print(f"\nLoaded sky:     {sky.shape}, range=[{sky.min():.1f}, {sky.max():.1f}], "
          f"mean={sky.mean():.1f}, σ={sky.std():.3f}")
    print(f"Loaded terrain: {terrain.shape}, range=[{terrain.min():.1f}, {terrain.max():.1f}], "
          f"mean={terrain.mean():.1f}, σ={terrain.std():.2f}")

    # Spatial variation analysis
    print(f"\n--- Spatial variation (THE key metric) ---")
    print(f"  Sky:     σ_spatial = {sky.std():.3f} (essentially zero — uniform)")
    print(f"  Terrain: σ_spatial = {terrain.std():.2f} (texture variation)")

    # Spatial gradient (how different adjacent pixels are)
    sky_grad_h = np.abs(np.diff(sky, axis=1)).mean()
    sky_grad_v = np.abs(np.diff(sky, axis=0)).mean()
    terrain_grad_h = np.abs(np.diff(terrain, axis=1)).mean()
    terrain_grad_v = np.abs(np.diff(terrain, axis=0)).mean()
    print(f"\n  Adjacent pixel gradient (mean |Δ|):")
    print(f"  Sky:     h={sky_grad_h:.3f}, v={sky_grad_v:.3f}")
    print(f"  Terrain: h={terrain_grad_h:.2f}, v={terrain_grad_v:.2f}")

    # Frame differencing on re-noised renders
    print(f"\n--- Frame differencing with sensor noise ---")
    for iso in [100, 400, 1600]:
        n_trials = 5
        sky_diffs = []
        terrain_diffs = []
        for _ in range(n_trials):
            sky_n1 = add_sensor_noise(sky, iso=iso)
            sky_n2 = add_sensor_noise(sky, iso=iso)
            terrain_n1 = add_sensor_noise(terrain, iso=iso)
            terrain_n2 = add_sensor_noise(terrain, iso=iso)

            sky_d = np.abs(sky_n1 - sky_n2)
            terrain_d = np.abs(terrain_n1 - terrain_n2)
            sky_diffs.append(sky_d.std())
            terrain_diffs.append(terrain_d.std())

        sky_diff_std = np.mean(sky_diffs)
        terrain_diff_std = np.mean(terrain_diffs)
        print(f"  ISO {iso:>4d}: sky diff σ={sky_diff_std:.2f}, "
              f"terrain diff σ={terrain_diff_std:.2f}, "
              f"ratio={terrain_diff_std/sky_diff_std:.1f}×")

    # Detection threshold: how much flux is needed for SNR≥3 on each background?
    print(f"\n--- Detection threshold (SNR≥3 and SNR≥5) ---")
    print(f"  Drone: 0.32px apparent, PSF σ=0.7px")
    print(f"  Method: frame differencing (ISO 400)")
    print()

    target_y, target_x = 540, 960  # center of frame
    fluxes = [2, 5, 10, 15, 20, 30, 50, 80]

    for bg_name, bg in [("sky", sky), ("terrain", terrain)]:
        print(f"  Background: {bg_name} (σ_spatial={bg.std():.2f})")
        print(f"  {'Flux':>6s} {'Single SNR':>12s} {'Diff SNR':>10s} {'Detectable?':>12s}")

        for flux in fluxes:
            # Add drone to two frames
            drone_frame1 = insert_drone(bg, target_y, target_x, flux)
            drone_frame2 = insert_drone(bg, target_y, target_x + 0.5, flux)  # moved 0.5px

            # Add sensor noise
            noisy1 = add_sensor_noise(drone_frame1, iso=400)
            noisy2 = add_sensor_noise(drone_frame2, iso=400)

            # Single-frame SNR
            single_snr, _, _, _ = measure_snr_at(noisy1, target_y, target_x)

            # Frame difference
            diff = np.abs(noisy1.astype(float) - noisy2.astype(float))
            diff_snr, _, _, _ = measure_snr_at(diff, target_y, target_x + 1)

            detectable = "YES" if diff_snr >= 3 else ("marginal" if diff_snr >= 2 else "no")
            print(f"  {flux:>5.0f}  {single_snr:>11.2f}  {diff_snr:>9.2f}  {detectable:>12s}")
        print()

    # Key finding: spatial variation as noise floor
    print("=" * 70)
    print("KEY FINDING: what limits detection on each background")
    print("=" * 70)
    print(f"""
  Sky background:
    Spatial variation: σ ≈ {sky.std():.3f} (essentially zero)
    Detection limited by: SENSOR NOISE ONLY
    At ISO 400: diff noise σ ≈ {np.mean([add_sensor_noise(sky, 400).__sub__(add_sensor_noise(sky, 400)).std() for _ in range(3)]):.2f}
    → Frame differencing works if drone flux > ~3× sensor noise

  Terrain background:
    Spatial variation: σ ≈ {terrain.std():.2f}
    Detection limited by: SPATIAL CLUTTER (texture variation)
    Even with perfect sensor: drone must exceed terrain texture variation
    → Frame differencing requires flux > ~3× terrain σ = {3 * terrain.std():.1f}

  Ratio: terrain requires {terrain.std() / max(sky.std(), 0.01):.0f}× more flux than sky
  (vs synthetic model's assumed 6×)
""")

    # Save results
    results = {
        "sky": {
            "mean": float(sky.mean()),
            "std_spatial": float(sky.std()),
            "grad_h": float(sky_grad_h),
            "grad_v": float(sky_grad_v),
        },
        "terrain": {
            "mean": float(terrain.mean()),
            "std_spatial": float(terrain.std()),
            "grad_h": float(terrain_grad_h),
            "grad_v": float(terrain_grad_v),
        },
        "detection_thresholds": {},
    }

    for bg_name, bg in [("sky", sky), ("terrain", terrain)]:
        thresholds = {}
        for flux in fluxes:
            drone_frame = insert_drone(bg, target_y, target_x, flux)
            drone_frame2 = insert_drone(bg, target_y, target_x + 0.5, flux)
            noisy1 = add_sensor_noise(drone_frame, iso=400)
            noisy2 = add_sensor_noise(drone_frame2, iso=400)
            single_snr, _, _, _ = measure_snr_at(noisy1, target_y, target_x)
            diff = np.abs(noisy1.astype(float) - noisy2.astype(float))
            diff_snr, _, _, _ = measure_snr_at(diff, target_y, target_x + 1)
            thresholds[flux] = {"single_snr": float(single_snr), "diff_snr": float(diff_snr)}
        results["detection_thresholds"][bg_name] = thresholds

    out_path = logs_dir / "real_render_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
