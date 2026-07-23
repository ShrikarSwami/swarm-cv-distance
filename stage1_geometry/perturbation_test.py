"""
Perturbation test: frame differencing under realistic deployment conditions.

Tests at what perturbation magnitude the flux≥5 detection threshold
breaks down, on both sky and terrain backgrounds.

Perturbations modeled:
1. Camera jitter (sub-pixel to 2px) — platform vibration, wind on mast
2. Moving cloud shadows — brightness modulation across frame
3. Atmospheric shimmer — high-frequency spatial noise varying frame-to-frame
4. Combined perturbations

All perturbations are applied to real Cycles renders (not synthetic backgrounds).
The drone is inserted as a known-position Gaussian blob so detection SNR
can be measured precisely.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
logs_dir = project_root / "logs" / "temporal_detection"


def load_image(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def insert_drone(frame, cy, cx, flux, sigma=0.7):
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]
    blob = flux * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return frame + blob


def add_sensor_noise(frame, iso=400):
    gain = iso / 100.0
    photons = frame * gain * 100.0
    shot = np.random.poisson(np.maximum(photons, 1).astype(np.int32)).astype(np.float64)
    read = np.random.normal(0, 2.0 * gain, frame.shape)
    return np.clip((shot + read) / (gain * 100.0), 0, 255)


def apply_jitter(frame, dx, dy):
    """Shift frame by sub-pixel amount (bilinear interpolation)."""
    from scipy.ndimage import shift
    return shift(frame, [dy, dx], order=1, mode='reflect')


def apply_cloud_shadow(frame, speed_px, frame_idx):
    """Moving brightness modulation — cloud shadow passing across."""
    h, w = frame.shape
    # Slow-moving dark band across the frame
    center = w * 0.3 + speed_px * frame_idx
    width = w * 0.4
    shadow = np.ones((h, w), dtype=np.float64)
    # Gaussian shadow profile
    xx = np.arange(w)[np.newaxis, :]
    shadow_profile = 1.0 - 0.15 * np.exp(-((xx - center)**2) / (2 * (width/3)**2))
    shadow *= shadow_profile
    return frame * shadow


def apply_atmospheric_shimmer(frame, strength, seed):
    """High-frequency spatial noise that changes frame-to-frame (heat haze)."""
    rng = np.random.default_rng(seed)
    h, w = frame.shape
    # Generate correlated noise at multiple scales
    shimmer = np.zeros_like(frame)
    for scale in [5, 15, 40]:
        sh, sw = max(2, h // scale), max(2, w // scale)
        noise = rng.uniform(-1, 1, (sh, sw))
        upsampled = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)
        shimmer += upsampled[:h, :w] / 3.0
    return frame + strength * shimmer


def measure_snr(detection_img, ty, tx, bg_radius=200, target_radius=3):
    h, w = detection_img.shape
    y1, y2 = max(0, ty - target_radius), min(h, ty + target_radius + 1)
    x1, x2 = max(0, tx - target_radius), min(w, tx + target_radius + 1)
    signal = detection_img[y1:y2, x1:x2].max()

    if tx < w // 2:
        nx1 = min(w - bg_radius, tx + 300)
    else:
        nx1 = max(0, tx - 300 - bg_radius)
    nx2 = nx1 + bg_radius
    ny1, ny2 = max(0, ty - 50), min(h, ty + 50)
    noise_region = detection_img[ny1:ny2, nx1:nx2]
    noise_std = max(noise_region.std(), 0.1)
    noise_mean = noise_region.mean()
    return (signal - noise_mean) / noise_std


def run_perturbation_sweep(bg_name, bg_frame, fluxes, n_trials=3):
    """Sweep perturbation magnitudes and measure detection SNR."""
    h, w = bg_frame.shape
    ty, tx = h // 2, w // 2  # drone at center

    results = []

    # --- Baseline: no perturbation ---
    print(f"\n  --- {bg_name}: Baseline (no perturbation) ---")
    for flux in fluxes:
        snrs = []
        for _ in range(n_trials):
            f1 = add_sensor_noise(insert_drone(bg_frame, ty, tx, flux))
            f2 = add_sensor_noise(insert_drone(bg_frame, ty, tx + 0.5, flux))
            diff = np.abs(f1.astype(float) - f2.astype(float))
            snrs.append(measure_snr(diff, ty, tx + 1))
        mean_snr = np.mean(snrs)
        detectable = "YES" if mean_snr >= 3 else ("marginal" if mean_snr >= 2 else "no")
        print(f"    flux={flux:>3.0f}: SNR={mean_snr:>6.2f} [{detectable}]")
        results.append({"perturbation": "none", "magnitude": 0, "flux": flux,
                        "mean_snr": float(mean_snr), "detectable": mean_snr >= 3})

    # --- Camera jitter ---
    print(f"\n  --- {bg_name}: Camera jitter ---")
    for jitter_px in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
        snrs_per_flux = {}
        for flux in fluxes:
            snrs = []
            for trial in range(n_trials):
                rng = np.random.default_rng(trial * 1000 + int(jitter_px * 100))
                dx1 = rng.uniform(-jitter_px, jitter_px)
                dy1 = rng.uniform(-jitter_px, jitter_px)
                dx2 = rng.uniform(-jitter_px, jitter_px)
                dy2 = rng.uniform(-jitter_px, jitter_px)

                drone_f1 = insert_drone(bg_frame, ty, tx, flux)
                drone_f2 = insert_drone(bg_frame, ty, tx + 0.5, flux)
                jittered_f1 = add_sensor_noise(apply_jitter(drone_f1, dx1, dy1))
                jittered_f2 = add_sensor_noise(apply_jitter(drone_f2, dx2, dy2))
                diff = np.abs(jittered_f1.astype(float) - jittered_f2.astype(float))
                snrs.append(measure_snr(diff, ty, tx + 1))
            mean_snr = np.mean(snrs)
            snrs_per_flux[flux] = mean_snr

        # Find threshold flux
        threshold_flux = None
        for flux in sorted(fluxes):
            if snrs_per_flux[flux] >= 3:
                threshold_flux = flux
                break

        status = f"threshold={threshold_flux:.0f}" if threshold_flux else "FAIL"
        snr_str = " ".join(f"f{f}:{snrs_per_flux[f]:.1f}" for f in fluxes)
        print(f"    jitter={jitter_px:.1f}px: {snr_str}  [{status}]")
        for flux in fluxes:
            results.append({"perturbation": "jitter", "magnitude": jitter_px,
                            "flux": flux, "mean_snr": float(snrs_per_flux[flux]),
                            "detectable": snrs_per_flux[flux] >= 3})

    # --- Cloud shadows ---
    print(f"\n  --- {bg_name}: Moving cloud shadows ---")
    for shadow_speed in [0.5, 1.0, 2.0, 5.0]:  # pixels per frame
        snrs_per_flux = {}
        for flux in fluxes:
            snrs = []
            for trial in range(n_trials):
                drone_f1 = insert_drone(bg_frame, ty, tx, flux)
                drone_f2 = insert_drone(bg_frame, ty, tx + 0.5, flux)
                shadowed_f1 = add_sensor_noise(apply_cloud_shadow(drone_f1, shadow_speed, 0))
                shadowed_f2 = add_sensor_noise(apply_cloud_shadow(drone_f2, shadow_speed, 1))
                diff = np.abs(shadowed_f1.astype(float) - shadowed_f2.astype(float))
                snrs.append(measure_snr(diff, ty, tx + 1))
            snrs_per_flux[flux] = np.mean(snrs)

        threshold_flux = None
        for flux in sorted(fluxes):
            if snrs_per_flux[flux] >= 3:
                threshold_flux = flux
                break
        status = f"threshold={threshold_flux:.0f}" if threshold_flux else "FAIL"
        snr_str = " ".join(f"f{f}:{snrs_per_flux[f]:.1f}" for f in fluxes)
        print(f"    shadow_speed={shadow_speed:.1f}px/fr: {snr_str}  [{status}]")
        for flux in fluxes:
            results.append({"perturbation": "cloud_shadow", "magnitude": shadow_speed,
                            "flux": flux, "mean_snr": float(snrs_per_flux[flux]),
                            "detectable": snrs_per_flux[flux] >= 3})

    # --- Atmospheric shimmer ---
    print(f"\n  --- {bg_name}: Atmospheric shimmer ---")
    for shimmer_strength in [0.5, 1.0, 2.0, 5.0, 10.0]:
        snrs_per_flux = {}
        for flux in fluxes:
            snrs = []
            for trial in range(n_trials):
                drone_f1 = insert_drone(bg_frame, ty, tx, flux)
                drone_f2 = insert_drone(bg_frame, ty, tx + 0.5, flux)
                shimmered_f1 = add_sensor_noise(apply_atmospheric_shimmer(drone_f1, shimmer_strength, trial * 10))
                shimmered_f2 = add_sensor_noise(apply_atmospheric_shimmer(drone_f2, shimmer_strength, trial * 10 + 1))
                diff = np.abs(shimmered_f1.astype(float) - shimmered_f2.astype(float))
                snrs.append(measure_snr(diff, ty, tx + 1))
            snrs_per_flux[flux] = np.mean(snrs)

        threshold_flux = None
        for flux in sorted(fluxes):
            if snrs_per_flux[flux] >= 3:
                threshold_flux = flux
                break
        status = f"threshold={threshold_flux:.0f}" if threshold_flux else "FAIL"
        snr_str = " ".join(f"f{f}:{snrs_per_flux[f]:.1f}" for f in fluxes)
        print(f"    shimmer={shimmer_strength:.1f}: {snr_str}  [{status}]")
        for flux in fluxes:
            results.append({"perturbation": "shimmer", "magnitude": shimmer_strength,
                            "flux": flux, "mean_snr": float(snrs_per_flux[flux]),
                            "detectable": snrs_per_flux[flux] >= 3})

    # --- Combined: jitter + shimmer (realistic deployment) ---
    print(f"\n  --- {bg_name}: Combined (jitter + shimmer) ---")
    for jitter_px, shimmer_str in [(0.3, 1.0), (0.5, 2.0), (1.0, 2.0), (1.0, 5.0)]:
        snrs_per_flux = {}
        for flux in fluxes:
            snrs = []
            for trial in range(n_trials):
                rng = np.random.default_rng(trial * 1000)
                dx1, dy1 = rng.uniform(-jitter_px, jitter_px, 2)
                dx2, dy2 = rng.uniform(-jitter_px, jitter_px, 2)

                drone_f1 = insert_drone(bg_frame, ty, tx, flux)
                drone_f2 = insert_drone(bg_frame, ty, tx + 0.5, flux)
                perturbed_f1 = apply_jitter(drone_f1, dx1, dy1)
                perturbed_f2 = apply_jitter(drone_f2, dx2, dy2)
                perturbed_f1 = apply_atmospheric_shimmer(perturbed_f1, shimmer_str, trial * 10)
                perturbed_f2 = apply_atmospheric_shimmer(perturbed_f2, shimmer_str, trial * 10 + 1)
                noisy_f1 = add_sensor_noise(perturbed_f1)
                noisy_f2 = add_sensor_noise(perturbed_f2)
                diff = np.abs(noisy_f1.astype(float) - noisy_f2.astype(float))
                snrs.append(measure_snr(diff, ty, tx + 1))
            snrs_per_flux[flux] = np.mean(snrs)

        threshold_flux = None
        for flux in sorted(fluxes):
            if snrs_per_flux[flux] >= 3:
                threshold_flux = flux
                break
        status = f"threshold={threshold_flux:.0f}" if threshold_flux else "FAIL"
        snr_str = " ".join(f"f{f}:{snrs_per_flux[f]:.1f}" for f in fluxes)
        print(f"    j={jitter_px:.1f}px s={shimmer_str:.1f}: {snr_str}  [{status}]")
        for flux in fluxes:
            results.append({"perturbation": "combined", "magnitude": f"j{jitter_px}_s{shimmer_str}",
                            "flux": flux, "mean_snr": float(snrs_per_flux[flux]),
                            "detectable": snrs_per_flux[flux] >= 3})

    return results


def main():
    print("=" * 70)
    print("PERTURBATION TEST: frame differencing under deployment conditions")
    print("=" * 70)

    # Load real renders
    sky_path = logs_dir / "real_render_sky_physical.png"
    terrain_path = logs_dir / "real_render_terrain.png"

    if not sky_path.exists():
        print(f"ERROR: {sky_path} not found. Run rerender_sky.py first.")
        sys.exit(1)
    if not terrain_path.exists():
        print(f"ERROR: {terrain_path} not found. Run validate_real_render.py first.")
        sys.exit(1)

    sky = load_image(sky_path)
    terrain = load_image(terrain_path)

    print(f"\nSky:     σ={sky.std():.2f}, mean={sky.mean():.1f}")
    print(f"Terrain: σ={terrain.std():.2f}, mean={terrain.mean():.1f}")

    fluxes = [2, 5, 8, 10, 15, 20, 30]

    all_results = {}

    for bg_name, bg in [("sky_physical", sky), ("terrain", terrain)]:
        print(f"\n{'='*70}")
        print(f"BACKGROUND: {bg_name}")
        print(f"{'='*70}")
        results = run_perturbation_sweep(bg_name, bg, fluxes, n_trials=3)
        all_results[bg_name] = results

    # Summary: detection threshold under each perturbation
    print(f"\n{'='*70}")
    print("SUMMARY: Minimum flux for SNR≥3 under each perturbation")
    print(f"{'='*70}")
    print(f"\n{'Perturbation':<30s} {'Sky threshold':>14s} {'Terrain threshold':>18s}")
    print("-" * 65)

    for perturbation_type in ["none", "jitter", "cloud_shadow", "shimmer", "combined"]:
        for bg_name in ["sky_physical", "terrain"]:
            bg_results = all_results[bg_name]
            pert_results = [r for r in bg_results if r["perturbation"] == perturbation_type and r["detectable"]]
            if pert_results:
                min_flux = min(r["flux"] for r in pert_results)
            else:
                min_flux = "FAIL"
            if bg_name == "sky_physical":
                sky_thresh = min_flux
            else:
                terr_thresh = min_flux

        # Get magnitudes
        magnitudes = set()
        for bg_name in ["sky_physical", "terrain"]:
            for r in all_results[bg_name]:
                if r["perturbation"] == perturbation_type:
                    magnitudes.add(str(r["magnitude"]))
        mag_str = ", ".join(sorted(magnitudes)[:3])

        label = f"{perturbation_type} ({mag_str})" if perturbation_type != "none" else "none (baseline)"
        sky_str = f"{sky_thresh}" if isinstance(sky_thresh, str) else f"flux≥{sky_thresh}"
        terr_str = f"{terr_thresh}" if isinstance(terr_thresh, str) else f"flux≥{terr_thresh}"
        print(f"  {label:<28s} {sky_str:>14s} {terr_str:>18s}")

    # Save
    out_path = logs_dir / "perturbation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
