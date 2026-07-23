# Sensor Noise Analysis: Does the 0.067 Signal Survive?

## The measurement

The inline Cycles render of a 0.5m emissive drone at 2km standoff produces a
signal of **0.067 render units** above the sky background (R=0.566 at brightest
pixel vs sky R=0.499). This is in Cycles' scene-referred linear HDR space.

## From first principles: render units to photons

Cycles output is proportional to received radiance. The conversion to physical
photon count depends on scene illuminance and camera exposure, but the
**signal-to-noise ratio is independent of the absolute conversion factor**.
Here's why:

- Signal in render units: `S_render = 0.067`
- Shot noise (electrons): `σ_shot = sqrt(N_photons)` where N_photons = render_value × K
- Shot noise (render units): `σ_shot_render = sqrt(render_value × K) / K = sqrt(render_value / K)`
- SNR = `S_render / σ_shot_render = 0.067 / sqrt(render_value / K)`
- Since K (the conversion factor) cancels in the ratio, **SNR depends only on
  the number of photons, not the units**

So I can compute SNR directly from photon counts without knowing the exact
render-to-photon conversion.

## Photon count model

For a 24mm f/2.8 lens, 1ms exposure, 50% quantum efficiency, imaging a sky
at luminance L (cd/m²):

```
N_photons = L × (π/4) × (D/f)² × t × A_pixel × QE / (hc/λ)
```

Approximate values:
- Sky luminance: 5,000 cd/m² (typical outdoor overcast), or 500 cd/m² (shaded)
- f/2.8, t=1ms, QE=50%
- N_photons ≈ 20,000–200,000 per pixel (depends on L)

## Noise in render units

| Source | Formula | Value (render units) |
|---|---|---|
| Shot noise at sky level | sqrt(N_sky) / K | sqrt(render_sky × K) / K |
| Read noise | σ_read / K | ~0.000003 (negligible) |
| **Total noise (single frame)** | | **≈ sqrt(0.5 / K)** |
| **Noise (frame difference)** | sqrt(2) × single-frame | **≈ sqrt(1.0 / K)** |

The key: noise scales as `1/sqrt(K)` where K = photons per render unit.
More photons → less noise → higher SNR.

## SNR calculation

```
SNR = signal / noise = 0.067 / sqrt(1.0 / K)
    = 0.067 × sqrt(K)
    = 0.067 × sqrt(N_photons_per_render_unit)
```

For a 5,000 cd/m² sky, K ≈ 40,000 photons per render unit:
```
SNR = 0.067 × sqrt(40,000) = 0.067 × 200 = 13.4
```

For a 500 cd/m² sky (darker), K ≈ 4,000:
```
SNR = 0.067 × sqrt(4,000) = 0.067 × 63 = 4.2
```

## Comparison with M1 thresholds

| Sky brightness | Photon count | SNR | M1 threshold | Survives? |
|---|---|---|---|---|
| 5,000 cd/m² (bright outdoor) | ~200K | ~13 | flux≥5 → SNR≈8 | **YES** |
| 2,000 cd/m² (overcast) | ~80K | ~8 | flux≥5 → SNR≈8 | **MARGINAL** |
| 500 cd/m² (shade) | ~20K | ~4 | flux≥5 → SNR≈8 | **NO** |

## Key caveat

This assumes:
1. The render's 0.067 signal corresponds to a physically plausible drone brightness
2. The sky luminance is in the expected range
3. No other noise sources (read noise is negligible; dark current negligible for
   short exposures)
4. The emission shader produces a realistic brightness relative to the sky

The emission shader (strength=100) may not match real drone reflectance. A
real drone reflecting sunlight would produce a different signal level. The
0.067 measurement is specific to this emission setup and may not generalize.

## Bottom line

**The margin is unresolved.** Whether the 0.067 signal survives depends on sky
brightness (photon count), which is scene-dependent. At bright outdoor lighting
(≥2,000 cd/m²), SNR is 8–13 and detection is plausible. In shade or overcast
conditions (≤500 cd/m²), SNR drops below 5 and detection becomes unreliable.

The earlier characterization of "thin margin" or "comfortable" was based on
unit-mismatched numbers and should not be relied upon. This analysis establishes
the correct framework but the actual photon count for this specific scene has
not been empirically measured.
