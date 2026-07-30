# B-Sweep Analytic Report

## Overview

- **Configs per scale:** 54 (6 n_views x 3 geometry_class x 3 noise_std_px)
- **Trials per config:** 20
- **Drones:** 5
- **Standoff:** 2000.0 m
- **Match threshold:** 1.5 m
- **Epipolar threshold:** 3.0 px

## Full scale (AREA_KM=5.0)

| n_views | Geometry | Noise (px) | Matched | Recall | Ghosts | F1 | Median err (m) | P95 err (m) | Coverage % |
|---------|----------|------------|---------|--------|--------|----|----------------|-------------|------------|
|  2 | all_ground   |    0 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   0.0 |
|  2 | all_ground   |    1 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   0.0 |
|  2 | all_ground   |    3 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   0.0 |
|  2 | mixed        |    0 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  10.0 |
|  2 | mixed        |    1 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  10.0 |
|  2 | mixed        |    3 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  10.0 |
|  2 | surround     |    0 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  2 | surround     |    1 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  2 | surround     |    3 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  4 | all_ground   |    0 |   0.0 | 0.000 |  1.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  4 | all_ground   |    1 |   0.0 | 0.000 |  0.5 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  4 | all_ground   |    3 |   0.0 | 0.000 |  0.3 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  4 | mixed        |    0 |   1.0 | 0.200 |  0.0 | 0.333 |   0.00 &plusmn; 0.00 |   0.00 |  25.0 |
|  4 | mixed        |    1 |   0.8 | 0.170 |  0.1 | 0.283 |   0.83 &plusmn; 0.45 |   0.83 |  25.0 |
|  4 | mixed        |    3 |   0.1 | 0.020 |  0.8 | 0.033 |   0.09 &plusmn; 0.29 |   0.09 |  25.0 |
|  4 | surround     |    0 |   2.0 | 0.400 |  0.0 | 0.571 |   0.00 &plusmn; 0.00 |   0.00 |  40.0 |
|  4 | surround     |    1 |   1.6 | 0.330 |  0.3 | 0.471 |   0.85 &plusmn; 0.20 |   0.94 |  40.0 |
|  4 | surround     |    3 |   0.3 | 0.070 |  1.6 | 0.101 |   0.29 &plusmn; 0.49 |   0.30 |  40.0 |
|  6 | all_ground   |    0 |   1.0 | 0.200 |  0.0 | 0.333 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  6 | all_ground   |    1 |   0.3 | 0.070 |  0.6 | 0.117 |   0.45 &plusmn; 0.62 |   0.45 |  20.0 |
|  6 | all_ground   |    3 |   0.0 | 0.000 |  0.6 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
|  6 | mixed        |    0 |   2.0 | 0.400 |  0.0 | 0.571 |   0.00 &plusmn; 0.00 |   0.00 |  26.7 |
|  6 | mixed        |    1 |   1.0 | 0.200 |  0.9 | 0.290 |   0.64 &plusmn; 0.27 |   0.64 |  26.7 |
|  6 | mixed        |    3 |   0.3 | 0.070 |  1.3 | 0.105 |   0.37 &plusmn; 0.53 |   0.37 |  26.7 |
|  6 | surround     |    0 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   3.3 |
|  6 | surround     |    1 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   3.3 |
|  6 | surround     |    3 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   3.3 |
|  8 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 |  42.5 |
|  8 | all_ground   |    1 |   1.9 | 0.380 |  3.0 | 0.384 |   0.71 &plusmn; 0.22 |   0.91 |  42.5 |
|  8 | all_ground   |    3 |   0.3 | 0.060 |  3.5 | 0.070 |   0.32 &plusmn; 0.51 |   0.32 |  42.5 |
|  8 | mixed        |    0 |   2.0 | 0.400 |  0.0 | 0.571 |   0.00 &plusmn; 0.00 |   0.00 |  25.0 |
|  8 | mixed        |    1 |   1.1 | 0.230 |  0.8 | 0.333 |   0.60 &plusmn; 0.23 |   0.64 |  25.0 |
|  8 | mixed        |    3 |   0.4 | 0.080 |  1.4 | 0.121 |   0.42 &plusmn; 0.55 |   0.42 |  25.0 |
|  8 | surround     |    0 |   2.0 | 0.400 |  0.0 | 0.571 |   0.00 &plusmn; 0.00 |   0.00 |  42.5 |
|  8 | surround     |    1 |   2.0 | 0.400 |  0.0 | 0.571 |   0.55 &plusmn; 0.14 |   0.66 |  42.5 |
|  8 | surround     |    3 |   0.6 | 0.120 |  1.8 | 0.170 |   0.55 &plusmn; 0.57 |   0.55 |  42.5 |
| 10 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 |  22.0 |
| 10 | all_ground   |    1 |   0.8 | 0.150 |  4.1 | 0.150 |   0.52 &plusmn; 0.57 |   0.56 |  22.0 |
| 10 | all_ground   |    3 |   0.0 | 0.000 |  3.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  22.0 |
| 10 | mixed        |    0 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   0.0 |
| 10 | mixed        |    1 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   0.0 |
| 10 | mixed        |    3 |   0.0 | 0.000 |  0.0 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |   0.0 |
| 10 | surround     |    0 |   4.0 | 0.800 |  0.0 | 0.889 |   0.00 &plusmn; 0.00 |   0.00 |  20.0 |
| 10 | surround     |    1 |   2.0 | 0.410 |  1.9 | 0.461 |   0.82 &plusmn; 0.26 |   0.98 |  20.0 |
| 10 | surround     |    3 |   0.2 | 0.040 |  2.8 | 0.049 |   0.18 &plusmn; 0.37 |   0.18 |  20.0 |
| 12 | all_ground   |    0 |   4.0 | 0.800 |  0.0 | 0.889 |   0.00 &plusmn; 0.00 |   0.00 |  13.3 |
| 12 | all_ground   |    1 |   0.2 | 0.040 |  3.7 | 0.044 |   0.25 &plusmn; 0.51 |   0.25 |  13.3 |
| 12 | all_ground   |    3 |   0.0 | 0.000 |  2.2 | 0.000 |   0.00 &plusmn; 0.00 |   0.00 |  13.3 |
| 12 | mixed        |    0 |   1.0 | 0.200 |  0.0 | 0.333 |   0.00 &plusmn; 0.00 |   0.00 |   6.7 |
| 12 | mixed        |    1 |   0.4 | 0.080 |  0.6 | 0.133 |   0.40 &plusmn; 0.53 |   0.40 |   6.7 |
| 12 | mixed        |    3 |   0.1 | 0.010 |  0.8 | 0.017 |   0.06 &plusmn; 0.27 |   0.06 |   6.7 |
| 12 | surround     |    0 |   3.0 | 0.600 |  0.0 | 0.750 |   0.00 &plusmn; 0.00 |   0.00 |  45.0 |
| 12 | surround     |    1 |   3.0 | 0.600 |  0.1 | 0.746 |   0.49 &plusmn; 0.17 |   0.67 |  45.0 |
| 12 | surround     |    3 |   1.1 | 0.220 |  2.5 | 0.255 |   0.72 &plusmn; 0.45 |   0.76 |  45.0 |

## Matched scale (AREA_KM=0.3)

| n_views | Geometry | Noise (px) | Matched | Recall | Ghosts | F1 | Median err (m) | P95 err (m) | Coverage % |
|---------|----------|------------|---------|--------|--------|----|----------------|-------------|------------|
|  2 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  2 | all_ground   |    1 |   1.9 | 0.380 |  2.9 | 0.392 |   0.96 &plusmn; 0.32 |   1.06 | 100.0 |
|  2 | all_ground   |    3 |   0.1 | 0.020 |  2.2 | 0.029 |   0.11 &plusmn; 0.34 |   0.11 | 100.0 |
|  2 | mixed        |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  2 | mixed        |    1 |   3.4 | 0.680 |  1.4 | 0.699 |   0.94 &plusmn; 0.20 |   1.11 | 100.0 |
|  2 | mixed        |    3 |   0.3 | 0.070 |  2.5 | 0.087 |   0.37 &plusmn; 0.56 |   0.37 | 100.0 |
|  2 | surround     |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  2 | surround     |    1 |   4.2 | 0.830 |  0.8 | 0.830 |   0.98 &plusmn; 0.19 |   1.21 | 100.0 |
|  2 | surround     |    3 |   0.3 | 0.070 |  2.8 | 0.081 |   0.38 &plusmn; 0.56 |   0.38 | 100.0 |
|  4 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  4 | all_ground   |    1 |   4.8 | 0.950 |  0.2 | 0.950 |   0.71 &plusmn; 0.13 |   0.96 | 100.0 |
|  4 | all_ground   |    3 |   0.6 | 0.120 |  4.8 | 0.115 |   0.55 &plusmn; 0.55 |   0.57 | 100.0 |
|  4 | mixed        |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  4 | mixed        |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.75 &plusmn; 0.16 |   1.09 | 100.0 |
|  4 | mixed        |    3 |   0.6 | 0.110 |  4.5 | 0.111 |   0.52 &plusmn; 0.56 |   0.53 | 100.0 |
|  4 | surround     |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  4 | surround     |    1 |   5.0 | 0.990 |  0.1 | 0.990 |   0.68 &plusmn; 0.14 |   1.05 | 100.0 |
|  4 | surround     |    3 |   0.4 | 0.080 |  4.8 | 0.079 |   0.44 &plusmn; 0.61 |   0.45 | 100.0 |
|  6 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  6 | all_ground   |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.63 &plusmn; 0.13 |   0.93 | 100.0 |
|  6 | all_ground   |    3 |   1.3 | 0.260 |  4.5 | 0.244 |   0.77 &plusmn; 0.47 |   0.90 | 100.0 |
|  6 | mixed        |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  6 | mixed        |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.67 &plusmn; 0.12 |   0.98 | 100.0 |
|  6 | mixed        |    3 |   1.4 | 0.290 |  4.0 | 0.282 |   1.06 &plusmn; 0.40 |   1.13 | 100.0 |
|  6 | surround     |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  6 | surround     |    1 |   5.0 | 1.000 |  0.1 | 0.996 |   0.63 &plusmn; 0.15 |   0.96 | 100.0 |
|  6 | surround     |    3 |   1.4 | 0.290 |  3.9 | 0.284 |   0.75 &plusmn; 0.46 |   0.87 | 100.0 |
|  8 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  8 | all_ground   |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.54 &plusmn; 0.11 |   0.86 | 100.0 |
|  8 | all_ground   |    3 |   1.6 | 0.320 |  4.2 | 0.296 |   0.96 &plusmn; 0.32 |   1.04 | 100.0 |
|  8 | mixed        |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  8 | mixed        |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.52 &plusmn; 0.11 |   0.79 | 100.0 |
|  8 | mixed        |    3 |   2.0 | 0.400 |  3.3 | 0.384 |   0.94 &plusmn; 0.45 |   1.08 | 100.0 |
|  8 | surround     |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
|  8 | surround     |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.46 &plusmn; 0.11 |   0.73 | 100.0 |
|  8 | surround     |    3 |   2.0 | 0.400 |  3.9 | 0.377 |   1.00 &plusmn; 0.34 |   1.15 | 100.0 |
| 10 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
| 10 | all_ground   |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.47 &plusmn; 0.10 |   0.68 | 100.0 |
| 10 | all_ground   |    3 |   2.1 | 0.430 |  3.9 | 0.395 |   1.01 &plusmn; 0.28 |   1.17 | 100.0 |
| 10 | mixed        |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
| 10 | mixed        |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.46 &plusmn; 0.11 |   0.74 | 100.0 |
| 10 | mixed        |    3 |   2.4 | 0.480 |  3.1 | 0.454 |   0.99 &plusmn; 0.38 |   1.13 | 100.0 |
| 10 | surround     |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
| 10 | surround     |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.48 &plusmn; 0.09 |   0.67 | 100.0 |
| 10 | surround     |    3 |   2.3 | 0.460 |  3.2 | 0.443 |   0.98 &plusmn; 0.39 |   1.14 | 100.0 |
| 12 | all_ground   |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
| 12 | all_ground   |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.45 &plusmn; 0.07 |   0.65 | 100.0 |
| 12 | all_ground   |    3 |   2.9 | 0.570 |  3.3 | 0.515 |   0.96 &plusmn; 0.24 |   1.17 | 100.0 |
| 12 | mixed        |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
| 12 | mixed        |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.45 &plusmn; 0.08 |   0.64 | 100.0 |
| 12 | mixed        |    3 |   3.1 | 0.630 |  2.2 | 0.607 |   0.98 &plusmn; 0.20 |   1.23 | 100.0 |
| 12 | surround     |    0 |   5.0 | 1.000 |  0.0 | 1.000 |   0.00 &plusmn; 0.00 |   0.00 | 100.0 |
| 12 | surround     |    1 |   5.0 | 1.000 |  0.0 | 1.000 |   0.42 &plusmn; 0.11 |   0.63 | 100.0 |
| 12 | surround     |    3 |   2.5 | 0.490 |  3.7 | 0.447 |   1.04 &plusmn; 0.16 |   1.22 | 100.0 |

## Key Takeaways

### Full scale

- **Full scale:** Error decreases with camera count. Mean error at 4 cameras: 0.23m vs 12 cameras: 0.21m (ratio 1.1x).
- **Matched scale:** Error decreases with camera count. Mean error at 4 cameras: 0.40m vs 12 cameras: 0.48m (ratio 0.8x).
- **Full scale:** At low camera counts (n_views ≤ 4), all_ground error = 0.00m, mixed = 0.15m, surround = 0.19m. All-ground performs competitively at this scale.
- **Matched scale:** At low camera counts (n_views ≤ 4), all_ground error = 0.39m, mixed = 0.43m, surround = 0.42m. All-ground performs competitively at this scale.
- **Full scale:** Diminishing returns above 8 views — error at 8 views = 0.35m vs 12 = 0.21m (38.8% improvement).
- **Matched scale:** Diminishing returns above 8 views — error at 8 views = 0.49m vs 12 = 0.48m (2.6% improvement).
- **Full scale:** Coverage < 95% for 54 config(s): 2v/all_ground, 2v/all_ground, 2v/all_ground, 2v/mixed, 2v/mixed, 2v/mixed, 2v/surround, 2v/surround, 2v/surround, 4v/all_ground, 4v/all_ground, 4v/all_ground, 4v/mixed, 4v/mixed, 4v/mixed, 4v/surround, 4v/surround, 4v/surround, 6v/all_ground, 6v/all_ground, 6v/all_ground, 6v/mixed, 6v/mixed, 6v/mixed, 6v/surround, 6v/surround, 6v/surround, 8v/all_ground, 8v/all_ground, 8v/all_ground, 8v/mixed, 8v/mixed, 8v/mixed, 8v/surround, 8v/surround, 8v/surround, 10v/all_ground, 10v/all_ground, 10v/all_ground, 10v/mixed, 10v/mixed, 10v/mixed, 10v/surround, 10v/surround, 10v/surround, 12v/all_ground, 12v/all_ground, 12v/all_ground, 12v/mixed, 12v/mixed, 12v/mixed, 12v/surround, 12v/surround, 12v/surround.
- **Full scale:** Zero-noise baseline recall > 0.8 for 2 config(s); median error range = 0.0000–0.0000m.
- **Matched scale:** Zero-noise baseline recall > 0.8 for 18 config(s); median error range = 0.0000–0.0000m.

