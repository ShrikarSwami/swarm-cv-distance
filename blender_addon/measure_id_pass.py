"""
Extract drone measurements from an ID-pass EXR rendered by render_config.py.

Usage: python blender_addon/measure_id_pass.py <exr_path> <n_drones> <standoff_m> <focal_mm> <sensor_width_mm> <h_px>
"""

import json
import sys
import numpy as np
import OpenEXR
import Imath


def measure(exr_path, n_drones, standoff_m, focal_mm, sensor_width_mm, h_px):
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    # Read ID pass — single FLOAT stored as .V; RGB stored as .R/.G/.B
    channels = list(header['channels'].keys())
    idx_channel = None
    for name in ['id_.V', 'id_.R', 'IndexOB.R', 'IndexOB.V']:
        if name in channels:
            idx_channel = name
            break
    if idx_channel is None:
        raise ValueError(f"No ID pass channel. Available: {channels}")

    idx_str = exr_file.channel(idx_channel)
    idx_arr = np.frombuffer(idx_str, dtype=np.float32).reshape(h, w)

    # Extract per-drone measurements
    measurements = []
    for drone_id in range(1, n_drones + 1):
        mask = idx_arr == drone_id
        pixel_count = int(np.sum(mask))
        if pixel_count == 0:
            measurements.append({
                "drone_id": drone_id - 1,
                "visible": False,
                "pixel_count": 0,
                "bbox_width_px": 0,
                "bbox_height_px": 0,
                "centroid_x": None,
                "centroid_y": None,
            })
            continue

        ys, xs = np.where(mask)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

        measurements.append({
            "drone_id": drone_id - 1,
            "visible": True,
            "pixel_count": pixel_count,
            "bbox_width_px": bbox_w,
            "bbox_height_px": bbox_h,
            "centroid_x": round(centroid_x, 2),
            "centroid_y": round(centroid_y, 2),
        })

    visible_count = sum(1 for m in measurements if m["visible"])

    # Angular resolution (primary metric)
    sensor_w_m = sensor_width_mm * 1e-3
    focal_m = focal_mm * 1e-3
    theta_um = (sensor_w_m / (focal_m * h_px)) * 1e6
    drone_size = 0.5
    expected_px = drone_size / (standoff_m * (theta_um * 1e-6))

    return {
        "n_drones": n_drones,
        "visible_count": visible_count,
        "visibility_fraction": round(visible_count / n_drones, 3),
        "theta_um_rad": round(theta_um, 4),
        "expected_apparent_px": round(expected_px, 2),
        "image_size": [w, h],
        "drone_measurements": measurements,
    }


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: measure_id_pass.py <exr_path> <n_drones> <standoff_m> <focal_mm> <sensor_width_mm> <h_px>")
        sys.exit(1)

    result = measure(
        sys.argv[1],
        int(sys.argv[2]),
        float(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        int(sys.argv[6]),
    )
    print(json.dumps(result, indent=2))
