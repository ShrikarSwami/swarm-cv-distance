"""
Verify FFV1/MKV round-trip is bit-exact.

Creates a synthetic frame, encodes to FFV1/MKV, decodes back, and compares.
Must pass before committing to FFV1/MKV as the master video format.

Run: python verify_ffv1.py
"""

import subprocess
import tempfile
import os
import numpy as np


def create_test_frames(n_frames=5, width=640, height=480, channels=3):
    """Create synthetic test frames with known content."""
    frames = []
    for i in range(n_frames):
        frame = np.zeros((height, width, channels), dtype=np.uint8)
        # Draw a unique pattern per frame
        frame[:, :, 0] = (i * 50) % 256  # R varies per frame
        frame[:, :, 1] = 128  # G constant
        frame[:, :, 2] = (i * 37 + 100) % 256  # B varies per frame
        # Add a small marker that's easy to verify
        y0, y1 = 100 + i * 10, 120 + i * 10
        frame[y0:y1, 100:120] = 255  # white box
        frames.append(frame)
    return frames


def encode_ffv1(frames, output_path, width, height):
    """Encode frames to FFV1/MKV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", "30",
        "-i", "-",
        "-c:v", "ffv1",
        "-level", "3",
        "-coder", "1",  # range coder (default, best compression)
        "-context", "1",
        output_path,
    ]
    proc = subprocess.run(
        cmd,
        input=b"".join(f.tobytes() for f in frames),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode()}")
    return output_path


def decode_ffv1(input_path, n_frames, width, height):
    """Decode MKV/FFV1 back to raw frames."""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()}")
    raw = proc.stdout
    frame_size = width * height * 3
    frames = []
    for i in range(n_frames):
        start = i * frame_size
        end = start + frame_size
        if end > len(raw):
            break
        frame = np.frombuffer(raw[start:end], dtype=np.uint8).reshape(height, width, 3)
        frames.append(frame)
    return frames


def main():
    print("=== FFV1/MKV Bit-Exact Verification ===\n")

    n_frames = 5
    width, height = 640, 480

    # Create original frames
    original = create_test_frames(n_frames, width, height)
    print(f"Created {n_frames} test frames ({width}x{height})")

    with tempfile.TemporaryDirectory() as tmpdir:
        mkv_path = os.path.join(tmpdir, "test.mkv")

        # Encode
        encode_ffv1(original, mkv_path, width, height)
        file_size = os.path.getsize(mkv_path)
        raw_size = n_frames * width * height * 3
        print(f"Encoded to MKV: {file_size:,} bytes (raw would be {raw_size:,} bytes, {raw_size/file_size:.1f}x)")

        # Decode
        decoded = decode_ffv1(mkv_path, n_frames, width, height)
        print(f"Decoded {len(decoded)} frames")

        # Compare
        all_match = True
        for i, (orig, dec) in enumerate(zip(original, decoded)):
            match = np.array_equal(orig, dec)
            if not match:
                diff = np.abs(orig.astype(int) - dec.astype(int))
                max_diff = diff.max()
                mean_diff = diff.mean()
                print(f"  Frame {i}: MISMATCH (max diff={max_diff}, mean diff={mean_diff:.2f})")
                all_match = False
            else:
                print(f"  Frame {i}: EXACT MATCH")

        if all_match:
            print("\n✓ FFV1/MKV round-trip is bit-exact. Safe to use as master format.")
        else:
            print("\n✗ FFV1/MKV round-trip is NOT bit-exact. Do not use as master format.")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
