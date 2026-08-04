"""Verify rendered frames are actually different across environments/weather."""
import subprocess
import numpy as np
from pathlib import Path
import tempfile

def decode_first_frame(mkv_path: str, output_dir: str) -> str:
    """Decode first frame from MKV to PNG."""
    output_path = Path(output_dir) / f"{Path(mkv_path).stem}_frame0.png"
    cmd = [
        "ffmpeg", "-y", "-i", str(mkv_path),
        "-vframes", "1", "-f", "image2",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return str(output_path)


def load_image_as_array(png_path: str) -> np.ndarray:
    """Load PNG as numpy array (using PIL)."""
    from PIL import Image
    img = Image.open(png_path)
    return np.array(img)


def compare_frames(frame1: np.ndarray, frame2: np.ndarray) -> dict:
    """Compare two frames and return statistics."""
    if frame1.shape != frame2.shape:
        return {"error": "Different shapes"}

    diff = np.abs(frame1.astype(float) - frame2.astype(float))
    return {
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "ssim_approx": float(1 - diff.mean() / 255.0),  # Rough SSIM approximation
    }


if __name__ == "__main__":
    output_dir = Path("dataset_smoke_test/frame_comparison")
    output_dir.mkdir(exist_ok=True)

    clips = [
        "desert_clear",
        "desert_overcast",
        "forest_clear",
        "forest_hazy",
        "city_dusk",
    ]

    print("=" * 70)
    print("Decoding first frame from each clip...")
    print("=" * 70)

    frames = {}
    for clip in clips:
        mkv_path = Path(f"dataset_smoke_test/clips/{clip}/frames.mkv")
        if mkv_path.exists():
            png_path = decode_first_frame(str(mkv_path), str(output_dir))
            frames[clip] = load_image_as_array(png_path)
            print(f"  {clip}: {frames[clip].shape}, mean={frames[clip].mean():.1f}")
        else:
            print(f"  {clip}: MKV not found!")

    print("\n" + "=" * 70)
    print("Comparing frames across clips...")
    print("=" * 70)

    # Compare desert_clear vs desert_overcast (same environment, different weather)
    if "desert_clear" in frames and "desert_overcast" in frames:
        stats = compare_frames(frames["desert_clear"], frames["desert_overcast"])
        print(f"\ndesert_clear vs desert_overcast (same env, different weather):")
        print(f"  Mean pixel diff: {stats['mean_diff']:.2f}")
        print(f"  Max pixel diff: {stats['max_diff']:.2f}")
        print(f"  Approx SSIM: {stats['ssim_approx']:.4f}")
        print(f"  Status: {'DIFFERENT' if stats['mean_diff'] > 5 else 'IDENTICAL!'}")

    # Compare desert_clear vs forest_clear (different environment, same weather)
    if "desert_clear" in frames and "forest_clear" in frames:
        stats = compare_frames(frames["desert_clear"], frames["forest_clear"])
        print(f"\ndesert_clear vs forest_clear (different env, same weather):")
        print(f"  Mean pixel diff: {stats['mean_diff']:.2f}")
        print(f"  Max pixel diff: {stats['max_diff']:.2f}")
        print(f"  Approx SSIM: {stats['ssim_approx']:.4f}")
        print(f"  Status: {'DIFFERENT' if stats['mean_diff'] > 5 else 'IDENTICAL!'}")

    # Compare forest_clear vs forest_hazy (same environment, different weather)
    if "forest_clear" in frames and "forest_hazy" in frames:
        stats = compare_frames(frames["forest_clear"], frames["forest_hazy"])
        print(f"\nforest_clear vs forest_hazy (same env, different weather):")
        print(f"  Mean pixel diff: {stats['mean_diff']:.2f}")
        print(f"  Max pixel diff: {stats['max_diff']:.2f}")
        print(f"  Approx SSIM: {stats['ssim_approx']:.4f}")
        print(f"  Status: {'DIFFERENT' if stats['mean_diff'] > 5 else 'IDENTICAL!'}")

    # Compare all pairs
    print("\n" + "=" * 70)
    print("All pairwise comparisons...")
    print("=" * 70)

    for i, clip1 in enumerate(clips):
        for clip2 in clips[i+1:]:
            if clip1 in frames and clip2 in frames:
                stats = compare_frames(frames[clip1], frames[clip2])
                status = "DIFFERENT" if stats['mean_diff'] > 5 else "IDENTICAL!"
                print(f"  {clip1} vs {clip2}: mean_diff={stats['mean_diff']:.2f} [{status}]")

    # Save a visual comparison
    print("\n" + "=" * 70)
    print("Saving visual comparison...")
    print("=" * 70)

    from PIL import Image, ImageDraw, ImageFont

    # Create side-by-side comparison
    frame_height = frames[list(frames.keys())[0]].shape[0]
    frame_width = frames[list(frames.keys())[0]].shape[1]

    # Resize for comparison
    resize_h, resize_w = 270, 480

    comparison = Image.new('RGB', (resize_w * len(frames), resize_h))
    for i, (clip, frame) in enumerate(frames.items()):
        img = Image.fromarray(frame)
        img = img.resize((resize_w, resize_h))
        comparison.paste(img, (i * resize_w, 0))

    comparison_path = output_dir / "comparison.png"
    comparison.save(str(comparison_path))
    print(f"  Saved to: {comparison_path}")
