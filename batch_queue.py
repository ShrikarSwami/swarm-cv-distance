"""
Resumable batch queue for rendering multi-view drone footage.

Features:
- Output root configurable at runtime (for 1TB drive)
- Durable per-clip state (crash resumes, not restarts)
- ExFAT-safe: few large files, never large directories of small ones

Usage:
    python batch_queue.py --config sweep_config.json --output-root /Volumes/1TB/drone_footage
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

# Default Blender path (macOS Steam)
DEFAULT_BLENDER = str(Path.home() / "Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender")

class ClipState:
    """State for a single clip render job."""

    def __init__(self, clip_name: str, config: Dict):
        self.clip_name = clip_name
        self.config = config
        self.status = "pending"  # pending, rendering, completed, failed
        self.start_time = None
        self.end_time = None
        self.output_path = None
        self.error = None
        self.retry_count = 0

    def to_dict(self) -> Dict:
        return {
            "clip_name": self.clip_name,
            "config": self.config,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output_path": self.output_path,
            "error": self.error,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClipState':
        state = cls(data["clip_name"], data["config"])
        state.status = data.get("status", "pending")
        state.start_time = data.get("start_time")
        state.end_time = data.get("end_time")
        state.output_path = data.get("output_path")
        state.error = data.get("error")
        state.retry_count = data.get("retry_count", 0)
        return state


class BatchQueue:
    """Resumable batch rendering queue."""

    def __init__(self, output_root: str, blender_path: str = DEFAULT_BLENDER):
        self.output_root = Path(output_root)
        self.blender_path = blender_path
        self.state_dir = self.output_root / ".queue_state"
        self.state_file = self.state_dir / "queue_state.json"
        self.clips: Dict[str, ClipState] = {}
        self.render_script = str(Path(__file__).parent / "render_clip.py")

        # Create state directory if it doesn't exist
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state if available
        self._load_state()

    def _load_state(self):
        """Load queue state from disk."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                for clip_data in data.get("clips", []):
                    state = ClipState.from_dict(clip_data)
                    self.clips[state.clip_name] = state
            print(f"Loaded state: {len(self.clips)} clips")

    def _save_state(self):
        """Save queue state to disk."""
        data = {
            "clips": [state.to_dict() for state in self.clips.values()],
            "last_updated": time.time(),
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_clip(self, clip_name: str, config: Dict):
        """Add a clip to the queue."""
        if clip_name not in self.clips:
            self.clips[clip_name] = ClipState(clip_name, config)
            self._save_state()
            print(f"Added clip: {clip_name}")

    def add_clips_from_config(self, config_path: str):
        """Add multiple clips from a config file."""
        with open(config_path) as f:
            config = json.load(f)

        clips = config.get("clips", [])
        for clip_config in clips:
            clip_name = clip_config["clip_name"]
            self.add_clip(clip_name, clip_config)

        print(f"Added {len(clips)} clips from {config_path}")

    def get_pending_clips(self) -> List[ClipState]:
        """Get all pending clips."""
        return [state for state in self.clips.values()
                if state.status == "pending"]

    def get_failed_clips(self, max_retries: int = 3) -> List[ClipState]:
        """Get failed clips that can be retried."""
        return [state for state in self.clips.values()
                if state.status == "failed" and state.retry_count < max_retries]

    def render_clip(self, state: ClipState) -> bool:
        """Render a single clip. Returns True on success."""
        clip_name = state.clip_name
        config = state.config

        # Create output directory
        clip_dir = self.output_root / "clips" / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Update config with output path
        config["dataset_root"] = str(self.output_root)

        # Save config to temporary file
        config_path = self.state_dir / f"{clip_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Update state
        state.status = "rendering"
        state.start_time = time.time()
        state.output_path = str(clip_dir)
        self._save_state()

        print(f"Rendering {clip_name}...")

        try:
            # Run Blender
            proc = subprocess.run(
                [self.blender_path, "--background", "--python", self.render_script, "--", str(config_path)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per clip
            )

            if proc.returncode == 0:
                state.status = "completed"
                state.end_time = time.time()
                self._save_state()
                print(f"  Completed {clip_name} in {state.end_time - state.start_time:.1f}s")
                return True
            else:
                state.status = "failed"
                state.end_time = time.time()
                state.error = proc.stderr[-500:] if proc.stderr else "Unknown error"
                state.retry_count += 1
                self._save_state()
                print(f"  Failed {clip_name}: {state.error[:100]}...")
                return False

        except subprocess.TimeoutExpired:
            state.status = "failed"
            state.end_time = time.time()
            state.error = "Timeout after 600s"
            state.retry_count += 1
            self._save_state()
            print(f"  Timeout {clip_name}")
            return False

        except Exception as e:
            state.status = "failed"
            state.end_time = time.time()
            state.error = str(e)
            state.retry_count += 1
            self._save_state()
            print(f"  Error {clip_name}: {e}")
            return False

    def run(self, max_clips: Optional[int] = None, max_retries: int = 3):
        """Run the queue until all clips are rendered or max_clips is reached."""
        print(f"Starting batch queue: {len(self.clips)} clips total")
        print(f"Output root: {self.output_root}")

        # Get pending clips
        pending = self.get_pending_clips()
        if max_clips:
            pending = pending[:max_clips]

        print(f"Pending clips: {len(pending)}")

        # Render each clip
        for i, state in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}] {state.clip_name}")
            self.render_clip(state)

        # Retry failed clips
        failed = self.get_failed_clips(max_retries)
        if failed:
            print(f"\nRetrying {len(failed)} failed clips...")
            for state in failed:
                print(f"\nRetry {state.retry_count}/{max_retries}: {state.clip_name}")
                self.render_clip(state)

        # Print summary
        completed = sum(1 for s in self.clips.values() if s.status == "completed")
        failed_count = sum(1 for s in self.clips.values() if s.status == "failed")
        pending_count = sum(1 for s in self.clips.values() if s.status == "pending")

        print(f"\n{'='*60}")
        print(f"Queue complete:")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed_count}")
        print(f"  Pending: {pending_count}")
        print(f"{'='*60}")

    def get_status(self) -> Dict:
        """Get queue status."""
        completed = sum(1 for s in self.clips.values() if s.status == "completed")
        failed = sum(1 for s in self.clips.values() if s.status == "failed")
        pending = sum(1 for s in self.clips.values() if s.status == "pending")

        return {
            "total": len(self.clips),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "output_root": str(self.output_root),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Resumable batch rendering queue")
    parser.add_argument("--config", required=True, help="Path to sweep config JSON")
    parser.add_argument("--output-root", required=True, help="Output root directory")
    parser.add_argument("--blender", default=DEFAULT_BLENDER, help="Path to Blender")
    parser.add_argument("--max-clips", type=int, help="Maximum clips to render")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for failed clips")
    parser.add_argument("--status", action="store_true", help="Print status and exit")

    args = parser.parse_args()

    queue = BatchQueue(args.output_root, args.blender)

    if args.status:
        status = queue.get_status()
        print(json.dumps(status, indent=2))
        return

    queue.add_clips_from_config(args.config)
    queue.run(max_clips=args.max_clips, max_retries=args.max_retries)


if __name__ == "__main__":
    main()
