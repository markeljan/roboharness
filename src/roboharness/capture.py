"""Webcam video recording and low-resolution preview frame extraction via ffmpeg."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class CaptureError(RuntimeError):
    """Raised when webcam capture, synthetic video generation, or frame extraction fails."""


def build_webcam_ffmpeg_cmd(
    duration_sec: int,
    device: str,
    output_path: Path,
) -> list[str]:
    """Construct the macOS avfoundation ffmpeg capture command."""
    return [
        "ffmpeg",
        "-y",
        "-f",
        "avfoundation",
        "-framerate",
        "30",
        "-video_size",
        "640x480",
        "-i",
        f"{device}:none",
        "-t",
        str(duration_sec),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def record_webcam_clip(
    duration_sec: int = 5,
    device: str = "0",
    output_path: Path | None = None,
) -> Path:
    """Record a short H.264 MP4 clip from a macOS webcam using ffmpeg avfoundation."""
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")

    if output_path is None:
        out_path = Path(tempfile.gettempdir()) / f"roboharness_clip_{int(time.time())}.mp4"
    else:
        out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_webcam_ffmpeg_cmd(duration_sec=duration_sec, device=device, output_path=out_path)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CaptureError(
            "ffmpeg executable not found on PATH. Install it via `brew install ffmpeg` "
            "or pass a pre-recorded video with `--video <path.mp4>`."
        ) from exc

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-400:]
        raise CaptureError(
            f"Webcam recording failed (ffmpeg exit {proc.returncode}) on device '{device}'.\n"
            "Troubleshooting on macOS:\n"
            "  1. Grant Camera permission to your terminal/IDE in "
            "System Settings -> Privacy & Security -> Camera.\n"
            "  2. List available avfoundation video devices with:\n"
            '     ffmpeg -f avfoundation -list_devices true -i ""\n'
            "  3. Retry with `--device <index>` or set `ROBOHARNESS_WEBCAM_DEVICE=<index>`.\n"
            f"ffmpeg stderr: {stderr_tail}"
        )

    return out_path


def _probe_video_duration_sec(video_path: Path) -> float | None:
    """Probe media duration in seconds using ffprobe, with ffmpeg stderr fallback."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            val = proc.stdout.strip()
            if val and val.lower() != "n/a":
                dur = float(val)
                if dur > 0.0:
                    return dur
    except (FileNotFoundError, ValueError):
        pass

    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(video_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        match = _DURATION_RE.search(proc.stderr or "")
        if match:
            hours = int(match.group(1))
            mins = int(match.group(2))
            secs = float(match.group(3))
            dur = hours * 3600.0 + mins * 60.0 + secs
            if dur > 0.0:
                return dur
    except FileNotFoundError:
        pass
    return None


def extract_preview_frames(video_path: Path, count: int = 4) -> list[bytes]:
    """Extract `count` evenly spaced low-resolution (320px wide) JPEG frames from a video."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    duration_sec = _probe_video_duration_sec(path)
    last_stderr = ""

    with tempfile.TemporaryDirectory(prefix="roboharness_frames_") as tmp:
        tmp_dir = Path(tmp)
        frames: list[bytes] = []

        if duration_sec is not None and duration_sec > 0.0 and count > 1:
            fps = max(count / duration_sec, 0.1)
            out_pattern = tmp_dir / "frame_%03d.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"fps={fps:.6f},scale=320:-2",
                "-frames:v",
                str(count),
                "-q:v",
                "5",
                "-f",
                "image2",
                str(out_pattern),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                last_stderr = (proc.stderr or "").strip()
            except FileNotFoundError as exc:
                raise CaptureError(
                    "ffmpeg executable not found on PATH; cannot extract preview frames."
                ) from exc

            frames = [p.read_bytes() for p in sorted(tmp_dir.glob("frame_*.jpg")) if p.stat().st_size > 0]

        if not frames:
            fallback_path = tmp_dir / "fallback_001.jpg"
            fallback_cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                "scale=320:-2",
                "-frames:v",
                "1",
                "-q:v",
                "5",
                "-f",
                "image2",
                str(fallback_path),
            ]
            try:
                proc = subprocess.run(fallback_cmd, capture_output=True, text=True, check=False)
                last_stderr = (proc.stderr or "").strip() or last_stderr
            except FileNotFoundError as exc:
                raise CaptureError(
                    "ffmpeg executable not found on PATH; cannot extract preview frames."
                ) from exc

            if fallback_path.exists() and fallback_path.stat().st_size > 0:
                frames = [fallback_path.read_bytes()]

        if not frames:
            raise CaptureError(
                f"Could not extract preview frames from '{path}'. "
                f"Ensure the file is a valid video. ffmpeg stderr: {last_stderr[-300:]}"
            )
        return frames[:count]


def make_synthetic_test_video(output_path: Path, duration_sec: float = 2.0) -> Path:
    """Generate a deterministic synthetic H.264 MP4 clip for offline testing without a webcam."""
    if duration_sec <= 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lavfi_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec:.2f}:size=320x240:rate=15",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        proc = subprocess.run(lavfi_cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise CaptureError(
            "ffmpeg executable not found on PATH; cannot generate synthetic test video."
        ) from exc

    if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    # Fallback: synthesize Pillow frames and encode with ffmpeg
    fps = 10
    total_frames = max(2, round(duration_sec * fps))
    with tempfile.TemporaryDirectory(prefix="roboharness_synth_") as tmp:
        tmp_dir = Path(tmp)
        for i in range(total_frames):
            img = Image.new("RGB", (320, 240), (20, 24, 34))
            draw = ImageDraw.Draw(img)
            x = 40 + int((240 * i) / max(1, total_frames - 1))
            draw.ellipse([x - 16, 104, x + 16, 136], fill=(56, 189, 248))
            draw.text((12, 12), f"ROBOHARNESS SYNTH FRAME {i + 1}/{total_frames}", fill=(226, 232, 240))
            img.save(tmp_dir / f"f_{i:04d}.png")

        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "f_%04d.png"),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        fb = subprocess.run(fallback_cmd, capture_output=True, text=True, check=False)
        if fb.returncode != 0 or not out_path.exists():
            raise CaptureError(
                f"Failed to generate synthetic test video at '{out_path}': "
                f"{(fb.stderr or proc.stderr or '').strip()[-300:]}"
            )
    return out_path
