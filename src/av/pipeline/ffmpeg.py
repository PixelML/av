"""FFmpeg/ffprobe utilities for video processing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from av.core.exceptions import FFmpegError


@dataclass
class VideoMeta:
    duration_sec: float
    width: int | None
    height: int | None
    fps: float | None
    codec: str | None
    bitrate: int | None
    file_size_bytes: int


def get_video_info(path: Path) -> VideoMeta:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except FileNotFoundError:
        raise FFmpegError("ffprobe not found. Install ffmpeg: brew install ffmpeg", cmd=" ".join(cmd))
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"ffprobe failed: {e.stderr}", cmd=" ".join(cmd), returncode=e.returncode)
    except subprocess.TimeoutExpired:
        raise FFmpegError("ffprobe timed out", cmd=" ".join(cmd))

    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    # Find video stream
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    duration = float(fmt.get("duration", 0))
    width = None
    height = None
    fps = None
    codec = None
    bitrate = int(fmt.get("bit_rate", 0)) or None

    if video_stream:
        width = video_stream.get("width")
        height = video_stream.get("height")
        codec = video_stream.get("codec_name")
        # Parse frame rate (e.g., "30/1" or "29.97")
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        if "/" in str(r_frame_rate):
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else None
        else:
            fps = float(r_frame_rate) if r_frame_rate else None

    return VideoMeta(
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        bitrate=bitrate,
        file_size_bytes=path.stat().st_size,
    )


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    """Extract audio as 16kHz mono WAV for transcription."""
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-f", "wav",
        "-y",  # overwrite
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    except FileNotFoundError:
        raise FFmpegError("ffmpeg not found. Install ffmpeg: brew install ffmpeg", cmd=" ".join(cmd))
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Audio extraction failed: {e.stderr}", cmd=" ".join(cmd), returncode=e.returncode)

    return output_path


def extract_frames(
    video_path: Path,
    fps_sample: float = 0.5,
    max_frames: int = 200,
    output_dir: Path | None = None,
) -> list[tuple[Path, float]]:
    """Extract frames at given FPS rate. Returns list of (frame_path, timestamp_sec)."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="av_frames_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration to calculate frame count
    meta = get_video_info(video_path)
    total_frames = int(meta.duration_sec * fps_sample)
    total_frames = min(total_frames, max_frames)

    if total_frames <= 0:
        return []

    print(f"  Extracting up to {total_frames} frames at {fps_sample} fps...", file=sys.stderr)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps_sample}",
        "-frames:v", str(total_frames),
        "-q:v", "2",
        "-y",
        str(output_dir / "frame_%06d.jpg"),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Frame extraction failed: {e.stderr}", cmd=" ".join(cmd), returncode=e.returncode)

    # Collect extracted frames with timestamps
    frames: list[tuple[Path, float]] = []
    for frame_path in sorted(output_dir.glob("frame_*.jpg")):
        # Frame number from filename (1-indexed)
        frame_num = int(frame_path.stem.split("_")[1])
        timestamp = (frame_num - 1) / fps_sample
        frames.append((frame_path, timestamp))

    return frames[:max_frames]
