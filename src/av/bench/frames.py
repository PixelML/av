"""Pinned frame extraction for benchmarking.

``pipeline/ffmpeg.extract_frames`` is tuned for ingestion: it caps frame counts and
uses ingest defaults. Benchmarking needs the invocation itself to be part of the
record, so these helpers keep their command line fixed, return it verbatim for the
receipt, and never silently change sampling behind the caller's back.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from av.core.exceptions import FFmpegError

# Fixed encoder settings. Changing any of these changes every measurement, so they
# live here as constants rather than as call-site defaults.
JPEG_QUALITY = "2"
SCALE_WIDTH_DEFAULT = 768

# Real surveillance footage is frequently limited-range YUV, which the mjpeg encoder
# refuses outright. Normalising the pixel format keeps the pinned invocation working
# on ordinary CCTV files instead of failing on the footage that matters most.
PIXEL_FORMAT = "yuvj420p"


@dataclass
class FrameSet:
    paths: list[Path]
    timestamps: list[float]
    commands: list[str] = field(default_factory=list)
    interval_sec: float | None = None

    def __len__(self) -> int:
        return len(self.paths)


def _run(cmd: list[str], timeout: int = 600) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    except FileNotFoundError:
        raise FFmpegError("ffmpeg not found. Install ffmpeg: brew install ffmpeg", cmd=" ".join(cmd))
    except subprocess.CalledProcessError as e:
        raise FFmpegError(
            f"Frame extraction failed: {e.stderr}", cmd=" ".join(cmd), returncode=e.returncode
        )
    except subprocess.TimeoutExpired:
        raise FFmpegError("Frame extraction timed out", cmd=" ".join(cmd))


def _vf(interval_sec: float, scale_width: int | None) -> str:
    parts = [f"fps=1/{interval_sec}"]
    if scale_width:
        parts.append(f"scale={scale_width}:-2")
    parts.append(f"format={PIXEL_FORMAT}")
    return ",".join(parts)


def sample_interval(
    video_path: Path,
    interval_sec: float,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    max_frames: int = 1024,
    scale_width: int | None = SCALE_WIDTH_DEFAULT,
    out_dir: Path | None = None,
) -> FrameSet:
    """Sample one frame every ``interval_sec`` seconds.

    ``max_frames`` is a hard ceiling, not a target: a request that would exceed it is
    truncated and the caller is expected to record the truncation.
    """
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="av_bench_frames_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(video_path)]
    if duration_sec is not None:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-vf", _vf(interval_sec, scale_width),
        "-frames:v", str(max_frames),
        "-q:v", JPEG_QUALITY,
        str(out_dir / "f_%06d.jpg"),
    ]
    _run(cmd)

    paths = sorted(out_dir.glob("f_*.jpg"))
    timestamps = [start_sec + i * interval_sec for i in range(len(paths))]
    return FrameSet(paths=paths, timestamps=timestamps, commands=[" ".join(cmd)], interval_sec=interval_sec)


def sample_at(
    video_path: Path,
    timestamps: list[float],
    *,
    scale_width: int | None = SCALE_WIDTH_DEFAULT,
    out_dir: Path | None = None,
) -> FrameSet:
    """Grab one frame at each requested timestamp — the agentic arm's fetch step."""
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="av_bench_targeted_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    kept: list[float] = []
    commands: list[str] = []
    for i, ts in enumerate(sorted(set(round(t, 3) for t in timestamps))):
        out_path = out_dir / f"t_{i:04d}_{int(ts * 1000):09d}.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{max(ts, 0.0):.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", JPEG_QUALITY,
        ]
        vf = [f"scale={scale_width}:-2"] if scale_width else []
        vf.append(f"format={PIXEL_FORMAT}")
        cmd += ["-vf", ",".join(vf), str(out_path)]
        try:
            _run(cmd, timeout=120)
        except FFmpegError:
            continue  # a seek past the end is not a harness failure
        if out_path.exists() and out_path.stat().st_size > 0:
            paths.append(out_path)
            kept.append(ts)
            commands.append(" ".join(cmd))

    return FrameSet(paths=paths, timestamps=kept, commands=commands, interval_sec=None)
