"""Optional FFmpeg clip rendering and playability validation.

Rendering is deliberately separate from provider selection: every second of
provider time and every token is accounted in the decision stages, and
``render_ms`` only ever measures local ffmpeg work.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from av.core.exceptions import FFmpegError

# Re-encode both streams so cuts land exactly on the requested seconds and
# audio stays synchronized; stream copy can only cut on keyframes.
_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "128k"]
_DURATION_TOLERANCE_SEC = 1.0
_START_TIME_TOLERANCE_SEC = 0.5


@dataclass
class ExportRequest:
    clip_rank: int
    start_sec: float
    end_sec: float


def _fmt_name(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}{m:02d}{s:02d}"


def _probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    except FileNotFoundError as exc:
        raise FFmpegError("ffprobe not found. Install ffmpeg: brew install ffmpeg / apt install ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"ffprobe failed: {exc.stderr}", cmd=" ".join(cmd)) from exc
    return json.loads(result.stdout)


def _stream_summary(probe: dict) -> dict:
    video = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
    return {
        "video_codec": video.get("codec_name") if video else None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "start_time_sec": float(probe.get("format", {}).get("start_time", 0) or 0),
    }


def export_clips(
    source: Path,
    clip_records: list[dict],
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[list[dict], float]:
    """Render each clip record to ``out_dir`` and validate the result.

    Returns (receipts, elapsed_seconds). Receipts carry the measured output
    duration, stream summary, and a ``valid`` flag with per-check warnings.
    """
    source = source.resolve()
    if not source.exists():
        raise FFmpegError(f"Source video not found: {source}")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        raise FFmpegError(f"Export target is not a directory: {out_dir}")

    source_audio = _probe(source)
    source_has_audio = any(
        s.get("codec_type") == "audio" for s in source_audio.get("streams", [])
    )

    receipts: list[dict] = []
    started = time.perf_counter()
    for clip in clip_records:
        rank = clip["rank"]
        start = float(clip["start_sec"])
        end = float(clip["end_sec"])
        duration = max(end - start, 0.0)
        out_path = out_dir / f"clip_{rank:02d}_{_fmt_name(start)}-{_fmt_name(end)}.mp4"
        if out_path.exists() and not overwrite:
            raise FFmpegError(
                f"Export file already exists: {out_path}. Pass --overwrite-export to replace it."
            )
        resolved = out_path.resolve()
        if resolved.parent != out_dir:
            raise FFmpegError(f"Refusing export outside the requested directory: {resolved}")
        # Create the output file before ffmpeg runs so a laggy NFS mount can
        # never fail ffmpeg's open() with ENOENT right after mkdir().
        resolved.touch(exist_ok=True)

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            *_VIDEO_ARGS,
            *_AUDIO_ARGS,
            "-movflags", "+faststart",
            "-y",
            str(resolved),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        except FileNotFoundError as exc:
            raise FFmpegError("ffmpeg not found. Install ffmpeg: brew install ffmpeg / apt install ffmpeg") from exc
        except subprocess.CalledProcessError as exc:
            raise FFmpegError(
                f"ffmpeg failed for clip {rank}: {exc.stderr}",
                cmd=" ".join(cmd),
            ) from exc

        probe = _probe(resolved)
        measured = float(probe.get("format", {}).get("duration", 0) or 0)
        streams = _stream_summary(probe)
        warnings: list[str] = []
        if abs(measured - duration) > _DURATION_TOLERANCE_SEC:
            warnings.append(
                f"Output duration {measured:.3f}s differs from requested {duration:.3f}s"
            )
        if not streams["video_codec"]:
            warnings.append("Output has no video stream")
        if source_has_audio and not streams["audio_codec"]:
            warnings.append("Source has audio but output does not; audio was lost")
        if abs(streams["start_time_sec"]) > _START_TIME_TOLERANCE_SEC:
            warnings.append(
                f"Output start_time {streams['start_time_sec']:.3f}s is not near zero"
            )
        receipts.append({
            "clip_rank": rank,
            "clip_id": clip.get("clip_id"),
            "path": str(resolved),
            "requested_start_sec": round(start, 3),
            "requested_end_sec": round(end, 3),
            "requested_duration_sec": round(duration, 3),
            "measured_duration_sec": round(measured, 3),
            "streams": streams,
            "valid": not warnings,
            "warnings": warnings,
            "command": "ffmpeg -ss <start> -i <source> -t <duration> "
            + " ".join(_VIDEO_ARGS + _AUDIO_ARGS + ["-movflags", "+faststart"]),
        })
    elapsed_ms = (time.perf_counter() - started) * 1000
    return receipts, elapsed_ms
