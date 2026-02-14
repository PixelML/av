"""Video file discovery and validation."""

from __future__ import annotations

from pathlib import Path

from av.core.constants import VIDEO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Check if path points to a video file by extension."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def discover_videos(path: Path) -> list[Path]:
    """Find all video files in a path (file or directory).

    If path is a file, returns [path] if it's a video.
    If path is a directory, recursively finds all videos.
    """
    if path.is_file():
        return [path] if is_video_file(path) else []
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if is_video_file(p))
    return []
