"""YouTube/download URL helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from av.core.exceptions import IngestError

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def is_url(value: str) -> bool:
    """Return True if value looks like an absolute URL."""
    return bool(_URL_RE.match(value.strip()))


def download_video(url: str) -> Path:
    """Download a video URL using yt-dlp and return local file path."""
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        raise IngestError(
            "URL ingest requires yt-dlp, but it was not found in PATH. "
            "Install with: brew install yt-dlp"
        )

    out_dir = Path(tempfile.mkdtemp(prefix="av_ytdlp_"))
    out_template = str(out_dir / "%(title).120s-%(id)s.%(ext)s")

    cmd = [
        yt_dlp,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*[height<=480]+ba/b[height<=480]/b",
        "-o",
        out_template,
        url,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise IngestError(f"Failed to download URL with yt-dlp (exit {e.returncode}).") from e

    files = [p for p in out_dir.iterdir() if p.is_file()]
    if not files:
        raise IngestError("yt-dlp finished but no downloaded file was found.")

    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[0]
