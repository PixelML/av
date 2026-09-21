"""Tests for ffmpeg clip export and playability validation (needs ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from av.core.exceptions import FFmpegError
from av.pipeline.clip_export import export_clips

ffmpeg = shutil.which("ffmpeg")


@pytest.fixture()
def source_media(tmp_path: Path) -> Path:
    assert ffmpeg
    out = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=duration=60:size=320x240:rate=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-c:a", "aac", "-b:a", "64k",
            "-y", str(out),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return out


@pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")
def test_export_clips_are_playable_and_synchronized(
    source_media: Path, tmp_path: Path
) -> None:
    clips = [
        {"rank": 1, "clip_id": "clip-a", "start_sec": 10.0, "end_sec": 25.0},
        {"rank": 2, "clip_id": "clip-b", "start_sec": 30.5, "end_sec": 45.5},
    ]
    out_dir = tmp_path / "render"
    receipts, elapsed = export_clips(source_media, clips, out_dir, overwrite=True)
    assert len(receipts) == 2
    assert elapsed > 0
    for receipt in clips and receipts:
        assert receipt["valid"], receipt["warnings"]
        assert abs(receipt["measured_duration_sec"] - receipt["requested_duration_sec"]) <= 1.0
        assert receipt["streams"]["video_codec"] == "h264"
        assert receipt["streams"]["audio_codec"] == "aac"
        assert Path(receipt["path"]).exists()


@pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")
def test_export_refuses_to_overwrite_without_flag(
    source_media: Path, tmp_path: Path
) -> None:
    clips = [{"rank": 1, "clip_id": "clip-a", "start_sec": 10.0, "end_sec": 20.0}]
    out_dir = tmp_path / "render"
    export_clips(source_media, clips, out_dir, overwrite=True)
    with pytest.raises(FFmpegError):
        export_clips(source_media, clips, out_dir, overwrite=False)


@pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")
def test_export_requires_existing_source(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError):
        export_clips(
            tmp_path / "missing.mp4",
            [{"rank": 1, "start_sec": 0.0, "end_sec": 5.0}],
            tmp_path / "out",
        )
