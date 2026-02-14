"""Tests for ingest pipeline graceful degradation with non-OpenAI providers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from av.core.config import AVConfig
from av.db.repository import Repository
from av.pipeline.ingest import ingest_video


@dataclass
class _FakeVideoMeta:
    duration_sec: float = 10.0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    codec: str = "h264"
    bitrate: int = 5000000
    file_size_bytes: int = 1024


@pytest.fixture()
def fake_video(tmp_path: Path) -> Path:
    """Create a tiny fake file to act as a 'video'."""
    f = tmp_path / "test.mp4"
    f.write_bytes(b"\x00" * 256)
    return f


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    db_path = tmp_path / "test.db"
    return Repository(db_path)


def _patch_ffmpeg():
    """Patch ffmpeg calls so we don't need a real video."""
    return patch(
        "av.pipeline.ingest.get_video_info",
        return_value=_FakeVideoMeta(),
    )


def test_anthropic_skips_transcription_and_embed(
    fake_video: Path, repo: Repository, capsys: pytest.CaptureFixture
) -> None:
    """Anthropic provider has no transcribe_model and no embed_model — both should be skipped."""
    config = AVConfig(
        provider="anthropic",
        api_base_url="https://api.anthropic.com/v1/",
        api_key="sk-ant-test",
        transcribe_model="",
        embed_model="",
        vision_model="claude-sonnet-4-5-20250929",
        chat_model="claude-sonnet-4-5-20250929",
    )

    with _patch_ffmpeg():
        result = ingest_video(fake_video, repo, config, no_embed=False)

    assert result["status"] == "complete_with_warnings"
    assert result["artifacts_count"] == 0
    assert "warnings" in result

    captured = capsys.readouterr()
    assert "Transcription disabled" in captured.err
    assert "Embeddings disabled" in captured.err


def test_gemini_skips_transcription_but_embeds(
    fake_video: Path, repo: Repository, capsys: pytest.CaptureFixture
) -> None:
    """Gemini has embed_model but no transcribe_model — skip transcription, but embedding would
    run if there were artifacts. With no artifacts, embedding step is a no-op."""
    config = AVConfig(
        provider="gemini",
        api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="AIza-test",
        transcribe_model="",
        embed_model="text-embedding-004",
        vision_model="gemini-2.5-flash",
        chat_model="gemini-2.5-flash",
    )

    with _patch_ffmpeg():
        result = ingest_video(fake_video, repo, config, no_embed=False)

    assert result["status"] == "complete_with_warnings"
    captured = capsys.readouterr()
    assert "Transcription disabled" in captured.err
    # No "Embeddings disabled" because embed_model is set (just no artifacts to embed)
    assert "Embeddings disabled" not in captured.err


def test_openai_does_not_skip(
    fake_video: Path, repo: Repository, capsys: pytest.CaptureFixture
) -> None:
    """OpenAI provider with all models set should attempt transcription (mocked)."""
    config = AVConfig(
        provider="openai",
        api_key="sk-test",
        transcribe_model="whisper",
        embed_model="text-embedding-3-small",
    )

    fake_audio = fake_video.parent / "audio.wav"
    fake_audio.write_bytes(b"\x00" * 100)

    mock_segments = [
        MagicMock(start_sec=0.0, end_sec=5.0, text="hello world"),
    ]
    # We need to set the attributes since TranscriptSegment is a dataclass
    for seg in mock_segments:
        seg.start_sec = 0.0
        seg.end_sec = 5.0
        seg.text = "hello world"

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = mock_segments

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 10]

    with _patch_ffmpeg(), \
         patch("av.pipeline.ingest.extract_audio", return_value=fake_audio), \
         patch("av.pipeline.ingest.OpenAITranscriber", return_value=mock_transcriber), \
         patch("av.pipeline.ingest.OpenAIEmbedder", return_value=mock_embedder):
        result = ingest_video(fake_video, repo, config, no_embed=False)

    assert result["status"] == "complete"
    assert result["artifacts_count"] == 1

    captured = capsys.readouterr()
    assert "Skipping transcription" not in captured.err
    assert "Skipping embeddings" not in captured.err


def test_dry_run_returns_early(fake_video: Path, repo: Repository) -> None:
    config = AVConfig(provider="openai", api_key="sk-test")

    with _patch_ffmpeg():
        result = ingest_video(fake_video, repo, config, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["filename"] == "test.mp4"
