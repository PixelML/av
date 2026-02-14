"""Tests for the three-layer captioning cascade."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from av.core.config import AVConfig
from av.core.constants import TOPIC_PRESETS, _GENERAL_TEMPLATE
from av.db.repository import Repository
from av.pipeline.cascade import (
    LAYER1_SYSTEM_PROMPT,
    LAYER2_SYSTEM_PROMPT,
    _build_chunk_prompt,
    run_cascade,
)
from av.pipeline.ingest import ingest_video


@dataclass
class _FakeVideoMeta:
    duration_sec: float = 30.0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    codec: str = "h264"
    bitrate: int = 5000000
    file_size_bytes: int = 1024


@pytest.fixture()
def fake_video(tmp_path: Path) -> Path:
    f = tmp_path / "test.mp4"
    f.write_bytes(b"\x00" * 256)
    return f


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    db_path = tmp_path / "test.db"
    return Repository(db_path)


@pytest.fixture()
def config() -> AVConfig:
    return AVConfig(
        provider="openai",
        api_key="sk-test",
        vision_model="gpt-4-1",
        chat_model="gpt-4-1",
        transcribe_model="",
        embed_model="",
    )


def _make_fake_frames(tmp_path: Path, n: int = 3) -> list[tuple[Path, float]]:
    """Create fake frame files for testing."""
    frames = []
    for i in range(n):
        fp = tmp_path / f"chunk_{i:03d}.jpg"
        # Write minimal JPEG header bytes
        fp.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        frames.append((fp, float(i * 10)))
    return frames


# --- Topic prompt formatting tests ---


def test_topic_preset_general():
    prompt = _build_chunk_prompt("general", 0.0, 30.0, 30, 3)
    assert "0.0s" in prompt
    assert "30.0s" in prompt
    assert "3 frames" in prompt
    assert "30-second" in prompt
    assert "STATIC" in prompt


def test_topic_preset_security():
    prompt = _build_chunk_prompt("security", 10.0, 40.0, 30, 3)
    assert "suspicious behavior" in prompt
    assert "10.0s" in prompt


def test_topic_preset_traffic():
    prompt = _build_chunk_prompt("traffic", 0.0, 30.0, 30, 3)
    assert "vehicle" in prompt.lower()
    assert "violation" in prompt.lower()


def test_custom_topic_string():
    prompt = _build_chunk_prompt("birds nesting on rooftop", 0.0, 30.0, 30, 3)
    assert "birds nesting on rooftop" in prompt
    assert "STATIC" in prompt  # Still has the general template structure


def test_all_presets_format_without_error():
    for name in TOPIC_PRESETS:
        prompt = _build_chunk_prompt(name, 0.0, 60.0, 60, 5)
        assert len(prompt) > 50
        assert "{" not in prompt  # No unresolved placeholders


# --- Cascade integration tests ---


def test_cascade_produces_three_layers(fake_video: Path, config: AVConfig, tmp_path: Path):
    """Patch VLM + LLM, assert all 3 layers are produced."""
    mock_captioner = MagicMock()
    mock_captioner.caption_chunk.return_value = "A person walks through a door at the far end of the hallway."

    mock_llm = MagicMock()
    mock_llm.summarize.side_effect = [
        "0:30:Person enters through hallway door",  # Layer 1
        "## Events\n- 00:00 Person enters\n\n## Summary\nBrief activity.\n\n## Categories\nperson_entry",  # Layer 2
    ]

    fake_frames = _make_fake_frames(tmp_path)

    with patch("av.pipeline.cascade.OpenAICaptioner", return_value=mock_captioner), \
         patch("av.pipeline.cascade.OpenAILLM", return_value=mock_llm), \
         patch("av.pipeline.cascade._extract_chunk_frames", return_value=fake_frames):
        l0, l1, l2 = run_cascade(
            fake_video, "test-video-id", config, duration_sec=30.0, topic="general",
        )

    assert len(l0) == 1
    assert l0[0].type == "caption"
    assert l0[0].start_sec == 0.0
    assert l0[0].end_sec == 30.0

    assert len(l1) == 1
    assert l1[0].type == "summary"

    assert len(l2) == 1
    assert l2[0].type == "report"


def test_cascade_multiple_chunks(fake_video: Path, config: AVConfig, tmp_path: Path):
    """A 90-second video should produce 3 chunks at 30s each."""
    mock_captioner = MagicMock()
    mock_captioner.caption_chunk.return_value = "Activity detected."

    mock_llm = MagicMock()
    mock_llm.summarize.side_effect = [
        "0:30:Event A\n30:60:Event B\n60:90:Event C",
        "## Events\n- Events\n\n## Summary\nThings happened.\n\n## Categories\nactivity",
    ]

    fake_frames = _make_fake_frames(tmp_path)

    with patch("av.pipeline.cascade.OpenAICaptioner", return_value=mock_captioner), \
         patch("av.pipeline.cascade.OpenAILLM", return_value=mock_llm), \
         patch("av.pipeline.cascade._extract_chunk_frames", return_value=fake_frames):
        l0, l1, l2 = run_cascade(
            fake_video, "test-video-id", config, duration_sec=90.0, topic="security",
        )

    assert len(l0) == 3
    assert l0[0].start_sec == 0.0
    assert l0[1].start_sec == 30.0
    assert l0[2].start_sec == 60.0
    assert len(l1) == 1
    assert len(l2) == 1


def test_cascade_graceful_degradation_vlm_failure(fake_video: Path, config: AVConfig, tmp_path: Path):
    """If VLM fails on every chunk, cascade returns empty layers without crashing."""
    mock_captioner = MagicMock()
    mock_captioner.caption_chunk.side_effect = RuntimeError("VLM is down")

    fake_frames = _make_fake_frames(tmp_path)

    with patch("av.pipeline.cascade.OpenAICaptioner", return_value=mock_captioner), \
         patch("av.pipeline.cascade._extract_chunk_frames", return_value=fake_frames):
        l0, l1, l2 = run_cascade(
            fake_video, "test-video-id", config, duration_sec=30.0,
        )

    assert l0 == []
    assert l1 == []
    assert l2 == []


def test_layer1_failure_preserves_layer0(fake_video: Path, config: AVConfig, tmp_path: Path):
    """If LLM fails in Layer 1, Layer 0 artifacts are still returned."""
    mock_captioner = MagicMock()
    mock_captioner.caption_chunk.return_value = "Person walks across the room."

    mock_llm = MagicMock()
    mock_llm.summarize.side_effect = RuntimeError("LLM is down")

    fake_frames = _make_fake_frames(tmp_path)

    with patch("av.pipeline.cascade.OpenAICaptioner", return_value=mock_captioner), \
         patch("av.pipeline.cascade.OpenAILLM", return_value=mock_llm), \
         patch("av.pipeline.cascade._extract_chunk_frames", return_value=fake_frames):
        l0, l1, l2 = run_cascade(
            fake_video, "test-video-id", config, duration_sec=30.0,
        )

    assert len(l0) == 1
    assert l0[0].text == "Person walks across the room."
    assert l1 == []
    assert l2 == []


def test_static_chunks_filtered(fake_video: Path, config: AVConfig, tmp_path: Path):
    """Chunks that return STATIC should be filtered out."""
    mock_captioner = MagicMock()
    mock_captioner.caption_chunk.return_value = "STATIC"

    fake_frames = _make_fake_frames(tmp_path)

    with patch("av.pipeline.cascade.OpenAICaptioner", return_value=mock_captioner), \
         patch("av.pipeline.cascade._extract_chunk_frames", return_value=fake_frames):
        l0, l1, l2 = run_cascade(
            fake_video, "test-video-id", config, duration_sec=30.0,
        )

    assert l0 == []


def test_cascade_artifacts_have_end_sec(fake_video: Path, config: AVConfig, tmp_path: Path):
    """Cascade caption artifacts should have end_sec populated (not None)."""
    mock_captioner = MagicMock()
    mock_captioner.caption_chunk.return_value = "Something happens."

    mock_llm = MagicMock()
    mock_llm.summarize.side_effect = ["Event log", "Report"]

    fake_frames = _make_fake_frames(tmp_path)

    with patch("av.pipeline.cascade.OpenAICaptioner", return_value=mock_captioner), \
         patch("av.pipeline.cascade.OpenAILLM", return_value=mock_llm), \
         patch("av.pipeline.cascade._extract_chunk_frames", return_value=fake_frames):
        l0, l1, l2 = run_cascade(
            fake_video, "test-video-id", config, duration_sec=60.0,
        )

    for art in l0:
        assert art.end_sec is not None
        assert art.end_sec > art.start_sec


# --- Frame captions (legacy) integration test ---


def _patch_ffmpeg():
    return patch("av.pipeline.ingest.get_video_info", return_value=_FakeVideoMeta())


def test_frame_captions_flag(fake_video: Path, repo: Repository):
    """--frame-captions triggers old per-frame behavior."""
    config = AVConfig(
        provider="openai",
        api_key="sk-test",
        vision_model="gpt-4-1",
        transcribe_model="",
        embed_model="",
    )

    mock_captioner = MagicMock()
    mock_captioner.caption_frames.return_value = [
        MagicMock(timestamp_sec=0.0, text="A frame caption", frame_path="/tmp/f.jpg"),
    ]

    fake_frames = [(fake_video.parent / "frame_000001.jpg", 0.0)]
    fake_frames[0][0].write_bytes(b"\xff\xd8" + b"\x00" * 50)

    with _patch_ffmpeg(), \
         patch("av.pipeline.ingest.extract_frames", return_value=fake_frames), \
         patch("av.pipeline.ingest.OpenAICaptioner", return_value=mock_captioner):
        result = ingest_video(
            fake_video, repo, config, frame_captions=True,
        )

    assert result["artifacts_count"] == 1
    mock_captioner.caption_frames.assert_called_once()


def test_captions_flag_triggers_cascade(fake_video: Path, repo: Repository):
    """--captions should trigger the cascade, not per-frame captioning."""
    config = AVConfig(
        provider="openai",
        api_key="sk-test",
        vision_model="gpt-4-1",
        chat_model="gpt-4-1",
        transcribe_model="",
        embed_model="",
    )

    with _patch_ffmpeg(), \
         patch("av.pipeline.ingest.run_cascade", return_value=([], [], [])) as mock_cascade:
        result = ingest_video(
            fake_video, repo, config, captions=True, topic="security",
        )

    mock_cascade.assert_called_once()
    call_kwargs = mock_cascade.call_args
    assert call_kwargs.kwargs["topic"] == "security"
