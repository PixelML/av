"""Offline receipts for API-only ingestion stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from av.core.config import AVConfig
from av.db.repository import Repository
from av.pipeline.ingest import ingest_video
from av.pipeline.transcript_sidecar import TranscriptSidecarError
from av.db.models import VideoRecord
from av.providers.base import Caption
from av.providers.usage import ProviderUsage


@dataclass
class _Meta:
    duration_sec: float = 20.0
    width: int = 640
    height: int = 360
    fps: float = 24.0
    codec: str = "h264"
    bitrate: int = 1_000_000
    file_size_bytes: int = 4


class _Captioner:
    def __init__(self, config: AVConfig) -> None:
        self.usage = ProviderUsage()

    def caption_frames(self, frame_paths, timestamps, prompt=None):
        for _ in frame_paths:
            usage = SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            )
            self.usage.record_success(usage)
        return [
            Caption(timestamp_sec=timestamp, text=f"frame at {timestamp}", frame_path=str(path))
            for path, timestamp in zip(frame_paths, timestamps)
        ]


def test_dense_ingest_reports_actual_usage_and_effective_budgets(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_a = frames_dir / "frame_000001.jpg"
    frame_b = frames_dir / "frame_000002.jpg"
    frame_a.write_bytes(b"jpg")
    frame_b.write_bytes(b"jpg")
    repo = Repository(tmp_path / "av.db")
    config = AVConfig(
        provider="openai",
        api_key="explicit",
        transcribe_model="",
        embed_model="",
        api_timeout_sec=19,
        api_max_retries=1,
        allow_oauth_fallback=False,
        allow_codex_fallback=False,
    )
    with patch("av.pipeline.ingest.get_video_info", return_value=_Meta()), \
         patch("av.pipeline.ingest.extract_frames", return_value=[(frame_a, 0.0), (frame_b, 10.0)]), \
         patch("av.pipeline.ingest.OpenAICaptioner", _Captioner), \
         patch("av.pipeline.ingest.export_dense_outputs"):
        result = ingest_video(
            video,
            repo,
            config,
            dense_vision=True,
            no_embed=True,
            max_frames=2,
            dense_output_dir=tmp_path / "dense",
        )
    usage = result["stage_usage"]["caption"]
    assert usage["requests"] == 2
    assert usage["successful_requests"] == 2
    assert usage["failed_requests"] == 0
    assert usage["input_tokens"] == 22
    assert usage["output_tokens"] == 6
    assert usage["cached_input_tokens"] == 4
    assert usage["input_tokens_complete"] is True
    settings = result["ingest_settings"]
    assert settings["api_timeout_sec"] == 19
    assert settings["api_max_retries"] == 1
    assert settings["allow_oauth_fallback"] is False
    assert settings["allow_codex_fallback"] is False
    assert settings["caption_concurrency"] == 1
    assert settings["max_frames"] == 2
    assert settings["dense_caption_frames"] == 2


def test_provider_usage_keeps_unknown_dimensions_null_after_missing_usage() -> None:
    usage = ProviderUsage()
    usage.record_success(SimpleNamespace(prompt_tokens=5, completion_tokens=2))
    usage.record_success(None)
    receipt = usage.snapshot()
    assert receipt["requests"] == 2
    assert receipt["input_tokens"] is None
    assert receipt["output_tokens"] is None
    assert receipt["cached_input_tokens"] is None
    assert receipt["input_tokens_complete"] is False
    assert receipt["cached_input_tokens_complete"] is False


def test_transcript_sidecar_bypasses_builtin_asr_and_uses_normal_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    sidecar = tmp_path / "transcript.json"
    sidecar.write_text(json.dumps({
        "model": "external-cheap-asr",
        "provenance": {"method": "public-script"},
        "segments": [
            {"start_sec": 1.0, "end_sec": 2.5, "text": " Exact accepted text "},
        ],
    }))
    repo = Repository(tmp_path / "av.db")
    config = AVConfig(
        api_key="explicit",
        transcribe_model="whisper-1",
        embed_model="",
    )
    with patch("av.pipeline.ingest.get_video_info", return_value=_Meta()), \
         patch("av.pipeline.ingest.extract_audio", side_effect=AssertionError("ASR audio must not run")), \
         patch("av.pipeline.ingest.OpenAITranscriber", side_effect=AssertionError("ASR provider must not run")):
        result = ingest_video(
            video,
            repo,
            config,
            transcript_json=sidecar,
            no_embed=True,
        )
    artifacts = repo.get_artifacts(result["video_id"], "transcript")
    assert len(artifacts) == 1
    assert artifacts[0].start_sec == 1.0
    assert artifacts[0].end_sec == 2.5
    assert artifacts[0].text == " Exact accepted text "
    assert json.loads(artifacts[0].meta_json) == {
        "model": "external-cheap-asr",
        "provenance": {"method": "public-script"},
    }
    assert result["stage_usage"]["transcription"]["requests"] == 0
    assert result["ingest_settings"]["transcription_source"] == "sidecar"


def test_invalid_sidecar_is_validated_before_force_deletes_existing_video(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    sidecar = tmp_path / "bad.json"
    sidecar.write_text('{"segments":[{"start_sec":0,"end_sec":99,"text":"too long"}]}')
    repo = Repository(tmp_path / "av.db")
    repo.insert_video(VideoRecord(
        id="existing",
        file_path=str(video),
        file_hash="same-hash",
        file_size_bytes=4,
        filename=video.name,
        duration_sec=20,
        status="complete",
    ))
    with patch("av.pipeline.ingest.file_hash", return_value="same-hash"), \
         patch("av.pipeline.ingest.get_video_info", return_value=_Meta()):
        with pytest.raises(TranscriptSidecarError):
            ingest_video(
                video,
                repo,
                AVConfig(transcribe_model="", embed_model=""),
                transcript_json=sidecar,
                force=True,
                no_embed=True,
            )
    assert repo.get_video("existing").id == "existing"
