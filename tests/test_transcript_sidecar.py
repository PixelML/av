"""Offline transcript-sidecar validation and CLI wiring tests."""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

import av.cli.ingest as ingest_cli
from av.core.exceptions import IngestError
from av.pipeline.transcript_sidecar import load_transcript_sidecar
from av.pipeline.transcript_sidecar import TranscriptSidecarError


VALID_SEGMENT = {"start_sec": 0, "end_sec": 2.4, "text": "Hello"}


def write_sidecar(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_object_preserves_segments_and_metadata(tmp_path):
    segments = [
        {"start_sec": 5, "end_sec": 10, "text": "  Second passage.\n"},
        {"start_sec": 0.125, "end_sec": 2.75, "text": "First passage — café"},
    ]
    provenance = {"source": "public-example", "settings": {"language": "en"}, "tags": [1, True, None]}
    path = write_sidecar(tmp_path, {"segments": segments, "model": "external-asr", "provenance": provenance})

    result = load_transcript_sidecar(path, duration_sec=10)

    assert [(s.start_sec, s.end_sec, s.text) for s in result.segments] == [
        (s["start_sec"], s["end_sec"], s["text"]) for s in segments
    ]
    assert isinstance(result.segments, tuple)
    assert result.model == "external-asr"
    assert result.provenance == provenance
    with pytest.raises(FrozenInstanceError):
        result.segments[0].text = "changed"


def test_simple_segment_list(tmp_path):
    result = load_transcript_sidecar(write_sidecar(tmp_path, [VALID_SEGMENT]), duration_sec=2.4)
    assert result.segments[0].end_sec == 2.4
    assert result.model is None
    assert result.provenance is None


@pytest.mark.parametrize("root", [[], {"segments": []}])
def test_empty_segments_are_valid_for_silence(tmp_path, root):
    assert load_transcript_sidecar(write_sidecar(tmp_path, root), duration_sec=10).segments == ()


@pytest.mark.parametrize("duration", [0, -1, True, False, "10", None, float("nan"), float("inf"), 10 ** 500])
def test_invalid_actual_duration(tmp_path, duration):
    with pytest.raises(TranscriptSidecarError, match="duration"):
        load_transcript_sidecar(write_sidecar(tmp_path, [VALID_SEGMENT]), duration_sec=duration)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-0.1, 1), (1, 1), (2, 1), (0, 10.000001), (True, 1), (0, False),
     ("0", 1), (0, "1"), (None, 1), ([], 1), ({}, 1), (0, float("nan")),
     (float("inf"), 1), (0, float("-inf")), (0, 10 ** 500)],
)
def test_rejects_invalid_timestamp_bounds_and_types(tmp_path, start, end):
    path = write_sidecar(tmp_path, [{"start_sec": start, "end_sec": end, "text": "Hello"}])
    with pytest.raises(TranscriptSidecarError):
        load_transcript_sidecar(path, duration_sec=10)


@pytest.mark.parametrize("text", ["", " \n\t ", None, 1, True, [], {}])
def test_rejects_invalid_text_without_echoing_it(tmp_path, text):
    segment = {**VALID_SEGMENT, "text": text}
    with pytest.raises(TranscriptSidecarError, match="text must be a nonempty string"):
        load_transcript_sidecar(write_sidecar(tmp_path, [segment]), duration_sec=10)


@pytest.mark.parametrize("root", [None, True, 1, "text", {}, {"text": "Hello"}, {"segments": {}}, {"segments": None}])
def test_rejects_invalid_root_and_segments_container(tmp_path, root):
    with pytest.raises(TranscriptSidecarError):
        load_transcript_sidecar(write_sidecar(tmp_path, root), duration_sec=10)


@pytest.mark.parametrize("segment", [None, True, "Hello", [], {}, {"start_sec": 0, "end_sec": 1}, {**VALID_SEGMENT, "video_id": "override"}])
def test_rejects_invalid_or_extra_segment_fields(tmp_path, segment):
    with pytest.raises(TranscriptSidecarError, match="requires only"):
        load_transcript_sidecar(write_sidecar(tmp_path, [segment]), duration_sec=10)


def test_rejects_extra_root_fields(tmp_path):
    with pytest.raises(TranscriptSidecarError, match="permits only"):
        load_transcript_sidecar(write_sidecar(tmp_path, {"segments": [], "video_id": "override"}), duration_sec=10)


@pytest.mark.parametrize("model", [None, True, 12, [], {}, "", "  "])
def test_rejects_invalid_model(tmp_path, model):
    with pytest.raises(TranscriptSidecarError, match="model"):
        load_transcript_sidecar(write_sidecar(tmp_path, {"segments": [], "model": model}), duration_sec=10)


@pytest.mark.parametrize("provenance", [None, True, 1, "source", []])
def test_rejects_nonobject_provenance(tmp_path, provenance):
    with pytest.raises(TranscriptSidecarError, match="provenance"):
        load_transcript_sidecar(write_sidecar(tmp_path, {"segments": [], "provenance": provenance}), duration_sec=10)


@pytest.mark.parametrize("raw", [
    "{",
    '{"segments": [], "segments": []}',
    '[{"start_sec": 0, "end_sec": 1, "text": "Hello", "text": "Other"}]',
    '{"segments": [], "provenance": {"nested": [1e999]}}',
    '{"segments": [], "provenance": {"nested": [NaN]}}',
    '{"segments": [], "provenance": {"same": 1, "same": 2}}',
])
def test_rejects_malformed_or_ambiguous_json(tmp_path, raw):
    path = tmp_path / "transcript.json"
    path.write_text(raw)
    with pytest.raises(TranscriptSidecarError):
        load_transcript_sidecar(path, duration_sec=10)


def test_file_read_errors_are_ingest_errors(tmp_path):
    with pytest.raises(IngestError, match="Could not read"):
        load_transcript_sidecar(tmp_path / "missing.json", duration_sec=10)
    path = tmp_path / "invalid.json"
    path.write_bytes(b"\xff")
    with pytest.raises(IngestError, match="Could not read"):
        load_transcript_sidecar(path, duration_sec=10)


@pytest.fixture
def cli_setup(tmp_path, monkeypatch):
    # Use real isolated config and SQLite; only replace the pipeline boundary.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", tmp_path / "config.json")
    for key in list(os.environ):
        if key.startswith("AV_") or key in {"OPENAI_API_KEY", "TYPESAFE_API_KEY", "TYPESAFE_DEFAULT_MODEL"}:
            monkeypatch.delenv(key)
    app = typer.Typer()

    @app.callback()
    def callback():
        pass

    ingest_cli.register(app)
    pipeline = Mock(return_value={"status": "complete", "artifacts_count": 1})
    monkeypatch.setattr(ingest_cli, "ingest_video", pipeline)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"offline-test-video")
    sidecar = write_sidecar(tmp_path, [VALID_SEGMENT])
    return app, pipeline, video, sidecar, tmp_path / "av.db"


def test_cli_forwards_resolved_sidecar_path(cli_setup):
    app, pipeline, video, sidecar, db = cli_setup
    result = CliRunner().invoke(app, ["ingest", str(video), "--transcript-json", sidecar.name, "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "complete"
    assert pipeline.call_args.kwargs["transcript_json"] == sidecar.resolve()
    assert pipeline.call_args.args[0] == video


def test_cli_without_sidecar_preserves_default(cli_setup):
    app, pipeline, video, _, db = cli_setup
    result = CliRunner().invoke(app, ["ingest", str(video), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert pipeline.call_args.kwargs["transcript_json"] is None


def test_cli_rejects_directory_even_with_one_video(cli_setup):
    app, pipeline, video, sidecar, db = cli_setup
    result = CliRunner().invoke(app, ["ingest", str(video.parent), "--transcript-json", str(sidecar), "--db", str(db)])
    assert result.exit_code == 1
    assert "not a directory" in result.output
    pipeline.assert_not_called()
    assert not db.exists()


def test_cli_rejects_multiple_discovered_inputs(cli_setup, monkeypatch):
    app, pipeline, video, sidecar, db = cli_setup
    monkeypatch.setattr(ingest_cli, "discover_videos", lambda _: [video, video])
    result = CliRunner().invoke(app, ["ingest", str(video), "--transcript-json", str(sidecar), "--db", str(db)])
    assert result.exit_code == 1
    assert "exactly one video" in result.output
    pipeline.assert_not_called()
    assert not db.exists()


@pytest.mark.parametrize("use_directory", [False, True])
def test_cli_rejects_invalid_sidecar_path_before_ingest(cli_setup, use_directory):
    app, pipeline, video, _, db = cli_setup
    path = video.parent if use_directory else video.parent / "missing.json"
    result = CliRunner().invoke(app, ["ingest", str(video), "--transcript-json", str(path), "--db", str(db)])
    assert result.exit_code == 2
    pipeline.assert_not_called()
    assert not db.exists()


def test_cli_accepts_one_downloaded_url(cli_setup, monkeypatch):
    app, pipeline, video, sidecar, db = cli_setup
    download = Mock(return_value=video)
    monkeypatch.setattr(ingest_cli, "download_video", download)
    url = "https://example.com/video.mp4"
    result = CliRunner().invoke(app, ["ingest", url, "--transcript-json", str(sidecar), "--db", str(db)])
    assert result.exit_code == 0, result.output
    download.assert_called_once_with(url)
    assert pipeline.call_args.kwargs["transcript_json"] == sidecar


def test_cli_reports_explicit_sidecar_validation_failure(cli_setup):
    app, pipeline, video, sidecar, db = cli_setup
    pipeline.side_effect = TranscriptSidecarError("Transcript segment 1 is invalid.")
    result = CliRunner().invoke(app, ["ingest", str(video), "--transcript-json", str(sidecar), "--db", str(db)])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "error"
    assert "Transcript segment 1 is invalid." in result.output
