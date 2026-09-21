"""Offline tests for clip candidate construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from av.db.models import ArtifactRecord, VideoRecord
from av.db.repository import Repository
from av.search.clip import build_candidates


def _video(video_id: str, path, duration: float = 120.0) -> VideoRecord:
    return VideoRecord(
        id=video_id,
        file_path=str(path),
        file_hash=f"hash-{video_id}",
        file_size_bytes=1024,
        filename=f"{video_id}.mp4",
        duration_sec=duration,
        status="complete",
    )


def _artifact(video_id: str, kind: str, n: int, start: float, end: float, text: str) -> ArtifactRecord:
    return ArtifactRecord(
        id=f"{video_id}-{kind}{n:02d}",
        video_id=video_id,
        type=kind,
        start_sec=start,
        end_sec=end,
        text=text,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "av.db")


def _seed_talk(repo: Repository, tmp_path, video_id: str = "v1") -> None:
    repo.insert_video(_video(video_id, tmp_path / f"{video_id}.mp4", duration=140.0))
    segments = [
        (0, 8, "Welcome everyone to the systems track."),
        (8, 16, "Today we discuss reliable large deployments."),
        (16, 24, "A quick story from our infrastructure."),
        (24, 34, "Quantum error correction is like backup batteries for qubits."),
        (34, 44, "When a qubit flips, the correction layer rewrites the state."),
        (44, 58, "On a live cluster the error rate collapsed by orders of magnitude."),
        (58, 66, "A short word from our sponsor about cloud credits."),
        (66, 78, "The grant committee wants monthly reports."),
        (78, 88, "Back to the technical track: redundancy is not resilience."),
        (88, 100, "Quantum error correction works when code distance grows."),
        (100, 112, "Three faulty gates in a row were recovered cleanly."),
        (112, 140, "Thanks for coming, questions at the booth."),
    ]
    repo.insert_artifacts_batch(
        [_artifact(video_id, "transcript", i + 1, s, e, t) for i, (s, e, t) in enumerate(segments)]
    )
    # A wide dense caption spanning half the video must never drive bounds.
    repo.insert_artifact(_artifact(video_id, "dense_caption", 1, 0, 80, "Speaker on stage; slides show lattice diagrams"))
    repo.insert_artifact(_artifact(video_id, "dense_caption", 2, 80, 140, "Slide shows code distance chart"))


def test_absent_topic_returns_no_candidates_without_provider(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "av.db")
    _seed_talk(repo, tmp_path)
    video = repo.get_video("v1")
    candidates, meta = build_candidates("stock market predictions", repo, video)
    assert candidates == []
    assert meta["retrieval_hits"] == 0


def test_candidates_are_bounded_and_keep_hit_inside(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "av.db")
    _seed_talk(repo, tmp_path)
    video = repo.get_video("v1")
    candidates, meta = build_candidates(
        "quantum error correction", repo, video, min_seconds=10.0, max_seconds=30.0
    )
    assert meta["retrieval_hits"] >= 2
    assert 0 < len(candidates) <= 24
    for candidate in candidates:
        span = candidate.end_sec - candidate.start_sec
        assert span <= 30.0 * 1.5  # hard cap plus documented judge slack
        assert any(
            artifact_id in candidate.hit_artifact_ids for artifact_id in candidate.artifact_ids
        )
        # Transcript boundaries only: the 0-80s caption never defines a bound.
        assert candidate.start_sec != 0.0 or candidate.end_sec != 80.0


def test_wide_caption_does_not_drive_timing_but_is_attached(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "av.db")
    _seed_talk(repo, tmp_path)
    video = repo.get_video("v1")
    candidates, _ = build_candidates(
        "quantum error correction", repo, video, min_seconds=10.0, max_seconds=30.0
    )
    assert candidates
    for candidate in candidates:
        for event in candidate.events:
            for row in event.get("rows", ()):
                assert row["source_type"] == "transcript"
        assert candidate.end_sec - candidate.start_sec <= 45.0


def test_video_isolation_and_deterministic_ids(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "av.db")
    _seed_talk(repo, tmp_path, "v1")
    repo.insert_video(_video("v2", tmp_path / "v2.mp4", duration=140.0))
    repo.insert_artifacts_batch(
        [_artifact("v2", "transcript", 1, 0, 8, "Cooking show intro about bread.")]
    )
    video = repo.get_video("v1")
    first, _ = build_candidates("quantum error correction", repo, video, max_seconds=30.0)
    second, _ = build_candidates("quantum error correction", repo, video, max_seconds=30.0)
    assert [c.candidate_id for c in first] == [c.candidate_id for c in second]
    assert all(c.video_id == "v1" for c in first)


def test_too_short_groups_are_rejected(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "av.db")
    repo.insert_video(_video("v1", tmp_path / "v1.mp4", duration=60.0))
    repo.insert_artifacts_batch(
        [
            _artifact("v1", "transcript", 1, 20, 22, "Cloud credits keep the lights on."),
            _artifact("v1", "transcript", 2, 30, 60, "Unrelated long closing remarks follow here."),
        ]
    )
    video = repo.get_video("v1")
    candidates, meta = build_candidates(
        "cloud credits", repo, video, min_seconds=10.0, max_seconds=30.0
    )
    assert candidates == []
    assert meta["too_short_groups"] == 1
