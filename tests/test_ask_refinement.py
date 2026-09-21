"""Offline tests for Jev search refinement and bounded sampled-frame inspection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from av.core.config import AVConfig, get_config
from av.db.models import ArtifactRecord, VideoRecord
from av.db.repository import Repository
from av.providers.base import CompletionResult
from av.providers.usage import ProviderUsage
from av.search.inspection import (
    ExplicitVisionClient,
    InspectionWindow,
    VisionResponse,
    inspect_with_stronger_vision,
    sample_window_timestamps,
)
from av.search.rag import _citations, ask
from av.search.refine import (
    RefinementError,
    Scene,
    SystemOneClient,
    _hydrate_scene_text,
    _ordered_contiguous_events,
    judge_relevance,
    merge_overlapping_scenes,
    refine_search_results,
)
from av.search.semantic import search


def _video(video_id: str, path: Path, duration: float = 120.0) -> VideoRecord:
    return VideoRecord(
        id=video_id,
        file_path=str(path),
        file_hash=f"hash-{video_id}",
        file_size_bytes=path.stat().st_size if path.exists() else 0,
        filename=path.name,
        duration_sec=duration,
        status="complete",
    )


def _artifact(
    artifact_id: str,
    video_id: str,
    start: float,
    end: float,
    text: str,
    artifact_type: str = "caption",
) -> ArtifactRecord:
    return ArtifactRecord(
        id=artifact_id,
        video_id=video_id,
        type=artifact_type,
        start_sec=start,
        end_sec=end,
        text=text,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "av.db")


def _seed_video(repo: Repository, tmp_path: Path, video_id: str, prefix: str = "event") -> Path:
    path = tmp_path / f"{video_id}.mp4"
    path.write_bytes(b"fake-video")
    repo.insert_video(_video(video_id, path))
    repo.insert_artifacts_batch([
        _artifact(f"{video_id}-{i}", video_id, i * 10.0, (i + 1) * 10.0, f"{prefix} scene {i}")
        for i in range(10)
    ])
    return path


class FakeSystemOne:
    def __init__(
        self,
        relevance: list[float] | None = None,
        *,
        support: float = 0.9,
        edge_boundaries: bool = False,
        usage: dict | None = None,
    ) -> None:
        self.relevance = list(relevance or [])
        self.support = support
        self.edge_boundaries = edge_boundaries
        self.usage = usage or {}
        self.calls: list[tuple[dict, dict]] = []
        self.relevance_offset = 0
        self.boundary_calls = 0

    def ask(self, state, questions):
        self.calls.append((state, questions))
        if "is_supported" in questions:
            return {"is_supported": {"type": "noul", "noul": self.support}}, self.usage
        if "start" in questions or "end" in questions:
            self.boundary_calls += 1
            answers = {}
            for side in ("start", "end"):
                if side not in questions:
                    continue
                labels = list(questions[side]["criteria"])
                if self.edge_boundaries and self.boundary_calls == 1:
                    choice = labels[0] if side == "start" else labels[-1]
                elif self.edge_boundaries:
                    if side == "start":
                        choice = "e-5" if "e-5" in labels else labels[0]
                    else:
                        choice = "e4" if "e4" in labels else labels[-1]
                else:
                    choice = "e0"
                answers[side] = {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 0.8 if side == "start" else 0.7,
                }
            return answers, self.usage
        answers = {}
        for index, key in enumerate(questions):
            position = self.relevance_offset + index
            probability = self.relevance[position] if position < len(self.relevance) else 1.0
            answers[key] = {"type": "noul", "noul": probability}
        self.relevance_offset += len(questions)
        return answers, self.usage


class FakeLLM:
    def __init__(self, config: AVConfig) -> None:
        self.config = config

    def complete(self, prompt: str, context: str) -> str:
        return "legacy answer"

    def complete_with_usage(self, prompt: str, context: str) -> CompletionResult:
        return CompletionResult("refined answer", input_tokens=None, output_tokens=None)


def test_temp_sqlite_search_preserves_end_times_and_video_isolation(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="door")
    _seed_video(repo, tmp_path, "v2", prefix="door")
    config = AVConfig(embed_model="")
    result = search("door", repo, config, limit=4, video_id="v1")
    assert result["results"]
    assert {item["video_id"] for item in result["results"]} == {"v1"}
    assert all(item["end_sec"] == item["timestamp_sec"] + 10 for item in result["results"])


def test_relevance_is_batched_ten_and_rejects_bad_probabilities() -> None:
    results = [
        {"artifact_id": f"a{i}", "video_id": "v", "timestamp_sec": i, "text": "x", "source_type": "caption"}
        for i in range(11)
    ]
    fake = FakeSystemOne([0.5] * 11)
    probabilities, usage = judge_relevance(fake, "query", results)
    assert len(probabilities) == 11
    assert len(fake.calls) == 2
    assert usage["requests"] == 2

    for bad in (-0.1, 1.1, "0.9", None):
        with pytest.raises(RefinementError):
            judge_relevance(FakeSystemOne([bad]), "query", results[:1])


def test_boundary_window_is_single_pass_and_configurable(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1")
    raw = search("event", repo, AVConfig(embed_model=""), limit=1, video_id="v1")["results"]
    # Put the hit in the middle so ±6 has real room on both sides.
    raw[0] = repo.search_fts("scene 5", limit=1, video_id="v1")[0].model_dump()
    fake = FakeSystemOne([0.9], edge_boundaries=True)
    _refined, meta, _ = refine_search_results(
        "event",
        raw,
        repo,
        AVConfig(typesafe_api_key="test", refine_context_events=4),
        client=fake,
    )
    assert fake.boundary_calls == 1
    boundary_questions = next(questions for _, questions in fake.calls if "start" in questions)
    assert len(boundary_questions["start"]["criteria"]) <= 5
    assert len(boundary_questions["end"]["criteria"]) <= 5
    assert meta["scene_count"] == 1

def test_edge_hit_resolves_singleton_side_locally_without_asking(
    repo: Repository, tmp_path: Path
) -> None:
    """A hit at the first temporal event has only `e0` on its start side. A
    structured provider rejects one-option questions, so that side must be
    resolved locally and never sent — otherwise every edge hit would fall
    back with a validation 422."""
    _seed_video(repo, tmp_path, "v1")
    raw = [repo.search_fts("scene 0", limit=1, video_id="v1")[0].model_dump()]
    fake = FakeSystemOne([0.9])
    refined, meta, _ = refine_search_results(
        "event",
        raw,
        repo,
        AVConfig(typesafe_api_key="test", embed_model=""),
        client=fake,
    )
    boundary = next(questions for _, questions in fake.calls if "start" in questions or "end" in questions)
    assert set(boundary) == {"end"}
    assert fake.boundary_calls == 1
    assert meta["scene_count"] == 1
    assert refined[0]["timestamp_sec"] == 0.0
    assert refined[0]["end_sec"] == 10.0


def test_overlap_merge_is_same_video_only_and_ranks_probability_times_score() -> None:
    def scene(artifact_id: str, video: str, start: float, end: float, score: float, p: float) -> Scene:
        return Scene(
            artifact_id=artifact_id,
            video_id=video,
            filename=f"{video}.mp4",
            source_type="caption",
            start_sec=start,
            end_sec=end,
            text=artifact_id,
            retrieval_score=score,
            relevance_p=p,
            chunk_start_sec=start,
            chunk_end_sec=end,
            merged_artifact_ids=[artifact_id],
        )

    merged, count = merge_overlapping_scenes([
        scene("a", "v1", 0, 20, 100, 0.5),
        scene("b", "v1", 10, 30, 60, 1.0),
        scene("c", "v2", 10, 30, 1.0, 1.0),
    ])
    assert count == 1
    assert len(merged) == 2
    v1 = next(item for item in merged if item.video_id == "v1")
    assert v1.artifact_id == "a"
    assert v1.relevance_p == 1.0
    assert v1.rank_score == 100
    assert set(v1.merged_artifact_ids) == {"a", "b"}


def test_refinement_caps_to_top_eight_scenes(repo: Repository, tmp_path: Path) -> None:
    raw: list[dict] = []
    for index in range(9):
        video_id = f"v{index}"
        path = tmp_path / f"{video_id}.mp4"
        path.write_bytes(b"fake")
        repo.insert_video(_video(video_id, path))
        artifact = _artifact(f"a{index}", video_id, 0, 10, f"target {index}")
        repo.insert_artifact(artifact)
        raw.append({
            "rank": index + 1,
            "score": 0.1 + index / 10,
            "video_id": video_id,
            "filename": path.name,
            "timestamp_sec": 0.0,
            "end_sec": 10.0,
            "timestamp_formatted": "00:00:00",
            "source_type": "caption",
            "text": artifact.text,
            "artifact_id": artifact.id,
        })
    refined, meta, _ = refine_search_results(
        "target",
        raw,
        repo,
        AVConfig(typesafe_api_key="test", refine_max_scenes=8),
        client=FakeSystemOne([1.0] * 9),
    )
    assert len(refined) == 8
    assert meta["capped_count"] == 1
    assert refined[0]["video_id"] == "v8"


def test_whole_video_summary_stays_broad_and_does_not_merge_with_local_scene(
    repo: Repository, tmp_path: Path
) -> None:
    path = tmp_path / "v1.mp4"
    path.write_bytes(b"fake")
    repo.insert_video(_video("v1", path, duration=100.0))
    repo.insert_artifacts_batch([
        _artifact("summary", "v1", 0, 100, "Whole-video overview", "summary"),
        _artifact("local", "v1", 10, 20, "Person opens the door", "caption"),
    ])
    raw = [item.model_dump() for item in repo.search_fts("overview OR door", limit=10, video_id="v1")]
    refined, meta, _ = refine_search_results(
        "door",
        raw,
        repo,
        AVConfig(typesafe_api_key="test", refine_context_events=2),
        client=FakeSystemOne([0.9, 0.9]),
    )
    assert meta["merged_count"] == 0
    assert len(refined) == 2
    broad = next(item for item in refined if item["artifact_id"] == "summary")
    local = next(item for item in refined if item["artifact_id"] == "local")
    assert broad["evidence_scope"] == "broad"
    assert broad["text"] == "Whole-video overview"
    assert broad["scene_confidence"] is None
    assert local["evidence_scope"] == "scene"
    assert "Person opens the door" in local["text"]


def test_temporal_events_attach_interleaved_modalities_without_false_gap() -> None:
    artifacts = [
        _artifact("cap0", "v", 0, 10, "caption zero", "caption"),
        _artifact("t0", "v", 0, 1, "hello", "transcript"),
        _artifact("t8", "v", 8, 9, "there", "transcript"),
        _artifact("cap10", "v", 10, 20, "caption ten", "caption"),
    ]
    events, hit_index = _ordered_contiguous_events(
        {
            "artifact_id": "cap10",
            "timestamp_sec": 10,
            "end_sec": 20,
            "source_type": "caption",
            "text": "caption ten",
        },
        artifacts,
    )
    assert [(event["start"], event["end"]) for event in events] == [(0, 10), (10, 20)]
    assert hit_index == 1
    assert "hello" in events[0]["text"]
    assert "there" in events[0]["text"]


@pytest.mark.parametrize(
    (
        "caption_start",
        "caption_end",
        "transcript_start",
        "transcript_end",
        "expected_start",
        "expected_end",
    ),
    [
        (10.0, 20.0, 5.0, 25.0, 5.0, 25.0),
        (10.0, 10.0, 10.0, 15.0, 10.0, 15.0),
    ],
)
def test_scene_uses_containing_temporal_event_bounds_without_boundary_call(
    repo: Repository,
    tmp_path: Path,
    caption_start: float,
    caption_end: float,
    transcript_start: float,
    transcript_end: float,
    expected_start: float,
    expected_end: float,
) -> None:
    path = tmp_path / "v.mp4"
    path.write_bytes(b"fake")
    repo.insert_video(_video("v", path))
    repo.insert_artifacts_batch([
        _artifact("caption-hit", "v", caption_start, caption_end, "target caption", "caption"),
        _artifact(
            "transcript-overlap",
            "v",
            transcript_start,
            transcript_end,
            "overlapping speech",
            "transcript",
        ),
    ])
    raw = [repo.search_fts("target", limit=1, video_id="v")[0].model_dump()]
    client = FakeSystemOne([0.9])
    refined, _, _ = refine_search_results(
        "target",
        raw,
        repo,
        AVConfig(typesafe_api_key="test", refine_context_events=3),
        client=client,
    )
    assert client.boundary_calls == 0
    assert refined[0]["timestamp_sec"] == expected_start
    assert refined[0]["end_sec"] == expected_end


def test_strict_hydration_excludes_touching_chunks_and_preserves_hit_texts(
    repo: Repository, tmp_path: Path
) -> None:
    path = tmp_path / "v.mp4"
    path.write_bytes(b"fake")
    repo.insert_video(_video("v", path))
    repo.insert_artifacts_batch([
        _artifact("before", "v", 0, 10, "outside before"),
        _artifact("hit", "v", 10, 20, "primary hit"),
        _artifact("merged", "v", 12, 14, "merged hit", "transcript"),
        _artifact("inside", "v", 15, 16, "surrounding inside", "transcript"),
        _artifact("after", "v", 20, 30, "outside after"),
    ])
    scene = Scene(
        artifact_id="hit",
        video_id="v",
        filename="v.mp4",
        source_type="caption",
        start_sec=10,
        end_sec=20,
        text="primary hit",
        retrieval_score=1,
        relevance_p=1,
        chunk_start_sec=10,
        chunk_end_sec=20,
        merged_artifact_ids=["hit", "merged"],
        hit_texts=["primary hit", "merged hit"],
    )
    _hydrate_scene_text(scene, repo)
    assert scene.text.startswith("[retrieved hit] primary hit\n[retrieved hit] merged hit")
    assert "surrounding inside" in scene.text
    assert "outside before" not in scene.text
    assert "outside after" not in scene.text


def test_citations_preserve_refinement_provenance() -> None:
    citation = _citations([{
        "video_id": "v",
        "timestamp_sec": 10,
        "end_sec": 20,
        "source_type": "caption",
        "text": "hit",
        "score": 0.7,
        "artifact_id": "a",
        "chunk_start_sec": 12,
        "chunk_end_sec": 14,
        "scene_confidence": 0.8,
        "merged_artifact_ids": ["a", "b"],
        "relevance_p": 0.9,
        "evidence_scope": "scene",
    }])[0]
    assert citation["artifact_id"] == "a"
    assert citation["chunk_start_sec"] == 12
    assert citation["chunk_end_sec"] == 14
    assert citation["scene_confidence"] == 0.8
    assert citation["merged_artifact_ids"] == ["a", "b"]
    assert citation["relevance_p"] == 0.9


def test_valid_all_irrelevant_is_no_results_not_raw_fallback(repo: Repository, tmp_path: Path) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")
    fake = FakeSystemOne([0.1] * 30)
    config = AVConfig(typesafe_api_key="test", embed_model="", refine_relevance_min=0.5)
    with patch("av.search.rag.open_decision_client", return_value=fake), \
         patch("av.search.rag.OpenAILLM", side_effect=AssertionError("answer model must not run")):
        result = ask("cake", repo, config, video_id="v1")
    assert result["route"] == "refined_no_results"
    assert result["evidence_status"] == "all_sources_irrelevant"
    assert result["citations"] == []
    assert not result["warnings"]


def test_fast_path_support_skips_vision_and_usage_stays_unknown(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")
    fake = FakeSystemOne([0.9] * 30, support=0.88)
    config = AVConfig(typesafe_api_key="test", embed_model="")
    with patch("av.search.rag.open_decision_client", return_value=fake), \
         patch("av.search.rag.OpenAILLM", FakeLLM), \
         patch("av.search.rag.inspect_with_stronger_vision") as inspect:
        result = ask("cake", repo, config, video_id="v1")
    inspect.assert_not_called()
    assert result["route"] == "refined"
    assert result["confidence"] == pytest.approx(0.88)
    assert result["stage_usage"]["answer"]["input_tokens"] is None
    assert result["stage_usage"]["relevance"]["input_tokens"] is None


def test_relevant_sources_can_still_fail_answer_support_and_escalate(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")
    fake = FakeSystemOne([0.9] * 30, support=0.1)
    inspection = {
        "status": "insufficient",
        "answer": None,
        "citations": [],
        "windows": [{"video_id": "v1", "start_sec": 0, "end_sec": 10}],
        "usage": {"requests": 1, "input_tokens": None, "output_tokens": None},
        "warnings": [],
    }
    with patch("av.search.rag.open_decision_client", return_value=fake), \
         patch("av.search.rag.OpenAILLM", FakeLLM), \
         patch("av.search.rag.inspect_with_stronger_vision", return_value=inspection) as inspect:
        result = ask("cake", repo, AVConfig(typesafe_api_key="test", embed_model=""), video_id="v1")
    inspect.assert_called_once()
    assert result["route"] == "refined_uncertain"
    assert result["evidence_status"] == "unsupported"
    assert "did not support" in result["answer"]


def test_refinement_outage_falls_back_raw_without_secret_leak(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")
    config = AVConfig(typesafe_api_key="top-secret", embed_model="")
    with patch("av.search.rag.open_decision_client"), \
         patch("av.search.rag.refine_search_results", side_effect=RefinementError("top-secret private input")), \
         patch("av.search.rag.OpenAILLM", FakeLLM):
        result = ask("cake", repo, config, video_id="v1")
    encoded = json.dumps(result)
    assert result["route"] == "refinement_fallback"
    assert result["evidence_status"] == "raw_unjudged"
    assert "top-secret" not in encoded
    assert "private input" not in encoded


def test_system_one_http_error_is_sanitized() -> None:
    response = MagicMock(status_code=401, ok=False)
    response.text = "secret-key and raw private state"
    session = MagicMock()
    session.post.return_value = response
    client = SystemOneClient(
        AVConfig(typesafe_api_key="secret-key", typesafe_max_retries=0), session=session
    )
    with pytest.raises(RefinementError) as exc:
        client.ask({"private": "state"}, {"q": {"type": "noul", "instructions": "x"}})
    assert "401" in str(exc.value)
    assert "secret-key" not in str(exc.value)
    assert "private state" not in str(exc.value)


@pytest.mark.parametrize("payload", [[], None, "invalid-root"])
def test_system_one_non_object_json_is_sanitized(payload) -> None:
    response = MagicMock(status_code=200, ok=True)
    response.json.return_value = payload
    session = MagicMock()
    session.post.return_value = response
    client = SystemOneClient(
        AVConfig(typesafe_api_key="secret", typesafe_max_retries=0),
        session=session,
    )
    with pytest.raises(RefinementError, match="invalid JSON document") as exc:
        client.ask({"private": "state"}, {"q": {"type": "noul", "instructions": "x"}})
    assert "secret" not in str(exc.value)


def test_partial_relevance_usage_is_preserved_when_later_retry_fails() -> None:
    class PartialFailureClient:
        calls = 0

        def ask(self, state, questions):
            self.calls += 1
            if self.calls == 1:
                return (
                    {key: {"type": "noul", "noul": 0.9} for key in questions},
                    {"input_tokens": 10, "output_tokens": 2},
                )
            raise RefinementError("unavailable", attempts=2)

    results = [
        {"artifact_id": f"a{i}", "video_id": "v", "timestamp_sec": i, "text": "x", "source_type": "caption"}
        for i in range(11)
    ]
    with pytest.raises(RefinementError) as exc:
        judge_relevance(PartialFailureClient(), "query", results, batch_size=10)
    usage = exc.value.stage_usage["relevance"]
    assert usage["requests"] == 3
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["input_tokens_complete"] is False


def test_refinement_fallback_returns_partial_stage_usage(repo: Repository, tmp_path: Path) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")
    partial = {
        "requests": 2,
        "input_tokens": None,
        "output_tokens": None,
        "input_tokens_complete": False,
        "output_tokens_complete": False,
    }
    error = RefinementError("failed", stage_usage={"relevance": partial})
    with patch("av.search.rag.open_decision_client"), \
         patch("av.search.rag.refine_search_results", side_effect=error), \
         patch("av.search.rag.OpenAILLM", FakeLLM):
        result = ask(
            "cake",
            repo,
            AVConfig(typesafe_api_key="test", embed_model=""),
            video_id="v1",
        )
    assert result["stage_usage"]["relevance"]["requests"] == 2
    assert result["stage_usage"]["relevance"]["input_tokens_complete"] is False


def test_refined_answer_failure_preserves_receipts_and_is_sanitized(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")

    class FailingLLM:
        def __init__(self, config: AVConfig) -> None:
            self.usage = ProviderUsage()

        def complete_with_usage(self, prompt: str, context: str) -> CompletionResult:
            self.usage.record_failure()
            raise RuntimeError("https://private.example/v1 secret-token")

    fake = FakeSystemOne([0.9] * 30)
    with patch("av.search.rag.open_decision_client", return_value=fake), \
         patch("av.search.rag.OpenAILLM", FailingLLM), \
         patch("av.search.rag.judge_answer_support") as support, \
         patch("av.search.rag.inspect_with_stronger_vision") as inspect:
        result = ask(
            "cake",
            repo,
            AVConfig(typesafe_api_key="test", embed_model=""),
            video_id="v1",
        )
    support.assert_not_called()
    inspect.assert_not_called()
    assert result["route"] == "refined_answer_failed"
    assert result["evidence_status"] == "answer_unavailable"
    assert result["stage_usage"]["relevance"]["requests"] > 0
    assert result["stage_usage"]["answer"]["requests"] == 1
    assert result["stage_usage"]["answer"]["failed_requests"] == 1
    assert result["ask_settings"]["chat_max_output_tokens"] == 1024
    encoded = json.dumps(result)
    assert "private.example" not in encoded
    assert "secret-token" not in encoded


def test_legacy_answer_failure_preserves_attempted_usage_and_is_sanitized(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")

    class FailingLLM:
        def __init__(self, config: AVConfig) -> None:
            self.usage = ProviderUsage()

        def complete_with_usage(self, prompt: str, context: str) -> CompletionResult:
            self.usage.record_failure()
            raise RuntimeError("https://private.example/v1 secret-token")

    with patch("av.search.rag.OpenAILLM", FailingLLM):
        result = ask(
            "cake",
            repo,
            AVConfig(embed_model=""),
            video_id="v1",
            refine=False,
        )
    assert result["route"] == "legacy_answer_failed"
    assert result["evidence_status"] == "answer_unavailable"
    assert "embedding" in result["stage_usage"]
    assert result["stage_usage"]["answer"]["requests"] == 1
    assert result["stage_usage"]["answer"]["failed_requests"] == 1
    encoded = json.dumps(result)
    assert "private.example" not in encoded
    assert "secret-token" not in encoded


def test_full_window_timestamp_plan_covers_start_and_end() -> None:
    windows = [
        InspectionWindow("v1", "a.mp4", Path("a.mp4"), 10.0, 40.0, 10.0, 40.0),
        InspectionWindow("v2", "b.mp4", Path("b.mp4"), 100.0, 120.0, 100.0, 120.0),
    ]
    plan = sample_window_timestamps(windows, 8)
    first, second = plan.values()
    assert first[0] == 10.0
    assert first[-1] == pytest.approx(39.999)
    assert second[0] == 100.0
    assert second[-1] == pytest.approx(119.999)
    assert sum(len(values) for values in plan.values()) == 8


def test_inspection_handles_provider_media_frames_and_budget_failures(
    repo: Repository, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.mp4"
    repo.insert_video(_video("v1", missing))
    result = [{"video_id": "v1", "timestamp_sec": 0.0, "end_sec": 10.0}]

    not_configured = inspect_with_stronger_vision("q", "a", result, repo, AVConfig())
    assert not_configured["status"] == "not_configured"

    missing_media = inspect_with_stronger_vision(
        "q",
        "a",
        result,
        repo,
        AVConfig(strong_vision_api_base_url="http://example/v1", strong_vision_model="model"),
    )
    assert missing_media["status"] == "unavailable"

    path = tmp_path / "v2.mp4"
    path.write_bytes(b"fake")
    repo.insert_video(_video("v2", path))
    available = [{"video_id": "v2", "timestamp_sec": 0.0, "end_sec": 10.0}]
    budget = inspect_with_stronger_vision(
        "q",
        "a",
        available,
        repo,
        AVConfig(
            strong_vision_api_base_url="http://example/v1",
            strong_vision_model="model",
            inspection_max_frames=0,
        ),
    )
    assert budget["status"] == "budget_exhausted"

    empty_frames = MagicMock(paths=[], timestamps=[])
    with patch("av.search.inspection.sample_at", return_value=empty_frames):
        empty = inspect_with_stronger_vision(
            "q",
            "a",
            available,
            repo,
            AVConfig(
                strong_vision_api_base_url="http://example/v1",
                strong_vision_model="model",
                inspection_max_frames=4,
            ),
        )
    assert empty["status"] == "insufficient"
    assert any("no usable" in warning for warning in empty["warnings"])


def test_inspection_validates_absolute_timestamped_evidence(
    repo: Repository, tmp_path: Path
) -> None:
    path = _seed_video(repo, tmp_path, "v1")
    requested: list[float] = []

    def fake_sample(video_path, timestamps, **kwargs):
        requested.extend(timestamps)
        frames = []
        for index, timestamp in enumerate(timestamps):
            frame = Path(kwargs["out_dir"]) / f"{index}.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"jpg")
            frames.append(frame)
        return MagicMock(paths=frames, timestamps=timestamps)

    class FakeVLM:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self, images, prompt):
            assert "absolute_time" in prompt
            return MagicMock(
                ok=True,
                text=json.dumps({
                    "supported": True,
                    "answer": "A person enters.",
                    "evidence": [{"video_id": "v1", "timestamp_sec": requested[0], "description": "Person in doorway"}],
                }),
                input_tokens=None,
                output_tokens=None,
            )

    config = AVConfig(
        strong_vision_api_base_url="http://example/v1",
        strong_vision_model="strong-model",
        inspection_max_frames=6,
    )
    with patch("av.search.inspection.sample_at", side_effect=fake_sample):
        out = inspect_with_stronger_vision(
            "Who enters?",
            "Unknown",
            [{"video_id": "v1", "timestamp_sec": 10.0, "end_sec": 40.0}],
            repo,
            config,
            provider_factory=FakeVLM,
        )
    assert path.exists()
    assert out["status"] == "supported"
    assert out["citations"][0]["source_type"] == "sampled_frame_inspection"
    assert min(requested) == 10.0
    assert max(requested) == pytest.approx(39.999)


def test_explicit_vision_client_uses_only_supplied_credential_and_one_attempt(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    response = MagicMock(ok=False, status_code=503)
    session = MagicMock()
    session.post.return_value = response
    config = AVConfig(
        strong_vision_api_base_url="https://public.example/v1",
        strong_vision_api_key="",
        strong_vision_model="cheap-vlm",
    )
    client = ExplicitVisionClient(config, session=session)
    result = client.ask([frame], "inspect")
    assert result.ok is False
    session.post.assert_called_once()
    kwargs = session.post.call_args.kwargs
    assert kwargs["headers"] == {"Content-Type": "application/json"}
    assert session.post.call_args.args[0] == "https://public.example/v1/chat/completions"


def test_inspection_provider_initialization_error_is_sanitized(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1")

    class BrokenProvider:
        def __init__(self, config):
            raise RuntimeError("https://private.example secret-token")

    out = inspect_with_stronger_vision(
        "q",
        "a",
        [{"video_id": "v1", "timestamp_sec": 0.0, "end_sec": 10.0}],
        repo,
        AVConfig(
            strong_vision_api_base_url="https://private.example/v1",
            strong_vision_api_key="secret-token",
            strong_vision_model="model",
        ),
        provider_factory=BrokenProvider,
    )
    encoded = json.dumps(out)
    assert out["status"] == "unavailable"
    assert "private.example" not in encoded
    assert "secret-token" not in encoded
    assert out["usage"]["requests"] == 0


def test_inspection_rejects_unsampled_timestamp_and_reports_truncation(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1")

    def fake_sample(video_path, timestamps, **kwargs):
        frame = Path(kwargs["out_dir"]) / "one.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"jpg")
        return MagicMock(paths=[frame], timestamps=[timestamps[0]])

    class UnsampledProvider:
        def __init__(self, config):
            pass

        def ask(self, images, prompt):
            assert "do not establish what happened between" in prompt
            return VisionResponse(
                ok=True,
                text=json.dumps({
                    "supported": True,
                    "answer": "unsupported timestamp",
                    "evidence": [{
                        "video_id": "v1",
                        "timestamp_sec": 15.0,
                        "description": "not actually sampled",
                    }],
                }),
                input_tokens=5,
                output_tokens=2,
            )

    config = AVConfig(
        strong_vision_api_base_url="https://public.example/v1",
        strong_vision_model="cheap-vlm",
        inspection_max_seconds=5,
        inspection_max_frames=4,
    )
    with patch("av.search.inspection.sample_at", side_effect=fake_sample):
        out = inspect_with_stronger_vision(
            "q",
            "a",
            [{"video_id": "v1", "timestamp_sec": 10.0, "end_sec": 40.0}],
            repo,
            config,
            provider_factory=UnsampledProvider,
        )
    assert out["status"] == "insufficient"
    window = out["windows"][0]
    assert window["requested_start_sec"] == 10.0
    assert window["requested_end_sec"] == 40.0
    assert window["start_sec"] == 10.0
    assert window["end_sec"] == 15.0
    assert window["truncated"] is True
    assert window["all_requested_frames_extracted"] is False


def test_inspection_attempt_cap_and_partial_usage_are_truthful(
    repo: Repository, tmp_path: Path
) -> None:
    _seed_video(repo, tmp_path, "v1")
    calls = 0

    def fake_sample(video_path, timestamps, **kwargs):
        frames = []
        for index, _ in enumerate(timestamps):
            frame = Path(kwargs["out_dir"]) / f"{index}.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"jpg")
            frames.append(frame)
        return MagicMock(paths=frames, timestamps=timestamps)

    class TwoAttemptProvider:
        def __init__(self, config):
            pass

        def ask(self, images, prompt):
            nonlocal calls
            calls += 1
            return VisionResponse(
                ok=True,
                text='{"supported": false, "answer": "", "evidence": []}',
                input_tokens=10 if calls == 1 else None,
                output_tokens=1 if calls == 1 else None,
            )

    config = AVConfig(
        strong_vision_api_base_url="https://public.example/v1",
        strong_vision_model="cheap-vlm",
        inspection_max_frames=8,
        inspection_max_attempts=2,
        inspection_dense_pass=True,
    )
    with patch("av.search.inspection.sample_at", side_effect=fake_sample):
        out = inspect_with_stronger_vision(
            "q",
            "a",
            [{"video_id": "v1", "timestamp_sec": 10.0, "end_sec": 40.0}],
            repo,
            config,
            provider_factory=TwoAttemptProvider,
        )
    assert calls == 2
    assert out["usage"]["requests"] == 2
    assert out["usage"]["input_tokens"] is None
    assert out["usage"]["input_tokens_complete"] is False


def test_broad_evidence_does_not_drive_inspection(repo: Repository, tmp_path: Path) -> None:
    _seed_video(repo, tmp_path, "v1")
    config = AVConfig(
        strong_vision_api_base_url="https://public.example/v1",
        strong_vision_model="cheap-vlm",
        inspection_max_windows=1,
        inspection_max_frames=0,
    )
    out = inspect_with_stronger_vision(
        "q",
        "a",
        [
            {"video_id": "v1", "timestamp_sec": 0.0, "end_sec": 100.0, "evidence_scope": "broad"},
            {"video_id": "v1", "timestamp_sec": 20.0, "end_sec": 30.0, "evidence_scope": "scene"},
        ],
        repo,
        config,
    )
    assert out["status"] == "budget_exhausted"
    assert any("Broad summary" in warning for warning in out["warnings"])


def test_config_file_env_priority_and_secret_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "typesafe_api_key": "file-key",
        "typesafe_model": "jev-file",
        "refine_relevance_min": 0.7,
        "strong_vision_api_key": "vision-file-key",
    }))
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", config_file)
    monkeypatch.setenv("AV_TYPESAFE_MODEL", "jev-env")
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    config = get_config()
    assert config.typesafe_api_key == "env-key"
    assert config.typesafe_model == "jev-env"
    assert config.refine_relevance_min == pytest.approx(0.7)
    assert config.strong_vision_api_key == "vision-file-key"


def test_no_refine_preserves_legacy_contract(repo: Repository, tmp_path: Path) -> None:
    _seed_video(repo, tmp_path, "v1", prefix="cake")
    raw = [repo.search_fts("cake", limit=1, video_id="v1")[0].model_dump()]
    embedding_usage = {
        "requests": 1,
        "input_tokens": 4,
        "output_tokens": 0,
        "input_tokens_complete": True,
        "output_tokens_complete": True,
    }
    with patch(
        "av.search.rag.search",
        return_value={"results": raw, "embedding_usage": embedding_usage},
    ), patch("av.search.rag.OpenAILLM", FakeLLM):
        result = ask(
            "cake",
            repo,
            AVConfig(typesafe_api_key="configured", embed_model=""),
            video_id="v1",
            refine=False,
        )
    assert result["answer"] == "refined answer"
    assert result["route"] == "legacy"
    assert result["evidence_status"] == "raw_unjudged"
    assert result["confidence_basis"] == "retrieval_heuristic"
    assert result["warnings"] == []
    assert result["stage_usage"]["embedding"] == embedding_usage
    assert result["stage_usage"]["answer"]["requests"] == 1
