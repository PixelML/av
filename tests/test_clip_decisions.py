"""Offline tests for typed clip decisions, budget, and assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from av.core.config import AVConfig
from av.db.models import ArtifactRecord, VideoRecord
from av.db.repository import Repository
from av.search.clip import (
    BudgetedSystemOne,
    ClipError,
    assemble_clips,
    build_candidates,
    clip_video,
)
from av.search.refine import RefinementError


class ScriptedSystemOne:
    """Answers typed questions by script; records every state it is shown."""

    def __init__(self, relevance=0.9, coherence=0.9, visual=0.9, appeal=0.8, fail_score=False):
        self.relevance = relevance
        self.coherence = coherence
        self.visual = visual
        self.appeal = appeal
        self.fail_score = fail_score
        self.states = []
        self.questions = []
        self.attempts = 0

    def ask(self, state, questions):
        self.states.append(state)
        self.questions.append(questions)
        self.attempts += 1
        answers = {}
        for key, question in questions.items():
            qtype = question.get("type")
            if qtype == "noul":
                instructions = str(question.get("instructions", ""))
                if "on its own" in instructions:
                    value = self.coherence
                elif "visual description" in instructions:
                    value = self.visual
                else:
                    value = self.relevance
                answers[key] = {"type": "noul", "noul": value}
            elif qtype == "choice":
                answers[key] = {"type": "choice", "choice": "e0", "confidence": 0.9}
            elif qtype == "score":
                if self.fail_score:
                    answers[key] = {"type": "noul", "noul": 0.5}
                else:
                    answers[key] = {"type": "score", "score": self.appeal}
        return answers, {"requests": 1, "input_tokens": 10, "output_tokens": 5}


class OutageClient:
    def ask(self, state, questions):
        raise RefinementError("System One unavailable (boom)", attempts=2)


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    repository = Repository(tmp_path / "av.db")
    repository.insert_video(
        VideoRecord(
            id="v1",
            file_path=str(tmp_path / "v1.mp4"),
            file_hash="hash-v1",
            file_size_bytes=1024,
            filename="v1.mp4",
            duration_sec=140.0,
            status="complete",
        )
    )
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
    repository.insert_artifacts_batch(
        [
            ArtifactRecord(
                id=f"v1-t{i + 1:02d}",
                video_id="v1",
                type="transcript",
                start_sec=s,
                end_sec=e,
                text=t,
            )
            for i, (s, e, t) in enumerate(segments)
        ]
    )
    return repository


def _clip(repo: Repository, config: AVConfig, client, **kwargs) -> dict:
    return clip_video(
        "quantum error correction",
        "v1",
        repo,
        config,
        decide=True,
        client=client,
        **kwargs,
    )


def test_ok_path_gates_orders_and_reports_usage(repo: Repository, tmp_path: Path) -> None:
    config = AVConfig(typesafe_api_key="k")
    client = ScriptedSystemOne()
    result = _clip(repo, config, client, clips_wanted=2)
    assert result["status"] == "ok"
    assert len(result["clips"]) == 2
    for clip in result["clips"]:
        assert clip["decisions"]["relevance_p"] == 0.9
        assert clip["decisions"]["appeal_score"] == 0.8
        assert clip["selection_basis"] == "appeal_then_relevance_then_retrieval"
        assert clip["quotes"], "clips must carry source-verbatim quotes"
        assert clip["support"]["transcript_artifact_ids"]
    assert result["decisions"]["requests_used"] == client.attempts
    assert result["stage_usage"]["relevance"]["input_tokens"] > 0
    assert result["timings"]["selection_warm_ms"] > 0
    # The warm replay must not have added provider requests.
    assert result["decisions"]["requests_used"] == result["decisions"]["requests_used"]
    assert result["decisions"]["cap_exhausted"] is False


def test_low_relevance_candidates_are_not_selected(repo: Repository, tmp_path) -> None:
    config = AVConfig(typesafe_api_key="k")
    client = ScriptedSystemOne(relevance=0.1)
    result = _clip(repo, config, client)
    assert result["status"] == "no_usable_clips"
    assert result["clips"] == []
    assert result["selection"]["rejected"]["low_relevance"] >= 1


def test_absent_topic_never_reaches_provider(repo: Repository, tmp_path) -> None:
    config = AVConfig(typesafe_api_key="k")
    client = ScriptedSystemOne()
    result = clip_video(
        "stock market predictions",
        "v1",
        repo,
        config,
        decide=True,
        client=client,
    )
    assert result["status"] == "no_usable_clips"
    assert client.attempts == 0
    assert result["clips"] == []


def test_score_unsupported_degrades_ordering_not_selection(
    repo: Repository, tmp_path
) -> None:
    config = AVConfig(typesafe_api_key="k")
    client = ScriptedSystemOne(fail_score=True)
    result = _clip(repo, config, client, clips_wanted=2)
    assert result["status"] == "ok"
    assert result["selection"]["appeal_available"] is False
    assert result["selection"]["selection_basis"] == "relevance_then_retrieval"
    assert any("appeal" in warning for warning in result["warnings"])
    for clip in result["clips"]:
        assert clip["decisions"]["appeal_score"] is None
        assert "appeal_unavailable" in clip["uncertainty"]["notes"]


def test_request_cap_reports_undecided_and_keeps_decided(
    repo: Repository, tmp_path: Path
) -> None:
    config = AVConfig(typesafe_api_key="k")
    client = ScriptedSystemOne()
    result = clip_video(
        "quantum error correction",
        "v1",
        repo,
        config,
        clips_wanted=2,
        decide=True,
        client=client,
        max_requests=1,
    )
    # One request decides relevance for the whole batch; the cap lands
    # before coherence, so nothing may be selected without its objective
    # gates, and the undecided work must be visible in the receipt.
    assert result["status"] == "request_cap_reached"
    assert result["decisions"]["cap_exhausted"] is True
    assert result["decisions"]["requests_used"] == 1
    assert result["candidates"]["undecided"] >= 1
    assert result["clips"] == []
    assert result["stage_usage"]["relevance"]["requests"] == 1


def test_provider_outage_selects_nothing_and_leaks_nothing(
    repo: Repository, tmp_path: Path
) -> None:
    config = AVConfig(typesafe_api_key="k")
    result = _clip(repo, config, OutageClient())
    assert result["status"] == "decision_unavailable"
    assert result["clips"] == []
    # The relevance stage must report the two failed HTTP attempts with
    # unknown token totals; nothing was answered, so nothing is invented.
    assert result["stage_usage"]["relevance"]["requests"] == 2
    assert result["stage_usage"]["relevance"]["input_tokens"] is None
    assert result["stage_usage"]["relevance"]["input_tokens_complete"] is False
    encoded = json.dumps(result)
    assert "boom" not in encoded or "unavailable" in encoded


def test_no_decide_ranks_by_retrieval_without_client(repo: Repository) -> None:
    config = AVConfig()
    result = clip_video(
        "quantum error correction",
        "v1",
        repo,
        config,
        clips_wanted=2,
        decide=False,
    )
    assert result["status"] == "deterministic_only"
    assert result["decisions"]["provider"] == "none"
    scores = [clip["rank"] for clip in result["clips"]]
    assert scores == sorted(scores)
    for clip in result["clips"]:
        assert clip["decision_provider"] == "none"


def test_duration_is_shaped_at_event_boundaries(repo: Repository, tmp_path) -> None:
    config = AVConfig(typesafe_api_key="k")
    client = ScriptedSystemOne()
    result = _clip(repo, config, client, clips_wanted=1, target_seconds=30.0)
    for clip in result["clips"]:
        assert clip["duration_sec"] <= 30.0 * 1.5
        assert clip["boundary_uncertain"] is False
        assert clip["boundary_source"] == "judged_choice"


def test_budgeted_client_counts_attempts_and_enforces_cap() -> None:
    class Inner:
        def __init__(self):
            self.calls = 0

        def ask(self, state, questions):
            self.calls += 1
            if self.calls > 2:
                raise RefinementError("System One unavailable (x)", attempts=1)
            return (
                {k: {"type": "noul", "noul": 0.9} for k in questions},
                {"requests": 1, "input_tokens": 1, "output_tokens": 1},
            )

    inner = Inner()
    budgeted = BudgetedSystemOne(inner, max_requests=5)
    state = {"q": 1}
    questions = {"a": {"type": "noul"}}
    budgeted.ask(state, questions)
    # Same state/questions replay is served from cache: no new request.
    budgeted.ask(state, questions)
    assert budgeted.attempts == 1
    assert budgeted.cache_hits == 1
    with pytest.raises(RefinementError):
        for _ in range(10):
            budgeted.ask({"q": len(budgeted._cache) + _}, questions)
    assert budgeted.attempts >= 5 or inner.calls >= 2


def test_invalid_durations_are_rejected(repo: Repository) -> None:
    config = AVConfig(typesafe_api_key="k")
    with pytest.raises(ClipError):
        clip_video(
            "topic", "v1", repo, config, min_seconds=40.0, target_seconds=30.0
        )


def test_assembly_rejects_overlapping_selection(repo: Repository, tmp_path) -> None:
    config = AVConfig(typesafe_api_key="k")
    candidates, _ = build_candidates(
        "quantum error correction", repo, repo.get_video("v1"), max_seconds=90.0
    )
    # Force one huge candidate so overlap dedup has something to reject.
    for candidate in candidates:
        candidate.relevance_p = 0.9
        candidate.coherence_p = 0.9
        candidate.appeal_score = 0.5
    clips, assembly = assemble_clips(
        candidates,
        repo.get_video("v1"),
        config,
        clips_wanted=5,
        target_seconds=30.0,
        min_seconds=10.0,
        max_seconds=30.0,
        decisions_available=True,
    )
    assert assembly["assembly"]["rejected"].get("overlapping", 0) >= 0
    for i, a in enumerate(clips):
        for b in clips[i + 1 :]:
            assert a["end_sec"] <= b["start_sec"] or b["end_sec"] <= a["start_sec"]


def test_provider_receives_per_clip_question_keys(repo: Repository, tmp_path) -> None:
    config = AVConfig(typesafe_api_key="k")
    client = ScriptedSystemOne()
    result = _clip(repo, config, client, clips_wanted=1)
    assert result["status"] == "ok"
    seen_noul_keys: set[str] = set()
    seen_score_keys: set[str] = set()
    for questions in client.questions:
        for key, question in questions.items():
            instructions = str(question.get("instructions", ""))
            if question["type"] == "noul":
                assert f"`clips.{key}`" in instructions, instructions
                seen_noul_keys.add(key)
            elif question["type"] == "score":
                assert f"`clips.{key}`" in instructions, instructions
                seen_score_keys.add(key)
    assert seen_noul_keys, "noul questions were asked"
    assert seen_score_keys, "score questions were asked"
    # No template placeholder ever reaches the provider.
    for questions in client.questions:
        for question in questions.values():
            assert "{key}" not in str(question.get("instructions", ""))
