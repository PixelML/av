"""Offline, deterministic compatibility tests for the djev-spark decision lane.

The wire fixtures here mirror the audited ``/v1/systemone`` behaviour of
``mmastrac/djev-spark`` at commit ``1444f3e927f83ba508e5b28a4fd4fdd9ecd0976b``
(derived from that source, not recorded from a live server). No test opens a
socket: every HTTP interaction goes through a fake ``requests`` session, and
the ask-path tests patch the client construction seam directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from av.core.config import AVConfig
from av.db.models import ArtifactRecord, VideoRecord
from av.db.repository import Repository
from av.providers.base import CompletionResult
from av.search.djev import DjevClient, decision_provider_name, open_decision_client
from av.search.rag import ask
from av.search.refine import RefinementError, judge_relevance

ENDPOINT = "http://10.1.2.3:8011/v1/systemone"


def _questions() -> dict[str, dict]:
    """The three-question Jev request shape from the upstream test suite."""
    return {
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer need a reply within the hour?",
            "criteria": {"true": "needs a reply now", "false": "can wait"},
        },
        "bucket": {
            "type": "choice",
            "instructions": "Which team owns this?",
            "criteria": {"billing": None, "outage": "service down", "feature": None},
        },
        "tone": {
            "type": "score",
            "instructions": "How angry is the customer?",
            "criteria": ["calm", "annoyed", "furious"],
        },
    }


def _answers() -> dict[str, dict]:
    """The answer shapes ``jev_answer`` produces for a confident first read.

    The exported score is the zero-based expected level ``sum(i * p)``:
    0·0.7 + 1·0.2 + 2·0.1 = 0.4 (the upstream test expects 0.45 for its own
    distribution on the same formula).
    """
    return {
        "urgent": {"type": "noul", "noul": 0.7},
        "bucket": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.7, "outage": 0.2, "feature": 0.1},
            "confidence": 0.7,
        },
        "tone": {
            "type": "score",
            "score": 0.4,
            "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
            "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
            "confidence": 0.7,
        },
    }


def _payload(**over: Any) -> dict:
    body: dict[str, Any] = {
        "model": "dgemma",
        "answers": _answers(),
        "usage": {"input_tokens": 321, "output_tokens": 12},
        "diagnostics": {"engine": "vllm", "timing": {"total_ms": 12.5, "reads": 1}},
    }
    body.update(over)
    return body


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, broken_json: bool = False):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self._broken_json = broken_json

    def json(self) -> Any:
        if self._broken_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _client(session: FakeSession, **config_over: Any) -> DjevClient:
    config = AVConfig(djev_endpoint=ENDPOINT, **config_over)
    return DjevClient(config, session=session)


def test_endpoint_is_required_and_never_defaulted() -> None:
    with pytest.raises(RefinementError, match="AV_DJEV_ENDPOINT"):
        DjevClient(AVConfig())


def test_request_shape_auth_is_optional_and_model_is_advisory() -> None:
    session = FakeSession(FakeResponse(payload=_payload()))
    client = _client(session, djev_api_key="test-key", djev_model="jev-latest")
    client.ask({"ticket": "down"}, _questions())

    call = session.calls[0]
    assert call["url"] == ENDPOINT
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"]["seed"] == 42
    assert call["json"]["state"] == {"ticket": "down"}
    assert set(call["json"]["questions"]) == {"urgent", "bucket", "tone"}

    # Without a configured key there is no Authorization header at all, and
    # without an advisory model the request carries no model field.
    anonymous = FakeSession(
        FakeResponse(payload=_payload(answers={"a": {"type": "noul", "noul": 0.5}}))
    )
    _client(anonymous).ask("state", {"a": {"type": "noul", "criteria": {}}})
    call = anonymous.calls[0]
    assert "Authorization" not in call["headers"]
    assert "model" not in call["json"]


def test_happy_path_records_served_identity_not_the_request_model() -> None:
    session = FakeSession(FakeResponse(payload=_payload()))
    client = _client(session, djev_model="jev-latest")
    answers, usage = client.ask({}, _questions())

    assert answers["urgent"] == {"type": "noul", "noul": 0.7}
    assert answers["bucket"]["choice"] == "billing"
    # The exported score is the zero-based expected level sum(i * p), in
    # [0, len(levels) - 1]: 0·0.7 + 1·0.2 + 2·0.1 = 0.4.
    assert answers["tone"]["score"] == pytest.approx(0.4)
    assert answers["tone"]["score"] == pytest.approx(
        sum(i * p for i, p in enumerate((0.7, 0.2, 0.1)))
    )

    # The request said "jev-latest"; the server ignored it and answered as
    # dgemma. The receipt must carry what actually served.
    assert client.served_model == "dgemma"
    assert client.server_engine == "vllm"
    assert usage["served_model"] == "dgemma"
    assert usage["server_engine"] == "vllm"
    assert usage["served_endpoint_host"] == "<private>"
    assert usage["input_tokens"] == 321
    assert usage["output_tokens"] == 12
    assert usage["_attempts"] == 1


def test_skipped_questions_are_reported_as_null() -> None:
    payload = _payload()
    payload["answers"]["bucket"] = None  # ask_if failed server-side
    questions = _questions()
    questions["bucket"]["ask_if"] = {"urgent": ["yes"]}
    session = FakeSession(FakeResponse(payload=payload))
    answers, _ = _client(session).ask({}, questions)
    assert answers["bucket"] is None
    assert answers["urgent"]["noul"] == 0.7


def test_null_answer_for_a_mandatory_question_is_rejected() -> None:
    payload = _payload()
    payload["answers"]["urgent"] = None  # relevance judgments cannot be skipped
    session = FakeSession(FakeResponse(payload=payload))
    with pytest.raises(RefinementError, match="mandatory question was skipped"):
        _client(session).ask({}, _questions())


@pytest.mark.parametrize("removed", ["model", "diagnostics"])
def test_success_without_served_identity_is_rejected(removed: str) -> None:
    payload = _payload()
    if removed == "diagnostics":
        payload["diagnostics"] = {}  # engine missing from diagnostics
    else:
        del payload[removed]
    session = FakeSession(FakeResponse(payload=payload))
    client = _client(session)
    with pytest.raises(RefinementError, match="did not identify the served runtime"):
        client.ask({}, _questions())
    # A failed identity check must not leave a default or stale value standing
    # in for this response.
    missing, present = (
        ("served_model", "server_engine")
        if removed == "model"
        else ("server_engine", "served_model")
    )
    assert getattr(client, missing) is None
    assert getattr(client, present) in (None, "vllm", "dgemma")


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda p, q: p["answers"].pop("tone"), "missing answer"),
        (lambda p, q: p["answers"].update(extra={"type": "noul", "noul": 0.5}), "not asked"),
        (lambda p, q: p["answers"].update(urgent={"type": "choice", "choice": "billing"}), "expected a noul"),
        (
            lambda p, q: p["answers"]["bucket"].update(choice="security"),
            "outside the offered options",
        ),
        (
            lambda p, q: p["answers"]["bucket"].update(probabilities={"billing": 1.0}),
            "do not match the offered options",
        ),
        (
            lambda p, q: p["answers"]["tone"].update(legend={"0": "calm", "1": "annoyed"}),
            "legend does not match",
        ),
        (
            lambda p, q: p["answers"]["tone"].update(probabilities={"0": 0.5, "1": 0.5}),
            "do not match the legend",
        ),
        (lambda p, q: p["answers"]["tone"].update(score=2.5), "outside the legend range"),
        (lambda p, q: p["answers"]["tone"].update(score=-0.5), "outside the legend range"),
        (
            lambda p, q: p["answers"]["tone"].update(score=0.9),
            "inconsistent with the legend probabilities",
        ),
        (lambda p, q: p["answers"]["urgent"].update(noul=1.5), "out of range"),
        (
            lambda p, q: p["answers"]["bucket"].update(
                probabilities={"billing": 0.4, "outage": 0.2, "feature": 0.1}
            ),
            "probabilities sum to",
        ),
        (
            lambda p, q: p["answers"]["bucket"].update(choice=[]),
            "outside the offered options",
        ),
        (
            lambda p, q: p["answers"]["bucket"].update(choice={}),
            "outside the offered options",
        ),
    ],
)
def test_malformed_responses_are_rejected_visibly(mutate, fragment) -> None:
    payload = _payload()
    questions = _questions()
    mutate(payload, questions)
    session = FakeSession(FakeResponse(payload=payload))
    with pytest.raises(RefinementError, match=fragment) as excinfo:
        _client(session).ask({}, questions)
    assert "djev-spark" in str(excinfo.value)

def test_http_401_is_not_retried_and_names_the_provider() -> None:
    session = FakeSession(FakeResponse(status_code=401))
    with pytest.raises(RefinementError, match="djev-spark request failed with HTTP 401"):
        _client(session).ask({}, _questions())
    assert len(session.calls) == 1


def test_retryable_503_then_success_counts_both_attempts() -> None:
    session = FakeSession(
        FakeResponse(status_code=503), FakeResponse(payload=_payload())
    )
    _, usage = _client(session).ask({}, _questions())
    assert len(session.calls) == 2
    assert usage["_attempts"] == 2


def test_retry_exhaustion_reports_unavailability() -> None:
    session = FakeSession(
        requests.ConnectionError("reset"), requests.ConnectionError("reset")
    )
    with pytest.raises(RefinementError, match="djev-spark unavailable"):
        _client(session, djev_max_retries=1).ask({}, _questions())


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(broken_json=True),
        FakeResponse(payload=[1, 2, 3]),
        FakeResponse(payload={"usage": {}}),  # no answers key
    ],
)
def test_unusable_successes_are_rejected(response: FakeResponse) -> None:
    session = FakeSession(response)
    with pytest.raises(RefinementError):
        _client(session).ask({}, _questions())


def test_judge_relevance_consumes_a_djev_client() -> None:
    payload = _payload()
    payload["answers"] = {
        "c0": {"type": "noul", "noul": 0.9},
        "c1": {"type": "noul", "noul": 0.2},
    }
    session = FakeSession(FakeResponse(payload=payload))
    client = _client(session)
    results = [
        {"artifact_id": f"a{i}", "video_id": "v", "timestamp_sec": i, "text": "x", "source_type": "caption"}
        for i in range(2)
    ]
    probabilities, usage = judge_relevance(client, "query", results)

    assert probabilities == {"a0": 0.9, "a1": 0.2}
    assert usage["requests"] == 1
    assert usage["served_model"] == "dgemma"
    # The batched relevance questions are exactly what went on the wire.
    sent = session.calls[0]["json"]["questions"]
    assert set(sent) == {"c0", "c1"}
    assert sent["c0"]["type"] == "noul"


def test_decision_lane_selection_and_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    hosted = AVConfig(typesafe_api_key="k")
    assert decision_provider_name(hosted) == "jev"
    assert type(open_decision_client(hosted)).__name__ == "SystemOneClient"

    self_hosted = AVConfig(typesafe_api_key="k", djev_endpoint=ENDPOINT)
    assert decision_provider_name(self_hosted) == "djev-spark"
    client = open_decision_client(self_hosted)
    assert isinstance(client, DjevClient)

    unconfigured = AVConfig()
    assert decision_provider_name(unconfigured) is None

    # Env var alone selects the lane with file/env priority intact.
    monkeypatch.setenv("AV_DJEV_ENDPOINT", ENDPOINT)
    from_env = AVConfig()
    assert decision_provider_name(from_env) == "djev-spark"


def test_config_file_and_env_priority_for_djev_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from av.core import config as config_module

    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"djev_endpoint": "http://file-host:8011/v1/systemone", "djev_seed": 7}'
    )
    monkeypatch.setattr(config_module, "CONFIG_FILE_PATH", config_file)

    config = config_module.get_config()
    assert config.djev_endpoint == "http://file-host:8011/v1/systemone"
    assert config.djev_seed == 7

    monkeypatch.setenv("AV_DJEV_ENDPOINT", "http://env-host:8011/v1/systemone")
    config = config_module.get_config()
    assert config.djev_endpoint == "http://env-host:8011/v1/systemone"
    assert config.djev_seed == 7  # file value survives where env is silent


# --- ask() wiring through the djev lane -------------------------------


def _video(video_id: str, path: Path) -> VideoRecord:
    return VideoRecord(
        id=video_id,
        file_path=str(path),
        file_hash=f"hash-{video_id}",
        file_size_bytes=path.stat().st_size,
        filename=path.name,
        duration_sec=100.0,
        status="complete",
    )


def _artifact(artifact_id: str, video_id: str, start: float, text: str) -> ArtifactRecord:
    return ArtifactRecord(
        id=artifact_id,
        video_id=video_id,
        type="caption",
        start_sec=start,
        end_sec=start + 10.0,
        text=text,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "av.db")


def _seed_video(repo: Repository, tmp_path: Path, video_id: str = "v1") -> Path:
    path = tmp_path / f"{video_id}.mp4"
    path.write_bytes(b"fake-video")
    repo.insert_video(_video(video_id, path))
    repo.insert_artifacts_batch([
        _artifact(f"{video_id}-{i}", video_id, i * 10.0, f"cake scene {i}")
        for i in range(10)
    ])
    return path


class FakeDjevDecision:
    """Duck-typed DjevClient: scripted decisions with served identity."""

    provider_label = "djev-spark"
    served_model = "dgemma"
    server_engine = "vllm"

    def __init__(self, relevance: float = 0.9, supports: list[float] | None = None) -> None:
        self.relevance = relevance
        self.supports = list(supports) if supports is not None else [0.95]
        self.calls: list[tuple[dict, dict]] = []

    def ask(self, state: Any, questions: dict[str, dict]) -> tuple[dict[str, dict], dict]:
        self.calls.append((state, questions))
        usage = {"input_tokens": 5, "output_tokens": 1, "_attempts": 1}
        if "is_supported" in questions:
            return {"is_supported": {"type": "noul", "noul": self.supports.pop(0)}}, usage
        answers: dict[str, dict] = {}
        for key, question in questions.items():
            if question["type"] == "choice":
                names = list(question["criteria"])
                share = 1.0 / len(names)
                answers[key] = {
                    "type": "choice",
                    "choice": names[0],
                    "probabilities": {name: share for name in names},
                    "confidence": share,
                }
            else:
                answers[key] = {"type": "noul", "noul": self.relevance}
        return answers, usage


class FakeAnswerLLM:
    def __init__(self, config: AVConfig) -> None:
        self.config = config

    def complete_with_usage(self, prompt: str, context: str) -> CompletionResult:
        return CompletionResult("refined answer", input_tokens=None, output_tokens=None)


def test_ask_over_djev_lane_reports_djev_identity_and_bases(
    repo: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_video(repo, tmp_path)
    fake = FakeDjevDecision(relevance=0.9, supports=[0.95])
    monkeypatch.setattr("av.search.rag.open_decision_client", lambda config: fake)
    monkeypatch.setattr("av.search.rag.OpenAILLM", FakeAnswerLLM)

    result = ask(
        "cake", repo, AVConfig(djev_endpoint=ENDPOINT, embed_model=""), video_id="v1"
    )

    assert result["route"] == "refined"
    assert result["confidence_basis"] == "djev_spark_answer_support"
    assert result["refinement"]["decision_provider"] == "djev-spark"
    assert result["refinement"]["served_model"] == "dgemma"
    assert result["refinement"]["server_engine"] == "vllm"
    assert result["ask_settings"]["decision_provider"] == "djev-spark"


def test_ask_all_irrelevant_reports_djev_relevance_basis(
    repo: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_video(repo, tmp_path)
    fake = FakeDjevDecision(relevance=0.1)
    monkeypatch.setattr("av.search.rag.open_decision_client", lambda config: fake)
    monkeypatch.setattr(
        "av.search.rag.OpenAILLM",
        lambda config: (_ for _ in ()).throw(AssertionError("answer model must not run")),
    )

    result = ask(
        "cake", repo, AVConfig(djev_endpoint=ENDPOINT, embed_model=""), video_id="v1"
    )

    assert result["route"] == "refined_no_results"
    assert result["confidence_basis"] == "djev_spark_relevance"
    assert result["refinement"]["decision_provider"] == "djev-spark"


def test_ask_inspected_answer_reports_djev_sampled_frames_basis(
    repo: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_video(repo, tmp_path)
    # First support judgment (plain answer) fails; the judgment on the
    # vision-inspected citations succeeds.
    fake = FakeDjevDecision(relevance=0.9, supports=[0.1, 0.95])
    monkeypatch.setattr("av.search.rag.open_decision_client", lambda config: fake)
    monkeypatch.setattr("av.search.rag.OpenAILLM", FakeAnswerLLM)
    from av.search.usage import new_usage

    monkeypatch.setattr(
        "av.search.rag.inspect_with_stronger_vision",
        lambda *args, **kwargs: {
            "status": "supported",
            "answer": "inspected answer",
            "citations": [{
                "video_id": "v1",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "source_type": "caption",
                "text": "cake scene 0",
            }],
            "windows": [],
            "usage": new_usage(),
            "warnings": [],
        },
    )

    result = ask(
        "cake", repo, AVConfig(djev_endpoint=ENDPOINT, embed_model=""), video_id="v1"
    )

    assert result["route"] == "vision_inspected"
    assert result["confidence_basis"] == "djev_spark_answer_support_after_sampled_frames"
    assert result["refinement"]["served_model"] == "dgemma"
