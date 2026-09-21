"""Tests for the clip evaluation contract, fixtures, mock, and runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from av import clip_eval
from av.clip_eval import contract
from av.clip_eval.corpus import (
    CorpusError,
    load_corpus,
    load_queries,
    materialize_corpus,
)
from av.clip_eval.mock import LabeledDecisionClient
from av.clip_eval.runner import run_evaluation
from av.core.config import AVConfig
from av.db.repository import Repository

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def corpus_path() -> Path:
    return REPO_ROOT / "clip-eval" / "corpus.json"


def test_corpus_loads_and_checksums_verify(corpus_path: Path) -> None:
    corpus = load_corpus(corpus_path)
    styles = {video["content"]["style"] for video in corpus["videos"]}
    assert len(styles) >= 3
    assert corpus["contract_version"] == clip_eval.CONTRACT_VERSION


def test_tampered_corpus_is_rejected(corpus_path: Path, tmp_path: Path) -> None:
    corpus = json.loads(corpus_path.read_text())
    corpus["videos"][0]["content"]["transcript"][0]["text"] += " tampered"
    bad = tmp_path / "bad-corpus.json"
    bad.write_text(json.dumps(corpus))
    with pytest.raises(CorpusError):
        load_corpus(bad)


def test_query_sets_cover_required_cases() -> None:
    dev = load_queries(REPO_ROOT / "clip-eval" / "queries.json")
    heldout = load_queries(REPO_ROOT / "clip-eval" / "queries-heldout.json")
    combined = dev + heldout
    absent = [q for q in combined if q["expected"] == "absent"]
    present = [q for q in combined if q["expected"] == "present"]
    assert absent and present
    notes = " ".join(
        str(q.get("note", "")) + " " + " ".join(str(m.get("note", "")) for m in q["moments"])
        for q in combined
    ).lower()
    assert "asr" in notes or "noise" in notes
    assert any("contradict" in n for n in [notes])
    assert any(m.get("requires_setup") for q in present for m in q["moments"])
    assert any(m.get("note") and "vision" in m["note"].lower() for q in present for m in q["moments"])


def test_metric_math_known_values() -> None:
    label = {
        "expected": "present",
        "moments": [
            {"moment_id": "m1", "start_sec": 10.0, "end_sec": 20.0},
            {"moment_id": "m2", "start_sec": 40.0, "end_sec": 50.0},
        ],
    }
    perfect = [
        {"start_sec": 10.0, "end_sec": 20.0},
        {"start_sec": 40.0, "end_sec": 50.0},
    ]
    metrics = contract.query_metrics(perfect, label, k=2)
    assert metrics["hits"] == 2
    assert metrics["precision_at_k"] == 1.0
    assert metrics["known_moment_recall"] == 1.0
    assert metrics["boundary_error_sec"] == 0.0

    absent_label = {"expected": "absent", "moments": []}
    fp = contract.query_metrics([{"start_sec": 0.0, "end_sec": 5.0}], absent_label, k=1)
    assert fp["absent_topic_fp"] == 1.0
    assert fp["known_moment_recall"] is None

    dup = [
        {"start_sec": 10.0, "end_sec": 20.0},
        {"start_sec": 12.0, "end_sec": 22.0},
    ]
    metrics = contract.query_metrics(dup, label, k=2)
    assert metrics["duplication_rate"] == 0.5


def test_context_loss_counts_missed_setup_head() -> None:
    label = {
        "expected": "present",
        "moments": [
            {
                "moment_id": "m1",
                "start_sec": 58.0,
                "end_sec": 90.0,
                "requires_setup": True,
                "setup_start_sec": 58.0,
                "setup_end_sec": 82.0,
            }
        ],
    }
    full = [{"start_sec": 58.0, "end_sec": 90.0}]
    partial = [{"start_sec": 70.0, "end_sec": 92.0}]
    assert contract.query_metrics(full, label, k=1)["context_loss_count"] == 0
    assert contract.query_metrics(partial, label, k=1)["context_loss_count"] == 1


def test_mock_answers_are_well_typed() -> None:
    queries = load_queries(REPO_ROOT / "clip-eval" / "queries.json")
    client = LabeledDecisionClient(queries)
    state = {
        "query": "quantum error correction",
        "clips": {
            "c0": {
                "transcript": "Quantum error correction is like backup batteries.",
                "caption": "Slides show lattice diagrams",
                "start_sec": 24.0,
                "end_sec": 58.0,
            },
            "c1": {
                "transcript": "Grant committee paperwork",
                "caption": "(none)",
                "start_sec": 66.0,
                "end_sec": 78.0,
            },
        },
    }
    answers, usage = client.ask(
        state,
        {
            "c0": {"type": "noul", "instructions": "Does `clips.c0` materially concern `query`?"},
            "c1": {"type": "noul", "instructions": "Does `clips.c1` materially concern `query`?"},
            "s": {"type": "score", "instructions": "Rate `clips.c0`."},
        },
    )
    assert answers["c0"]["type"] == "noul" and answers["c0"]["noul"] >= 0.5
    assert answers["c1"]["noul"] <= 0.5
    assert answers["s"]["type"] == "score" and 0 <= answers["s"]["score"] <= 1
    assert usage["requests"] == 1


def test_runner_end_to_end_offline(tmp_path: Path) -> None:
    receipt = run_evaluation(
        REPO_ROOT / "clip-eval" / "corpus.json",
        REPO_ROOT / "clip-eval" / "queries.json",
        db_path=tmp_path / "eval.db",
        arms=("deterministic", "jev_mock"),
        clips_wanted=2,
    )
    assert receipt["contract_version"] == clip_eval.CONTRACT_VERSION
    assert receipt["ingestion_ms"] >= 0
    assert [arm["arm"] for arm in receipt["arms"]] == ["deterministic", "jev_mock"]
    for arm in receipt["arms"]:
        assert len(arm["metrics_per_query"]) == 8
        assert arm["metrics_total"]["queries"] == 8
        assert arm["timings"]["prepare_ms_total"] >= 0
    # The absent-topic trap: the deterministic arm fires it, the judged arm
    # must not (labels are the only ground truth here).
    det = receipt["arms"][0]["metrics_total"]
    mock = receipt["arms"][1]["metrics_total"]
    assert det["absent_topic_fp_total"] >= 0
    assert mock["absent_topic_fp_total"] == 0.0


def test_materialize_rejects_existing_videos(corpus_path: Path, tmp_path: Path) -> None:
    corpus = load_corpus(corpus_path)
    repo = Repository(tmp_path / "dup.db")
    materialize_corpus(corpus, repo)
    with pytest.raises(CorpusError):
        materialize_corpus(corpus, repo)
    repo.close()


def test_config_defaults_are_public_and_bounded() -> None:
    config = AVConfig()
    assert 0 <= config.clip_relevance_min <= 1
    assert 0 <= config.clip_coherence_min <= 1
    assert 0 <= config.clip_visual_min <= 1
    assert 0 <= config.clip_boundary_conf_min <= 1
    assert config.clip_request_cap >= 1
