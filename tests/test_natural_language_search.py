"""Offline regressions for question candidate retrieval before Jev refinement."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from av.core.config import AVConfig
from av.db.models import ArtifactRecord, VideoRecord
from av.db.repository import Repository
from av.providers.base import CompletionResult
from av.providers.usage import ProviderUsage
from av.search.query import MAX_CANDIDATE_TERMS, natural_language_fts_query
from av.search.rag import ask
from av.search.semantic import search


@pytest.fixture()
def repo(tmp_path: Path):
    repository = Repository(tmp_path / "questions.db")
    for video_id in ("local", "other"):
        repository.insert_video(VideoRecord(
            id=video_id, file_path=f"{video_id}.mp4", file_hash=video_id,
            file_size_bytes=0, filename=f"{video_id}.mp4", duration_sec=120,
            status="complete",
        ))
    repository.insert_artifacts_batch([
        ArtifactRecord(id="blue", video_id="local", type="caption", start_sec=10,
                       end_sec=20, text="The truck was blue"),
        ArtifactRecord(id="irrelevant", video_id="local", type="caption", start_sec=60,
                       end_sec=70, text="The truck advertisement listed rental prices"),
        ArtifactRecord(id="red", video_id="other", type="caption", start_sec=10,
                       end_sec=20, text="The truck was red"),
        ArtifactRecord(id="unicode", video_id="local", type="caption", start_sec=90,
                       end_sec=100, text="A café named Étoile appears beside a 東京 sign"),
    ])
    try:
        yield repository
    finally:
        repository.close()


@pytest.mark.parametrize("question", ["What color was the truck?", "What color was the truck"])
def test_natural_question_retrieves_candidates_and_preserves_video_isolation(repo, question):
    result = search(question, repo, AVConfig(embed_model=""), video_id="local", natural_language=True)
    assert result["query"] == question
    assert {row["artifact_id"] for row in result["results"]} == {"blue", "irrelevant"}
    assert {row["video_id"] for row in result["results"]} == {"local"}


def test_unicode_punctuation_and_fts_operators_are_literal_candidates(repo):
    result = search('What is "CAFÉ"? OR 東京: (Étoile*)', repo, AVConfig(embed_model=""),
                    video_id="local", natural_language=True)
    assert [row["artifact_id"] for row in result["results"]] == ["unicode"]
    assert natural_language_fts_query('truck NOT "red"') == '"truck" OR "not" OR "red"'


@pytest.mark.parametrize("question", ["What is it?", "?! () :", "Where did the submarine surface?"])
def test_empty_or_unrelated_question_returns_no_hits_or_model_calls(repo, question):
    with patch("av.search.rag.open_decision_client") as judge, patch("av.search.rag.OpenAILLM") as llm:
        result = ask(question, repo, AVConfig(typesafe_api_key="test", embed_model=""), video_id="local")
    assert result["route"] == "refined_no_results"
    assert result["citations"] == []
    judge.assert_not_called()
    llm.assert_not_called()


def test_candidate_budget_applies_after_question_words_and_duplicates():
    question = "what " * 100 + "truck " * 100 + " ".join(f"object{i}" for i in range(40))
    terms = natural_language_fts_query(question).split(" OR ")
    assert len(terms) == MAX_CANDIDATE_TERMS
    assert terms[0] == '"truck"'
    assert terms[1] == '"object0"'
    assert len(set(terms)) == len(terms)


def test_advanced_fts_search_semantics_stay_unchanged(repo):
    config = AVConfig(embed_model="")
    result = search("truck NOT red", repo, config)
    assert {row["artifact_id"] for row in result["results"]} == {"blue", "irrelevant"}
    phrase = search('"truck was blue"', repo, config)
    assert [row["artifact_id"] for row in phrase["results"]] == ["blue"]
    assert search("What color was the truck", repo, config)["results"] == []


def test_embedding_receives_original_question(repo):
    question = "What color was the truck?"
    embedder = MagicMock()
    embedder.embed.return_value = [[1.0, 0.0]]
    embedder.usage.snapshot.return_value = {"requests": 1}
    with patch.object(repo, "get_embeddings_for_artifacts", return_value={"blue": [1.0, 0.0]}), \
         patch("av.search.semantic.OpenAIEmbedder", return_value=embedder):
        result = search(question, repo, AVConfig(), video_id="local", natural_language=True)
    embedder.embed.assert_called_once_with([question])
    assert result["query"] == question
    assert result["results"][0]["artifact_id"] == "blue"


@pytest.mark.parametrize("reject_all", [False, True])
def test_ask_passes_original_question_to_jev_and_filters_broad_candidates(repo, reject_all):
    question = "What color was the truck?"
    relevance_texts = []

    class Judge:
        def ask(self, state, questions):
            assert state.get("query", state.get("question")) == question
            if "is_supported" in questions:
                return {"is_supported": {"type": "noul", "noul": 0.99}}, {}
            assert "clips" in state
            relevance_texts.extend(clip["caption"] for clip in state["clips"].values())
            return {
                key: {"type": "noul", "noul": 0.99 if "blue" in clip["caption"] and not reject_all else 0.01}
                for key, clip in state["clips"].items()
            }, {}

    class Answer:
        def __init__(self, config):
            self.usage = ProviderUsage()

        def complete_with_usage(self, prompt, context):
            assert prompt == question
            assert "blue" in context
            assert "rental" not in context
            return CompletionResult("The truck was blue", input_tokens=20, output_tokens=5)

    with patch("av.search.rag.open_decision_client", return_value=Judge()), \
         patch("av.search.rag.OpenAILLM", side_effect=Answer) as llm:
        result = ask(question, repo, AVConfig(typesafe_api_key="test", embed_model=""), video_id="local")
    assert len(relevance_texts) == 2
    if reject_all:
        assert result["route"] == "refined_no_results"
        assert result["citations"] == []
        llm.assert_not_called()
    else:
        assert result["route"] == "refined"
        assert result["answer"] == "The truck was blue"
        assert [citation["artifact_id"] for citation in result["citations"]] == ["blue"]
