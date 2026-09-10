"""Tests for task loading, scoring, arms, and public-benchmark adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from av.bench.datasets import (
    ADAPTERS,
    SOURCES,
    adapt_lvbench,
    adapt_minerva,
    load_annotations,
    subset_by_video,
)
from av.bench.tasks.events import EVENT_PROBES, parse_presence
from av.bench.tasks.videoqa import (
    Question,
    build_question_prompt,
    build_selection_prompt,
    extract_choice,
    load_questions,
    parse_timestamps,
    score_answer,
)


# ---------------------------------------------------------------------------
# Task file loading
# ---------------------------------------------------------------------------

def _write_task(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "task.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_questions_resolves_relative_videos(tmp_path: Path) -> None:
    path = _write_task(tmp_path, [
        {"id": "q1", "video": "clips/a.mp4", "question": "what?", "options": ["A. x", "B. y"], "answer": "B"}
    ])
    questions = load_questions(path)
    assert questions[0].video == (tmp_path / "clips/a.mp4").resolve()
    assert questions[0].is_multiple_choice


def test_load_questions_honours_video_root(tmp_path: Path) -> None:
    path = _write_task(tmp_path, [{"video": "a.mp4", "question": "q"}])
    root = tmp_path / "elsewhere"
    assert load_questions(path, root)[0].video == (root / "a.mp4").resolve()


def test_load_questions_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    path.write_text('# a note\n\n{"video": "a.mp4", "question": "q"}\n')
    assert len(load_questions(path)) == 1


def test_load_questions_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    path.write_text('{"video": "a.mp4", "question": "q"}\nnot json\n')
    with pytest.raises(ValueError, match="task.jsonl:2"):
        load_questions(path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reply,expected",
    [("B", "B"), ("b", "B"), ("B.", "B"), ("The answer is C", "C"), ("", None), ("hmm", None)],
)
def test_extract_choice(reply: str, expected: str | None) -> None:
    assert extract_choice(reply, ["A. x", "B. y", "C. z"]) == expected


def test_extract_choice_falls_back_to_option_text() -> None:
    assert extract_choice("a red car", ["A. a blue van", "B. a red car"]) == "B"


def test_score_multiple_choice() -> None:
    q = Question(id="q", video=Path("a.mp4"), question="?", options=["A. x", "B. y"], answer="B")
    assert score_answer(q, "B")[0]
    assert not score_answer(q, "A")[0]


def test_unparseable_reply_scores_incorrect_not_skipped() -> None:
    q = Question(id="q", video=Path("a.mp4"), question="?", options=["A. x", "B. y"], answer="B")
    correct, parsed = score_answer(q, "I cannot tell from these frames")
    assert not correct
    assert parsed is None


def test_score_open_ended_is_substring_match() -> None:
    q = Question(id="q", video=Path("a.mp4"), question="?", answer="a red car")
    assert score_answer(q, "It is a red car.")[0]
    assert not score_answer(q, "a blue van")[0]


# ---------------------------------------------------------------------------
# Prompts and agentic selection
# ---------------------------------------------------------------------------

def test_question_prompt_carries_timestamps() -> None:
    q = Question(id="q", video=Path("a.mp4"), question="what?", options=["A. x"], answer="A")
    prompt = build_question_prompt(q, [0.0, 1.0, 2.0])
    assert "0.0s, 1.0s, 2.0s" in prompt
    assert "A. x" in prompt


def test_selection_prompt_withholds_the_answer_step() -> None:
    q = Question(id="q", video=Path("a.mp4"), question="what?", answer="x")
    prompt = build_selection_prompt(q, [0.0, 60.0], budget=4, duration=120.0)
    assert "Do not answer it yet" in prompt
    assert "at most 4" in prompt


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"timestamps": [1.0, 2.5]}', [1.0, 2.5]),
        ('```json\n{"timestamps": [3]}\n```', [3.0]),
        ("I want 5 and 10 seconds", [5.0, 10.0]),
        ("", []),
    ],
)
def test_parse_timestamps(reply: str, expected: list[float]) -> None:
    assert parse_timestamps(reply, duration=100.0, budget=8) == expected


def test_parse_timestamps_drops_out_of_range_and_respects_budget() -> None:
    assert parse_timestamps('{"timestamps": [1, 5, 500]}', duration=100.0, budget=8) == [1.0, 5.0]
    assert len(parse_timestamps('{"timestamps": [1,2,3,4,5]}', duration=100.0, budget=2)) == 2


# ---------------------------------------------------------------------------
# Event presence parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"present": true}', True),
        ('{"present": false}', False),
        ('```json\n{"present": true}\n```', True),
        ('```json\n{"present', None),          # truncated before the value
        ('```json\n{"present": true', True),   # truncated after it
        ("yes", True),
        ("no", False),
        ("", None),
    ],
)
def test_parse_presence(reply: str, expected: bool | None) -> None:
    assert parse_presence(reply) is expected


def test_every_probe_has_a_question_and_terms() -> None:
    for name, probe in EVENT_PROBES.items():
        assert probe["question"].endswith("?"), name
        assert probe["reference_terms"], name


# ---------------------------------------------------------------------------
# Public benchmark adapters
# ---------------------------------------------------------------------------

MINERVA_ROW = {
    "key": "vid1:abc",
    "video_id": "vid1",
    "question": "What happens?",
    "answer_choice_0": "nothing",
    "answer_choice_1": "something",
    "answer_choice_2": "everything",
    "answer_id": 1,
    "question_type": "Event Occurence",
    "split": "Sports",
    "category": "Basketball",
}

LVBENCH_ROW = {
    "key": "vidA",
    "type": "cartoon",
    "qa": [
        {
            "uid": "55",
            "question": "What year?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633",
            "answer": "D",
            "question_type": ["key information retrieval"],
            "time_reference": "00:15-00:19",
        }
    ],
}


def test_adapt_minerva_maps_index_to_letter() -> None:
    rows = adapt_minerva([MINERVA_ROW])
    assert len(rows) == 1
    assert rows[0].answer == "B"
    assert rows[0].options[1] == "B. something"
    assert rows[0].video_id == "vid1"


def test_adapt_minerva_skips_rows_with_a_bad_gold_index() -> None:
    assert adapt_minerva([{**MINERVA_ROW, "answer_id": 9}]) == []


def test_adapt_lvbench_splits_inline_options_and_timespan() -> None:
    rows = adapt_lvbench([LVBENCH_ROW])
    assert len(rows) == 1
    row = rows[0]
    assert row.question == "What year?"
    assert row.options == ["A. 1636", "B. 1366", "C. 1363", "D. 1633"]
    assert row.answer == "D"
    assert (row.start_sec, row.end_sec) == (15.0, 19.0)


def test_adapt_lvbench_handles_hour_length_timespans() -> None:
    row = {**LVBENCH_ROW}
    row["qa"] = [{**LVBENCH_ROW["qa"][0], "time_reference": "01:02:03-01:02:10"}]
    assert adapt_lvbench([row])[0].start_sec == 3723.0


def test_task_row_uses_the_video_template() -> None:
    row = adapt_minerva([MINERVA_ROW])[0].to_task_row("videos/{video_id}.mp4")
    assert row["video"] == "videos/vid1.mp4"
    assert row["meta"]["video_id"] == "vid1"


def test_subset_groups_by_video_to_limit_downloads() -> None:
    rows = adapt_minerva([
        {**MINERVA_ROW, "key": f"v{v}:{q}", "video_id": f"v{v}"}
        for v in range(4) for q in range(3)
    ])
    chosen = subset_by_video(rows, max_questions=6, max_videos=2)
    assert len({r.video_id for r in chosen}) == 2
    assert len(chosen) == 6


def test_subset_is_deterministic() -> None:
    rows = adapt_minerva([
        {**MINERVA_ROW, "key": f"v{v}:{q}", "video_id": f"v{v}"}
        for v in range(4) for q in range(3)
    ])
    first = [r.id for r in subset_by_video(rows, max_questions=5)]
    second = [r.id for r in subset_by_video(rows, max_questions=5)]
    assert first == second


def test_load_annotations_reads_both_json_and_jsonl(tmp_path: Path) -> None:
    as_json = tmp_path / "a.json"
    as_json.write_text(json.dumps([MINERVA_ROW]))
    as_jsonl = tmp_path / "b.jsonl"
    as_jsonl.write_text(json.dumps(LVBENCH_ROW) + "\n")
    assert len(load_annotations(as_json)) == 1
    assert len(load_annotations(as_jsonl)) == 1


def test_every_adapter_has_a_documented_source() -> None:
    """Licences differ sharply between these datasets, so each must name its terms."""
    for name in ADAPTERS:
        assert name in SOURCES
        assert SOURCES[name]["annotations_licence"]
        assert SOURCES[name]["annotations_url"].startswith("https://")
