"""Tests for synthetic fixtures, the ordering gate, and receipt hygiene."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from av.bench.fixtures import (
    COLOR_PALETTE,
    FIXTURE_KINDS,
    MAX_N,
    fixture_prompt,
    generate_fixture,
)
from av.bench.receipts import (
    COMMUNITY,
    MEASURED,
    Claim,
    Receipt,
    redact_endpoint,
    sha256_file,
    write_receipt,
)
from av.bench.tasks.ordering import normalise_answer, score_ordering

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_color_fixture_frames_and_labels(tmp_path: Path) -> None:
    fixture = generate_fixture("color", 4, tmp_path)
    assert len(fixture.frame_paths) == 4
    assert fixture.labels == [name for name, _ in COLOR_PALETTE[:4]]
    assert all(p.exists() and p.stat().st_size > 0 for p in fixture.frame_paths)


@needs_ffmpeg
def test_fixtures_are_byte_identical_across_runs(tmp_path: Path) -> None:
    """Determinism is the whole point: the same fixture must hash the same."""
    first = generate_fixture("count", 3, tmp_path / "a")
    second = generate_fixture("count", 3, tmp_path / "b")
    assert [sha256_file(p) for p in first.frame_paths] == [
        sha256_file(p) for p in second.frame_paths
    ]


@needs_ffmpeg
def test_frames_within_a_fixture_are_distinct(tmp_path: Path) -> None:
    fixture = generate_fixture("motion", 4, tmp_path)
    hashes = [sha256_file(p) for p in fixture.frame_paths]
    assert len(set(hashes)) == 4


@needs_ffmpeg
@pytest.mark.parametrize("kind", FIXTURE_KINDS)
def test_every_kind_generates(kind: str, tmp_path: Path) -> None:
    fixture = generate_fixture(kind, 2, tmp_path / kind)
    assert len(fixture.frame_paths) == 2
    assert fixture.ffmpeg_commands and all("ffmpeg" in c for c in fixture.ffmpeg_commands)


def test_fixture_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_fixture("rainbow", 2, tmp_path)


def test_fixture_rejects_degenerate_and_oversized(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_fixture("color", 1, tmp_path)
    with pytest.raises(ValueError):
        generate_fixture("color", MAX_N["color"] + 1, tmp_path)


def test_prompt_lists_allowed_colour_names() -> None:
    prompt = fixture_prompt("color", 3)
    assert "magenta" in prompt
    assert "3 images" in prompt


# ---------------------------------------------------------------------------
# Answer parsing and scoring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("red, green, blue", ["red", "green", "blue"]),
        ("Answer: red,green,blue", ["red", "green", "blue"]),
        ("1. red\n2. green\n3. blue", ["red", "green", "blue"]),
        ("`red, green`", ["red", "green"]),
        ("", []),
    ],
)
def test_normalise_colour_answers(raw: str, expected: list[str]) -> None:
    assert normalise_answer(raw, "color") == expected


def test_normalise_numeric_answers() -> None:
    assert normalise_answer("1, 2, 3, 4", "count") == ["1", "2", "3", "4"]


def test_score_exact_match() -> None:
    exact, prefix = score_ordering(["red", "green"], ["red", "green"])
    assert exact and prefix == 2


def test_score_records_prefix_on_truncated_answer() -> None:
    """The upstream failure to detect: a model that reports only the first frame."""
    exact, prefix = score_ordering(["red", "blue", "green"], ["red"])
    assert not exact
    assert prefix == 1


def test_score_wrong_order_is_not_exact() -> None:
    exact, prefix = score_ordering(["red", "blue"], ["blue", "red"])
    assert not exact
    assert prefix == 0


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

def test_claim_requires_a_source_when_not_measured() -> None:
    Claim(label=MEASURED, statement="ran it here")
    with pytest.raises(ValueError):
        Claim(label=COMMUNITY, statement="someone else's number")
    Claim(label=COMMUNITY, statement="someone else's number", source="a published chart")


def test_claim_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        Claim(label="probably", statement="...")


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.openai.com/v1", "api.openai.com"),
        ("http://localhost:30000/v1", "<private>"),
        ("http://10.1.2.3:8000/v1", "<private>"),
        ("http://box.tail1234.ts.net:30000/v1", "<private>"),
        ("http://100.64.1.2:30000/v1", "<private>"),
        (None, None),
    ],
)
def test_private_endpoints_never_reach_a_receipt(url: str | None, expected: str | None) -> None:
    assert redact_endpoint(url) == expected


def test_receipt_round_trips_to_disk(tmp_path: Path) -> None:
    receipt = Receipt(kind="gate", summary={"verdict": "pass"})
    receipt.add_claim(MEASURED, "it passed")
    path = write_receipt(receipt, tmp_path)
    assert path.exists()

    import json

    data = json.loads(path.read_text())
    assert data["kind"] == "gate"
    assert data["claims"][0]["label"] == MEASURED
    assert data["receipt_version"] >= 1
    assert "av_version" in data
