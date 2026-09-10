"""Tests for benchmark cost models, spreads, and the frontier summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from av.bench.cost import (
    CellUsage,
    HourlyCost,
    PerTokenCost,
    parse_cost_model,
    video_hours_per_dollar,
)
from av.bench.runner import (
    Spread,
    collapse_point,
    interpret_delta,
    is_saturated,
    noise_floor,
)


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------

def test_per_token_cost() -> None:
    model = PerTokenCost(input_usd_per_mtok=0.30, output_usd_per_mtok=2.50)
    usage = CellUsage(tokens_in=1_000_000, tokens_out=100_000)
    assert model.cost_usd(usage) == pytest.approx(0.30 + 0.25)
    assert model.basis() == "measured"


def test_per_token_cost_ignores_wall_clock() -> None:
    model = PerTokenCost(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0)
    fast = CellUsage(tokens_in=1000, wall_total_sec=1.0)
    slow = CellUsage(tokens_in=1000, wall_total_sec=1000.0)
    assert model.cost_usd(fast) == model.cost_usd(slow)


def test_hourly_cost_uses_wall_clock_without_throughput() -> None:
    model = HourlyCost(hourly_usd=25.0)
    usage = CellUsage(tokens_in=999_999, wall_total_sec=3600.0)
    assert model.cost_usd(usage) == pytest.approx(25.0)
    assert model.basis() == "measured"


def test_hourly_cost_with_throughput_is_derived() -> None:
    model = HourlyCost(hourly_usd=25.0, prefill_tok_per_s=20_000)
    usage = CellUsage(tokens_in=20_000 * 3600, wall_total_sec=0.0)
    assert model.cost_usd(usage) == pytest.approx(25.0)
    assert model.basis() == "derived"


def test_hourly_and_per_token_are_not_interchangeable() -> None:
    """A busy box and an idle one bill identically per token; an API does not."""
    hourly = HourlyCost(hourly_usd=25.0)
    per_token = PerTokenCost(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0)
    usage = CellUsage(tokens_in=1000, wall_total_sec=3600.0)
    assert hourly.cost_usd(usage) != pytest.approx(per_token.cost_usd(usage))


def test_video_hours_per_dollar() -> None:
    assert video_hours_per_dollar(3600.0, 1.0) == pytest.approx(1.0)
    assert video_hours_per_dollar(3600.0, 0.0) is None


def test_target_arithmetic_matches_the_stated_frontier() -> None:
    """1,024 tokens/frame at 20k tok/s on a $25/hr box: 1 fps is well under 1 vh/$."""
    model = HourlyCost(hourly_usd=25.0, prefill_tok_per_s=20_000)
    one_video_hour_at_1fps = CellUsage(tokens_in=3600 * 1024)
    vhpd = video_hours_per_dollar(3600.0, model.cost_usd(one_video_hour_at_1fps))
    assert vhpd == pytest.approx(0.78, abs=0.02)

    # And one frame every 128 s reaches 100 video-hours per dollar.
    sparse = CellUsage(tokens_in=int(3600 / 128 * 1024))
    assert video_hours_per_dollar(3600.0, model.cost_usd(sparse)) == pytest.approx(100, rel=0.01)


# ---------------------------------------------------------------------------
# Cost spec parsing
# ---------------------------------------------------------------------------

def test_parse_cost_model_empty_is_none() -> None:
    assert parse_cost_model("") is None
    assert parse_cost_model(None) is None


def test_parse_hourly_spec() -> None:
    model = parse_cost_model("hourly:25.0:20000:1200")
    assert isinstance(model, HourlyCost)
    assert model.hourly_usd == 25.0
    assert model.prefill_tok_per_s == 20_000
    assert model.decode_tok_per_s == 1200


def test_parse_token_spec() -> None:
    model = parse_cost_model("token:0.30:2.50")
    assert isinstance(model, PerTokenCost)
    assert model.output_usd_per_mtok == 2.50


def test_parse_cost_model_from_file(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"mode": "per_hour", "hourly_usd": 12.5, "prefill_tok_per_s": 5000}))
    model = parse_cost_model(f"@{path}")
    assert isinstance(model, HourlyCost)
    assert model.hourly_usd == 12.5


def test_parse_cost_model_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        parse_cost_model("guess:12")
    with pytest.raises(ValueError):
        parse_cost_model("hourly")


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------

def test_noise_floor_collects_spread() -> None:
    values = [0.7, 0.8, 0.75, 0.9, 0.7]
    spread = noise_floor(lambda i: values[i], repeats=5, unit=" score")
    assert spread.n == 5
    assert spread.range == pytest.approx(0.2)


def test_noise_floor_drops_failed_runs() -> None:
    spread = noise_floor(lambda i: None if i % 2 else 1.0, repeats=4)
    assert spread.n == 2


def test_delta_below_spread_is_called_noise() -> None:
    spread = Spread(values=[0.70, 0.74], unit=" score")
    assert "not a real difference" in interpret_delta(0.02, spread)
    assert "exceeds the noise floor" in interpret_delta(0.5, spread)


def test_saturated_spread_is_not_a_noise_floor() -> None:
    """A cell solved perfectly every time cannot report variance."""
    spread = Spread(values=[1.0, 1.0, 1.0])
    assert is_saturated(spread)
    assert "measures nothing" in interpret_delta(0.01, spread)


def test_unmeasured_spread_says_so() -> None:
    assert "unverified" in interpret_delta(0.5, Spread(values=[]))


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------

def test_collapse_point_finds_widest_safe_interval() -> None:
    rows = [
        {"interval_sec": 1.0, "score": 0.90},
        {"interval_sec": 5.0, "score": 0.85},
        {"interval_sec": 10.0, "score": 0.83},
        {"interval_sec": 30.0, "score": 0.40},
    ]
    result = collapse_point(rows, tolerance=0.1)
    assert result["safe_interval_sec"] == 10.0
    assert result["collapse_interval_sec"] == 30.0


def test_collapse_point_without_scores() -> None:
    assert collapse_point([{"interval_sec": 1.0}])["safe_interval_sec"] is None
