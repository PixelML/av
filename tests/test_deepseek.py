"""Tests for the DeepSeek-V4.1-Flash provider and its image-token model."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from av.core.config import AVConfig
from av.core.constants import PROVIDER_PRESETS
from av.providers.deepseek import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MAX_IMAGE_TOKENS,
    MIN_IMAGE_TOKENS,
    PLACEHOLDER_KEY,
    CapabilityRecord,
    make_config,
    plan_image_tokens,
    resolve_api_key,
    resolve_base_url,
    tokens_for_grid,
    widths_for_token_budgets,
)
from av.providers.openai import _client, _resolve_api_key


# ---------------------------------------------------------------------------
# Preset and configuration
# ---------------------------------------------------------------------------

def test_preset_is_registered() -> None:
    preset = PROVIDER_PRESETS["deepseek"]
    assert preset["vision_model"] == DEFAULT_MODEL
    assert preset["transcribe_model"] == ""  # not served by this deployment
    assert preset["embed_model"] == ""


def test_preset_ships_no_private_endpoint() -> None:
    """A hosted endpoint must never be baked into source; only a local placeholder."""
    url = PROVIDER_PRESETS["deepseek"]["api_base_url"]
    assert url == DEFAULT_BASE_URL
    assert "localhost" in url


def test_base_url_prefers_env_then_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AVConfig(provider="deepseek", api_base_url="http://from-config:30000/v1")
    monkeypatch.delenv("AV_API_BASE_URL", raising=False)
    assert resolve_base_url(config) == "http://from-config:30000/v1"

    monkeypatch.setenv("AV_API_BASE_URL", "http://from-env:30000/v1")
    assert resolve_base_url(config) == "http://from-env:30000/v1"


def test_base_url_falls_back_to_the_local_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AV_API_BASE_URL", raising=False)
    assert resolve_base_url(None) == DEFAULT_BASE_URL


def test_api_key_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AV_API_KEY", "DEEPSEEK_API_KEY", "SGLANG_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert resolve_api_key(AVConfig(api_key="explicit")) == "explicit"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    assert resolve_api_key(AVConfig(api_key="")) == "from-env"
    assert resolve_api_key(AVConfig(api_key=PLACEHOLDER_KEY)) == "from-env"


def test_api_key_placeholder_when_server_needs_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AV_API_KEY", "DEEPSEEK_API_KEY", "SGLANG_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_api_key(AVConfig(api_key="")) == PLACEHOLDER_KEY


def test_openai_client_routes_deepseek_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared OpenAI-compatible client must pick up the provider's own env var."""
    monkeypatch.delenv("AV_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    config = AVConfig(provider="deepseek", api_key="")
    assert _resolve_api_key(config) == "sk-deepseek"


def test_make_config_disables_unserved_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AV_API_BASE_URL", raising=False)
    config = make_config(model="deepseek-v4.1-flash")
    assert config.provider == "deepseek"
    assert config.transcribe_model == ""
    assert config.embed_model == ""
    assert config.vision_model == config.chat_model == "deepseek-v4.1-flash"


def test_client_sends_no_anthropic_header() -> None:
    config = AVConfig(provider="deepseek", api_key="k", api_base_url=DEFAULT_BASE_URL)
    assert _client(config)._custom_headers.get("anthropic-version") is None


# ---------------------------------------------------------------------------
# Image token model
# ---------------------------------------------------------------------------

def test_token_formula() -> None:
    """One token per grid cell, one newline per row, plus start and end."""
    assert tokens_for_grid(13, 13) == 13 * 14 + 2 == 184


@pytest.mark.parametrize(
    "width,height,expected_tokens",
    [
        (512, 512, 184),      # below the pixel floor, upscaled
        (640, 480, 206),
        (800, 600, 317),
        (1024, 1024, 652),
        (1280, 720, 578),
        (100, 3000, 290),     # extreme aspect ratio
    ],
)
def test_plan_reproduces_published_examples(width: int, height: int, expected_tokens: int) -> None:
    assert plan_image_tokens(width, height).tokens == expected_tokens


def test_small_frames_hit_the_upscale_floor() -> None:
    """Shrinking past the floor buys nothing: a thumbnail costs what 544x544 costs."""
    tiny = plan_image_tokens(64, 64)
    small = plan_image_tokens(512, 512)
    assert tiny.upscaled and small.upscaled
    assert tiny.tokens == small.tokens == MIN_IMAGE_TOKENS


def test_huge_frames_are_capped_and_flagged_approximate() -> None:
    big = plan_image_tokens(5000, 5000)
    assert big.tokens <= MAX_IMAGE_TOKENS
    assert big.downscaled
    assert big.approximate  # a shrink search ran; measure it rather than trust it


def test_extra_resolution_past_the_ceiling_is_discarded() -> None:
    assert plan_image_tokens(2000, 2000).tokens == plan_image_tokens(5000, 5000).tokens


def test_tokens_rise_with_resolution_between_the_walls() -> None:
    counts = [plan_image_tokens(w, int(w * 9 / 16)).tokens for w in (768, 1024, 1280)]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_plan_rejects_degenerate_sizes() -> None:
    with pytest.raises(ValueError):
        plan_image_tokens(0, 100)


def test_widths_for_budgets_stay_within_budget() -> None:
    solved = widths_for_token_budgets([200, 400, 800])
    for budget, width in solved.items():
        height = max(int(round(width / (16 / 9))), 1)
        assert plan_image_tokens(width, height).tokens <= budget
    assert solved[200] < solved[400] < solved[800]


# ---------------------------------------------------------------------------
# Capability record
# ---------------------------------------------------------------------------

def test_capability_record_defaults_to_unestablished() -> None:
    """Nothing is assumed: a field we have not measured stays None."""
    record = CapabilityRecord()
    assert record.context_tokens is None
    assert record.image_tokens_per_frame is None
    assert record.multi_image_supported is None
    assert record.max_frames_per_request() is None


def test_max_frames_per_request_when_both_inputs_known() -> None:
    record = CapabilityRecord(context_tokens=1_048_576, image_tokens_per_frame=1024)
    frames = record.max_frames_per_request(prompt_overhead_tokens=0)
    assert frames == 1024
    # 1,024 frames at 1 fps is roughly 17 minutes in a single request.
    assert 16.5 < frames / 60 < 17.5
