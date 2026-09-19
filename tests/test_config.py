"""Tests for config loading: env vars > config.json > defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from av.cli.config_cmd import config_show
from av.core.config import AVConfig, _load_config_file, get_config, get_openai_config, save_config
from av.core.constants import CONFIG_FILE_PATH, PROVIDER_PRESETS


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_config = tmp_path / "config.json"
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", fake_config)

    data = {"provider": "gemini", "api_key": "test-key-123"}
    result = save_config(data)
    assert result == fake_config
    assert fake_config.exists()

    loaded = json.loads(fake_config.read_text())
    assert loaded["provider"] == "gemini"
    assert loaded["api_key"] == "test-key-123"


def test_load_config_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", tmp_path / "nope.json")
    assert _load_config_file() == {}


def test_load_config_file_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "config.json"
    bad.write_text("not json {{{")
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", bad)
    assert _load_config_file() == {}


# ---------------------------------------------------------------------------
# Priority: env vars > config.json > defaults
# ---------------------------------------------------------------------------

def test_defaults_without_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", tmp_path / "nope.json")
    # Clear any AV_ env vars that might interfere
    for key in list(os.environ):
        if key.startswith("AV_"):
            monkeypatch.delenv(key, raising=False)

    config = get_config()
    assert config.provider == ""
    assert config.api_base_url == "https://api.openai.com/v1"
    assert config.transcribe_model == "whisper-1"
    assert config.chat_model == "gpt-4.1"


def test_config_file_overrides_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "provider": "anthropic",
        "api_base_url": "https://api.anthropic.com/v1/",
        "chat_model": "claude-sonnet-4-5-20250929",
        "transcribe_model": "",
    }))
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", cfg_file)
    for key in list(os.environ):
        if key.startswith("AV_"):
            monkeypatch.delenv(key, raising=False)

    config = get_config()
    assert config.provider == "anthropic"
    assert config.api_base_url == "https://api.anthropic.com/v1/"
    assert config.chat_model == "claude-sonnet-4-5-20250929"
    assert config.transcribe_model == ""


def test_env_var_overrides_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "provider": "anthropic",
        "chat_model": "claude-sonnet-4-5-20250929",
    }))
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", cfg_file)
    # Clear any stale AV_ env vars, then set the one we want to test
    for key in list(os.environ):
        if key.startswith("AV_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AV_CHAT_MODEL", "my-custom-model")

    config = get_config()
    # Env var wins
    assert config.chat_model == "my-custom-model"
    # Config file value still applies for non-overridden fields
    assert config.provider == "anthropic"


def test_chat_output_cap_loads_from_config_and_env_with_positive_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"chat_max_output_tokens": 700}))
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", cfg_file)
    monkeypatch.delenv("AV_CHAT_MAX_OUTPUT_TOKENS", raising=False)
    assert get_config().chat_max_output_tokens == 700

    monkeypatch.setenv("AV_CHAT_MAX_OUTPUT_TOKENS", "900")
    assert get_config().chat_max_output_tokens == 900

    with pytest.raises(ValueError):
        AVConfig(chat_max_output_tokens=0)


def test_provider_token_limit_settings_load_from_config_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "api_token_limit_parameter": "max_completion_tokens",
        "vision_max_output_tokens": 32,
        "vision_chunk_max_output_tokens": 64,
    }))
    monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", cfg_file)
    for name in (
        "AV_API_TOKEN_LIMIT_PARAMETER",
        "AV_VISION_MAX_OUTPUT_TOKENS",
        "AV_VISION_CHUNK_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = get_config()
    assert config.api_token_limit_parameter == "max_completion_tokens"
    assert config.vision_max_output_tokens == 32
    assert config.vision_chunk_max_output_tokens == 64

    monkeypatch.setenv("AV_API_TOKEN_LIMIT_PARAMETER", "max_tokens")
    monkeypatch.setenv("AV_VISION_MAX_OUTPUT_TOKENS", "48")
    monkeypatch.setenv("AV_VISION_CHUNK_MAX_OUTPUT_TOKENS", "80")
    config = get_config()
    assert config.api_token_limit_parameter == "max_tokens"
    assert config.vision_max_output_tokens == 48
    assert config.vision_chunk_max_output_tokens == 80


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("api_token_limit_parameter", "unsupported"),
        ("vision_max_output_tokens", 0),
        ("vision_chunk_max_output_tokens", 0),
    ],
)
def test_provider_token_limit_settings_reject_invalid_values(setting: str, value: object) -> None:
    with pytest.raises(ValueError):
        AVConfig(**{setting: value})


def test_config_show_includes_chat_output_cap() -> None:
    config = AVConfig(
        api_token_limit_parameter="max_completion_tokens",
        vision_max_output_tokens=32,
        vision_chunk_max_output_tokens=64,
        chat_max_output_tokens=777,
    )
    with patch("av.cli.config_cmd.get_config", return_value=config), \
         patch("av.cli.config_cmd.output_json") as output:
        config_show()
    shown = output.call_args.args[0]
    assert shown["api_token_limit_parameter"] == "max_completion_tokens"
    assert shown["vision_max_output_tokens"] == 32
    assert shown["vision_chunk_max_output_tokens"] == 64
    assert shown["chat_max_output_tokens"] == 777


def test_openai_fallback_does_not_read_oauth_unless_enabled() -> None:
    config = AVConfig(provider="anthropic", openai_api_key="", allow_oauth_fallback=False)
    with patch("av.providers.openai._openclaw_oauth_token") as openclaw, \
         patch("av.providers.openai._codex_oauth_token") as codex:
        assert get_openai_config(config) is None
    openclaw.assert_not_called()
    codex.assert_not_called()


def test_openai_fallback_can_use_oauth_only_when_explicitly_enabled() -> None:
    config = AVConfig(
        provider="anthropic",
        openai_api_key="",
        allow_oauth_fallback=True,
        api_token_limit_parameter="max_completion_tokens",
        vision_max_output_tokens=32,
        vision_chunk_max_output_tokens=64,
        chat_max_output_tokens=96,
    )
    with patch("av.providers.openai._openclaw_oauth_token", return_value="oauth-explicit"), \
         patch("av.providers.openai._codex_oauth_token") as codex:
        fallback = get_openai_config(config)
    assert fallback is not None
    assert fallback.api_key == "oauth-explicit"
    assert fallback.api_token_limit_parameter == "max_completion_tokens"
    assert fallback.vision_max_output_tokens == 32
    assert fallback.vision_chunk_max_output_tokens == 64
    assert fallback.chat_max_output_tokens == 96
    codex.assert_not_called()


def test_db_path_override() -> None:
    custom = Path("/tmp/test.db")
    config = get_config(db_path=custom)
    assert config.db_path == custom


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

def test_all_provider_presets_have_required_keys() -> None:
    required = {"api_base_url", "transcribe_model", "vision_model", "embed_model", "chat_model"}
    for name, preset in PROVIDER_PRESETS.items():
        assert required.issubset(preset.keys()), f"Preset {name!r} missing keys: {required - preset.keys()}"


def test_openai_presets_have_transcription() -> None:
    assert PROVIDER_PRESETS["openai"]["transcribe_model"] == "whisper-1"
    assert PROVIDER_PRESETS["openai-oauth"]["transcribe_model"] == "whisper-1"


def test_anthropic_preset_disables_transcription_and_embed() -> None:
    p = PROVIDER_PRESETS["anthropic"]
    assert p["transcribe_model"] == ""
    assert p["embed_model"] == ""


def test_gemini_preset_disables_transcription() -> None:
    p = PROVIDER_PRESETS["gemini"]
    assert p["transcribe_model"] == ""
    assert p["embed_model"] == "text-embedding-004"  # Gemini supports embeddings
