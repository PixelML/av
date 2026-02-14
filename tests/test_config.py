"""Tests for config loading: env vars > config.json > defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from av.core.config import AVConfig, _load_config_file, get_config, save_config
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
    assert config.chat_model == "gpt-4-1"


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
    monkeypatch.setenv("AV_CHAT_MODEL", "my-custom-model")

    config = get_config()
    # Env var wins
    assert config.chat_model == "my-custom-model"
    # Config file value still applies for non-overridden fields
    assert config.provider == "anthropic"


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
