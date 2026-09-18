"""Tests for provider-aware OpenAI client."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from av.core.config import AVConfig
from av.providers.openai import OpenAICaptioner, _client, _resolve_api_key


def test_client_default_no_extra_headers() -> None:
    config = AVConfig(provider="openai", api_key="sk-test", api_base_url="https://api.openai.com/v1")
    client = _client(config)
    # No anthropic-version header for OpenAI
    assert client._custom_headers.get("anthropic-version") is None


def test_client_anthropic_has_version_header() -> None:
    config = AVConfig(provider="anthropic", api_key="sk-ant-test", api_base_url="https://api.anthropic.com/v1/")
    client = _client(config)
    assert client._custom_headers.get("anthropic-version") == "2023-06-01"


def test_client_gemini_no_extra_headers() -> None:
    config = AVConfig(provider="gemini", api_key="AIza-test", api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    client = _client(config)
    assert client._custom_headers.get("anthropic-version") is None


def test_resolve_api_key_explicit() -> None:
    config = AVConfig(api_key="sk-my-real-key")
    assert _resolve_api_key(config) == "sk-my-real-key"


def test_resolve_api_key_no_key_placeholder() -> None:
    config = AVConfig(api_key="no-key")
    # "no-key" is treated as placeholder, falls through to oauth
    with patch("av.providers.openai._openclaw_oauth_token", return_value=None), \
         patch("av.providers.openai._codex_oauth_token", return_value=None):
        assert _resolve_api_key(config) == "no-key"


def test_resolve_api_key_oauth_fallback() -> None:
    config = AVConfig(api_key="", allow_oauth_fallback=True)
    with patch("av.providers.openai._openclaw_oauth_token", return_value="oauth-token-123"), \
         patch("av.providers.openai._codex_oauth_token", return_value=None):
        assert _resolve_api_key(config) == "oauth-token-123"


def test_resolve_api_key_default_does_not_read_oauth() -> None:
    config = AVConfig(api_key="", allow_oauth_fallback=False)
    with patch("av.providers.openai._openclaw_oauth_token") as openclaw, \
         patch("av.providers.openai._codex_oauth_token") as codex:
        assert _resolve_api_key(config) == "no-key"
    openclaw.assert_not_called()
    codex.assert_not_called()


def test_client_disables_sdk_retries_and_uses_configured_timeout() -> None:
    config = AVConfig(
        api_key="explicit",
        api_timeout_sec=17.0,
        api_max_retries=2,
    )
    client = _client(config)
    assert client.max_retries == 0
    assert client.timeout == 17.0


def test_caption_fallback_is_disabled_by_default(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("model_not_found")
    config = AVConfig(
        api_key="explicit",
        api_max_retries=0,
        allow_codex_fallback=False,
    )
    with patch("av.providers.openai._client", return_value=fake_client), \
         patch("av.providers.openai._codex_cli_caption") as codex:
        captioner = OpenAICaptioner(config)
        assert captioner.caption_frames([frame], [0.0]) == []
        with pytest.raises(RuntimeError, match="model_not_found"):
            captioner.caption_chunk([frame], [0.0], "describe")
    codex.assert_not_called()
    assert captioner.usage.snapshot()["requests"] == 2


def test_caption_retries_are_bounded_and_usage_marks_failed_attempt_unknown(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="caption"))],
        usage=usage,
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [RuntimeError("temporary"), response]
    config = AVConfig(api_key="explicit", api_max_retries=1)
    with patch("av.providers.openai._client", return_value=fake_client), \
         patch("av.providers.openai.time.sleep"):
        captioner = OpenAICaptioner(config)
        assert captioner.caption_chunk([frame], [0.0], "describe") == "caption"
    receipt = captioner.usage.snapshot()
    assert fake_client.chat.completions.create.call_count == 2
    assert receipt["requests"] == 2
    assert receipt["failed_requests"] == 1
    assert receipt["successful_requests"] == 1
    assert receipt["input_tokens"] is None
    assert receipt["input_tokens_complete"] is False
    assert receipt["cached_input_tokens"] is None
