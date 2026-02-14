"""Tests for provider-aware OpenAI client."""

from __future__ import annotations

from unittest.mock import patch

from av.core.config import AVConfig
from av.providers.openai import _client, _resolve_api_key


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
    config = AVConfig(api_key="")
    with patch("av.providers.openai._openclaw_oauth_token", return_value="oauth-token-123"), \
         patch("av.providers.openai._codex_oauth_token", return_value=None):
        assert _resolve_api_key(config) == "oauth-token-123"
