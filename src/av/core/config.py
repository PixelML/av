"""Configuration via environment variables, config.json, and .env files."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from av.core.constants import (
    CONFIG_FILE_PATH,
    DEFAULT_CHAT_MODEL,
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_MODEL,
    DEFAULT_FPS_SAMPLE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_TRANSCRIBE_MODEL,
    DEFAULT_VISION_MODEL,
)


def _load_config_file() -> dict:
    """Read ~/.config/av/config.json if it exists, return as dict."""
    if not CONFIG_FILE_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE_PATH.read_text())
    except Exception:
        return {}


def save_config(data: dict) -> Path:
    """Write config dict to ~/.config/av/config.json. Returns the path."""
    CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE_PATH.write_text(json.dumps(data, indent=2) + "\n")
    return CONFIG_FILE_PATH


class AVConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider
    provider: str = Field(default="")

    # API
    api_base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # Models
    transcribe_model: str = Field(default=DEFAULT_TRANSCRIBE_MODEL)
    vision_model: str = Field(default=DEFAULT_VISION_MODEL)
    embed_model: str = Field(default=DEFAULT_EMBED_MODEL)
    chat_model: str = Field(default=DEFAULT_CHAT_MODEL)

    # Database
    db_path: Path = Field(default=DEFAULT_DB_PATH)

    # Ingest defaults
    fps_sample: float = Field(default=DEFAULT_FPS_SAMPLE)
    max_frames: int = Field(default=DEFAULT_MAX_FRAMES)


def get_config(db_path: Path | None = None) -> AVConfig:
    """Create config with priority: env vars > config.json > defaults."""
    file_data = _load_config_file()

    # Build init kwargs from config.json values, but skip keys where an env var is set
    # (env vars should always win, and pydantic treats __init__ kwargs as highest priority)
    init_kwargs: dict = {}
    for key in (
        "provider",
        "api_base_url",
        "api_key",
        "openai_api_key",
        "transcribe_model",
        "vision_model",
        "embed_model",
        "chat_model",
    ):
        env_name = f"AV_{key.upper()}"
        if key in file_data and env_name not in os.environ:
            init_kwargs[key] = file_data[key]

    config = AVConfig(**init_kwargs)

    # If openai_api_key not set explicitly, try OPENAI_API_KEY env var as fallback
    if not config.openai_api_key:
        config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    if db_path is not None:
        config.db_path = db_path
    return config


def get_openai_config(config: AVConfig) -> AVConfig | None:
    """Return an OpenAI-direct config for embeddings/transcription, or None if unavailable.

    When a non-OpenAI provider is active (e.g. PixelML, Anthropic) but an OpenAI key
    is available (explicit, env var, or Codex OAuth), this returns a config pointing at
    api.openai.com with standard model names.
    """
    # Already using OpenAI directly — no need for a separate config
    if config.provider in ("openai", "openai-oauth", ""):
        return None

    # Try explicit openai_api_key first
    key = (config.openai_api_key or "").strip()

    # Fallback: Codex OAuth tokens (same mechanism as _resolve_api_key in openai.py)
    if not key:
        from av.providers.openai import _codex_oauth_token, _openclaw_oauth_token
        key = _openclaw_oauth_token() or _codex_oauth_token() or ""

    if not key:
        return None

    return AVConfig(
        provider="openai",
        api_base_url="https://api.openai.com/v1",
        api_key=key,
        transcribe_model="whisper-1",
        embed_model="text-embedding-3-small",
        vision_model=config.vision_model,
        chat_model=config.chat_model,
    )
