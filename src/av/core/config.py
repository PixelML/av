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
        "transcribe_model",
        "vision_model",
        "embed_model",
        "chat_model",
    ):
        env_name = f"AV_{key.upper()}"
        if key in file_data and env_name not in os.environ:
            init_kwargs[key] = file_data[key]

    config = AVConfig(**init_kwargs)

    if db_path is not None:
        config.db_path = db_path
    return config
