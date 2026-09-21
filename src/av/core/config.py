"""Configuration via environment variables, config.json, and .env files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

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
    api_timeout_sec: float = Field(default=120.0, gt=0)
    api_max_retries: int = Field(default=1, ge=0, le=3)
    # Select one request field; provider semantics and enforcement can differ.
    api_token_limit_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    allow_oauth_fallback: bool = Field(default=False)
    allow_codex_fallback: bool = Field(default=False)

    # Models
    transcribe_model: str = Field(default=DEFAULT_TRANSCRIBE_MODEL)
    vision_model: str = Field(default=DEFAULT_VISION_MODEL)
    vision_max_output_tokens: int = Field(default=200, gt=0)
    vision_chunk_max_output_tokens: int = Field(default=500, gt=0)
    embed_model: str = Field(default=DEFAULT_EMBED_MODEL)
    chat_model: str = Field(default=DEFAULT_CHAT_MODEL)
    chat_max_output_tokens: int = Field(default=1024, gt=0)

    # Optional System One query refinement. A credential enables refinement by
    # default; callers can still opt out per request.
    typesafe_api_key: str = Field(default="")
    typesafe_endpoint: str = Field(default="https://api.typesafe.ai/v1/systemone")
    typesafe_model: str = Field(default="jev-latest")
    typesafe_timeout_sec: float = Field(default=30.0, gt=0)
    typesafe_max_retries: int = Field(default=1, ge=0, le=3)
    # Optional self-hosted djev-spark decision endpoint speaking the same
    # documented /v1/systemone contract. No default endpoint ships with av:
    # an explicit endpoint selects this lane over hosted System One.
    djev_endpoint: str = Field(default="")
    djev_api_key: str = Field(default="")
    # Advisory request model. djev-spark ignores it and reports the model it
    # actually served; responses carry that identity into usage records.
    djev_model: str = Field(default="")
    # Cold structured reads are slow on long states (upstream documents ~105 s
    # at 110k tokens), so the default is more generous than the hosted lane's.
    djev_timeout_sec: float = Field(default=180.0, gt=0)
    djev_max_retries: int = Field(default=1, ge=0, le=3)
    djev_seed: int = Field(default=42, ge=0)
    refine_enabled: bool = Field(default=True)
    refine_relevance_min: float = Field(default=0.5, ge=0, le=1)
    refine_support_min: float = Field(default=0.5, ge=0, le=1)
    refine_max_scenes: int = Field(default=8, ge=1, le=50)
    refine_batch_size: int = Field(default=10, ge=1, le=50)
    refine_context_events: int = Field(default=3, ge=0, le=12)

    # Optional stronger sampled-frame inspection after an unsupported answer.
    strong_vision_api_base_url: str = Field(default="")
    strong_vision_api_key: str = Field(default="")
    strong_vision_model: str = Field(default="")
    inspection_max_windows: int = Field(default=2, ge=0, le=8)
    inspection_max_seconds: float = Field(default=120.0, ge=0)
    inspection_max_frames: int = Field(default=12, ge=0, le=128)
    inspection_max_attempts: int = Field(default=1, ge=1, le=2)
    inspection_dense_pass: bool = Field(default=False)

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
        "api_timeout_sec",
        "api_max_retries",
        "api_token_limit_parameter",
        "allow_oauth_fallback",
        "allow_codex_fallback",
        "transcribe_model",
        "vision_model",
        "vision_max_output_tokens",
        "vision_chunk_max_output_tokens",
        "embed_model",
        "chat_model",
        "chat_max_output_tokens",
        "typesafe_api_key",
        "typesafe_endpoint",
        "typesafe_model",
        "typesafe_timeout_sec",
        "djev_endpoint",
        "djev_api_key",
        "djev_model",
        "djev_timeout_sec",
        "djev_max_retries",
        "djev_seed",
        "typesafe_max_retries",
        "refine_enabled",
        "refine_relevance_min",
        "refine_support_min",
        "refine_max_scenes",
        "refine_batch_size",
        "refine_context_events",
        "strong_vision_api_base_url",
        "strong_vision_api_key",
        "strong_vision_model",
        "inspection_max_windows",
        "inspection_max_seconds",
        "inspection_max_frames",
        "inspection_max_attempts",
        "inspection_dense_pass",
    ):
        env_name = f"AV_{key.upper()}"
        alias_is_set = key == "typesafe_api_key" and "TYPESAFE_API_KEY" in os.environ
        if key in file_data and env_name not in os.environ and not alias_is_set:
            init_kwargs[key] = file_data[key]

    config = AVConfig(**init_kwargs)

    # If openai_api_key not set explicitly, try OPENAI_API_KEY env var as fallback
    if not config.openai_api_key:
        config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    if "AV_TYPESAFE_API_KEY" not in os.environ:
        config.typesafe_api_key = os.environ.get("TYPESAFE_API_KEY", config.typesafe_api_key)
    if "AV_TYPESAFE_MODEL" not in os.environ:
        config.typesafe_model = os.environ.get("TYPESAFE_DEFAULT_MODEL", config.typesafe_model)

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
    if not key and config.allow_oauth_fallback:
        from av.providers.openai import _codex_oauth_token, _openclaw_oauth_token
        key = _openclaw_oauth_token() or _codex_oauth_token() or ""

    if not key:
        return None

    return AVConfig(
        provider="openai",
        api_base_url="https://api.openai.com/v1",
        api_key=key,
        api_timeout_sec=config.api_timeout_sec,
        api_max_retries=config.api_max_retries,
        api_token_limit_parameter=config.api_token_limit_parameter,
        allow_oauth_fallback=False,
        allow_codex_fallback=config.allow_codex_fallback,
        transcribe_model="whisper-1",
        embed_model="text-embedding-3-small",
        vision_model=config.vision_model,
        vision_max_output_tokens=config.vision_max_output_tokens,
        vision_chunk_max_output_tokens=config.vision_chunk_max_output_tokens,
        chat_model=config.chat_model,
        chat_max_output_tokens=config.chat_max_output_tokens,
    )
