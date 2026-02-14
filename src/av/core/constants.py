"""Default model names and constants."""

from pathlib import Path

# Default models (OpenAI-compatible, GPT-4.1 family)
# Model names match LiteLLM / proxy conventions used in production
DEFAULT_TRANSCRIBE_MODEL = "whisper-1"
DEFAULT_VISION_MODEL = "gpt-4-1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4-1"

# Paths
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "av"
DEFAULT_DB_PATH = DEFAULT_CONFIG_DIR / "av.db"

# Ingest defaults
DEFAULT_FPS_SAMPLE = 0.5
DEFAULT_MAX_FRAMES = 200
MAX_AUDIO_CHUNK_BYTES = 25 * 1024 * 1024  # 25MB Whisper API limit
LONG_VIDEO_WARN_MINUTES = 60

# Hash
HASH_PREFIX_BYTES = 64 * 1024  # First 64KB for fast hashing

# Search
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_TOP_K = 10

# Config file
CONFIG_FILE_PATH = DEFAULT_CONFIG_DIR / "config.json"

# Provider presets
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai-oauth": {
        "api_base_url": "https://api.openai.com/v1",
        "transcribe_model": "whisper-1",
        "vision_model": "gpt-4-1",
        "embed_model": "text-embedding-3-small",
        "chat_model": "gpt-4-1",
    },
    "openai": {
        "api_base_url": "https://api.openai.com/v1",
        "transcribe_model": "whisper-1",
        "vision_model": "gpt-4-1",
        "embed_model": "text-embedding-3-small",
        "chat_model": "gpt-4-1",
    },
    "anthropic": {
        "api_base_url": "https://api.anthropic.com/v1/",
        "transcribe_model": "",
        "vision_model": "claude-sonnet-4-5-20250929",
        "embed_model": "",
        "chat_model": "claude-sonnet-4-5-20250929",
    },
    "gemini": {
        "api_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "transcribe_model": "",
        "vision_model": "gemini-2.5-flash",
        "embed_model": "text-embedding-004",
        "chat_model": "gemini-2.5-flash",
    },
}

# Video extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",
    ".wmv", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts",
}
