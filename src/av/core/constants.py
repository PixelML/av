"""Default model names and constants."""

from pathlib import Path

# Default models (OpenAI-compatible, GPT-4.1 family)
# Model names match LiteLLM / proxy conventions used in production
DEFAULT_TRANSCRIBE_MODEL = "whisper-1"
DEFAULT_VISION_MODEL = "gpt-4.1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4.1"

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
        "vision_model": "gpt-4.1",
        "embed_model": "text-embedding-3-small",
        "chat_model": "gpt-4.1",
    },
    "openai": {
        "api_base_url": "https://api.openai.com/v1",
        "transcribe_model": "whisper-1",
        "vision_model": "gpt-4.1",
        "embed_model": "text-embedding-3-small",
        "chat_model": "gpt-4.1",
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

# Cascade defaults
DEFAULT_CHUNK_DURATION_SEC = 30
DEFAULT_FRAMES_PER_CHUNK = 3

# Topic presets — prompt templates for cascade captioning.
# Placeholders: {start_sec}, {end_sec}, {chunk_duration}, {frames_per_chunk}, {custom_focus}
_GENERAL_TEMPLATE = (
    "You are analyzing {frames_per_chunk} frames sampled from a {chunk_duration}-second video chunk "
    "(timestamps {start_sec}s–{end_sec}s). "
    "Describe what HAPPENS across these frames — focus on temporal changes, actions, and events, "
    "not static scene descriptions. "
    "{custom_focus}"
    "Be specific about what changed between frames. "
    "If nothing meaningful happens, reply with exactly: STATIC"
)

TOPIC_PRESETS: dict[str, str] = {
    "general": _GENERAL_TEMPLATE.replace("{custom_focus}", ""),
    "security": _GENERAL_TEMPLATE.replace(
        "{custom_focus}",
        "Focus on: people entering/leaving, door/gate activity, suspicious behavior, "
        "unattended objects, unauthorized access, loitering. "
    ),
    "traffic": _GENERAL_TEMPLATE.replace(
        "{custom_focus}",
        "Focus on: vehicle movements, traffic violations, near-misses, collisions, "
        "pedestrian crossings, signal changes, lane changes, speeding. "
    ),
    "warehouse": _GENERAL_TEMPLATE.replace(
        "{custom_focus}",
        "Focus on: worker activity, PPE compliance (helmets, vests, gloves), "
        "forklift/equipment operation, spills, blocked exits, loading/unloading. "
    ),
    "retail": _GENERAL_TEMPLATE.replace(
        "{custom_focus}",
        "Focus on: customer flow, staff interactions, crowd density changes, "
        "checkout activity, shelf restocking, potential theft indicators. "
    ),
    "meeting": _GENERAL_TEMPLATE.replace(
        "{custom_focus}",
        "Focus on: speaker changes, slide transitions, audience reactions, "
        "whiteboard/screen content changes, gestures, Q&A activity. "
    ),
}

# Video extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",
    ".wmv", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts",
}
