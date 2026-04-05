# av — Agentic Video Intelligence

**Index. Search. Detect.** Video intelligence toolkit for AI agents by [Pixel ML](https://pixelml.com).

```
pip install pixelml-av
```

## What av Does

**Video Memory** — Ingest videos, search by natural language, ask questions with RAG citations.

**Surveillance Intelligence** — Detect falls, long queues, crowd gathering, and wheelchair compliance in CCTV footage using temporal reasoning.

## Quick Start

### Video Search

```bash
# 1. Set up your provider
av config setup

# 2. Ingest a video
av ingest video.mp4

# 3. Search
av search "person with red bag"

# 4. Ask questions
av ask "what happened at 2:30?"
```

### Surveillance Detection

```bash
# Cloud (quick start — Gemini free tier)
export AV_API_KEY=your-gemini-key
av sentinel video.mp4

# Local (free, private — runs on your Mac/GPU)
ollama pull mistral-small3.2
av sentinel video.mp4 --provider ollama

# Specific alerts
av sentinel video.mp4 --alerts FALL,LONG_QUEUE

# Batch a directory
av sentinel videos/ --camera cam_lobby
```

### All Commands

```bash
# Video memory
av ingest video.mp4             # Index video content
av search "what was discussed"  # Semantic search
av ask "key decisions?"         # RAG Q&A with citations
av list                         # List indexed videos
av transcript <id> --format vtt # Get transcript
av export --format jsonl        # Export all data
av export --format jsonl

# Surveillance intelligence
av sentinel video.mp4              # Detect events (all 4 alert types)
av sentinel video.mp4 --alerts FALL # Fall detection only
av sentinel video.mp4 -p ollama    # Self-hosted (free)
av sentinel videos/ -c cam_lobby   # Batch with camera tracking
```

## Sentinel — Surveillance Event Detection

Detects 4 event types using temporal reasoning over VLM observations:

| Alert | Detection | How It Works |
|-------|-----------|-------------|
| **FALL** | Position tracking | `standing→lying` transition across frames (F1=0.944) |
| **LONG_QUEUE** | Temporal persistence | Queue detected in 3+ consecutive chunks (90s) |
| **CROWD_GATHERING** | Density + growth | Sustained crowd or rapid person count increase |
| **WHEELCHAIR_COMPLIANCE** | Service timing | Wheelchair user unattended > threshold |

### Providers for Sentinel

| Provider | Setup | Cost | Speed |
|----------|-------|------|-------|
| **Gemini** (cloud) | `export AV_API_KEY=key` | Free tier available | ~5s/chunk |
| **OpenRouter** | `export OPENROUTER_API_KEY=key` | $0.04-0.14/1M tokens | ~10s/chunk |
| **Ollama** (local) | `ollama pull mistral-small3.2` | Free | ~25s/chunk |
| **OpenAI** | `export AV_API_KEY=key` | $$$ | ~5s/chunk |

Auto-detection: if no provider specified, av tries Gemini → OpenRouter → ollama → OpenAI.

### How It Works

```
Video → 30s chunks (5s overlap)
  → 8 frames per chunk
  → VLM perception (positions, queue, crowd, wheelchair)
  → Temporal agent (state across chunks)
  → Alert rules (transition detection, persistence, growth)
  → JSON output
```

Built on 107 experiments across 21 vision models. Key insight: structural extraction + temporal rules beats generic "detect anomalies" prompts.

## Configuration

### Interactive Setup (Recommended)

```bash
av config setup
```

Choose from four providers:

| # | Provider | Auth | Transcription | Embeddings |
|---|----------|------|---------------|------------|
| 1 | **OpenAI (Codex OAuth)** | Auto-detected | Whisper | text-embedding-3-small |
| 2 | **OpenAI (API key)** | `sk-...` key | Whisper | text-embedding-3-small |
| 3 | **Anthropic (Claude)** | API key | Not supported | Not supported |
| 4 | **Google (Gemini)** | API key | Not supported | text-embedding-004 |

Config is saved to `~/.config/av/config.json` and persists across sessions.

**Note:** Anthropic and Gemini don't support Whisper transcription. With these providers, use `av ingest --captions` for frame-based captioning, or set `AV_OPENAI_API_KEY` for transcription fallback.

### Environment Variables

Env vars always override config.json:

```bash
export AV_API_KEY="sk-..."
export AV_API_BASE_URL="https://api.openai.com/v1"  # or any OpenAI-compatible endpoint
export AV_TRANSCRIBE_MODEL="whisper"
export AV_VISION_MODEL="gpt-4-1"
export AV_EMBED_MODEL="text-embedding-3-small"
export AV_CHAT_MODEL="gpt-4-1"
```

## Requirements

- Python 3.11+
- FFmpeg (`brew install ffmpeg`)
- An API key from OpenAI, Anthropic, or Google — or Codex CLI OAuth

## Commands

| Command | Description |
|---------|-------------|
| `av config setup` | Interactive provider setup wizard |
| `av config show` | Show current configuration |
| `av ingest <path>` | Ingest video file(s) into the index |
| `av search <query>` | Full-text + semantic search |
| `av ask <question>` | RAG Q&A with citations |
| `av list` | List all indexed videos |
| `av info <video_id>` | Detailed video metadata |
| `av transcript <id>` | Output transcript (VTT/SRT/text) |
| `av export` | Export as JSONL/VTT/SRT |
| `av open <id> --at <sec>` | Open video at timestamp |
| `av version` | Print version JSON |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
