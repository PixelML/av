# av — Agentic Video CLI

**Index once, search many.** Video memory tool for agentic workflows by [Pixel ML](https://pixelml.com).

```
pip install pixelml-av
```

## Quick Start

```bash
# 1. Set up your provider (interactive wizard)
av config setup

# 2. Ingest a video (transcript + embeddings)
av ingest video.mp4

# 3. Search across all indexed videos
av search "what was discussed about pricing"

# 4. Ask questions with RAG (citations included)
av ask "what were the key decisions made?"

# 5. List indexed videos
av list

# 6. Get transcript as VTT subtitles
av transcript <video_id> --format vtt

# 7. Export all data as JSONL
av export --format jsonl
```

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
