---
name: agentic-video-memory
version: 0.1.0
description: Video memory for AI agents. Ingest, search, and ask questions over video content.
homepage: https://agentic.video
---

# Agentic Video Memory

You are setting up `av` — a CLI that turns video files into searchable, queryable memory. Index once, search many times. What `jq` is for JSON, but for video.

## Quick Start

Run these three commands:

```bash
pip install pixelml-av
av config setup
av ingest <video-file-or-url>
```

After ingestion, you can search and ask questions:

```bash
av search "what was discussed about pricing"
av ask "what were the key decisions?"
```

## Requirements

- **Python 3.11+**
- **ffmpeg** — must be installed and on PATH (`brew install ffmpeg` / `apt install ffmpeg`)
- **API key** — one of: OpenAI, Anthropic (Claude), or Google (Gemini)

## Configuration

### Interactive Setup

Run `av config setup` to launch the interactive wizard. It will ask you to:

1. Choose a provider (OpenAI, Anthropic, Gemini)
2. Enter your API key
3. Validate the key works

Configuration is saved to `~/.config/av/config.json`.

### Environment Variable Overrides

You can override any config value with environment variables. These take priority over `config.json`.

| Variable | Default | Description |
|----------|---------|-------------|
| `AV_PROVIDER` | (none) | Provider name: `openai`, `anthropic`, `gemini` |
| `AV_API_KEY` | (none) | API key (overrides config.json) |
| `AV_API_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `AV_TRANSCRIBE_MODEL` | `whisper-1` | Transcription model |
| `AV_VISION_MODEL` | `gpt-4-1` | Vision/caption model |
| `AV_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `AV_CHAT_MODEL` | `gpt-4-1` | Chat/RAG model |
| `AV_DB_PATH` | `~/.config/av/av.db` | Database file location |

### Provider Capabilities

| Provider | Transcription | Vision / Chat | Embeddings |
|----------|--------------|---------------|------------|
| OpenAI | whisper-1 | gpt-4-1 | text-embedding-3-small |
| Anthropic | — | claude-sonnet-4-5 | — |
| Gemini | — | gemini-2.5-flash | text-embedding-004 |

When a capability is unavailable (e.g., Anthropic has no transcription), the pipeline skips that stage and warns. Use `AV_OPENAI_API_KEY` as a transcription fallback for non-OpenAI providers.

## Commands

### `av ingest <path>`

Ingest a video file, directory, or YouTube URL into the index.

```bash
# Single file
av ingest meeting.mp4

# With frame captions (vision model describes video frames)
av ingest meeting.mp4 --captions

# Dense visual captioning timeline
av ingest meeting.mp4 --dense-vision

# YouTube URL
av ingest "https://youtu.be/dQw4w9WgXcQ"

# Batch — ingest all videos in a directory
av ingest /path/to/videos/

# Skip embedding generation (transcript only)
av ingest meeting.mp4 --no-embed

# Re-ingest even if file hash matches
av ingest meeting.mp4 --force

# Preview what would happen without executing
av ingest meeting.mp4 --dry-run
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--captions` | Enable frame captioning via vision model |
| `--dense-vision` | Enable dense visual captioning timeline |
| `--fps-sample FLOAT` | Frames per second to sample (default: 0.5) |
| `--max-frames INT` | Maximum frames to caption (default: 200) |
| `--no-embed` | Skip embedding generation |
| `--force` | Re-ingest even if file hash matches existing record |
| `--dry-run` | Show what would happen without executing |
| `--principles PATH` | Path to custom principles file (.yaml/.json/.txt) |
| `--dense-output-dir PATH` | Directory for dense caption outputs |
| `--db PATH` | Override database path |

**Output (stdout, JSON):**

```json
{
  "status": "complete",
  "video_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "meeting.mp4",
  "duration_sec": 3600.0,
  "artifacts_count": 847,
  "elapsed_sec": 45.2
}
```

**On partial failure:**

```json
{
  "status": "complete_with_warnings",
  "video_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "meeting.mp4",
  "duration_sec": 3600.0,
  "artifacts_count": 320,
  "elapsed_sec": 30.1,
  "warnings": ["Transcription skipped: provider does not support whisper"]
}
```

### `av search <query>`

Full-text + semantic search across all indexed videos.

```bash
av search "what was discussed about pricing"
av search "action items" --limit 5
av search "deployment plan" --video-id a1b2c3d4
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--limit INT`, `-n INT` | Maximum results (default: 10) |
| `--video-id STR`, `-v STR` | Restrict search to a specific video |
| `--db PATH` | Override database path |

**Output (stdout, JSON):**

```json
{
  "query": "what was discussed about pricing",
  "results": [
    {
      "rank": 1,
      "score": 0.87,
      "video_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "filename": "meeting.mp4",
      "timestamp_sec": 1455.0,
      "timestamp_formatted": "00:24:15",
      "source_type": "transcript",
      "text": "We agreed on the $49/mo tier for the starter plan...",
      "artifact_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890"
    }
  ],
  "total_results": 5,
  "search_time_ms": 12
}
```

### `av ask <question>`

RAG question-answering over your indexed videos. Returns an answer with timestamped citations.

```bash
av ask "what were the key decisions?"
av ask "summarize the meeting" --video-id a1b2c3d4
av ask "what did they say about the budget?" --top-k 20
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--video-id STR`, `-v STR` | Restrict to a specific video |
| `--top-k INT`, `-k INT` | Number of context chunks for RAG (default: 10) |
| `--db PATH` | Override database path |

**Output (stdout, JSON):**

```json
{
  "answer": "Three key decisions were made in the meeting: 1) Launch the starter plan at $49/mo, 2) Push the release to Q2, 3) Hire two more engineers for the platform team.",
  "citations": [
    {
      "video_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "start_sec": 1455.0,
      "source_type": "transcript",
      "text": "We agreed on the $49/mo tier for the starter plan...",
      "score": 0.91
    }
  ],
  "confidence": 0.85
}
```

### `av list`

List all indexed videos.

```bash
av list
```

**Output (stdout, JSON):**

```json
{
  "videos": [
    {
      "video_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "filename": "meeting.mp4",
      "duration_formatted": "01:00:00",
      "status": "complete",
      "artifacts_count": 847
    }
  ],
  "total": 1
}
```

### `av info <video_id>`

Show detailed metadata for a specific video.

```bash
av info a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Output (stdout, JSON):**

```json
{
  "video_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "meeting.mp4",
  "file_path": "/Users/you/Videos/meeting.mp4",
  "duration_sec": 3600.0,
  "duration_formatted": "01:00:00",
  "resolution": "1920x1080",
  "status": "complete",
  "artifacts": {
    "transcript": 320,
    "caption": 42
  },
  "ingested_at": "2026-02-14T10:30:00Z"
}
```

### `av transcript <video_id>`

Output the transcript for a video.

```bash
av transcript a1b2c3d4 --format vtt
av transcript a1b2c3d4 --format srt
av transcript a1b2c3d4 --format text
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--format STR`, `-f STR` | Output format: `vtt`, `srt`, or `text` (default: `vtt`) |
| `--db PATH` | Override database path |

**Output (stdout, text — not JSON):**

```
WEBVTT

00:00:00.000 --> 00:00:05.500
Welcome to the Q1 planning meeting

00:00:05.500 --> 00:00:10.000
Today we'll discuss the product roadmap
```

### `av export`

Export all video memory as JSONL or subtitle formats.

```bash
av export
av export --format jsonl
av export --format srt --video-id a1b2c3d4
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--format STR`, `-f STR` | Export format: `jsonl`, `srt`, or `vtt` (default: `jsonl`) |
| `--video-id STR`, `-v STR` | Export a specific video only |
| `--db PATH` | Override database path |

**Output (stdout, JSONL — one JSON object per line):**

```jsonl
{"video_id": "a1b2c3d4", "filename": "meeting.mp4", "type": "transcript", "start_sec": 0.0, "end_sec": 5.5, "timestamp_formatted": "00:00:00", "text": "Welcome to the Q1 planning meeting"}
{"video_id": "a1b2c3d4", "filename": "meeting.mp4", "type": "transcript", "start_sec": 5.5, "end_sec": 10.0, "timestamp_formatted": "00:00:05", "text": "Today we'll discuss the product roadmap"}
```

### `av open <video_id>`

Open a video file at a specific timestamp.

```bash
av open a1b2c3d4
av open a1b2c3d4 --at 90.5
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--at FLOAT` | Timestamp in seconds to seek to (default: 0.0) |
| `--db PATH` | Override database path |

### `av config setup`

Interactive wizard to configure your AI provider and API key. No flags — fully interactive.

### `av config show`

Show current configuration.

**Output (stdout, JSON):**

```json
{
  "provider": "openai",
  "api_base_url": "https://api.openai.com/v1",
  "api_key": "***",
  "transcribe_model": "whisper-1",
  "vision_model": "gpt-4-1",
  "embed_model": "text-embedding-3-small",
  "chat_model": "gpt-4-1",
  "db_path": "~/.config/av/av.db"
}
```

### `av config path`

Show the path to the database file.

**Output (stdout, plain text):**

```
/Users/you/.config/av/av.db
```

### `av version`

Print version info.

**Output (stdout, JSON):**

```json
{"version": "0.1.0", "package": "pixelml-av"}
```

## Agent Integration Patterns

### JSON stdout / stderr separation

All data-producing commands output valid JSON to stdout. Progress messages, warnings, and errors go to stderr. This means you can safely parse stdout without filtering out log noise.

```bash
# Capture just the JSON output
result=$(av search "pricing" 2>/dev/null)

# Or pipe directly to jq
av search "pricing" 2>/dev/null | jq '.results[0].text'
```

### Batch directory ingestion

Point `av ingest` at a directory to ingest all video files:

```bash
av ingest /path/to/recordings/
```

Supported video extensions: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.mpg`, `.mpeg`, `.3gp`, `.ts`

### Chaining search and ask

Use search to find relevant videos, then ask questions about specific ones:

```bash
# Find which video mentions pricing
av search "pricing discussion" 2>/dev/null | jq -r '.results[0].video_id'

# Ask a question about that specific video
av ask "what was the final price agreed?" --video-id <video_id_from_above>
```

### Piping to other tools

```bash
# Export transcript and pipe to another tool
av transcript <video_id> --format text | wc -w

# Export all data as JSONL for processing
av export --format jsonl > all_videos.jsonl
```

## Enterprise

Pixel ML offers managed video intelligence infrastructure for teams that need:

- Custom model hosting and fine-tuning
- SLA guarantees and priority support
- On-premise deployment
- High-volume ingestion pipelines

Contact **hello@pixelml.com** for enterprise pricing and setup.

## Troubleshooting

### `ffmpeg: command not found`

Install ffmpeg:
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

### `av: command not found` after pip install

Make sure your Python scripts directory is on PATH. Try:
```bash
python -m av --help
```

Or install with pipx:
```bash
pipx install pixelml-av
```

### API key errors / authentication failures

Run `av config setup` again to re-enter your API key. Check that:
- The key is valid and not expired
- The key has access to the models listed in the provider table above
- You're using the correct provider

### Transcription not working with Anthropic or Gemini

These providers don't have a Whisper-equivalent transcription API. The pipeline will skip transcription and warn. To get transcripts with a non-OpenAI provider, set an OpenAI API key as a fallback:

```bash
export AV_OPENAI_API_KEY=sk-...
```

### Embeddings not working with Anthropic

Anthropic does not offer an embeddings API. The pipeline will skip embedding generation. FTS5 full-text search will still work. For semantic search, use OpenAI or Gemini as your provider.

### If you're stuck

Tell your human owner to email **hello@pixelml.com** with details about the issue. We'll help.

## Reliability Policy

- **Best-effort pipeline**: If a stage fails (transcription, captioning, embedding), the pipeline continues with what's available and warns. Partial results are better than no results.
- **Idempotent ingestion**: Running `av ingest` on the same file twice is a no-op (matched by file hash). Use `--force` to re-ingest.
- **Skip and warn**: When a provider capability is unavailable, the pipeline skips that stage and includes a warning in the output. It does not error out.
- **Single file database**: All data lives in one SQLite file. No external services required. Back up by copying `~/.config/av/av.db`.
