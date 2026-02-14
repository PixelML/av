# av — Agent Contract

**Package**: `pixelml-av` | **CLI**: `av` | **By**: Pixel ML

## Overview

`av` is a CLI tool for indexing video files and searching their content. It extracts transcripts (via Whisper), optionally captions frames (via vision models), generates embeddings, and enables full-text + semantic search and RAG Q&A over video content.

**Design principles**:
- JSON to stdout, progress/warnings to stderr
- OpenAI-compatible API (swappable via `AV_API_BASE_URL`)
- Single SQLite database at `~/.config/av/av.db`
- FTS5 as primary search; cosine reranking when embeddings exist

## Quick Workflow

```bash
# 1. Ingest a video
av ingest meeting.mp4

# 2. Search content
av search "budget discussion"

# 3. Ask questions (RAG with citations)
av ask "what decisions were made about Q3 budget?"

# 4. Export for downstream processing
av export --format jsonl
```

## Commands & JSON Contracts

### `av version`
```json
{"version": "0.1.0", "package": "pixelml-av"}
```

### `av ingest <path>`
```bash
av ingest video.mp4                     # transcript + embeddings (default)
av ingest video.mp4 --captions          # + frame captions
av ingest video.mp4 --no-embed          # transcript only
av ingest video.mp4 --dry-run           # preview
av ingest /folder/                      # batch
av ingest video.mp4 --force             # re-ingest
```
```json
{"status": "complete", "video_id": "uuid", "filename": "video.mp4", "duration_sec": 120.5, "artifacts_count": 42, "elapsed_sec": 15.3}
```

### `av list`
```json
{"videos": [{"video_id": "uuid", "filename": "video.mp4", "duration_formatted": "00:02:00", "status": "complete", "artifacts_count": 42}], "total": 1}
```

### `av info <video_id>`
```json
{"video_id": "uuid", "filename": "video.mp4", "file_path": "/path/to/video.mp4", "duration_sec": 120.5, "duration_formatted": "00:02:00", "resolution": "1920x1080", "status": "complete", "artifacts": {"transcript": 42, "caption": 10}, "ingested_at": "2025-01-01 00:00:00"}
```

### `av search <query>`
```json
{"query": "machine learning", "results": [{"rank": 1, "score": 0.87, "video_id": "uuid", "filename": "video.mp4", "timestamp_sec": 123.4, "timestamp_formatted": "00:02:03", "source_type": "transcript", "text": "...matching text..."}], "total_results": 5, "search_time_ms": 12}
```

### `av ask <question>`
```json
{"answer": "The team decided to...", "citations": [{"video_id": "uuid", "start_sec": 120.0, "end_sec": 135.0, "source_type": "transcript", "text": "...relevant excerpt...", "score": 0.91}], "confidence": 0.85}
```

### `av transcript <video_id>`
Outputs VTT (default), SRT, or plain text to stdout. **Not JSON.**
```bash
av transcript <id> --format vtt    # WebVTT
av transcript <id> --format srt    # SubRip
av transcript <id> --format text   # Plain text
```

### `av export`
JSONL lines to stdout (default), or VTT/SRT.
```bash
av export --format jsonl
av export --format vtt --video-id <id>
```

Each JSONL line:
```json
{"video_id": "uuid", "filename": "video.mp4", "type": "transcript", "start_sec": 10.0, "end_sec": 20.0, "timestamp_formatted": "00:00:10", "text": "..."}
```

### `av open <video_id> --at <seconds>`
Opens video file in system player. Uses mpv for timestamp seeking if available.

### `av config show`
```json
{"api_base_url": "https://api.openai.com/v1", "api_key": "***", "transcribe_model": "whisper-1", "vision_model": "gpt-4o-mini", "embed_model": "text-embedding-3-small", "chat_model": "gpt-4o-mini", "db_path": "/Users/you/.config/av/av.db"}
```

## Calling from an Agent

```bash
# Ingest and capture video_id
VIDEO_ID=$(av ingest video.mp4 | python -c "import sys,json; print(json.load(sys.stdin)['video_id'])")

# Search and parse results
av search "keyword" | python -c "
import sys, json
data = json.load(sys.stdin)
for r in data['results']:
    print(f'{r[\"timestamp_formatted\"]} [{r[\"source_type\"]}] {r[\"text\"][:80]}')
"

# Ask with jq
av ask "what happened?" | jq '.answer'

# Export all data
av export --format jsonl > video_memory.jsonl
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AV_API_KEY` | (none) | OpenAI API key |
| `AV_API_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `AV_TRANSCRIBE_MODEL` | `whisper` | Transcription model |
| `AV_VISION_MODEL` | `gpt-4-1` | Vision model for captions |
| `AV_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `AV_CHAT_MODEL` | `gpt-4-1` | Chat/RAG model |
| `AV_DB_PATH` | `~/.config/av/av.db` | Database location |

## Database Schema

Single SQLite file with FTS5. Key tables:
- `videos` — indexed video files with metadata
- `artifacts` — unified table for transcripts, captions, scenes
- `embeddings` — stored vectors as BLOBs (dim per row)
- `artifacts_fts` — FTS5 virtual table for text search

## Development

```bash
uv sync
uv run av version
uv run av --help
```
