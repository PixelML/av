# av — Agent Contract

**Package**: `pixelml-av` | **CLI**: `av` | **License**: Apache 2.0

## Project Goal

`av` is an open-source CLI that turns video files into searchable, queryable memory for AI agents and developers. The core loop: **ingest once, search many times**.

Long-term vision: become the standard developer tool for video understanding — what `jq` is for JSON or `ripgrep` is for code search, but for video content. It should work with any AI provider, run locally, and produce structured output that agents can consume.

## Architecture

```
video file / URL
       │
       ▼
┌─────────────────────────────┐
│  av ingest                  │
│  ├─ ffmpeg → audio → Whisper (transcript segments)
│  ├─ ffmpeg → frames → Vision model (captions)
│  ├─ Embeddings (batch)      │
│  └─ SQLite (FTS5 + vectors) │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  av search / av ask         │
│  ├─ FTS5 full-text match    │
│  ├─ Cosine reranking        │
│  └─ RAG Q&A with citations  │
└─────────────────────────────┘
```

### Key Design Decisions

- **JSON to stdout, progress to stderr** — agents parse stdout, humans read stderr
- **Single SQLite file** at `~/.config/av/av.db` — no external DB dependencies
- **FTS5 as primary search** — works without embeddings; cosine reranking optional
- **Provider-agnostic** — OpenAI, Anthropic, Gemini via `av config setup`
- **Best-effort pipeline** — if a stage fails (auth, model access), continue and warn
- **Config priority**: env vars > `~/.config/av/config.json` > defaults

## Code Layout

```
src/av/
├── cli/              # Typer commands (one file per command)
│   ├── app.py        # Root app, wires all subcommands
│   ├── config_cmd.py # av config show/setup (interactive wizard)
│   ├── ingest.py     # av ingest (file, directory, URL)
│   ├── search.py     # av search
│   ├── ask.py        # av ask (RAG)
│   └── output.py     # JSON/text output helpers
├── core/
│   ├── config.py     # AVConfig (pydantic-settings), config.json loader
│   ├── constants.py  # Provider presets, model defaults
│   └── exceptions.py # Error hierarchy
├── db/
│   ├── connection.py # SQLite setup (WAL, foreign keys)
│   ├── models.py     # Pydantic data models
│   ├── repository.py # CRUD + FTS search
│   └── schema.py     # Migrations
├── pipeline/
│   ├── ingest.py     # Orchestrator (the main pipeline)
│   ├── ffmpeg.py     # ffmpeg/ffprobe wrappers
│   ├── chunker.py    # Text chunking for embeddings
│   └── dense_caption.py  # Structured event export
├── bench/            # Cost/accuracy frontier measurement
│   ├── cost.py       # Per-hour vs per-token cost models (never conflated)
│   ├── datasets.py   # Adapters for public benchmarks (no data vendored)
│   ├── fixtures.py   # Deterministic ffmpeg fixtures for the ordering gate
│   ├── frames.py     # Pinned frame extraction (command recorded in the receipt)
│   ├── receipts.py   # Labelled evidence + endpoint redaction
│   ├── runner.py     # Sweep axes, noise floor, collapse point
│   ├── vlm.py        # Provider-agnostic multi-image calls with usage accounting
│   └── tasks/
│       ├── ordering.py  # Temporal-ordering capability gate
│       ├── videoqa.py   # Dense vs agentic arms
│       └── events.py    # Event recall vs sampling interval
├── providers/
│   ├── base.py       # Abstract interfaces
│   ├── deepseek.py   # DeepSeek-V4.1-Flash (SGLang) config + image-token model
│   └── openai.py     # OpenAI-compatible client (works for all providers)
├── search/
│   ├── semantic.py   # FTS5 + cosine reranking
│   └── rag.py        # RAG Q&A with citations
└── utils/
    ├── video.py      # Video file discovery
    ├── youtube.py     # yt-dlp download helper
    ├── hashing.py     # Fast file hashing (idempotency)
    └── principles.py  # Custom prompt principles loader
```

## Commands & JSON Contracts

### `av config setup`
Interactive wizard — choose provider, enter API key, validates, saves to `~/.config/av/config.json`.

### `av config show`
```json
{"provider": "openai", "api_base_url": "https://api.openai.com/v1", "api_key": "***", "transcribe_model": "whisper-1", "vision_model": "gpt-4-1", "embed_model": "text-embedding-3-small", "chat_model": "gpt-4-1", "db_path": "~/.config/av/av.db"}
```

### `av ingest <path>`
```bash
av ingest video.mp4                     # transcript + embeddings
av ingest video.mp4 --captions          # + frame captions
av ingest video.mp4 --no-embed          # transcript only
av ingest video.mp4 --dry-run           # preview
av ingest /folder/                      # batch directory
av ingest video.mp4 --force             # re-ingest
av ingest "https://youtu.be/..."        # YouTube URL
av ingest video.mp4 --dense-vision      # structured dense captions
av ingest video.mp4 --transcript-json transcript.json  # validated external transcript; skips built-in ASR
```
```json
{"status": "complete", "video_id": "uuid", "filename": "video.mp4", "duration_sec": 120.5, "artifacts_count": 42, "elapsed_sec": 15.3}
```
On partial failure: `{"status": "complete_with_warnings", ..., "warnings": ["Transcription skipped: ..."]}`

### `av search <query>`
```json
{"query": "...", "results": [{"rank": 1, "score": 0.87, "video_id": "uuid", "filename": "video.mp4", "timestamp_sec": 123.4, "timestamp_formatted": "00:02:03", "source_type": "transcript", "text": "..."}], "total_results": 5, "search_time_ms": 12}
```

### `av ask <question>`
```json
{"answer": "...", "citations": [{"video_id": "uuid", "start_sec": 120.0, "source_type": "transcript", "text": "...", "score": 0.91}], "confidence": 0.85}
```

With `AV_TYPESAFE_API_KEY` (or `TYPESAFE_API_KEY`) configured, `av ask`
automatically runs Jev source relevance, configurable bounded scene grouping, answer
synthesis, and a separate answer-support Noul. `--no-refine` preserves legacy RAG
for one request. Refined responses retain answer/citations/confidence and add route,
evidence, refinement, warning, inspected-window, and stage-usage metadata.

### `av list` / `av info <id>` / `av transcript <id>` / `av export` / `av open <id>`
See `av <command> --help` for details.

### `av bench`
Measures the cost/accuracy frontier. Headline axes are **tokens per query** and
**accuracy**, plus **dollars per query on hardware you own** — the axis a per-token
API vendor cannot report.

```bash
av bench probe                       # is tokens-per-frame tunable on this endpoint?
av bench gate --sizes 2,4,8          # can the model order frames at all?
av bench plan --budgets 200,400,800  # predicted token cost per resolution (offline)
av bench prepare minerva ann.json -o task.jsonl
av bench run task.jsonl --arms dense,agentic --cost hourly:25.0:20000
av bench sweep captions.jsonl videos/ --intervals 1,2,5,10,30
av bench noise --repeats 5
av bench cost --tokens-per-frame 1024 --prefill-tok-s 20000 --hourly-usd 25
```

Every subcommand writes a JSON receipt to `./bench-receipts/`.

**Rules that are not optional here:**
- **Run the gate before quoting any score.** A model that cannot order eight frames
  is not being measured on temporal understanding, and its throughput is irrelevant.
- **Label every claim.** `measured` / `derived` / `documented` / `community-reported`
  / `untested`. Non-measured claims must cite a source; `Claim` raises otherwise.
- **Publish the noise floor**, and never on a saturated cell.
- **Never conflate `$/hr` and `$/token`.** They are different economics.
- **Never put someone else's published number and ours in one cell as a ratio.**
  Different model, different hardware, different methodology — it is a comparison of
  approaches, not a head-to-head.
- **No endpoints in source.** Receipts record a hostname; private hosts are redacted.
- **No benchmark data is vendored.** Adapters read files the user fetched, under the
  upstream licence.

## Provider Support

| Provider | Transcription | Vision/Chat | Embeddings | Setup |
|----------|--------------|-------------|------------|-------|
| OpenAI (OAuth) | whisper-1 | gpt-4-1 | text-embedding-3-small | Auto-detect Codex token |
| OpenAI (API key) | whisper-1 | gpt-4-1 | text-embedding-3-small | Paste `sk-...` |
| Anthropic | -- | claude-sonnet-4-5 | -- | Paste API key |
| Gemini | -- | gemini-2.5-flash | text-embedding-004 | Paste API key |
| DeepSeek-V4.1-Flash | -- | deepseek-v4.1-flash (self-hosted SGLang) | -- | `AV_API_BASE_URL` + `DEEPSEEK_API_KEY` |

When a capability is unavailable (e.g. Anthropic has no Whisper), the pipeline skips that stage and warns.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AV_API_KEY` | (none) | API key (overrides config.json) |
| `AV_API_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `AV_API_TIMEOUT_SEC` | `120` | Per-attempt API timeout |
| `AV_API_MAX_RETRIES` | `1` | Explicit retry count; SDK retries stay disabled |
| `AV_ALLOW_OAUTH_FALLBACK` | `false` | Allow reading local OAuth caches only when explicitly enabled |
| `AV_ALLOW_CODEX_FALLBACK` | `false` | Allow spawning Codex for caption fallback only when explicitly enabled |
| `AV_PROVIDER` | (none) | Provider name |
| `AV_TRANSCRIBE_MODEL` | `whisper-1` | Transcription model |
| `AV_VISION_MODEL` | `gpt-4-1` | Vision/caption model |
| `AV_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `AV_CHAT_MODEL` | `gpt-4-1` | Chat/RAG model |
| `AV_CHAT_MAX_OUTPUT_TOKENS` | `1024` | Positive output-token cap for each answer response |
| `AV_DB_PATH` | `~/.config/av/av.db` | Database location |
| `AV_TYPESAFE_API_KEY` | (none) | Jev/System One key; enables ask refinement by default |
| `AV_TYPESAFE_ENDPOINT` | `https://api.typesafe.ai/v1/systemone` | Explicit System One endpoint |
| `AV_TYPESAFE_MODEL` | `jev-latest` | System One model |
| `AV_REFINE_RELEVANCE_MIN` | `0.5` | Minimum source-relevance Noul probability |
| `AV_REFINE_SUPPORT_MIN` | `0.5` | Minimum answer-support Noul probability |
| `AV_REFINE_MAX_SCENES` | `8` | Maximum merged scenes sent to synthesis |
| `AV_REFINE_BATCH_SIZE` | `10` | Sources per System One relevance request |
| `AV_REFINE_CONTEXT_EVENTS` | `3` | Maximum temporal events on each side of a hit |
| `AV_STRONG_VISION_API_BASE_URL` | (none) | Explicit OpenAI-compatible sampled-frame endpoint |
| `AV_STRONG_VISION_API_KEY` | (none) | Key for the stronger sampled-frame endpoint |
| `AV_STRONG_VISION_MODEL` | (none) | Explicit stronger vision model; never selected implicitly |
| `DEEPSEEK_API_KEY` | (none) | Key for a self-hosted DeepSeek-V4.1-Flash server |
| `SGLANG_API_KEY` | (none) | Alias for the same, matching SGLang's own naming |

## Database

Single SQLite file, WAL mode, FTS5. Tables:
- `videos` — file metadata, ingest status, hash (idempotency)
- `artifacts` — transcripts, captions, dense_captions (unified, timestamped)
- `embeddings` — float32 vectors as BLOBs
- `artifacts_fts` — FTS5 virtual table with auto-sync triggers

## Development

```bash
uv sync --extra dev       # install with test deps
uv run av --help          # verify CLI
uv run pytest tests/ -v   # run tests (21 tests)
```

### Conventions

- **Tests**: `tests/test_*.py`, use `pytest` + `monkeypatch` for config isolation
- **No mocks for config**: use `monkeypatch.setattr("av.core.config.CONFIG_FILE_PATH", ...)` + `tmp_path`
- **Ingest tests**: patch `av.pipeline.ingest.get_video_info` to avoid needing real video files
- **All output to stdout must be valid JSON** (except `transcript` and `export` text formats)
- **Imports**: one class/function per line, stdlib → third-party → local

### Adding a New Provider

1. Add preset to `PROVIDER_PRESETS` in `core/constants.py`
2. Add menu entry in `cli/config_cmd.py` `_PROVIDER_MENU`
3. If non-OpenAI-compatible: add header/auth logic in `providers/openai.py` `_client()`
4. Set empty string for unsupported capabilities (e.g. `"transcribe_model": ""`)
5. Pipeline will auto-skip disabled stages
6. Add test in `tests/test_config.py`

## Roadmap

Near-term priorities for contributors:
- [ ] Audio chunking for files > 25MB (split before Whisper API call)
- [ ] `av config reset` command to clear saved config
- [ ] Cross-video search improvements (search across all indexed videos at once)
- [ ] Streaming ingest progress (SSE-style output for long videos)
- [ ] Profile presets for dense captioning (security, retail, meeting, etc.)
- [ ] `av bench` precision measurement (currently only recall against reference windows)
- [ ] Video fetch helper for benchmark task files (yt-dlp, with link-rot reporting)
- [x] CI/CD with GitHub Actions (lint + test on PR) — `.github/workflows/ci.yml`
- [x] PyPI publish workflow — `.github/workflows/publish.yml` + `PUBLISHING.md`
