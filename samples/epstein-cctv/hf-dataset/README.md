---
license: cc-by-4.0
task_categories:
  - video-text-to-text
  - visual-question-answering
tags:
  - video-memory
  - surveillance
  - cctv
  - epstein-files
  - av
  - video-search
  - video-qa
  - dense-captioning
  - cascade-captioning
pretty_name: "Epstein Files CCTV — Video Memory Index"
size_categories:
  - n<1K
---

# Epstein Files CCTV — Video Memory Index

Pre-built video memory index of DOJ Epstein Files Dataset 8 (MCC prison CCTV surveillance footage), created with [`av`](https://github.com/PixelML/av).

**10 videos | ~4 hours of footage | 209 temporal event captions + 7 structured summaries + 7 analysis reports | searchable + queryable**

## Quickstart

Install `av` and download the pre-built database:

```bash
pip install pixelml-av
huggingface-cli download PixelML/epstein-files-cctv-video-memory av.db --local-dir .
```

Start querying immediately — no ingestion needed:

```bash
# Search across all indexed footage
av search "person entering through door" --db av.db

# Ask questions with citations
av ask "what activity is visible in the corridor?" --db av.db

# List all indexed videos
av list --db av.db
```

## What's in this dataset

### Source

[DOJ Epstein Files — Data Set 8](https://www.justice.gov/epstein/doj-disclosures/data-set-8-files): 419 MP4 surveillance videos from the Metropolitan Correctional Center (MCC) in New York, recorded between July 6, 2019 (arrest) and August 11, 2019 (day after death).

### This subset

10 representative clips from Dataset 8 covering multiple camera positions:

| File | Duration | Resolution | Audio | Description |
|------|----------|------------|-------|-------------|
| EFTA00028842.mp4 | 12s | 854x480 | Yes | Higher-resolution clip |
| EFTA00029996.mp4 | 8s | 352x240 | No | Short surveillance clip |
| EFTA00029997.mp4 | 18s | 352x240 | No | Short surveillance clip |
| EFTA00033226.mp4 | 60min | 352x240 | Yes | Long CCTV recording |
| EFTA00033244.mp4 | 60min | 352x240 | Yes | Long CCTV recording |
| EFTA00033246.mp4 | 60min | 352x240 | Yes | Long CCTV recording |
| EFTA00033262.mp4 | 60min | 352x240 | Yes | Long CCTV recording |
| EFTA00033280.mp4 | 59min | 352x240 | Yes | Long CCTV recording |
| EFTA00033368.mp4 | 60min | 352x240 | Yes | Long CCTV recording |
| EFTA00033396.mp4 | 60min | 352x240 | Yes | Long CCTV recording |

### Files

| File | Format | Description |
|------|--------|-------------|
| `av.db` | SQLite | Drop-in database for `av` CLI — instant search and Q&A |
| `captions.jsonl` | JSONL | 209 temporal event captions with start/end timestamps |
| `reports.jsonl` | JSONL | 7 structured summaries + 7 full analysis reports per video |
| `transcripts.jsonl` | JSONL | Audio transcripts (Whisper) where available |
| `all_artifacts.jsonl` | JSONL | Complete export of all 224 artifacts |

### Processing — Three-Layer Cascade

Unlike per-frame captioning (which produces repetitive static scene descriptions), this dataset uses `av`'s **three-layer captioning cascade** with the `security` topic:

1. **Layer 0 — Chunk VLM**: Video split into 30-second chunks, 3 frames extracted per chunk, sent to GPT-4.1 vision as a multi-image call. Prompt focuses on temporal changes, people entering/leaving, door activity, suspicious behavior. Static chunks are filtered out.
2. **Layer 1 — Structured Summary**: All Layer 0 captions aggregated and summarized by GPT-4.1 into a structured event log with `START:END:EVENT` format.
3. **Layer 2 — Analysis Report**: Layer 1 output consolidated into a final report with timestamped events, summary, and categorized tags.

**Result**: 209 meaningful event descriptions (vs 1,418 repetitive frame-by-frame captions previously), each with `start_sec` and `end_sec` for temporal ranges.

### Artifact types

| Type | Count | Description |
|------|-------|-------------|
| `caption` | 209 | Temporal event descriptions (30-second chunks) |
| `summary` | 7 | Structured event logs per video |
| `report` | 7 | Full analysis reports with categories |
| `transcript` | 1 | Audio transcript |

## How it was built

```bash
pip install pixelml-av
av config setup  # OpenAI provider

# Ingest with cascade captioning + security topic
av ingest videos/ --captions --topic security --db epstein.db

# Export for distribution
av export --format jsonl --db epstein.db > all_artifacts.jsonl
```

See [agentic.video](https://agentic.video) for more about `av`.

## Use cases

- **Journalism**: Search surveillance footage by description rather than scrubbing through hours of video
- **Research**: Query what's visible across multiple camera angles simultaneously
- **Demonstration**: Show how AI video memory works on real-world, publicly available footage
- **Agent tooling**: Give your AI agent the ability to answer questions about this footage

## License

The underlying videos are U.S. government works released by the DOJ under FOIA. The AI-generated captions, transcripts, and embeddings in this dataset are released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Links

- [`av` CLI](https://github.com/PixelML/av) — the tool that built this index
- [agentic.video](https://agentic.video) — project homepage
- [DOJ Epstein Files](https://www.justice.gov/epstein) — official source
- [Pixel ML](mailto:hello@pixelml.com) — enterprise video intelligence
