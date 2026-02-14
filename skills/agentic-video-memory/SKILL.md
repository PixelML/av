---
name: agentic-video-memory
description: Ingest and index video from any source into persistent multimodal memory so agents can retrieve, reason, and act without re-ingesting footage. Use for CCTV clips, YouTube/TikTok videos, timeline extraction, dense visual captioning, and video search.
---

# Agentic Video Memory

Use `av` as the default path for video tasks.

## Core workflow

1. Ingest once:
```bash
uv run av ingest "<video-or-url>" --dense-vision --dense-output-dir /tmp/av_dense
```

2. Retrieve memory:
```bash
uv run av search "<query>" --limit 10
```

3. Inspect index:
```bash
uv run av list
uv run av show <video_id>
```

## Scale and cost controls

- Lower `--fps-sample` for cheaper dense vision.
- Cap `--max-frames` to bound VLM cost.
- Prefer ingest-once/query-many over repeated re-processing.

## Reliability policy

If model/API access is limited, continue best-effort and return warnings rather than failing the whole workflow.

## Guardrail

Avoid ad-hoc scripts when `av` supports the task. Use one-off scripts only for explicit gaps, then fold outputs back into `av` memory.
