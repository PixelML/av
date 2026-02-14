---
title: Epstein Files CCTV — Video Memory Search
emoji: "🔍"
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
pinned: true
license: cc-by-4.0
hardware: cpu-basic
short_description: Search temporal event captions from Epstein CCTV
tags:
  - video-memory
  - surveillance
  - epstein-files
  - av
  - video-search
---

# Epstein Files CCTV — Video Memory Search

Search 209 AI-generated temporal event captions + 7 structured analysis reports from DOJ Epstein Files Dataset 8 (MCC prison CCTV surveillance footage).

GPT-4.1 vision analyzed 30-second video chunks to describe *what happens across frames* — temporal changes, people entering/leaving, door activity. Static scenes are filtered out.

Built with [`av`](https://github.com/PixelML/av) — video memory for AI agents.

## How it works

1. 10 CCTV clips from MCC New York (July–August 2019) were ingested with `av ingest --captions --topic security`
2. Three-layer cascade: chunk VLM (GPT-4.1 vision) → structured event log → analysis report
3. 209 temporal event captions + 7 summaries + 7 reports indexed with SQLite FTS5
4. This Space lets you search those captions with natural language

## Links

- [Dataset](https://huggingface.co/datasets/PixelML/epstein-files-cctv-video-memory)
- [av CLI](https://github.com/PixelML/av)
- [agentic.video](https://agentic.video)
