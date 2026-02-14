# Sample: Epstein Files CCTV — Surveillance Video Intelligence

Turn 4 hours of DOJ prison CCTV footage into searchable, queryable video memory using `av`'s three-layer captioning cascade.

**Live demo**: [HuggingFace Space](https://huggingface.co/spaces/PixelML/epstein-files-cctv-video-memory-search)
**Dataset**: [HuggingFace](https://huggingface.co/datasets/PixelML/epstein-files-cctv-video-memory)

## What this shows

- **Cascade captioning** (`--captions --topic security`): 30-second chunks analyzed for temporal events instead of per-frame static descriptions
- **Topic-driven prompts**: The `security` topic focuses the VLM on people entering/leaving, door activity, suspicious behavior
- **Three-layer output**: 209 event captions → 7 structured summaries → 7 analysis reports
- **HuggingFace deployment**: Pre-built DB + Gradio Space for instant search

## Source data

10 clips from [DOJ Epstein Files — Data Set 8](https://www.justice.gov/epstein/doj-disclosures/data-set-8-files): 419 CCTV surveillance videos from MCC New York (July–August 2019).

## Reproduce

```bash
# 1. Install av
pip install pixelml-av
av config setup  # Choose OpenAI, enter API key

# 2. Get the videos (or use your own)
# Videos are in the HF dataset: huggingface-cli download PixelML/epstein-files-cctv-video-memory videos/ --local-dir .

# 3. Ingest with cascade + security topic
av ingest videos/ --captions --topic security --db epstein.db

# 4. Search
av search "someone being escorted out" --db epstein.db
av ask "what happens with the person in the red shirt?" --db epstein.db

# 5. Export for distribution
av export --format jsonl --db epstein.db > all_artifacts.jsonl
```

## Files

```
epstein-cctv/
├── README.md           # This file
├── subset.json         # Video manifest (from DOJ zip)
├── hf-dataset/         # HuggingFace dataset repo contents
│   ├── README.md       # Dataset card
│   ├── captions.jsonl  # 209 temporal event captions
│   ├── reports.jsonl   # 7 summaries + 7 reports
│   ├── transcripts.jsonl
│   └── all_artifacts.jsonl
├── hf-space/           # HuggingFace Space (Gradio app)
│   ├── README.md       # Space card
│   ├── app.py          # Search UI
│   └── requirements.txt
└── videos/             # (gitignored) 10 MP4 files, ~180MB
```

## Results comparison

| Metric | Old (per-frame) | New (cascade) |
|--------|----------------|---------------|
| API calls | ~1,400 VLM | ~850 VLM + 14 LLM |
| Artifacts | 1,418 static descriptions | 209 events + 7 summaries + 7 reports |
| `end_sec` | Always NULL | 30-second ranges |
| Quality | "In this video frame, three uniformed individuals are present..." | "A group in tactical gear enters, one person is escorted toward the door under supervision, then both exit..." |
