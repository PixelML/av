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

### Refined `av ask` (optional)

Configure a TypeSafe System One key to make Jev refinement automatic for `av ask`:

```bash
export AV_TYPESAFE_API_KEY="..."       # TYPESAFE_API_KEY is also accepted
av ask "when does the person enter the room?"
av ask "when does the person enter the room?" --no-refine  # legacy RAG for this call
```

The refined path uses Jev only for typed decisions: it filters source relevance,
uses a configurable bounded temporal neighborhood to form local scenes, merges
overlapping same-video scenes, and ranks them by `relevance probability × retrieval
score`. A separate Jev Noul checks whether the answer is supported; source relevance
is not treated as answer correctness.

If Jev is unavailable, `av ask` visibly warns and falls back to raw retrieval. If
Jev validly rejects every hit, the result is empty instead of restoring rejected
hits. Refined JSON includes `route`, `evidence_status`, `refinement`, `warnings`,
`inspected_windows`, and per-stage token usage when providers report it. Unknown
usage remains `null`.

Missing, malformed, or out-of-range System One probabilities are treated as a
refinement failure: `av` reports the fallback and does not invent a confidence.

See the [AV ask refinement cookbook](cookbook/README.md) for a reproducible recipe,
offline cost arithmetic, and receipt provenance.

FTS5 remains the first retrieval stage. An unscoped query with zero FTS matches does
not scan the video archive or invoke sampled-frame inspection.

### API-only reproducible ingest

Use an explicit OpenAI-compatible endpoint/key and keep both fallback flags disabled:

```bash
export AV_API_BASE_URL="https://your-provider.example/v1"
export AV_API_KEY="..."
export AV_VISION_MODEL="your-cheap-vision-model"
export AV_ALLOW_OAUTH_FALLBACK="false"
export AV_ALLOW_CODEX_FALLBACK="false"

av ingest video.mp4 --dense-vision --max-frames 120 --no-embed
```

To import a timestamped transcript produced by a separate public ASR script, pass a
validated sidecar instead of running built-in ASR:

```bash
av ingest video.mp4 --dense-vision --transcript-json transcript.json
```

The sidecar contains `segments` with `start_sec`, `end_sec`, and `text`, plus optional
`model` and public `provenance`. It is validated against the probed video duration
before database changes or API calls. AV records the import as local work with zero
provider requests; external ASR usage or cost is not attributed to this ingest.

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

# Benchmarking
av bench probe                     # What can this deployment actually do?
av bench gate                      # Can it order frames at all? Run this first.
av bench run task.jsonl            # Dense vs agentic, with tokens and dollars
av bench sweep captions.jsonl vids/ # Where does recall collapse as frames thin out?
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

## Bench — Cost/Accuracy Frontier

Selling video understanding on hardware you own means one number decides everything:
**video-hours analysed per dollar**. `av bench` measures it, and measures what it
costs you in accuracy to get there.

Two headline axes, chosen so results read against published agentic-video
comparisons: **tokens per query** and **accuracy**. Alongside them sits the axis an
API vendor cannot report — **dollars per query on your own box** — because per-token
billing and per-hour hardware are different economics and the tool never conflates
them.

### Run the gate first

```bash
av bench gate --sizes 2,4,8
```

Deterministic ffmpeg fixtures carrying a known order, one question, exact-match
scoring. A model that cannot report the order of eight flat colours cannot be
meaningfully scored on long-video reasoning, and any throughput number measured
against it describes a machine doing the wrong thing quickly. The gate costs cents
and it can save the whole exercise.

### Establish what is tunable before sweeping it

```bash
av bench probe
av bench plan --widths 512,768,1024,1536 --budgets 200,400,800
```

`probe` tests two candidate knobs against your live endpoint — the OpenAI `detail`
hint and the resolution actually uploaded — because a server may honour one and
silently ignore the other. If neither moves the per-frame token count, the
tokens-per-frame axis is reported as fixed rather than faked. `plan` predicts the
same thing offline from a published preprocessor algorithm, and shows the two walls
worth knowing: an upscale floor below which shrinking frames buys nothing, and a
token ceiling above which extra resolution is discarded.

### Dense versus agentic

```bash
av bench prepare minerva minerva.json --out task.jsonl --max-questions 40 --max-videos 6
av bench run task.jsonl --arms dense,agentic --cost hourly:25.0:20000
```

The **dense** arm samples the whole window at a fixed rate and asks once. The
**agentic** arm takes a cheap coarse look, decides which moments it needs, then
fetches only those — and is charged for both requests. Nothing else differs between
them.

`av bench prepare` adapts a public benchmark's annotations into the task format.
**No benchmark data ships with av and no videos are downloaded.** Fetch annotations
yourself and mind their licences: MINERVA's are CC BY 4.0, LVBench's are
CC BY-NC-SA with an explicit commercial-use prohibition, and neither grants any
rights to the videos themselves.

### Where does it collapse?

```bash
av bench sweep captions.jsonl videos/ --intervals 1,2,5,10,30 --cost token:0.30:2.50
```

Event detection against sampling interval on real footage. The interval at which
detection collapses is the cheapest safe sampling rate — and it is a per-task
answer, not a global one. Smoke tolerates sparse frames; a door opening does not.

### Noise floor

```bash
av bench noise --repeats 5
```

Runs one unchanged cell repeatedly and publishes the spread. This is the number that
makes every other number readable: a delta smaller than the spread is noise. Point it
at a cell the model does not already solve perfectly — a saturated cell has no
headroom to vary, and the tool says so rather than reporting a meaningless zero.

### Receipts

Every subcommand writes a JSON receipt to `./bench-receipts/` carrying the provider,
the determinism controls, the exact ffmpeg invocations, fixture hashes, the cost
model, and every cell. Claims are labelled `measured`, `derived`, `documented`,
`community-reported`, or `untested`, and a non-measured claim must cite a source.
Endpoints are reduced to a hostname, and private or tunnelled hosts never appear at
all — receipts are meant to be published.

### Cost model

```bash
av bench cost --tokens-per-frame 1024 --context-tokens 1048576 \
  --prefill-tok-s 20000 --hourly-usd 25 --kv-bytes-per-token 890 \
  --source "your measurements"
```

Pure arithmetic, no API calls, every input recorded. Supply `--cost hourly:RATE` for
hardware you own or `--cost token:IN:OUT` for a vendor API — they are different
shapes and reporting one in the other's units produces a number that means nothing.

## Configuration

### Interactive Setup (Recommended)

```bash
av config setup
```

Choose from six providers:

| # | Provider | Auth | Transcription | Embeddings |
|---|----------|------|---------------|------------|
| 1 | **OpenAI (Codex OAuth)** | Auto-detected | Whisper | text-embedding-3-small |
| 2 | **OpenAI (API key)** | `sk-...` key | Whisper | text-embedding-3-small |
| 3 | **PixelML (OpenRouter)** | API key | Not supported | Not supported |
| 4 | **Anthropic (Claude)** | API key | Not supported | Not supported |
| 5 | **Google (Gemini)** | API key | Not supported | text-embedding-004 |
| 6 | **DeepSeek-V4.1-Flash** | Your own endpoint | Not supported | Not supported |

**DeepSeek-V4.1-Flash** talks to an OpenAI-compatible SGLang server that you run.
No endpoint ships with `av` — the preset defaults to SGLang's own local bind
address, and you point `AV_API_BASE_URL` at your deployment. Set `DEEPSEEK_API_KEY`
if your server requires one; leave it unset if it does not.

Config is saved to `~/.config/av/config.json` and persists across sessions.

**Note:** Anthropic and Gemini don't support Whisper transcription. With these providers, use `av ingest --captions` for frame-based captioning, or set `AV_OPENAI_API_KEY` for transcription fallback.

### Environment Variables

Env vars always override config.json:

```bash
export AV_API_KEY="sk-..."
export AV_API_BASE_URL="https://api.openai.com/v1"  # or any OpenAI-compatible endpoint
export AV_API_TIMEOUT_SEC="120"
export AV_API_MAX_RETRIES="1"
export AV_ALLOW_OAUTH_FALLBACK="false"  # never read local auth caches unless explicitly enabled
export AV_ALLOW_CODEX_FALLBACK="false"  # never spawn Codex unless explicitly enabled
export AV_TRANSCRIBE_MODEL="whisper"
export AV_VISION_MODEL="gpt-4-1"
export AV_EMBED_MODEL="text-embedding-3-small"
export AV_CHAT_MODEL="gpt-4-1"
export AV_CHAT_MAX_OUTPUT_TOKENS="1024"  # positive cap for each answer response

# Optional Jev/System One refinement (automatic when a key is present)
export AV_TYPESAFE_API_KEY="..."  # TYPESAFE_API_KEY also works
export AV_TYPESAFE_ENDPOINT="https://api.typesafe.ai/v1/systemone"
export AV_TYPESAFE_MODEL="jev-latest"
export AV_REFINE_RELEVANCE_MIN="0.5"
export AV_REFINE_SUPPORT_MIN="0.5"
export AV_REFINE_MAX_SCENES="8"
export AV_REFINE_BATCH_SIZE="10"
export AV_REFINE_CONTEXT_EVENTS="3"

# Optional bounded sampled-frame fallback after an unsupported answer
export AV_STRONG_VISION_API_BASE_URL="https://your-explicit-endpoint.example/v1"
export AV_STRONG_VISION_API_KEY="..."
export AV_STRONG_VISION_MODEL="your-explicit-model"
export AV_INSPECTION_MAX_WINDOWS="2"
export AV_INSPECTION_MAX_SECONDS="120"
export AV_INSPECTION_MAX_FRAMES="12"
export AV_INSPECTION_MAX_ATTEMPTS="1"
export AV_INSPECTION_DENSE_PASS="false"

# Self-hosted DeepSeek-V4.1-Flash via SGLang
export AV_PROVIDER="deepseek"
export AV_API_BASE_URL="http://your-sglang-host:30000/v1"
export DEEPSEEK_API_KEY="..."   # only if your server requires one
```

API requests use the configured timeout and explicit retry limit. Ingestion JSON
includes `stage_usage` for transcription, captioning, caption summarization, and
embeddings, plus the effective frame/request settings. Request failures are counted;
token totals become `null` with a completeness flag when any provider omits usage.
Ask JSON likewise reports the effective chat model/output cap and per-stage usage.
No dollar total is inferred.

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
| `av ask <question>` | RAG Q&A; automatically refines with Jev when configured |
| `av list` | List all indexed videos |
| `av info <video_id>` | Detailed video metadata |
| `av transcript <id>` | Output transcript (VTT/SRT/text) |
| `av export` | Export as JSONL/VTT/SRT |
| `av open <id> --at <sec>` | Open video at timestamp |
| `av bench gate` | Temporal-ordering capability gate |
| `av bench probe` | Measure a deployment's image-token and multi-image behaviour |
| `av bench plan` | Predict per-frame token cost against resolution (offline) |
| `av bench prepare` | Adapt a public benchmark's annotations into a task file |
| `av bench run` | Dense vs agentic arms, with tokens and dollars |
| `av bench sweep` | Event recall against sampling interval |
| `av bench noise` | Spread across identical runs |
| `av bench cost` | Cost arithmetic with labelled inputs (offline) |
| `av version` | Print version JSON |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
