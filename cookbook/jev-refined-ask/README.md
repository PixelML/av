# Ask with Jev evidence refinement

Use AV's indexed moments as candidate evidence, then refine relevance and scene
context before answering. This recipe describes the open-source CLI behavior;
it does not claim that an AV run reproduces historical Composer cost or quality.
See the [cost model](https://github.com/PixelML/av/tree/codex/jev-query-cascade/cookbook/cost-model)
and [sanitized receipts](https://github.com/PixelML/av/tree/codex/jev-query-cascade/cookbook/receipts)
for separate stage accounting and evidence provenance.

## Run it

Configure the ordinary AV answer provider first, then set your TypeSafe key in
the environment. Do not save real credentials in notebooks or receipts.

    av config setup
    export AV_TYPESAFE_API_KEY='your-typesafe-key'
    av ingest video.mp4 --captions
    av ask "What happens when the speaker approaches the microphone?"
    av ask "What happens when the speaker approaches the microphone?" --no-refine

These commands make provider calls. Ingestion, answer generation, and configured
judgments can incur charges; the cost calculator itself stays offline. The
**TYPESAFE_API_KEY** alias also enables refinement. If **AV_TYPESAFE_API_KEY** is
set, it takes precedence. Without either key, AV uses the existing retrieval and
answer path. **--no-refine** opts out for a request; **AV_REFINE_ENABLED=false**
disables refinement through configuration.

## Import an externally generated transcript

If your caption provider does not transcribe audio, generate a transcript with a
separate ASR tool and use the public import seam:

    av ingest video.mp4 --transcript-json transcript.json --no-embed

The input can be a root segment array or an object with a required **segments**
array and optional **model** string and **provenance** object:

    {"segments": [{"start_sec": 0, "end_sec": 2.4, "text": "Hello"}],
     "model": "your-asr-model", "provenance": {"method": "external ASR"}}

Timestamps must be numeric, finite, ordered within each segment, non-negative,
and within the actual video duration. The import preserves segment order, text,
and timestamps. It accepts one video at a time; an empty array is valid for
silence. Import does not meter the external ASR run or verify transcript accuracy.
Keep that run's token usage, approximate-timestamp caveat, and price separate from
AV's ingestion receipt. Use only public metadata in a published sidecar.

The included [`transcribe_gemini.py`](https://github.com/PixelML/av/blob/codex/jev-query-cascade/cookbook/jev-refined-ask/transcribe_gemini.py)
is a standard-library helper for Gemini audio transcription. It does not bundle
media or transcripts. It accepts any readable local source supported by ffmpeg:

    export GEMINI_API_KEY='your-key'
    python3 cookbook/jev-refined-ask/transcribe_gemini.py \
      ./video.mp4 ./asr-run --model gemini-3.5-flash-lite \
      --budget-usd 0.50 --prior-reserved-usd 0
    av ingest ./video.mp4 --transcript-json ./asr-run/transcript.json --no-embed

The helper supports positive sub-second inputs, folds only a remainder shorter
than one second into the preceding window, and uses no automatic retries. Before
either provider action it writes a conservative reservation to an fsync'd ledger.
Reservations survive failures and restarts; use `--prior-reserved-usd` for attempts
made outside the output directory.

`run-manifest.json` binds the output directory to the source hash, source size,
duration, model, prompt, generation configuration, pricing inputs, window plan,
and recipe revision. Completed chunk and raw-response caches also carry that
binding. JSON artifacts are atomically replaced. A valid saved provider response
can be recovered after interruption without issuing a duplicate generation call;
a mismatch stops and requires a new directory.

For a non-default model, pass both `--input-usd-per-million` and
`--output-usd-per-million`. These rates enforce a local reservation cap and
produce a list-rate estimate only. Actual billed/account cost, compute, storage,
and network remain unknown unless separately evidenced.

## Dense captions with a separate transcript

For a bounded image-caption recipe, select the caption provider and model with
**AV_API_BASE_URL**, **AV_API_KEY**, and **AV_VISION_MODEL**, then import the
external transcript in the same ingestion:

    av ingest video.mp4 --transcript-json transcript.json --dense-vision \
      --fps-sample 0.0666667 --max-frames 300 --no-embed --db ./recipe.db
    av ask "Your question" --top-k 5 --db ./recipe.db

The example samples about one still every 15 seconds, with a maximum of 300
frames. That budget is appropriate for about 75 minutes; the frame limit caps
extraction instead of redistributing samples across a longer video. Record the
actual **dense_caption_frames** and sampling settings from ingestion output.
Use a fresh database for this FTS-only example: **--no-embed** prevents new
embeddings, but does not remove embeddings from an existing database.

**--dense-vision** runs this dense-frame path. **--captions** additionally selects
the cascade caption/summary path; combining them changes the workload and cost.
Set **AV_CHAT_MODEL** and, if needed, the provider environment for the answer
command separately. The caption VLM, external ASR, answer model, and direct-video
baseline are independent choices. Neither this command example nor an output
receipt establishes answer quality without checking the answer against the media.

## What happens

1. Retrieve timestamped index artifacts for the question. The index may contain
   transcripts, captions, or both; material not represented in it can be missed.
2. Ask Jev for source relevance. Relevant text is not yet proof of an answer.
3. Refine bounded scene context, combine overlaps, rank candidates, and cap the
   context supplied to the configured answer model.
4. Generate an answer with citations, then ask a separate support question about
   whether that evidence supports the answer's material claims.
5. If support is insufficient or unavailable, optionally inspect bounded sampled
   frames using an explicitly configured stronger model; otherwise return an
   uncertain result. A successful inspection also receives a support check.

A successful judgment that rejects every source returns no supported evidence;
it does not silently revert to rejected raw hits. If refinement itself fails,
AV warns and can answer from raw retrieval with heuristic confidence. A failed
support judgment is labeled unknown, not supported.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| AV_CHAT_MAX_OUTPUT_TOKENS | 1024 | Output-token cap passed to the configured answer provider |
| AV_TYPESAFE_MODEL | jev-latest | Jev model used for judgments |
| AV_TYPESAFE_TIMEOUT_SEC | 30 | Timeout for a judgment request |
| AV_TYPESAFE_MAX_RETRIES | 1 | Retry bound |
| AV_REFINE_RELEVANCE_MIN | 0.5 | Source relevance threshold |
| AV_REFINE_SUPPORT_MIN | 0.5 | Separate answer-support threshold |
| AV_REFINE_MAX_SCENES | 8 | Maximum merged scenes sent to synthesis |
| AV_REFINE_BATCH_SIZE | 10 | Maximum hits in a relevance request |
| AV_REFINE_CONTEXT_EVENTS | 3 | Bounded neighboring-event context on each side |

The explicit TypeSafe endpoint is configurable with **AV_TYPESAFE_ENDPOINT**.
The thresholds are routing settings, not calibrated probabilities of correctness.
FTS retrieval remains the first stage: an unscoped query with no matching hits
does not inspect the whole archive with the stronger model.

## Optional stronger inspection

Configure **AV_STRONG_VISION_API_BASE_URL**, **AV_STRONG_VISION_MODEL**, and
**AV_STRONG_VISION_API_KEY** when the endpoint requires authentication. AV does
not choose a stronger paid provider automatically. Source media must still be
available at the indexed location, and ffmpeg must be installed.

| Budget setting | Default | Meaning |
|---|---|---|
| AV_INSPECTION_MAX_WINDOWS | 2 | Maximum selected scene windows |
| AV_INSPECTION_MAX_SECONDS | 120 | Total duration represented by selected windows |
| AV_INSPECTION_MAX_FRAMES | 12 | Total sampled-frame budget across attempts |
| AV_INSPECTION_MAX_ATTEMPTS | 1 | At most 2 attempts may be configured |
| AV_INSPECTION_DENSE_PASS | false | Allow a second sampling pass within the total budget |

This path sends sampled still images. It does not send native video or audio and
cannot establish continuity, timing between unsampled frames, or inaudible spoken
content. More frames are not a guarantee of better evidence. Record stronger
inspection costs and its invocation rate separately when comparing workloads.

## Read the result

The answer, citations, and confidence remain available. Refined results also expose
**route**, **evidence_status**, **confidence_basis**, **refinement**, **warnings**,
**inspected_windows**, **ask_settings**, and **stage_usage**. Treat confidence
according to its basis. The ordinary path also reports answer/query-embedding
usage and its answer settings. A route ending in **answer_failed** preserves
available stage usage and sets **evidence_status=answer_unavailable**; inspect
these fields instead of treating any JSON response as a successful answer.

| Result | Interpretation |
|---|---|
| refined / supported | Jev support check met the configured threshold |
| refined_no_results | No retrieval hits or all retrieved sources rejected |
| refinement_fallback / raw_unjudged | Refinement failed; raw retrieval answer has not been judged |
| refined_uncertain | Evidence did not support a reliable answer or support remains unknown |
| vision_inspected / sampled_frames_supported | Bounded frames yielded an answer that passed support checking |
| legacy, refined, or refinement_fallback route ending in answer_failed | Answer provider failed; usage can be incomplete and incurred costs remain possible |

Stage usage covers relevance, boundary, answer, support, vision, and query
embedding when reported.
Request counts can include retry attempts; inspect metering completeness before
using aggregate usage as a full cost receipt.
Missing token counts and unpriced retrieval infrastructure are unknown, not zero. The
metadata describes execution and evidence routing; it is not a complete billing
receipt. No benchmark in this recipe establishes quality improvement, parity
with direct-video models, or a universal cost ratio.

The current receipts include completed ASR and Gemini 3.8 baseline calls, an
aborted caption attempt, a successful image smoke call, and a 32-token cap probe
blocked by HTTP 502/no route. **No completed AV Grok+Jev comparison exists yet.**
Do not claim speed, cost, or quality parity from these component attempts.
