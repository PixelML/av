# Jev topic clipping

Recipe for the `av clip` command: find and optionally export topic-specific
highlight clips from one already-indexed video — no re-ingestion, no
model-invented timestamps.

## What it demonstrates

- Candidate construction from retrieval plus bounded temporal neighborhoods,
  with timing anchored on transcript segment boundaries (vision captions are
  attached as evidence, never as clock sources).
- Typed Jev decisions — Noul relevance, Noul standalone coherence, Noul
  visual evidence, Choice boundaries, Score highlight appeal — kept strictly
  separate: objective source support gates selection, appeal only orders.
- An explicit per-run request ceiling, a truthful per-stage usage receipt,
  and a warm replay that measures harness overhead with zero provider calls.
- Optional ffmpeg export with ffprobe playability/synchronization checks.

## Run

```bash
uv sync --extra dev
uv run av ingest examples/lecture.mp4          # once; any supported ingest path
uv run av clip "quantum error correction" \
  --video-id <id> --clips 2 --target-seconds 30 \
  --export ./clips --overwrite-export
```

stdout is a single JSON document: clips with source-verbatim quotes,
supporting artifact IDs, decision scores, uncertainty, provider/model, the
request-cap state, per-stage usage, and prepare/selection/warm/render
timings. `--no-decide` skips Jev and ranks retrieval candidates
deterministically; absent topics return `no_usable_clips` instead of guesses.

## Evaluation

The shared contract lives in [`clip-eval/`](../../clip-eval/README.md):
synthetic corpus with frozen checksums, labeled dev and held-out query sets,
frozen metric definitions, and a runner (`python -m av.clip_eval`) that
compares the deterministic arm against the Jev-decided arm on identical
candidates. Committed receipts are offline mock runs; see the evidence
status section there.

## Evidence status

Runnable recipe with deterministic tests and an offline mock evaluation
receipt. **No live Jev calls were made for this recipe** — no authorized
allowance was available in this environment, so live selection quality,
live token/request totals, and live charges are pending, unmeasured, and
not claimed. The "two-second, two-cent" social claim about clipping cost is
treated as unverified community folklore; it is not an acceptance target and
is not reproduced here. No speed, cost, or quality parity claim is made
against any other tool.

## Advisor receipt

Advisor review was sought through the enabled advisor runtime at two points
(repository orientation before settling the candidate schema, and at final
verification). The runtime reported no advisor peers on both checks, so no
advice was observed or applied; the unavailable category is recorded here
rather than synthesized.
