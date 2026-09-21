# Clip evaluation contract (v1)

This directory is the initial public shared evaluation contract for AV topic
clipping (`av clip`). It exists so that deterministic retrieval/windows and
Jev-decided selection can be compared on **identical candidates** with
**independent labels**, and so future lanes can extend the corpus without
redefining the metrics.

## What is frozen here

| File | Role |
|---|---|
| `corpus.json` | Synthetic corpus: 3 content styles, inline transcript/vision segments, stable artifact IDs, per-video SHA-256 checksums, and the deterministic ffmpeg media-generation spec |
| `queries.json` | Labeled dev query set: present topics, absent topics, repeated topics, noisy ASR, missing vision, contradictory vision, context-dependent soundbites |
| `queries-heldout.json` | Held-out labels. Run at most once per evidence cycle; never against prompt or threshold iteration |
| `receipts/` | Committed receipts from offline runs (small JSON, no media) |

`contract_version` is `1`. The loader (`av.clip_eval.corpus.load_corpus`)
verifies every checksum and rejects any modified corpus. The metric
definitions live in `av.clip_eval.contract` and are quoted verbatim in this
file; changing either requires bumping the contract version.

## Rights and provenance

All text in `corpus.json` was invented for this contract. No real persons,
events, footage, transcripts, or captions are included. No private source,
prompts, rubrics, customer assets, or private experiment data was used. Media
files are **not stored in the repository**: `corpus.json` records
deterministic ffmpeg commands (`testsrc2` color source + sine tone) so any
ffmpeg build can regenerate synthetic media outside the working tree for
export-validity checks. The corpus is therefore synthetic and rights-cleared
by construction.

## Metrics (contract v1)

Every metric compares returned clips against frozen labels only. Jev decision
scores are never ground truth.

- `precision_at_k` — hits in the first k returned clips / k. A hit means
  interval IoU ≥ 0.5 against a labeled moment.
- `known_moment_recall` — labeled moments covered (IoU ≥ 0.5) / total moments.
- `absent_topic_fp` — 1.0 when any clip is returned for an `absent` query.
  The dev set includes an incidental-mention trap: the sponsor line mentions
  "cloud credits" once; it is labeled absent.
- `boundary_error_sec` — mean of (|start delta| + |end delta|) / 2 over hits.
- `duplication_rate` — returned clip pairs with IoU ≥ 0.2 / clip count.
- `context_loss_count` — returned hits that start after a labeled setup head
  (more than 1s of slack). q-005 ("laminating trick") is the canonical case.

Timing is reported as ingestion (corpus materialization), prepare
(retrieval + candidates), selection (typed decisions + assembly),
selection_warm (identical replay from the in-run decision cache, zero
provider calls), and render (ffmpeg export) milliseconds. Percentiles are
reported only when at least five samples support them.

## How to run

```bash
# Offline (deterministic + labeled-mock Jev pipeline exercise)
python -m av.clip_eval \
  --corpus clip-eval/corpus.json --queries clip-eval/queries.json \
  --db /path/eval.db --media-dir /path/media --export-dir /path/render \
  --arms deterministic,jev_mock \
  --out clip-eval/receipts/offline-mock-dev.json
```

## Evidence status

Committed receipts are **offline mock runs**: the `jev_mock` arm exercises
the typed Noul/Choice/Score pipeline against frozen labels. It proves the
harness works and quantifies the deterministic baseline; it is **not** a
measurement of Jev quality. Live Jev runs additionally require an authorized
allowance, an explicit per-run request ceiling, and usage tracking; none was
available for the initial contract, so live quality evidence is pending.

Measured at target 30s / min 10s / 2 clips (dev, `receipts/offline-mock-dev.json`):

| Arm | P@k (k=2) | Known-moment recall | Absent-topic FP | Boundary error (s) | Duplication |
|---|---|---|---|---|---|
| deterministic | 0.1875 | 0.625 | 1.0 | 3.667 | 0.0 |
| jev_mock | 0.25 | 0.875 | 0.0 | 2.75 | 0.0 |

Held-out, run once after the final harness state (`receipts/offline-mock-heldout.json`):
deterministic P@k 0.25 / recall 0.667 / boundary error 5.0s versus
jev_mock P@k 0.375 / recall 1.0 / boundary error 3.0s. Both arms have zero
absent-topic false positives, zero duplication, and no context-loss cases.

Known gaps, kept deliberately: q-005 measures the duration-vs-context
tradeoff — the 32s labeled moment cannot be covered at a 30s cap without
losing part of the setup head, and both arms report it (the Jev arm returns a
partial-setup clip that scores a context loss; the deterministic arm misses
entirely). Tiny synthetic fixtures are smoke evidence for the contract, not
proof of broad quality.
