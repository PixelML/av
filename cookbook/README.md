# AV cookbook

Runnable recipes for the open-source **av** CLI live here alongside the code.

| Recipe | What it demonstrates | Evidence status |
|---|---|---|
| [Cost model](cost-model/README.md) | Separate tokens, estimates, unknown costs, ingestion, queries, failures, reservations, and cap accounting | Completed component receipts; paired AV comparison incomplete |
| [Jev-refined ask](jev-refined-ask/README.md) | Build a source-bound transcript sidecar, retrieve indexed moments, refine evidence, answer, and check support | Runnable recipe; no speed, cost, or quality parity claim |
| [Sanitized receipts](receipts/README.md) | Completed ASR/baseline, caption smoke/abort, incompatible cap probes, and the fresh zero-request local media-probe failure | No media, transcript/caption corpus, credentials, upload URIs, or private routes; baseline question and returned answer/rationale retained |

The [original public cost notebook](https://github.com/PixelML/cookbook/tree/main/agentic-video/cost-model)
remains available at its existing URL. Its Composer demonstrations are historical
context, not measurements of this CLI. New AV reproduction results belong in
this cookbook with their own media, model, configuration, and cost provenance.

The current evidence includes a completed Gemini 3.8 direct-video baseline. The
restored Grok route returned the exact requested model and usage, but returned
260 completion tokens against `max_completion_tokens=32`. A later explicitly
authorized fresh attempt imported the completed transcript, then stopped on a
local media-probe timeout before frame extraction: 0 captions and 0 provider
requests. There is **no completed AV Grok+Jev
comparison yet**; do not infer speed, cost, or quality parity from component
receipts.
