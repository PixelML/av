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

The current evidence includes a completed Gemini 3.8 direct-video baseline and a
completed AV Grok-only path: 300/300 caption requests, 75 transcript windows, and
one correct answer with transcript citation. The selected route ignored output
caps in both 32-token probes, and six of the 300 caption responses exceeded the
requested 200-token advisory cap. No Jev request ran because the required
credential was absent, so there is **no completed AV Grok+Jev comparison**. Do
not infer all-in speed, cost, or quality parity from component receipts.
