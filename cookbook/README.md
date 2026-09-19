# AV cookbook

Runnable recipes for the open-source **av** CLI live here alongside the code.

| Recipe | What it demonstrates | Evidence status |
|---|---|---|
| [Cost model](cost-model/README.md) | Separate tokens, estimates, unknown costs, ingestion, queries, failures, reservations, and cap accounting | Completed one-question comparison; no aggregate parity claim |
| [Jev-refined ask](jev-refined-ask/README.md) | Build a source-bound transcript sidecar, retrieve indexed moments, refine evidence, answer, and check support | Runnable recipe; no speed, cost, or quality parity claim |
| [Sanitized receipts](receipts/README.md) | Completed ASR/baseline, caption smoke/abort, incompatible cap probes, Grok ingestion, and Jev-refined query | No media, transcript/caption corpus, credentials, upload URIs, or private routes; baseline question and returned answer/rationale retained |

The [original public cost notebook](https://github.com/PixelML/cookbook/tree/main/agentic-video/cost-model)
remains available at its existing URL. Its Composer demonstrations are historical
context, not measurements of this CLI. New AV reproduction results belong in
this cookbook with their own media, model, configuration, and cost provenance.

The current evidence includes a completed Gemini 3.8 direct-video baseline, a
completed 300/300 Grok caption ingestion with 75 transcript windows, one Grok-only
answer, and one later Jev-refined answer for the same question. The selected route
ignored output caps in both 32-token probes, and six of the 300 caption responses
exceeded the requested 200-token advisory cap. The paired result is one question;
do not infer aggregate speed, cost, or quality parity from it.
