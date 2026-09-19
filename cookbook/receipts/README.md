# Sanitized reproduction receipts

These JSON files contain selected sanitized result evidence plus usage and
execution metadata. They do not include media, transcript or caption corpora,
credentials, upload URIs, or private routes. The source media is not redistributed.
The native-video baseline receipt intentionally retains its benchmark question
and returned answer, including the short explanatory rationale and timestamp,
because those fields are needed to interpret the recorded result.

| Receipt | Result |
|---|---|
| [asr.json](asr.json) | Completed 75-window external ASR plus one retained failed alignment attempt |
| [gemini38-baseline.json](gemini38-baseline.json) | Completed native-video Gemini 3.8 baseline query |
| [caption-aborted.json](caption-aborted.json) | Four metered caption responses; ingestion aborted and no captions persisted |
| [caption-smoke.json](caption-smoke.json) | Successful one-frame caption smoke request |
| [cap-probe-32-incompatible.json](cap-probe-32-incompatible.json) | 32-token cap probe blocked by HTTP 502/no available route; usage unknown |
| [cap-probe-32-direct-incompatible.json](cap-probe-32-direct-incompatible.json) | Direct exact-model 32-token cap probe returned usage but produced 309 completion tokens; output cap ignored |

The receipts are evidence for those individual attempts only. No completed AV
Grok+Jev comparison exists yet. They do not establish speed, cost, or quality
parity between the direct-video baseline and an AV pipeline.
