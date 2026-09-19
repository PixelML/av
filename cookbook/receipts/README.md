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
| [cap-probe-32-restored-incompatible.json](cap-probe-32-restored-incompatible.json) | Restored-route exact-model probe at commit `6a1cde2` produced 260 completion tokens against cap 32; ingestion/query stopped |
| [live-ingestion-local-probe-timeout.json](live-ingestion-local-probe-timeout.json) | Fresh 300-frame attempt at commit `e24845d` stopped locally before provider request 1; transcript imported, 0 captions persisted, $0 provider estimate |
| [grok-live-ingestion.json](grok-live-ingestion.json) | Completed 300-frame Grok ingestion at `dd9dfa2`; six responses exceeded the requested 200-token advisory cap |
| [grok-legacy-query.json](grok-legacy-query.json) | Completed single Grok-only answer with transcript citation; no Jev relevance/support request |
| [jev-credential-blocked.json](jev-credential-blocked.json) | Jev arm stopped before request because `AV_TYPESAFE_API_KEY` was absent |

The receipts are evidence for those individual attempts only. The completed query
was Grok-only through AV’s legacy route. No completed AV Grok+Jev comparison
exists. They do not establish speed, cost, or quality
parity between the direct-video baseline and an AV pipeline.
