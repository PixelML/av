# djev-spark refined ask — self-hosted decisions over the same contract

This recipe runs the [Jev-refined ask](../jev-refined-ask/README.md) workflow
against a **self-hosted [djev-spark](https://github.com/mmastrac/djev-spark)**
server instead of the hosted TypeSafe System One endpoint. Both speak the same
documented `POST /v1/systemone` contract, so `av ask` refinement, evidence
grouping, and answer-support checking work unchanged — only the decision
provider changes.

**Evidence status:** adapter and offline compatibility tests are complete; no
live djev quality or latency measurement has been made in this repository.
Everything below about runtime performance is upstream's own documentation and
is labelled as such.

## What the adapter guarantees

Audited upstream revision: `mmastrac/djev-spark`
`1444f3e927f83ba508e5b28a4fd4fdd9ecd0976b`.

- **Identity honesty.** djev-spark ignores a request's `model` field and
  reports the model it actually served. `DjevClient` records
  `served_model`, `server_engine`, and the redacted `served_endpoint_host`
  into every stage-usage receipt and into the `refinement` metadata of
  `av ask` output. A djev answer is never presented as a Jev measurement.
- **Visible rejection.** Responses that are malformed or incomplete for the
  questions actually asked — missing answers, answers to unasked questions,
  type mismatches, choices outside the offered options, score legends that do
  not match the levels, probabilities out of range or not summing to one —
  fail with a `djev-spark response failed validation` error instead of being
  silently coerced. A question the server skipped (`ask_if`) is preserved as
  `null`.
- **Same failure semantics as the hosted lane.** Timeouts, explicit retries
  (429/5xx and connection errors, exponential backoff), sanitized error
  messages, and provider-accurate fallback warnings ("djev-spark refinement
  was unavailable…") match `SystemOneClient` conventions.
- **Reproducibility knob.** Every request carries `AV_DJEV_SEED`
  (default `42`), the server-side default seed.

## Configuration

No endpoint ships with av. djev-spark is software you host; an explicit
endpoint selects this lane over hosted Jev (the endpoint wins even if a
TypeSafe key is also set).

| Variable | Example | Purpose |
|---|---|---|
| `AV_DJEV_ENDPOINT` | `http://10.1.2.3:8011/v1/systemone` | Structured server endpoint (compose default port `8011`) |
| `AV_DJEV_API_KEY` | any string | Sent as `Authorization: Bearer …`; needed only when the server sets `API_KEY` |
| `AV_DJEV_MODEL` | `dgemma` | Advisory only; the server ignores it. Omit unless you want it recorded in request logs |
| `AV_DJEV_TIMEOUT_SEC` | `180` | Cold structured reads on long states are slow (see below) |
| `AV_DJEV_MAX_RETRIES` | `1` | Retry count, mirroring the hosted lane's default |
| `AV_DJEV_SEED` | `42` | Sampler seed sent with every decision |

Verify wiring offline:

```bash
av config show | jq '.djev_endpoint, .djev_api_key'
```

Then ask exactly as in the Jev recipe:

```bash
av ask "When does the door open?" --video-id <id>
```

The response's `ask_settings.decision_provider` reads `"djev-spark"`, and
`refinement.served_model` / `refinement.server_engine` carry the identity the
server reported. If the server is unreachable, `route` becomes
`refinement_fallback`, the warning names the provider, and the answer is
produced from raw retrieval with `evidence_status: raw_unjudged`.

## Running the server safely (read before you start it)

These are requirements this project imposes on itself; the upstream defaults
are more permissive.

- **Networking.** Upstream's compose file uses host networking and both
  servers listen on all interfaces. Put the box behind a firewall or tailnet
  and reach it over a private address.
- **Authentication.** Upstream serves POST routes with no API key unless its
  `API_KEY` env var is set. Set one, and set `AV_DJEV_API_KEY` to match.
  `/health` (GET) stays open by upstream design.
- **Playground.** Upstream's test page (`TEST_PAGE=1`) is off by default;
  leave it off on any shared host.
- **Resources.** The model is DiffusionGemma 26B-A4B NVFP4 and wants a
  compatible GPU, verified non-boot storage for weights and build caches, and
  the upstream entrypoint's own headroom check. Never place weights, Docker
  caches, or media on the av control plane's boot disk. Do not start this
  next to workloads you do not own; do not evict anything to make room.
- **Model terms.** Gemma weight licence terms apply to the checkpoint; the
  server source files carry Apache-2.0 headers. av implements the wire
  protocol independently and distributes no upstream code or weights.

## Performance claims (upstream-documented, untested here)

Upstream's README reports roughly 0.1-second warm reads and, on its 128k
profile, a 104.94-second cold versus 0.44-second warm selection at a
110,707-token state. These are the author's numbers on their hardware — not
measurements from this repository, and not a guarantee that any full video
clips in two seconds. Measure your own cold/warm/export timings before
relying on any latency figure; record p50/p95 with sample counts when you do.

## Comparison protocol (pending)

A fair comparison against hosted Jev and the plain retrieval baseline must use
the same frozen corpus, queries, candidate windows, and transcript/vision text
as the sibling Jev clipping task, with independent labels — not provider
scores as ground truth. Measure relevance, recall, absent-topic false
positives, timing boundaries, duplication, and export validity separately per
lane. That evaluation is **not started here**: it is gated on the shared
fixture contract and on an authorized runtime. This recipe will gain a
results section only from actual recorded runs.

## Known limitations

- Image input (`data:` URLs / multipart parts) is supported by the upstream
  server but not sent by this adapter; the primary comparison is text-to-text.
- Clip candidate construction and export belong to the shared clipping
  contract, not to this provider adapter.
- Offered-label probabilities are diagnostics; nothing here treats them as
  calibrated quality scores.
