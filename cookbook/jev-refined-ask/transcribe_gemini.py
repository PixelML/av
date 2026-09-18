#!/usr/bin/env python3
"""Create an AV transcript sidecar using metered Gemini audio windows.

Requires Python 3.11+, ffmpeg, ffprobe, and GEMINI_API_KEY. No automatic retries.
Run --help for arguments. Timestamps are audio-window bounds, not word alignment.
Every cache artifact is bound to the source hash and complete run configuration.
"""
import argparse
import base64
import concurrent.futures
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid

DEFAULT_MODEL = "gemini-3.5-flash-lite"
API = "https://generativelanguage.googleapis.com/v1beta/models/"
PROMPT = ('Transcribe every audible spoken word faithfully in its original language. '
          'Return JSON only: {"text":"verbatim speech","silence":false}. '
          'Do not summarize or infer missing words. Preserve numbers and percentages as heard. '
          'Use [inaudible] for unintelligible speech. If there is no speech, return '
          '{"text":"","silence":true}. Do not provide timestamps.')
GENERATION = {
    "temperature": 0, "maxOutputTokens": 2048, "responseMimeType": "application/json",
    "responseSchema": {"type": "OBJECT", "properties": {
        "text": {"type": "STRING"}, "silence": {"type": "BOOLEAN"}},
        "required": ["text", "silence"]},
    "thinkingConfig": {"thinkingLevel": "minimal"},
}
MAX_INPUT_TOKENS = 3500
MAX_AUDIO_BYTES = 3000000
RECIPE_REVISION = 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def amount(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("amount must be finite and non-negative") from None
    require(result.is_finite() and result >= 0, "amount must be finite and non-negative")
    return result


def windows(duration):
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool)
            and math.isfinite(duration) and 0 < duration <= 4501,
            "source duration must be positive and at most 4501 seconds")
    starts = list(range(0, math.ceil(duration), 60))
    if len(starts) > 1 and duration - starts[-1] < 1:
        starts.pop()  # Fold a short remainder into the preceding audio window.
    require(len(starts) <= 75, "maximum 75 audio windows")
    return [(start, starts[i + 1] if i + 1 < len(starts) else duration)
            for i, start in enumerate(starts)]


def parse_transcription(value):
    # The measured run included one singleton-array response. Preserve its raw
    # response while accepting exactly this normalization, not arbitrary arrays.
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    require(isinstance(value, dict) and isinstance(value.get("text"), str)
            and isinstance(value.get("silence"), bool), "invalid response schema")
    text = value["text"].strip()
    require(value["silence"] == (not bool(text)), "inconsistent silence/text")
    return text, value["silence"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def write_json(path, value):
    """Atomically replace a JSON artifact so interrupted writes are never cache hits."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def bind_directory(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run-manifest.json"
    if path.exists():
        require(json.loads(path.read_text()) == manifest, "cached manifest mismatch; use a new directory")
    else:
        require(not any(directory.iterdir()), "use an empty output directory")
        write_json(path, manifest)


class Ledger:
    """Persist reservations before sending requests, including abandoned attempts.

    A reservation is a conservative list-price guard, not a measured bill. It is
    never released on failure, so a manual resume cannot forget prior attempts.
    """
    def __init__(self, path, budget, prior):
        self.path = path
        self.lock = threading.Lock()
        self.budget = amount(budget)
        self.reserved = amount(prior)
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["status"] == "reserved":
                    self.reserved += amount(row["reserved_max_usd"])
        require(self.reserved <= self.budget, "prior reservations exceed budget")

    def _write(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, base, ceiling):
        with self.lock:
            ceiling = amount(ceiling)
            require(self.reserved + ceiling <= self.budget, "ASR reservation budget exceeded")
            attempt = dict(base, attempt_id=uuid.uuid4().hex,
                           reserved_max_usd=str(ceiling))
            self._write(dict(attempt, status="reserved"))
            self.reserved += ceiling
            return attempt

    def record(self, row):
        with self.lock:
            self._write(row)

    @property
    def remaining(self):
        return self.budget - self.reserved


def post(key, model, action, payload):
    request = urllib.request.Request(
        API + urllib.parse.quote(model, safe="") + ":" + action,
        data=json.dumps(payload).encode(),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def response_artifact(response, binding, elapsed):
    usage = response.get("usageMetadata") or {}
    candidate = (response.get("candidates") or [{}])[0]
    finish = candidate.get("finishReason")
    text = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []))
    return dict(binding, usage=usage, finish_reason=finish, text=text,
                wall_seconds=elapsed)


def parse_response_artifact(artifact, binding):
    for name, expected in binding.items():
        require(artifact.get(name) == expected, "cached response mismatch; use a new directory")
    require(artifact.get("finish_reason") == "STOP",
            "cached generation is incomplete; use a new directory")
    return parse_transcription(json.loads(artifact["text"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget-usd", default="0.50",
                        help="Conservative upstream list-price reservation limit, not a billing limit")
    parser.add_argument("--prior-reserved-usd", default="0",
                        help="Reservations from earlier attempts outside this output directory")
    parser.add_argument("--input-usd-per-million")
    parser.add_argument("--output-usd-per-million")
    parser.add_argument("--first-only", action="store_true",
                        help="Process only the first window; do not produce a full transcript")
    args = parser.parse_args(argv)
    require(bool(args.model.strip()), "model must not be empty")
    budget, prior = amount(args.budget_usd), amount(args.prior_reserved_usd)
    require(budget > 0 and prior <= budget, "invalid budget or prior reservation")
    require(bool(args.input_usd_per_million) == bool(args.output_usd_per_million),
            "supply both custom rate arguments")
    custom_rates = args.input_usd_per_million is not None
    require(args.model == DEFAULT_MODEL or custom_rates,
            "a different model requires explicit input and output rates and compatible API semantics")
    input_rate = amount(args.input_usd_per_million if custom_rates else "0.30")
    output_rate = amount(args.output_usd_per_million if custom_rates else "2.50")
    rates = {"input_per_million": str(input_rate), "output_per_million": str(output_rate),
             "basis": "upstream list-price estimate; account billing is unverified",
             "source": "user supplied" if custom_rates else "https://ai.google.dev/gemini-api/docs/pricing",
             "as_of": None if custom_rates else "2026-09-18",
             "input_policy": "all input conservatively priced at the audio rate; no cache discount"}
    key = os.environ.get("GEMINI_API_KEY", "")
    require(bool(key), "GEMINI_API_KEY is required")
    require(args.source.is_file(), "source must be a readable file")
    duration = float(json.loads(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_format", "-of", "json", str(args.source)]))["format"]["duration"])
    intervals = windows(duration)
    digest = hashlib.sha256()
    with args.source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    source_hash = digest.hexdigest()
    manifest = {"schema_version": 1, "recipe_revision": RECIPE_REVISION,
                "source_sha256": source_hash, "source_bytes": args.source.stat().st_size,
                "model": args.model,
                "source_duration_sec": duration, "windows": intervals,
                "prompt": PROMPT, "generation_config": GENERATION,
                "max_input_tokens": MAX_INPUT_TOKENS, "max_audio_bytes": MAX_AUDIO_BYTES,
                "audio": {"format": "flac", "channels": 1, "sample_rate_hz": 16000},
                "max_workers": 2, "retry_count": 0, "rates": rates,
                "prior_reserved_usd": str(prior), "budget_usd": str(budget)}
    # Normalize tuples to JSON arrays before equality checking on resume.
    manifest = json.loads(json.dumps(manifest))
    manifest_id = fingerprint(manifest)
    bind_directory(args.output_dir, manifest)
    ledger = Ledger(args.output_dir / "asr-usage.jsonl", budget, prior)
    stopped = threading.Event()

    def one(index):
        require(not stopped.is_set(), "another window failed; no new request started")
        start, end = intervals[index]
        output = args.output_dir / f"chunk-{index:03d}.json"
        binding = {"manifest_sha256": manifest_id, "source_sha256": source_hash,
                   "model": args.model, "chunk": index, "start_sec": start,
                   "end_sec": end}
        if output.exists():
            cached = json.loads(output.read_text())
            require(all(cached.get(name) == expected for name, expected in binding.items()),
                    "cached chunk mismatch; use a new directory")
            parse_transcription(cached)
            return cached
        response_path = args.output_dir / f"chunk-{index:03d}.response.json"
        if response_path.exists():
            artifact = json.loads(response_path.read_text())
            text, silent = parse_response_artifact(artifact, binding)
            row = dict(binding, text=text, silence=silent)
            write_json(output, row)
            print(json.dumps({"chunk": index, "status": "recovered_from_response"}), flush=True)
            return row
        audio = args.output_dir / f"chunk-{index:03d}.flac"
        # Rebuild unfinished windows; an existing audio file is not a valid cache.
        subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(start),
                        "-i", str(args.source), "-t", str(end - start), "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "flac", "-y", str(audio)], check=True, timeout=90)
        require(0 < audio.stat().st_size < MAX_AUDIO_BYTES, "audio bytes ceiling")
        contents = [{"role": "user", "parts": [{"text": PROMPT}, {"inlineData": {
            "mimeType": "audio/flac", "data": base64.b64encode(audio.read_bytes()).decode()}}]}]
        # Reserve a conservative maximum before either provider request. Count-token
        # calls are not assumed free, and reservations remain charged to the local
        # cap after failures or interrupted runs.
        ceiling = (MAX_INPUT_TOKENS * input_rate
                   + GENERATION["maxOutputTokens"] * output_rate) / 1000000
        base = ledger.reserve(dict(binding, stage="asr", provider_requests_max=2,
                                   automatic_retries=0, rates=rates), ceiling)
        probe_began = time.monotonic()
        try:
            count_response = post(key, args.model, "countTokens", {"contents": contents})
        except Exception as exc:
            ledger.record(dict(base, status="failed", failed_action="countTokens",
                               error_type=type(exc).__name__, http_status=getattr(exc, "code", None),
                               input_tokens=None, output_tokens=None, completeness="unknown",
                               wall_seconds=time.monotonic() - probe_began))
            raise RuntimeError("ASR token-count request failed; reservation retained") from None
        count = count_response.get("totalTokens")
        require(isinstance(count, int) and not isinstance(count, bool)
                and 0 < count <= MAX_INPUT_TOKENS, "input token ceiling")
        require(not stopped.is_set(), "another window failed; no new generation started")
        request_base = dict(base, input_count_probe=count,
                            count_tokens_wall_seconds=time.monotonic() - probe_began)
        began = time.monotonic()
        try:
            response = post(key, args.model, "generateContent", {
                "contents": contents, "generationConfig": GENERATION})
        except Exception as exc:
            ledger.record(dict(request_base, status="failed", failed_action="generateContent",
                               error_type=type(exc).__name__,
                               http_status=getattr(exc, "code", None), input_tokens=None,
                               output_tokens=None, completeness="unknown",
                               wall_seconds=time.monotonic() - began))
            raise RuntimeError("ASR request failed; receipt saved") from None
        elapsed = time.monotonic() - began
        artifact = response_artifact(response, binding, elapsed)
        # Persist the provider response before validating it. A safe resume can
        # recover a valid response without issuing a duplicate paid request.
        write_json(response_path, artifact)
        usage = artifact["usage"]
        ledger.record(dict(request_base, status="response",
                           finish_reason=artifact["finish_reason"],
                           input_tokens=usage.get("promptTokenCount"),
                           output_tokens=usage.get("candidatesTokenCount"),
                           thinking_tokens=usage.get("thoughtsTokenCount"),
                           cached_tokens=usage.get("cachedContentTokenCount"),
                           raw_usage=usage, wall_seconds=elapsed,
                           cache_policy="no explicit cache requested; absent cache meter is unknown"))
        text, silent = parse_response_artifact(artifact, binding)
        row = dict(binding, text=text, silence=silent)
        write_json(output, row)
        print(json.dumps({"chunk": index, "status": "valid_silence" if silent else "transcribed"}), flush=True)
        return row

    def guarded(index):
        try:
            return one(index)
        except Exception:
            stopped.set()
            raise

    indices = [0] if args.first_only else range(len(intervals))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(guarded, indices))
    if not args.first_only:
        rows.sort(key=lambda row: row["chunk"])
        segments = [{key: row[key] for key in ("start_sec", "end_sec", "text")}
                    for row in rows if not row["silence"]]
        write_json(args.output_dir / "transcript.json", {
            "model": args.model, "segments": segments,
            "provenance": {"source_sha256": source_hash, "source_duration_sec": duration,
                           "stage": "external Gemini ASR sidecar; separate AV import",
                           "window_seconds": 60, "chunk_count": len(rows),
                           "silent_windows": sum(row["silence"] for row in rows),
                           "entire_audio_submitted": True,
                           "timestamp_quality": "coarse fixed audio windows, not speech alignment",
                           "human_transcript_review": False,
                           "omissions": "model transcription may contain errors or omissions; coverage means input submitted, not verified word completeness",
                           "rates": rates}})
        print(json.dumps({"status": "completed", "windows": len(rows), "artifacts": len(segments),
                          "source_duration_sec": duration, "reserved_usd": str(ledger.reserved),
                          "reservation_remaining_usd": str(ledger.remaining)}))


if __name__ == "__main__":
    main()
