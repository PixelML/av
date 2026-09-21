"""Topic clipping: retrieval-grounded candidates with typed Jev decisions.

AV owns candidate construction, timing, deduplication, and export. Jev
(System One) owns typed relevance, coherence, visual-evidence, boundary, and
appeal decisions, expressed as Noul/Choice/Score operations over
source-verbatim text. No generative completion is presented as a decision and
no model is asked to invent timestamps: every returned second is an artifact
boundary already present in the database.

Objective source support (relevance, coherence, visual evidence) is decided
and gated separately from subjective highlight appeal, which only orders
candidates that already passed the objective gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from typing import Any

from av.core.config import AVConfig
from av.db.models import VideoRecord
from av.db.repository import Repository, _fmt_timestamp
from av.search.query import natural_language_fts_query
from av.search.refine import (
    RefinementError,
    SystemOneClient,
    _judge_bounds,
    _ordered_contiguous_events,
    _probability,
)
from av.search.usage import new_usage, record_usage

DEFAULT_RETRIEVAL_LIMIT = 24
DEFAULT_MAX_CANDIDATES = 24
MAX_QUOTES_PER_MODALITY = 12
# Candidates are built generously around the hit (this multiple of the
# duration cap) so the boundary judge can choose tight setup/payoff bounds
# inside them; assembly still enforces the hard cap at event boundaries.
CLIP_BOUND_SLACK = 1.5
_VISION_TYPES = frozenset({"caption", "dense_caption", "scene"})
_STAGE_NAMES = ("relevance", "coherence", "visual", "boundary", "appeal")


class ClipError(RuntimeError):
    """A clipping request that cannot be served as specified."""


class DecisionBudgetExhausted(ClipError):
    """The explicit per-run decision request ceiling was reached."""

    def __init__(self, message: str, *, stage_usage: dict[str, dict] | None = None) -> None:
        super().__init__(message)
        self.stage_usage = stage_usage or {}


class BudgetedSystemOne:
    """SystemOneClient wrapper enforcing an explicit per-run request ceiling.

    Requests are counted as actual HTTP attempts, retries included. Cached
    replays (warm selection) perform no provider call and add no request.
    """

    def __init__(self, inner: SystemOneClient, *, max_requests: int) -> None:
        if max_requests < 1:
            raise ClipError("Decision request ceiling must be at least 1")
        self.inner = inner
        self.max_requests = max_requests
        self.attempts = 0
        self.cache_hits = 0
        self.usage = new_usage()
        self._cache: dict[str, tuple[dict, dict]] = {}

    def ask(self, state: Any, questions: dict[str, dict]) -> tuple[dict, dict]:
        key = json.dumps(
            {"state": state, "questions": questions}, sort_keys=True, default=str
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.attempts >= self.max_requests:
            raise DecisionBudgetExhausted(
                f"Decision request ceiling reached ({self.max_requests}); "
                f"{len(questions)} question(s) left undecided"
            )
        try:
            answers, usage = self.inner.ask(state, questions)
        except RefinementError as exc:
            attempts = max(exc.attempts, 1)
            self.attempts += attempts
            record_usage(
                self.usage,
                exc.raw_usage,
                requests=attempts,
                ambiguous_attempts=attempts > 1 or exc.raw_usage is None,
            )
            raise
        attempts = 1
        cleaned: dict[str, Any] | None = usage
        if isinstance(usage, dict):
            raw_attempts = usage.get("_attempts", 1)
            attempts = raw_attempts if isinstance(raw_attempts, int) and raw_attempts > 0 else 1
            cleaned = {key_: value for key_, value in usage.items() if key_ != "_attempts"}
        self.attempts += attempts
        record_usage(self.usage, cleaned, requests=attempts, ambiguous_attempts=attempts > 1)
        self._cache[key] = (answers, usage)
        return answers, usage


@dataclass
class ClipCandidate:
    """One contiguous candidate moment, built only from artifact boundaries."""

    candidate_id: str
    video_id: str
    filename: str
    start_sec: float
    end_sec: float
    retrieval_score: float
    hit_artifact_ids: list[str]
    artifact_ids: list[str]
    transcript_ids: list[str]
    vision_ids: list[str]
    vision_rows: list[dict]
    transcript_text: str
    vision_text: str
    events: list[dict]
    hit_event_index: int
    relevance_p: float | None = None
    coherence_p: float | None = None
    visual_support_p: float | None = None
    boundary_start_sec: float | None = None
    boundary_end_sec: float | None = None
    boundary_confidence: float | None = None
    appeal_score: float | None = None


def _candidate_id(video_id: str, start: float, end: float) -> str:
    digest = hashlib.sha256(f"{video_id}|{start:.3f}|{end:.3f}".encode()).hexdigest()
    return f"clip-{digest[:12]}"


def _transcript_rows(events: list[dict]) -> list[dict]:
    transcripts: list[dict] = []
    for event in events:
        for row in event.get("rows", ()):
            if row["source_type"] == "transcript":
                transcripts.append(row)
    return transcripts


def _bounded_run(
    events: list[dict],
    hit_index: int,
    max_seconds: float,
) -> tuple[list[dict], int]:
    """Trim a contiguous event run around its hit event.

    Transcript segments usually chain back-to-back, so the raw contiguous run
    can span minutes. Candidates expand from the hit event in alternating
    setup-first order (previous event, then next) while the span stays within
    ``max_seconds``. Every accepted boundary is an existing event boundary.
    """
    lo = hi = hit_index

    def fits(new_lo: int, new_hi: int) -> bool:
        start = min(events[lo]["start"], events[new_lo]["start"])
        end = max(events[hi]["end"], events[new_hi]["end"])
        return end - start <= max_seconds

    setup_first = True
    while True:
        placed = False
        sides = (True, False) if setup_first else (False, True)
        for side in sides:
            if side and lo > 0 and fits(lo - 1, hi):
                lo -= 1
                placed = True
                break
            if not side and hi < len(events) - 1 and fits(lo, hi + 1):
                hi += 1
                placed = True
                break
        if not placed:
            return events[lo : hi + 1], hit_index - lo
        setup_first = not setup_first


def build_candidates(
    topic: str,
    repo: Repository,
    video: VideoRecord,
    *,
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    context_events: int = 3,
    min_seconds: float = 0.0,
    max_seconds: float | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[list[ClipCandidate], dict]:
    """Deterministic candidates: FTS hits expanded into temporal neighborhoods
    bounded by ``max_seconds`` around each hit. No provider calls and no
    invented timestamps."""
    meta: dict[str, Any] = {
        "retrieval_hits": 0,
        "groups": 0,
        "too_short_groups": 0,
        "candidate_count": 0,
        "retrieval_limit": retrieval_limit,
    }
    query = natural_language_fts_query(topic)
    if not query:
        return [], meta
    hits = repo.search_fts(query, limit=retrieval_limit, video_id=video.id)
    meta["retrieval_hits"] = len(hits)
    if not hits:
        return [], meta

    # Group hits sharing one contiguous temporal neighborhood so one moment
    # yields one candidate, not one candidate per retrieval hit.
    groups: list[dict] = []
    for hit in sorted(hits, key=lambda h: (h.timestamp_sec, h.artifact_id or "")):
        result = {
            "timestamp_sec": hit.timestamp_sec,
            "end_sec": hit.end_sec,
            "artifact_id": hit.artifact_id,
            "source_type": hit.source_type,
            "text": hit.text,
        }
        window = repo.get_refinement_window(
            video.id, hit.timestamp_sec, before=context_events, after=context_events
        )
        # Timing anchors on transcript boundaries when they exist; wide vision
        # captions never drive clip bounds. Captions remain attached evidence
        # and take over as the timing basis only without any transcript.
        transcript_artifacts = [
            artifact for artifact in window if artifact.type == "transcript"
        ]
        events: list[dict] = []
        hit_index = -1
        if transcript_artifacts:
            events, hit_index = _ordered_contiguous_events(
                result, transcript_artifacts, append_missing_hit=False
            )
        if not events or hit_index < 0:
            events, hit_index = _ordered_contiguous_events(result, window)
        if max_seconds is not None:
            events, hit_index = _bounded_run(
                events, hit_index, max_seconds * CLIP_BOUND_SLACK
            )
        hit_event = events[hit_index]
        for group in groups:
            group_start = min(event["start"] for event in group["events"])
            group_end = max(event["end"] for event in group["events"])
            if hit_event["start"] < group_end and hit_event["end"] > group_start:
                merged: dict[tuple[float, float, str], dict] = {}
                for event in group["events"] + events:
                    merged[(event["start"], event["end"], event["text"])] = event
                group["events"] = sorted(
                    merged.values(), key=lambda e: (e["start"], e["end"])
                )
                if hit.artifact_id:
                    group["hit_ids"].add(hit.artifact_id)
                group["best_score"] = max(group["best_score"], hit.score)
                if max_seconds is not None:
                    # Re-bound the union around its earliest hit event so
                    # merged groups keep the bounded-candidate invariant.
                    anchor = next(
                        (
                            index
                            for index, event in enumerate(group["events"])
                            if group["hit_ids"] & set(event["artifact_ids"])
                        ),
                        0,
                    )
                    group["events"], _ = _bounded_run(
                        group["events"], anchor, max_seconds * CLIP_BOUND_SLACK
                    )
                break
        else:
            hit_ids = {hit.artifact_id} if hit.artifact_id else set()
            groups.append({
                "events": list(events),
                "hit_ids": hit_ids,
                "best_score": hit.score,
                "first_hit_sec": hit.timestamp_sec,
            })
    meta["groups"] = len(groups)

    candidates: list[ClipCandidate] = []
    for group in groups:
        events = group["events"]
        start = min(event["start"] for event in events)
        end = max(event["end"] for event in events)
        if end - start < min_seconds:
            meta["too_short_groups"] += 1
            continue
        transcripts = _transcript_rows(events)
        artifact_ids: list[str] = []
        for event in events:
            artifact_ids.extend(event["artifact_ids"])
        # Attach overlapping vision rows as evidence; they never move bounds.
        vision_rows: list[dict] = []
        for artifact in repo.get_artifacts_overlapping(video.id, start, end):
            if artifact.type in _VISION_TYPES:
                vision_rows.append({
                    "artifact_id": artifact.id,
                    "source_type": artifact.type,
                    "start": artifact.start_sec,
                    "end": artifact.end_sec if artifact.end_sec is not None else artifact.start_sec,
                    "text": artifact.text,
                })
        vision_rows.sort(key=lambda row: (row["start"], row["artifact_id"]))
        hit_event_index = next(
            (
                index
                for index, event in enumerate(events)
                if group["hit_ids"] & set(event["artifact_ids"])
            ),
            0,
        )
        candidates.append(
            ClipCandidate(
                candidate_id=_candidate_id(video.id, start, end),
                video_id=video.id,
                filename=video.filename,
                start_sec=start,
                end_sec=end,
                retrieval_score=group["best_score"],
                hit_artifact_ids=sorted(group["hit_ids"]),
                artifact_ids=list(dict.fromkeys(artifact_ids)),
                transcript_ids=[row["artifact_id"] for row in transcripts],
                vision_ids=[row["artifact_id"] for row in vision_rows],
                vision_rows=vision_rows,
                transcript_text="\n".join(row["text"] for row in transcripts),
                vision_text="\n".join(row["text"] for row in vision_rows),
                events=events,
                hit_event_index=hit_event_index,
            )
        )
    candidates.sort(key=lambda c: (-c.retrieval_score, c.start_sec, c.candidate_id))
    candidates = candidates[: max_candidates]
    meta["candidate_count"] = len(candidates)
    return candidates, meta


def _clip_payload(candidate: ClipCandidate, key: str) -> dict:
    return {
        "transcript": candidate.transcript_text or "(none)",
        "caption": candidate.vision_text or "(none)",
        "start_sec": candidate.start_sec,
        "end_sec": candidate.end_sec,
    }


def _ask_noul_batch(
    client: BudgetedSystemOne,
    topic: str,
    candidates: list[ClipCandidate],
    *,
    instructions: str,
    true_criteria: str,
    false_criteria: str,
    usage: dict,
) -> dict[str, float]:
    """Batched Noul questions; returns candidate_id -> probability."""
    probabilities: dict[str, float] = {}
    state = {
        "query": topic,
        "clips": {
            f"c{index}": _clip_payload(candidate, f"c{index}")
            for index, candidate in enumerate(candidates)
        },
    }
    questions = {
        f"c{index}": {
            "type": "noul",
            "instructions": instructions.format(key=f"c{index}"),
            "criteria": {"true": true_criteria, "false": false_criteria},
        }
        for index in range(len(candidates))
    }
    try:
        answers, call_usage = client.ask(state, questions)
    except RefinementError as exc:
        # The wrapper already counted these attempts in its aggregate; the
        # affected stage must also report them before the error propagates.
        attempts = max(exc.attempts, 1)
        record_usage(
            usage,
            exc.raw_usage,
            requests=attempts,
            ambiguous_attempts=attempts > 1 or exc.raw_usage is None,
        )
        raise
    record_usage(usage, {k: v for k, v in (call_usage or {}).items() if k != "_attempts"})
    for index, candidate in enumerate(candidates):
        key = f"c{index}"
        answer = answers.get(key)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise RefinementError(f"System One returned an invalid Noul answer for {key}")
        probabilities[candidate.candidate_id] = _probability(answer.get("noul"), key)
    return probabilities


_RELEVANCE_INSTRUCTIONS = (
    "Does `clips.{key}` contain spoken or visual source evidence that materially "
    "concerns the subject or event described by `query`?"
)
_RELEVANCE_TRUE = "The clip content materially concerns the requested subject or event."
_RELEVANCE_FALSE = (
    "The clip is unrelated, or mentions the subject only incidentally."
)
_COHERENCE_INSTRUCTIONS = (
    "Could `clips.{key}` be understood on its own, without surrounding video "
    "context, as a complete spoken or visual moment about `query`?"
)
_COHERENCE_TRUE = (
    "The clip carries its own setup and payoff and does not depend on missing "
    "context before or after it."
)
_COHERENCE_FALSE = (
    "The clip references unavailable context, starts or ends mid-thought, or "
    "would confuse a viewer who saw only this clip."
)
_VISUAL_INSTRUCTIONS = (
    "Does the visual description in `clips.{key}` show visual evidence about the "
    "subject or event described by `query`?"
)
_VISUAL_TRUE = "The described visuals materially show the requested subject or event."
_VISUAL_FALSE = (
    "The described visuals are unrelated, absent, or contradict the requested "
    "subject or event."
)
_APPEAL_INSTRUCTIONS = (
    "Rate how compelling `clips.{key}` is as a standalone highlight for a viewer "
    "interested in `query`, from 0.0 (not compelling) to 1.0 (exceptional)."
)
_APPEAL_CRITERIA = {
    "1.0": "An exceptional, self-contained highlight for this topic.",
    "0.5": "A usable but ordinary moment for this topic.",
    "0.0": "Not compelling as a highlight for this topic.",
}


def _ask_appeal(
    client: BudgetedSystemOne,
    topic: str,
    candidates: list[ClipCandidate],
    usage: dict,
) -> dict[str, float]:
    state = {
        "query": topic,
        "clips": {
            f"c{index}": _clip_payload(candidate, f"c{index}")
            for index, candidate in enumerate(candidates)
        },
    }
    questions = {
        f"c{index}": {
            "type": "score",
            "instructions": _APPEAL_INSTRUCTIONS.format(key=f"c{index}"),
            "criteria": _APPEAL_CRITERIA,
        }
        for index in range(len(candidates))
    }
    try:
        answers, call_usage = client.ask(state, questions)
    except RefinementError as exc:
        attempts = max(exc.attempts, 1)
        record_usage(
            usage,
            exc.raw_usage,
            requests=attempts,
            ambiguous_attempts=attempts > 1 or exc.raw_usage is None,
        )
        raise
    record_usage(usage, {k: v for k, v in (call_usage or {}).items() if k != "_attempts"})
    scores: dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        key = f"c{index}"
        answer = answers.get(key)
        if not isinstance(answer, dict) or answer.get("type") != "score":
            raise RefinementError(f"System One returned an invalid Score answer for {key}")
        value = answer.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RefinementError(f"System One returned an invalid score for {key}")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise RefinementError(f"System One returned an out-of-range score for {key}")
        scores[candidate.candidate_id] = value
    return scores


def decide_candidates(
    candidates: list[ClipCandidate],
    topic: str,
    client: BudgetedSystemOne,
    *,
    batch_size: int = 10,
    context_events: int = 3,
) -> tuple[dict[str, dict], dict[str, dict], dict]:
    """Run typed decision stages over the candidates in place.

    ``client`` is the budgeted wrapper so warm replays hit the same decision
    cache and make zero provider calls. Returns (per-candidate decision
    snapshots, per-stage usage, budget info). Decisions mutate the candidates.
    """
    stage_usage = {name: new_usage() for name in _STAGE_NAMES}

    def batches(items: list[ClipCandidate]) -> list[list[ClipCandidate]]:
        return [items[offset : offset + batch_size] for offset in range(0, len(items), batch_size)]

    try:
        for batch in batches(candidates):
            probabilities = _ask_noul_batch(
                client,
                topic,
                batch,
                instructions=_RELEVANCE_INSTRUCTIONS,
                true_criteria=_RELEVANCE_TRUE,
                false_criteria=_RELEVANCE_FALSE,
                usage=stage_usage["relevance"],
            )
            for candidate in batch:
                candidate.relevance_p = probabilities[candidate.candidate_id]

        for batch in batches(candidates):
            probabilities = _ask_noul_batch(
                client,
                topic,
                batch,
                instructions=_COHERENCE_INSTRUCTIONS,
                true_criteria=_COHERENCE_TRUE,
                false_criteria=_COHERENCE_FALSE,
                usage=stage_usage["coherence"],
            )
            for candidate in batch:
                candidate.coherence_p = probabilities[candidate.candidate_id]

        with_vision = [c for c in candidates if c.vision_text.strip()]
        for batch in batches(with_vision):
            probabilities = _ask_noul_batch(
                client,
                topic,
                batch,
                instructions=_VISUAL_INSTRUCTIONS,
                true_criteria=_VISUAL_TRUE,
                false_criteria=_VISUAL_FALSE,
                usage=stage_usage["visual"],
            )
            for candidate in batch:
                candidate.visual_support_p = probabilities[candidate.candidate_id]

        for candidate in candidates:
            if len(candidate.events) < 2:
                # A single-event candidate has no neighboring event to judge;
                # its containing-group bounds are the only honest boundary.
                continue
            start, end, confidence, _, _, _ = _judge_bounds(
                client, topic, candidate.events, candidate.hit_event_index,
                context_events, stage_usage["boundary"],
            )
            candidate.boundary_start_sec = start
            candidate.boundary_end_sec = end
            candidate.boundary_confidence = confidence

        try:
            for batch in batches(candidates):
                scores = _ask_appeal(client, topic, batch, stage_usage["appeal"])
                for candidate in batch:
                    candidate.appeal_score = scores[candidate.candidate_id]
        except RefinementError:
            # Appeal is subjective ordering, never an objective gate; a
            # provider that cannot answer Score questions degrades ordering,
            # not selection. Recorded as unavailable, with the failed request
            # already counted in the stage usage.
            stage_usage["appeal"]["appeal_available"] = False

        decisions = {
            candidate.candidate_id: {
                "relevance_p": candidate.relevance_p,
                "coherence_p": candidate.coherence_p,
                "visual_support_p": candidate.visual_support_p,
                "boundary_confidence": candidate.boundary_confidence,
                "appeal_score": candidate.appeal_score,
            }
            for candidate in candidates
        }
        budget = {
            "requests_used": client.attempts,
            "request_cap": client.max_requests,
            "cache_hits_warm": client.cache_hits,
        }
        return decisions, stage_usage, budget
    except DecisionBudgetExhausted as exc:
        exc.stage_usage = {name: usage for name, usage in stage_usage.items()}
        exc.total_usage = client.usage
        exc.requests_used = client.attempts
        raise
    except RefinementError as exc:
        exc.stage_usage = {name: usage for name, usage in stage_usage.items()}
        exc.total_usage = client.usage
        exc.requests_used = client.attempts
        raise


def _shaped_window(
    candidate: ClipCandidate,
    *,
    min_seconds: float,
    max_seconds: float,
) -> tuple[float, float, str, list[str], bool]:
    """Apply judged bounds, then shape duration at event boundaries only.

    Extension preference is setup first (earlier events), then payoff.
    Trimming drops the earliest non-hit events first, preserving payoff.
    Returns (start, end, boundary_source, dropped_events, duration_shaped).
    """
    events = candidate.events
    if candidate.boundary_start_sec is not None and candidate.boundary_end_sec is not None:
        start = candidate.boundary_start_sec
        end = candidate.boundary_end_sec
        boundary_source = "judged_choice"
    else:
        start = candidate.start_sec
        end = candidate.end_sec
        boundary_source = "containing_group"
    original_window = (start, end)
    hit_ids = set(candidate.hit_artifact_ids)

    def contains_hit(event: dict) -> bool:
        return bool(hit_ids & set(event["artifact_ids"]))

    # Trim toward max_seconds, earliest-first, never dropping a hit event.
    kept = [event for event in events if event["end"] > start and event["start"] < end]
    dropped: list[str] = []
    while len(kept) > 1:
        span = max(event["end"] for event in kept) - min(event["start"] for event in kept)
        if span <= max_seconds:
            break
        first = kept[0]
        if contains_hit(first):
            break
        dropped.append(f"{first['start']:.3f}-{first['end']:.3f}")
        kept = kept[1:]
    start = min(event["start"] for event in kept)
    end = max(event["end"] for event in kept)

    # Extend toward min_seconds inside the candidate group: setup first.
    group_before = [event for event in events if event["end"] <= start + 1e-9]
    group_after = [event for event in events if event["start"] >= end - 1e-9]
    while end - start < min_seconds and group_before:
        start = group_before.pop()["start"]
    while end - start < min_seconds and group_after:
        end = group_after.pop()["end"]

    if end - start > max_seconds and len(kept) == 1:
        # A single oversized source event: source alignment wins over the
        # duration target; the violation is exposed instead of hidden.
        boundary_source = "oversized_event"
    return start, end, boundary_source, dropped, (start, end) != original_window

def _clip_record(
    candidate: ClipCandidate,
    rank: int,
    *,
    config: AVConfig,
    start: float,
    end: float,
    boundary_source: str,
    dropped_events: list[str],
    duration_shaped: bool,
    thresholds: dict,
    selection_basis: str,
) -> dict:
    duration = end - start
    overlap_rows: list[dict] = [
        row
        for event in candidate.events
        for row in event.get("rows", ())
        if row["end"] > start and row["start"] < end
    ]
    overlap_rows.sort(key=lambda row: (row["start"], row["artifact_id"]))
    vision_rows = [
        row for row in candidate.vision_rows if row["end"] > start and row["start"] < end
    ]
    quotes: list[dict] = []
    truncated = False
    for modality, rows in (
        ("transcript", overlap_rows),
        ("vision", vision_rows),
    ):
        for row in rows[:MAX_QUOTES_PER_MODALITY]:
            quotes.append({
                "artifact_id": row["artifact_id"],
                "source_type": row["source_type"],
                "modality": modality,
                "start_sec": round(row["start"], 3),
                "end_sec": round(row["end"], 3),
                "text": row["text"],
            })
        if len(rows) > MAX_QUOTES_PER_MODALITY:
            truncated = True

    boundary_uncertain = (
        candidate.boundary_confidence is None
        or candidate.boundary_confidence < config.clip_boundary_conf_min
        or boundary_source in {"containing_group", "oversized_event"}
        or duration_shaped
    )
    uncertainty: dict[str, Any] = {
        "boundary": (
            round(1.0 - candidate.boundary_confidence, 4)
            if candidate.boundary_confidence is not None
            else None
        ),
        "visual_support": (
            round(1.0 - candidate.visual_support_p, 4)
            if candidate.visual_support_p is not None
            else None
        ),
        "appeal": (
            round(1.0 - candidate.appeal_score, 4)
            if candidate.appeal_score is not None
            else None
        ),
        "notes": [],
    }
    if candidate.visual_support_p is None:
        uncertainty["notes"].append("visual_support_unavailable")
    if candidate.appeal_score is None:
        uncertainty["notes"].append("appeal_unavailable")
    if boundary_uncertain:
        uncertainty["notes"].append("boundary_uncertain")

    return {
        "clip_id": candidate.candidate_id,
        "rank": rank,
        "video_id": candidate.video_id,
        "filename": candidate.filename,
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(duration, 3),
        "timestamp_formatted": _fmt_timestamp(start),
        "quotes": quotes,
        "quotes_truncated": truncated,
        "support": {
            "transcript_artifact_ids": [
                artifact_id
                for artifact_id in candidate.transcript_ids
            ],
            "vision_artifact_ids": [artifact_id for artifact_id in candidate.vision_ids],
        },
        "decisions": {
            "relevance_p": candidate.relevance_p,
            "coherence_p": candidate.coherence_p,
            "visual_support_p": candidate.visual_support_p,
            "boundary_confidence": candidate.boundary_confidence,
            "appeal_score": candidate.appeal_score,
        },
        "uncertainty": uncertainty,
        "boundary_source": boundary_source,
        "boundary_uncertain": boundary_uncertain,
        "duration_shaped": duration_shaped,
        "dropped_events_for_duration": dropped_events,
        "selection_basis": selection_basis,
        "thresholds": thresholds,
        "decision_provider": "typesafe" if candidate.relevance_p is not None else "none",
        "decision_model": config.typesafe_model if candidate.relevance_p is not None else None,
    }


def assemble_clips(
    candidates: list[ClipCandidate],
    video: VideoRecord,
    config: AVConfig,
    *,
    clips_wanted: int,
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
    decisions_available: bool,
) -> tuple[list[dict], dict]:
    """Deterministic gating, duration shaping, dedup, ranking, selection."""
    thresholds = {
        "relevance_min": config.clip_relevance_min,
        "coherence_min": config.clip_coherence_min,
        "visual_min": config.clip_visual_min,
        "boundary_conf_min": config.clip_boundary_conf_min,
    }
    rejected: dict[str, int] = {}
    gated: list[ClipCandidate] = []
    if decisions_available:
        for candidate in candidates:
            if candidate.relevance_p is None:
                rejected["undecided"] = rejected.get("undecided", 0) + 1
                continue
            if candidate.relevance_p < thresholds["relevance_min"]:
                rejected["low_relevance"] = rejected.get("low_relevance", 0) + 1
                continue
            if candidate.coherence_p is None or candidate.coherence_p < thresholds["coherence_min"]:
                rejected["low_coherence"] = rejected.get("low_coherence", 0) + 1
                continue
            if (
                candidate.visual_support_p is not None
                and candidate.visual_support_p < thresholds["visual_min"]
            ):
                rejected["low_visual_support"] = rejected.get("low_visual_support", 0) + 1
                continue
            gated.append(candidate)
    else:
        gated = list(candidates)

    shaped: list[tuple[ClipCandidate, float, float, str, list[str], bool]] = []
    for candidate in gated:
        start, end, boundary_source, dropped, duration_shaped = _shaped_window(
            candidate, min_seconds=min_seconds, max_seconds=max_seconds
        )
        shaped.append((candidate, start, end, boundary_source, dropped, duration_shaped))

    appeal_available = decisions_available and all(c.appeal_score is not None for c in gated) and bool(gated)
    if appeal_available:
        selection_basis = "appeal_then_relevance_then_retrieval"
        shaped.sort(
            key=lambda item: (
                -item[0].appeal_score,
                -(item[0].relevance_p or 0.0),
                -item[0].retrieval_score,
                item[1],
                item[0].candidate_id,
            )
        )
    else:
        selection_basis = (
            "relevance_then_retrieval" if decisions_available else "retrieval_only"
        )
        shaped.sort(
            key=lambda item: (
                -(item[0].relevance_p if item[0].relevance_p is not None else 0.0),
                -item[0].retrieval_score,
                item[1],
                item[0].candidate_id,
            )
        )

    selected: list[tuple[ClipCandidate, float, float, str, list[str], bool]] = []
    warnings: list[str] = []
    for item in shaped:
        _, start, end, _, _, _ = item
        if any(start < other[2] and end > other[1] for other in selected):
            rejected["overlapping"] = rejected.get("overlapping", 0) + 1
            continue
        selected.append(item)
    selected = selected[:clips_wanted]

    clips: list[dict] = []
    for rank, (candidate, start, end, boundary_source, dropped, duration_shaped) in enumerate(selected, 1):
        record = _clip_record(
            candidate,
            rank,
            config=config,
            start=start,
            end=end,
            boundary_source=boundary_source,
            dropped_events=dropped,
            duration_shaped=duration_shaped,
            thresholds=thresholds,
            selection_basis=selection_basis,
        )
        if record["boundary_source"] == "oversized_event":
            warnings.append(
                f"Clip {rank} exceeds the duration target because a single source "
                "event spans more than the limit; no uncut boundary was available."
            )
        clips.append(record)

    meta = {
        "thresholds": thresholds,
        "selection_basis": selection_basis,
        "appeal_available": appeal_available,
        "rejected": rejected,
        "target_seconds": target_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
    }
    return clips, {"assembly": meta, "warnings": warnings}


def clip_video(
    topic: str,
    video_id: str,
    repo: Repository,
    config: AVConfig,
    *,
    clips_wanted: int = 3,
    target_seconds: float = 30.0,
    min_seconds: float = 10.0,
    max_seconds: float | None = None,
    decide: bool = True,
    client: SystemOneClient | None = None,
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_requests: int | None = None,
) -> dict:
    """Find topic-specific highlight clips in one indexed video.

    Selection only; rendering is a separate pipeline stage. With ``decide``
    the candidate set is judged with typed System One operations under an
    explicit per-run request ceiling; without it, ranking is deterministic
    retrieval order and no provider call is made.
    """
    if clips_wanted < 1:
        raise ClipError("Clip count must be at least 1")
    if target_seconds <= 0 or min_seconds <= 0:
        raise ClipError("Target and minimum durations must be positive")
    if max_seconds is None:
        max_seconds = target_seconds
    if not min_seconds <= target_seconds <= max_seconds:
        raise ClipError("Durations must satisfy min_seconds <= target_seconds <= max_seconds")
    if max_requests is None:
        max_requests = config.clip_request_cap

    video = repo.get_video(video_id)
    if max_seconds > video.duration_sec > 0:
        max_seconds = video.duration_sec

    warnings: list[str] = []
    prepare_start = time.perf_counter()
    candidates, build_meta = build_candidates(
        topic,
        repo,
        video,
        retrieval_limit=retrieval_limit,
        context_events=config.refine_context_events,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        max_candidates=max_candidates,
    )
    prepare_ms = (time.perf_counter() - prepare_start) * 1000

    empty_usage = {name: new_usage() for name in _STAGE_NAMES}
    decisions_block: dict[str, Any] = {
        "enabled": decide,
        "provider": "typesafe" if decide else "none",
        "model": config.typesafe_model if decide else None,
        "request_cap": max_requests if decide else 0,
        "requests_used": 0,
        "cap_exhausted": False,
        "cache_hits_warm": 0,
    }

    def receipt(status: str, clips: list[dict], assembly: dict) -> dict:
        return {
            "status": status,
            "topic": topic,
            "video_id": video.id,
            "filename": video.filename,
            "source_duration_sec": video.duration_sec,
            "clips": clips,
            "candidates": build_meta,
            "decisions": decisions_block,
            "selection": assembly["assembly"],
            "stage_usage": assembly.get("stage_usage", empty_usage),
            "timings": {
                "prepare_ms": round(prepare_ms, 1),
                "selection_ms": round(assembly.get("selection_ms", 0.0), 1),
                "selection_warm_ms": round(assembly.get("selection_warm_ms", 0.0), 1),
                "render_ms": 0.0,
            },
            "warnings": warnings + assembly.get("warnings", []),
        }

    if not candidates:
        return receipt("no_usable_clips", [], {
            "assembly": {
                "thresholds": {},
                "selection_basis": "none",
                "appeal_available": False,
                "rejected": {},
                "target_seconds": target_seconds,
                "min_seconds": min_seconds,
                "max_seconds": max_seconds,
            },
            "warnings": [],
        })

    if not decide:
        clips, assembly = assemble_clips(
            candidates,
            video,
            config,
            clips_wanted=clips_wanted,
            target_seconds=target_seconds,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            decisions_available=False,
        )
        return receipt("deterministic_only", clips, {
            "assembly": assembly["assembly"],
            "warnings": assembly["warnings"]
            + ["Decisions were disabled; clips are retrieval-ranked and unjudged."],
        })

    if client is None:
        client = SystemOneClient(config)
    budgeted = BudgetedSystemOne(client, max_requests=max_requests)
    decide_start = time.perf_counter()
    try:
        _decision_snapshots, stage_usage, budget = decide_candidates(
            candidates,
            topic,
            budgeted,
            batch_size=config.refine_batch_size,
            context_events=config.refine_context_events,
        )
    except DecisionBudgetExhausted as exc:
        partial_usage = {
            name: exc.stage_usage.get(name, new_usage())
            for name in _STAGE_NAMES
        }
        warnings.append(str(exc))
        decisions_block["requests_used"] = getattr(exc, "requests_used", 0)
        decisions_block["cap_exhausted"] = True
        warnings.append(
            "The decision request ceiling was reached before every candidate was "
            "judged; undecided candidates are reported, not selected."
        )
        decided = [
            candidate
            for candidate in candidates
            if candidate.relevance_p is not None and candidate.coherence_p is not None
        ]
        undecided_count = len(candidates) - len(decided)
        # Objective gates may be decided while later stages were truncated;
        # those clips carry reduced-confidence boundaries and no appeal.
        partial_count = sum(
            1
            for candidate in decided
            if candidate.boundary_confidence is None or candidate.appeal_score is None
        )
        clips, assembly = assemble_clips(
            decided,
            video,
            config,
            clips_wanted=clips_wanted,
            target_seconds=target_seconds,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            decisions_available=True,
        )
        status = "request_cap_reached"
        selection_ms = (time.perf_counter() - decide_start) * 1000
        result = receipt(status, clips, {
            "assembly": assembly["assembly"],
            "warnings": assembly["warnings"],
            "stage_usage": partial_usage,
            "selection_ms": selection_ms,
        })
        result["candidates"]["undecided"] = undecided_count
        result["candidates"]["partial_decisions"] = partial_count
        return result
    except RefinementError as exc:
        warnings.append(
            "Typed decisions were unavailable from the configured provider; no "
            "clips were selected. Rerun with --no-decide for retrieval-only output."
        )
        return receipt("decision_unavailable", [], {
            "assembly": {
                "thresholds": {},
                "selection_basis": "none",
                "appeal_available": False,
                "rejected": {},
                "target_seconds": target_seconds,
                "min_seconds": min_seconds,
                "max_seconds": max_seconds,
            },
            "warnings": [],
            "stage_usage": getattr(exc, "stage_usage", {}) or empty_usage,
            "selection_ms": (time.perf_counter() - decide_start) * 1000,
        })
    selection_ms = (time.perf_counter() - decide_start) * 1000

    decisions_block["requests_used"] = budget["requests_used"]
    decisions_block["cache_hits_warm"] = budget["cache_hits_warm"]
    decisions_block["cap_exhausted"] = budget["requests_used"] >= budget["request_cap"]

    # Warm replay: identical selection served from the in-run decision cache,
    # zero provider calls. Measures harness overhead, not provider latency.
    warm_start = time.perf_counter()
    warm_candidates = [replace(candidate) for candidate in candidates]
    _, _, warm_budget = decide_candidates(
        warm_candidates,
        topic,
        budgeted,
        batch_size=config.refine_batch_size,
        context_events=config.refine_context_events,
    )
    warm_ms = (time.perf_counter() - warm_start) * 1000
    decisions_block["cache_hits_warm"] = warm_budget["cache_hits_warm"]

    clips, assembly = assemble_clips(
        candidates,
        video,
        config,
        clips_wanted=clips_wanted,
        target_seconds=target_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        decisions_available=True,
    )
    if assembly["assembly"]["appeal_available"] is False:
        warnings.append(
            "Highlight appeal ordering was unavailable; candidates are ordered by "
            "objective relevance and retrieval score."
        )
    return receipt("ok" if clips else "no_usable_clips", clips, {
        "assembly": assembly["assembly"],
        "warnings": assembly["warnings"],
        "stage_usage": stage_usage,
        "selection_ms": selection_ms,
        "selection_warm_ms": warm_ms,
    })
