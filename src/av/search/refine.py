"""Jev/System One search refinement with bounded scene expansion.

Source relevance, temporal grouping, and answer support are deliberately separate
decisions. Window sizes and batch sizes are public configuration, not hidden policy.
All provider calls use the documented ``/v1/systemone`` contract; no generative
completion is presented as Jev.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from av.core.config import AVConfig
from av.db.models import ArtifactRecord
from av.db.repository import Repository, _fmt_timestamp
from av.search.usage import new_usage, record_usage

CONTIGUOUS_GAP_SEC = 1.5
MAX_SCENE_TEXT_CHARS = 6000
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RefinementError(RuntimeError):
    """A sanitized System One request or schema failure."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        raw_usage: dict[str, Any] | None = None,
        stage_usage: dict[str, dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.raw_usage = raw_usage
        self.stage_usage = stage_usage or {}


def _record_client_error(usage: dict, error: RefinementError) -> None:
    attempts = max(error.attempts, 1)
    record_usage(
        usage,
        error.raw_usage,
        requests=attempts,
        ambiguous_attempts=attempts > 1 or error.raw_usage is None,
    )


def _record_call_usage(usage: dict, call_usage: dict[str, Any] | None) -> None:
    attempts = 1
    cleaned = call_usage
    if isinstance(call_usage, dict):
        attempts_value = call_usage.get("_attempts", 1)
        attempts = attempts_value if isinstance(attempts_value, int) and attempts_value > 0 else 1
        cleaned = {key: value for key, value in call_usage.items() if key != "_attempts"}
    record_usage(
        usage,
        cleaned,
        requests=attempts,
        ambiguous_attempts=attempts > 1,
    )
    # Server-reported identity travels with the receipt so a reader can tell
    # which runtime actually answered (djev-spark reports the model it served).
    for key in ("served_model", "server_engine", "served_endpoint_host"):
        value = cleaned.get(key) if isinstance(cleaned, dict) else None
        if isinstance(value, str) and value:
            usage[key] = value


def _probability(value: Any, name: str, label: str = "System One") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefinementError(f"{label} returned an invalid probability for {name}")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RefinementError(f"{label} returned an out-of-range probability for {name}")
    return value


class SystemOneClient:
    """Small synchronous client for the documented TypeSafe System One endpoint."""
    provider_label = "Jev/System One"

    def __init__(self, config: AVConfig, *, session: requests.Session | None = None) -> None:
        if not config.typesafe_api_key:
            raise RefinementError("TypeSafe API key is not configured")
        self.endpoint = config.typesafe_endpoint
        self.api_key = config.typesafe_api_key
        self.model = config.typesafe_model
        self.timeout = config.typesafe_timeout_sec
        self.max_retries = config.typesafe_max_retries
        self.session = session or requests.Session()

    def ask(self, state: Any, questions: dict[str, dict]) -> tuple[dict[str, dict], dict]:
        payload = {"state": state, "model": self.model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = "request failed"
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts += 1
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = type(exc).__name__
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                raise RefinementError(
                    f"System One unavailable ({last_error})",
                    attempts=attempts,
                ) from exc
            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                time.sleep(min(0.25 * (2**attempt), 1.0))
                continue
            if not response.ok:
                raise RefinementError(
                    f"System One request failed with HTTP {response.status_code}",
                    attempts=attempts,
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise RefinementError(
                    "System One returned invalid JSON",
                    attempts=attempts,
                ) from exc
            if not isinstance(data, dict):
                raise RefinementError(
                    "System One returned an invalid JSON document",
                    attempts=attempts,
                )
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            answers = data.get("answers")
            if not isinstance(answers, dict):
                raise RefinementError(
                    "System One response is missing answers",
                    attempts=attempts,
                    raw_usage=usage,
                )
            usage = {**usage, "_attempts": attempts}
            return answers, usage
        raise RefinementError(f"System One unavailable ({last_error})", attempts=attempts)


@dataclass
class Scene:
    artifact_id: str
    video_id: str
    filename: str
    source_type: str
    start_sec: float
    end_sec: float
    text: str
    retrieval_score: float
    relevance_p: float
    chunk_start_sec: float
    chunk_end_sec: float
    scene_confidence: float | None = None
    merged_artifact_ids: list[str] = field(default_factory=list)
    hit_texts: list[str] = field(default_factory=list)
    is_broad: bool = False

    @property
    def rank_score(self) -> float:
        return self.relevance_p * self.retrieval_score

    def to_result(self, rank: int) -> dict:
        return {
            "rank": rank,
            "score": self.retrieval_score,
            "video_id": self.video_id,
            "filename": self.filename,
            "timestamp_sec": self.start_sec,
            "end_sec": self.end_sec,
            "timestamp_formatted": _fmt_timestamp(self.start_sec),
            "source_type": self.source_type,
            "text": self.text,
            "artifact_id": self.artifact_id,
            "relevance_p": self.relevance_p,
            "scene_confidence": self.scene_confidence,
            "merged_artifact_ids": self.merged_artifact_ids,
            "chunk_start_sec": self.chunk_start_sec,
            "chunk_end_sec": self.chunk_end_sec,
            "rank_score": self.rank_score,
            "evidence_scope": "broad" if self.is_broad else "scene",
        }


def _artifact_end(artifact: ArtifactRecord | dict) -> float:
    if isinstance(artifact, dict):
        start = float(artifact.get("timestamp_sec", 0))
        end = artifact.get("end_sec")
    else:
        start = artifact.start_sec
        end = artifact.end_sec
    return max(start, float(end)) if end is not None else start


def _ordered_contiguous_events(
    result: dict,
    artifacts: list[ArtifactRecord],
) -> tuple[list[dict], int]:
    hit_start = float(result.get("timestamp_sec", 0))
    hit_end = _artifact_end(result)
    hit_id = str(result.get("artifact_id") or "")
    rows = [
        {
            "artifact_id": artifact.id,
            "source_type": artifact.type,
            "start": artifact.start_sec,
            "end": _artifact_end(artifact),
            "text": artifact.text,
        }
        for artifact in artifacts
    ]
    if hit_id and not any(row["artifact_id"] == hit_id for row in rows):
        rows.append({
            "artifact_id": hit_id,
            "source_type": str(result.get("source_type") or "artifact"),
            "start": hit_start,
            "end": hit_end,
            "text": str(result.get("text") or ""),
        })

    # Build temporal events rather than treating every modality row as a chunk.
    # Strictly overlapping rows attach to one event; touching fixed chunks remain
    # separate events and are linked later by the contiguous-gap rule.
    rows.sort(key=lambda row: (row["start"], -(row["end"] - row["start"]), row["artifact_id"]))
    events: list[dict] = []
    for row in rows:
        point = row["end"] <= row["start"]
        target = None
        for event in reversed(events):
            overlaps = row["start"] < event["end"] and row["end"] > event["start"]
            point_inside = point and event["start"] <= row["start"] <= event["end"]
            if overlaps or point_inside:
                target = event
                break
            if event["end"] < row["start"]:
                break
        label = (
            f"[{row['source_type']} {_fmt_timestamp(row['start'])}-{_fmt_timestamp(row['end'])}] "
            f"{row['text']}"
        )
        if target is None:
            events.append({
                "artifact_ids": [row["artifact_id"]],
                "start": row["start"],
                "end": row["end"],
                "texts": [label],
                "text": label,
            })
        else:
            target["start"] = min(target["start"], row["start"])
            target["end"] = max(target["end"], row["end"])
            target["artifact_ids"].append(row["artifact_id"])
            if label not in target["texts"]:
                target["texts"].append(label)
            target["text"] = "\n".join(target["texts"])
    events.sort(key=lambda event: (event["start"], event["end"]))
    hit_index = next(
        (index for index, event in enumerate(events) if hit_id in event["artifact_ids"]),
        -1,
    )
    if hit_index < 0:
        hit_index = next(
            (
                index
                for index, event in enumerate(events)
                if event["start"] <= hit_start < max(event["end"], event["start"] + 0.001)
            ),
            -1,
        )
    if hit_index < 0:
        label = f"[{result.get('source_type', 'artifact')}] {result.get('text', '')}"
        events.append({
            "artifact_ids": [hit_id],
            "start": hit_start,
            "end": hit_end,
            "texts": [label],
            "text": label,
        })
        events.sort(key=lambda event: (event["start"], event["end"]))
        hit_index = next(index for index, event in enumerate(events) if hit_id in event["artifact_ids"])

    lo = hit_index
    while lo > 0 and events[lo]["start"] - events[lo - 1]["end"] <= CONTIGUOUS_GAP_SEC:
        lo -= 1
    hi = hit_index
    while hi < len(events) - 1 and events[hi + 1]["start"] - events[hi]["end"] <= CONTIGUOUS_GAP_SEC:
        hi += 1
    return events[lo : hi + 1], hit_index - lo


def _boundary_input(events: list[dict], hit_index: int, query: str, window: int) -> dict:
    lo = max(0, hit_index - window)
    hi = min(len(events) - 1, hit_index + window)
    candidates: dict[str, dict] = {}
    start_labels: list[str] = []
    end_labels: list[str] = []
    for index in range(lo, hi + 1):
        offset = index - hit_index
        label = f"e{offset}"
        event = events[index]
        candidates[label] = {
            "start": event["start"],
            "end": event["end"],
            "text": event["text"],
        }
        if offset <= 0:
            start_labels.append(label)
        if offset >= 0:
            end_labels.append(label)
    return {
        "query": query,
        "hit_event": events[hit_index]["text"],
        "surrounding_events": candidates,
        "start_labels": start_labels,
        "end_labels": end_labels,
        "lo": lo,
        "hi": hi,
    }


def _choice_criteria(labels: list[str], candidates: dict[str, dict]) -> dict:
    out = {}
    for label in labels:
        event = candidates[label]
        out[label] = (
            {"what": "the hit event itself; the scene does not extend farther on this side", "text": event["text"]}
            if label == "e0"
            else {"seconds": f"{event['start']}-{event['end']}", "text": event["text"]}
        )
    return out


def _read_choice(
    answer: Any, name: str, allowed: list[str], label: str = "System One"
) -> tuple[str, float]:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise RefinementError(f"{label} returned an invalid Choice answer for {name}")
    choice = answer.get("choice")
    if choice not in allowed:
        raise RefinementError(f"{label} returned an invalid boundary choice for {name}")
    confidence = _probability(answer.get("confidence"), f"{name}.confidence", label)
    return str(choice), confidence


def _judge_bounds(
    client: SystemOneClient,
    query: str,
    events: list[dict],
    hit_index: int,
    window: int,
    usage: dict,
) -> tuple[float, float, float, str, str, dict]:
    data = _boundary_input(events, hit_index, query, window)
    candidates = data["surrounding_events"]
    label = getattr(client, "provider_label", "System One")

    def side_question(side: str) -> dict:
        direction, other = ("earliest", "earlier") if side == "start" else ("latest", "later")
        return {
            "type": "choice",
            "instructions": (
                f"Select the {direction} candidate that belongs to the same continuous video moment "
                f"as `hit_event` for the user's `query`. Use e0 when {other} candidates do not belong "
                "to that moment."
            ),
            "criteria": _choice_criteria(data[f"{side}_labels"], candidates),
        }

    # A side whose only candidate is the hit itself has no alternative to
    # choose. Structured providers reject one-option questions outright, so
    # such sides are resolved locally (the scene simply cannot extend that
    # way) and only sides with a real decision are sent to the provider.
    forced = {
        side: data[f"{side}_labels"][0]
        for side in ("start", "end")
        if len(data[f"{side}_labels"]) < 2
    }
    questions = {
        side: side_question(side) for side in ("start", "end") if side not in forced
    }
    resolved: dict[str, str] = dict(forced)
    confidences: dict[str, float] = {side: 1.0 for side in forced}
    if questions:
        state = {
            "query": query,
            "hit_event": data["hit_event"],
            "surrounding_events": candidates,
            "note": "Candidates are ordered temporal events from one video. e0 contains the retrieved hit; negative labels are earlier and positive labels are later.",
        }
        try:
            answers, call_usage = client.ask(state, questions)
        except RefinementError as exc:
            _record_client_error(usage, exc)
            exc.stage_usage["boundary"] = usage
            raise
        _record_call_usage(usage, call_usage)
        try:
            for side in questions:
                resolved[side], confidences[side] = _read_choice(
                    answers.get(side), side, data[f"{side}_labels"], label
                )
        except RefinementError as exc:
            exc.stage_usage["boundary"] = usage
            raise
    start_event = events[hit_index + int(resolved["start"][1:])]
    end_event = events[hit_index + int(resolved["end"][1:])]
    return (
        start_event["start"],
        end_event["end"],
        min(confidences.values()),
        resolved["start"],
        resolved["end"],
        data,
    )


def judge_relevance(
    client: SystemOneClient,
    query: str,
    results: list[dict],
    *,
    batch_size: int = 10,
) -> tuple[dict[str, float], dict[str, int | None]]:
    usage = new_usage()
    label = getattr(client, "provider_label", "System One")
    probabilities: dict[str, float] = {}
    for offset in range(0, len(results), batch_size):
        batch = results[offset : offset + batch_size]
        clips: dict[str, dict] = {}
        questions: dict[str, dict] = {}
        ids: dict[str, str] = {}
        for index, result in enumerate(batch):
            key = f"c{index}"
            artifact_id = str(result.get("artifact_id") or f"result-{offset + index}")
            ids[key] = artifact_id
            clips[key] = {
                "caption": result.get("text") if result.get("source_type") != "transcript" else "(none)",
                "transcript": result.get("text") if result.get("source_type") == "transcript" else "(none)",
                "source_type": result.get("source_type"),
                "video_id": result.get("video_id"),
                "start_sec": result.get("timestamp_sec"),
                "end_sec": result.get("end_sec"),
            }
            questions[key] = {
                "type": "noul",
                "instructions": f"Does `clips.{key}` contain visual or spoken evidence about the subject or event requested by `query`?",
                "criteria": {
                    "true": "The clip content materially concerns the requested subject or event.",
                    "false": "The clip is unrelated or mentions the subject only incidentally.",
                },
            }
        try:
            answers, call_usage = client.ask({"query": query, "clips": clips}, questions)
        except RefinementError as exc:
            _record_client_error(usage, exc)
            exc.stage_usage["relevance"] = usage
            raise
        _record_call_usage(usage, call_usage)
        try:
            for key, artifact_id in ids.items():
                answer = answers.get(key)
                if not isinstance(answer, dict) or answer.get("type") != "noul":
                    raise RefinementError(f"{label} response is missing Noul answer {key}")
                probabilities[artifact_id] = _probability(answer.get("noul"), key, label)
        except RefinementError as exc:
            exc.stage_usage["relevance"] = usage
            raise
    return probabilities, usage


def _scene_from_result(result: dict, relevance_p: float) -> Scene:
    start = float(result.get("timestamp_sec", 0))
    end = _artifact_end(result)
    artifact_id = str(result.get("artifact_id") or "")
    source_type = str(result.get("source_type") or "")
    text = str(result.get("text") or "")
    return Scene(
        artifact_id=artifact_id,
        video_id=str(result.get("video_id") or ""),
        filename=str(result.get("filename") or ""),
        source_type=source_type,
        start_sec=start,
        end_sec=end,
        text=text,
        retrieval_score=float(result.get("score") or 0),
        relevance_p=relevance_p,
        chunk_start_sec=start,
        chunk_end_sec=end,
        merged_artifact_ids=[artifact_id] if artifact_id else [],
        hit_texts=[text] if text else [],
        is_broad=source_type in {"summary", "report"},
    )


def _expand_scene(
    query: str,
    result: dict,
    relevance_p: float,
    repo: Repository,
    client: SystemOneClient,
    usage: dict,
    context_events: int,
) -> Scene:
    scene = _scene_from_result(result, relevance_p)
    if scene.is_broad:
        return scene
    artifacts = repo.get_refinement_window(
        scene.video_id,
        scene.chunk_start_sec,
        before=context_events,
        after=context_events,
    )
    events, hit_index = _ordered_contiguous_events(result, artifacts)
    containing_event = events[hit_index]
    scene.start_sec = min(scene.start_sec, containing_event["start"])
    scene.end_sec = max(scene.end_sec, containing_event["end"])
    if len(events) < 2:
        return scene
    start, end, confidence, _, _, _ = _judge_bounds(
        client, query, events, hit_index, context_events, usage
    )
    scene.start_sec = min(start, scene.start_sec)
    scene.end_sec = max(end, scene.end_sec)
    scene.scene_confidence = confidence
    return scene


def merge_overlapping_scenes(scenes: list[Scene]) -> tuple[list[Scene], int]:
    by_video: dict[str, list[Scene]] = {}
    for scene in scenes:
        if scene.is_broad:
            continue
        by_video.setdefault(scene.video_id, []).append(scene)
    merged: list[Scene] = [scene for scene in scenes if scene.is_broad]
    merged_count = 0
    for video_scenes in by_video.values():
        video_scenes.sort(key=lambda scene: (scene.start_sec, scene.end_sec))
        current = video_scenes[0]
        for next_scene in video_scenes[1:]:
            if next_scene.start_sec <= current.end_sec:
                merged_count += 1
                # The higher retrieval score supplies representative fields;
                # merged relevance uses the maximum accepted probability.
                best = next_scene if next_scene.retrieval_score > current.retrieval_score else current
                current = Scene(
                    artifact_id=best.artifact_id,
                    video_id=best.video_id,
                    filename=best.filename,
                    source_type=best.source_type,
                    start_sec=min(current.start_sec, next_scene.start_sec),
                    end_sec=max(current.end_sec, next_scene.end_sec),
                    text=best.text,
                    retrieval_score=best.retrieval_score,
                    relevance_p=max(current.relevance_p, next_scene.relevance_p),
                    chunk_start_sec=best.chunk_start_sec,
                    chunk_end_sec=best.chunk_end_sec,
                    scene_confidence=min(
                        value for value in (current.scene_confidence, next_scene.scene_confidence) if value is not None
                    ) if current.scene_confidence is not None or next_scene.scene_confidence is not None else None,
                    merged_artifact_ids=current.merged_artifact_ids + next_scene.merged_artifact_ids,
                    hit_texts=list(dict.fromkeys(current.hit_texts + next_scene.hit_texts)),
                )
            else:
                merged.append(current)
                current = next_scene
        merged.append(current)
    return merged, merged_count


def _hydrate_scene_text(scene: Scene, repo: Repository) -> None:
    if scene.is_broad:
        return
    artifacts = repo.get_artifacts_overlapping(scene.video_id, scene.start_sec, scene.end_sec)
    parts = [f"[retrieved hit] {text}" for text in scene.hit_texts if text.strip()]
    seen_text = {text.strip() for text in scene.hit_texts if text.strip()}
    for artifact in artifacts:
        text = artifact.text.strip()
        if artifact.id in scene.merged_artifact_ids or not text or text in seen_text:
            continue
        part = f"[{_fmt_timestamp(artifact.start_sec)}-{_fmt_timestamp(_artifact_end(artifact))} {artifact.type}] {text}"
        if part not in parts:
            parts.append(part)
            seen_text.add(text)
    hydrated = "\n".join(parts)
    if hydrated:
        scene.text = hydrated[:MAX_SCENE_TEXT_CHARS]


def refine_search_results(
    query: str,
    results: list[dict],
    repo: Repository,
    config: AVConfig,
    *,
    client: SystemOneClient | None = None,
) -> tuple[list[dict], dict, dict[str, dict[str, int | None]]]:
    client = client or SystemOneClient(config)
    label = getattr(client, "provider_label", "System One")
    probabilities, relevance_usage = judge_relevance(
        client,
        query,
        results,
        batch_size=config.refine_batch_size,
    )
    kept = []
    for result in results:
        artifact_id = str(result.get("artifact_id") or "")
        probability = probabilities.get(artifact_id)
        if probability is None:
            raise RefinementError(
                f"{label} omitted a source relevance probability",
                stage_usage={"relevance": relevance_usage},
            )
        if probability >= config.refine_relevance_min:
            kept.append((result, probability))
    meta = {
        "status": "success" if kept else "no_relevant_evidence",
        "raw_count": len(results),
        "dropped_count": len(results) - len(kept),
        "merged_count": 0,
        "capped_count": 0,
        "scene_count": 0,
        "video_count": 0,
        "relevance_min": config.refine_relevance_min,
    }
    boundary_usage = new_usage()
    if not kept:
        return [], meta, {"relevance": relevance_usage, "boundary": boundary_usage}

    try:
        expanded = [
            _expand_scene(
                query,
                result,
                probability,
                repo,
                client,
                boundary_usage,
                config.refine_context_events,
            )
            for result, probability in kept
        ]
    except RefinementError as exc:
        exc.stage_usage.setdefault("relevance", relevance_usage)
        exc.stage_usage["boundary"] = boundary_usage
        raise
    merged, merged_count = merge_overlapping_scenes(expanded)
    for scene in merged:
        _hydrate_scene_text(scene, repo)
    merged.sort(key=lambda scene: scene.rank_score, reverse=True)
    capped = merged[: config.refine_max_scenes]
    meta.update({
        "merged_count": merged_count,
        "capped_count": len(merged) - len(capped),
        "scene_count": len(capped),
        "video_count": len({scene.video_id for scene in capped}),
    })
    return (
        [scene.to_result(index + 1) for index, scene in enumerate(capped)],
        meta,
        {"relevance": relevance_usage, "boundary": boundary_usage},
    )


def judge_answer_support(
    client: SystemOneClient,
    question: str,
    answer: str,
    evidence: list[dict],
) -> tuple[float, dict[str, int | None]]:
    state = {
        "question": question,
        "answer": answer,
        "evidence": [
            {
                "video_id": item.get("video_id"),
                "start_sec": item.get("timestamp_sec"),
                "end_sec": item.get("end_sec"),
                "source_type": item.get("source_type"),
                "text": item.get("text"),
            }
            for item in evidence
        ],
    }
    questions = {
        "is_supported": {
            "type": "noul",
            "instructions": "Is `answer` directly supported by `evidence` and sufficient to answer `question` without adding unsupported facts?",
            "criteria": {
                "true": "The evidence directly supports the answer's material claims and answers the question.",
                "false": "The evidence is merely relevant, incomplete, contradictory, or does not support the answer's material claims.",
            },
        }
    }
    usage = new_usage()
    label = getattr(client, "provider_label", "System One")
    try:
        answers, raw_usage = client.ask(state, questions)
    except RefinementError as exc:
        _record_client_error(usage, exc)
        exc.stage_usage["support"] = usage
        raise
    _record_call_usage(usage, raw_usage)
    try:
        answer_data = answers.get("is_supported")
        if not isinstance(answer_data, dict) or answer_data.get("type") != "noul":
            raise RefinementError(f"{label} response is missing the support Noul")
        probability = _probability(answer_data.get("noul"), "is_supported", label)
    except RefinementError as exc:
        exc.stage_usage["support"] = usage
        raise
    return probability, usage
