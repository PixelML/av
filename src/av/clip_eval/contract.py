"""Frozen metric definitions for the clip evaluation contract.

Every metric compares returned clips against independently frozen labels.
Jev decision scores are never used as ground truth, and absent-topic queries
carry no moments at all.

Contract version 1 metrics:
- precision_at_k          hits in the first k returned clips / k
- known_moment_recall     labeled moments covered (IoU >= 0.5) / total moments
- absent_topic_fp         1.0 when any clip is returned for an absent query
- boundary_error_sec      mean of (|start delta| + |end delta|) / 2 over hits
- duplication_rate        returned clip pairs with IoU >= 0.2 / clip count
- context_loss_count      returned hits missing more than one second of a
                          labeled setup head
"""

from __future__ import annotations

import math
from typing import Any

IOU_HIT_THRESHOLD = 0.5
IOU_DUPLICATION_THRESHOLD = 0.2
SETUP_SLACK_SEC = 1.0


def interval_iou(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """Intersection-over-union of two closed intervals (0.0 when disjoint)."""
    intersection = min(a_end, b_end) - max(a_start, b_start)
    if intersection <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


def best_match(
    clip: dict, moments: list[dict]
) -> tuple[dict | None, float]:
    """Return the labeled moment with the highest IoU against one clip."""
    best: dict | None = None
    best_iou = 0.0
    for moment in moments:
        iou = interval_iou(
            clip["start_sec"], clip["end_sec"],
            float(moment["start_sec"]), float(moment["end_sec"]),
        )
        if iou > best_iou:
            best, best_iou = moment, iou
    return best, best_iou


def query_metrics(clips: list[dict], label: dict, *, k: int | None = None) -> dict:
    """Compute all contract-v1 metrics for one query's returned clips."""
    moments = label.get("moments", [])
    expected_absent = label.get("expected") == "absent"
    returned = clips[:k] if k else clips

    hits = 0
    boundary_errors: list[float] = []
    covered: set[str] = set()
    context_losses = 0
    for clip in returned:
        moment, iou = best_match(clip, moments)
        if moment is None or iou < IOU_HIT_THRESHOLD:
            continue
        hits += 1
        covered.add(moment["moment_id"])
        boundary_errors.append(
            (
                abs(clip["start_sec"] - float(moment["start_sec"]))
                + abs(clip["end_sec"] - float(moment["end_sec"]))
            )
            / 2.0
        )
        if moment.get("requires_setup") and moment.get("setup_start_sec") is not None:
            setup_start = float(moment["setup_start_sec"])
            if clip["start_sec"] > setup_start + SETUP_SLACK_SEC:
                context_losses += 1

    duplication_pairs = 0
    for i, a in enumerate(returned):
        for b in returned[i + 1 :]:
            if (
                interval_iou(
                    a["start_sec"], a["end_sec"], b["start_sec"], b["end_sec"]
                )
                >= IOU_DUPLICATION_THRESHOLD
            ):
                duplication_pairs += 1

    k_denominator = k if k else max(len(returned), 1)
    return {
        "clips_returned": len(returned),
        "hits": hits,
        "precision_at_k": round(hits / k_denominator, 4) if returned else 0.0,
        "known_moment_recall": (
            round(len(covered) / len(moments), 4) if moments else None
        ),
        "absent_topic_fp": 1.0 if expected_absent and returned else 0.0,
        "boundary_error_sec": (
            round(sum(boundary_errors) / len(boundary_errors), 3)
            if boundary_errors
            else None
        ),
        "duplication_rate": (
            round(duplication_pairs / len(returned), 4) if returned else 0.0
        ),
        "context_loss_count": context_losses,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if len(values) < 5:
        # Percentiles are only reported when the sample supports them.
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return round(ordered[low], 3)
    return round(ordered[low] * (high - position) + ordered[high] * (position - low), 3)


def aggregate(per_query: list[dict]) -> dict:
    """Aggregate per-query metric blocks into arm-level totals."""
    def collect(field: str) -> list[float]:
        return [entry[field] for entry in per_query if entry.get(field) is not None]

    present_recall = [entry["known_moment_recall"] for entry in per_query if entry["known_moment_recall"] is not None]
    boundary = collect("boundary_error_sec")
    totals = {
        "queries": len(per_query),
        "precision_at_k_mean": round(sum(collect("precision_at_k")) / len(per_query), 4) if per_query else None,
        "known_moment_recall_mean": round(sum(present_recall) / len(present_recall), 4) if present_recall else None,
        "absent_topic_fp_total": round(sum(collect("absent_topic_fp")), 2),
        "boundary_error_sec_mean": round(sum(boundary) / len(boundary), 3) if boundary else None,
        "duplication_rate_mean": round(sum(collect("duplication_rate")) / len(per_query), 4) if per_query else None,
        "context_loss_total": sum(collect("context_loss_count")),
    }
    for name, values in (
        ("precision_at_k_p50", collect("precision_at_k")),
        ("boundary_error_sec_p50", boundary),
    ):
        totals[name] = _percentile(values, 0.5)
    totals["boundary_error_sec_p95"] = _percentile(boundary, 0.95)
    return totals


def merge_stage_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-stage usage dicts across queries (requests always add; token
    totals collapse to unknown if any contributing stage was incomplete)."""
    merged: dict[str, Any] = {}
    for usage in usages:
        for stage, numbers in usage.items():
            if not isinstance(numbers, dict):
                continue
            target = merged.setdefault(
                stage,
                {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_complete": True,
                    "output_tokens_complete": True,
                },
            )
            for key in ("requests", "input_tokens", "output_tokens"):
                target[key] = int(target.get(key) or 0) + int(numbers.get(key) or 0)
            for key in ("input_tokens_complete", "output_tokens_complete"):
                if numbers.get(key) is False:
                    target[key] = False
    for stage in merged.values():
        for prefix in ("input", "output"):
            if stage.get(f"{prefix}_tokens_complete") is False:
                stage[f"{prefix}_tokens"] = None
    return merged
