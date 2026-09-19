"""Validate timestamped transcript sidecars before importing transcript artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from av.core.exceptions import IngestError


class TranscriptSidecarError(IngestError):
    """An explicitly supplied transcript sidecar is invalid."""


@dataclass(frozen=True)
class TranscriptSidecarSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class TranscriptSidecar:
    segments: tuple[TranscriptSidecarSegment, ...]
    model: str | None = None
    provenance: dict | None = None


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _reject_constant(_value: str) -> None:
    raise TranscriptSidecarError("Transcript sidecar must contain finite JSON numbers.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TranscriptSidecarError("Transcript sidecar contains duplicate object keys.")
        result[key] = value
    return result


def _validate_json_numbers(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TranscriptSidecarError("Transcript sidecar must contain finite JSON numbers.")
    if isinstance(value, dict):
        for child in value.values():
            _validate_json_numbers(child)
    elif isinstance(value, list):
        for child in value:
            _validate_json_numbers(child)


def load_transcript_sidecar(path: Path, *, duration_sec: float) -> TranscriptSidecar:
    """Read one video's transcript without rewriting its times, order, or text.

    Accept a segment list, or an object with required segments and optional
    model and provenance fields. Empty lists represent no speech. Segment
    timestamps are in seconds relative to the media start and must fall entirely
    within its probed duration. No other fields are accepted for import.
    """
    if not _finite_number(duration_sec) or duration_sec <= 0:
        raise TranscriptSidecarError("Video duration must be a positive finite number.")

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError):
        raise TranscriptSidecarError("Could not read transcript sidecar as UTF-8 JSON.") from None
    except (ValueError, RecursionError):
        raise TranscriptSidecarError("Transcript sidecar is not valid JSON.") from None

    try:
        _validate_json_numbers(raw)
    except RecursionError:
        raise TranscriptSidecarError("Transcript sidecar JSON is nested too deeply.") from None

    model = None
    provenance = None
    if isinstance(raw, dict):
        if "segments" not in raw or set(raw) - {"segments", "model", "provenance"}:
            raise TranscriptSidecarError(
                "Transcript sidecar object requires segments and permits only model and provenance."
            )
        segments = raw["segments"]
        if "model" in raw:
            model = raw["model"]
            if not isinstance(model, str) or not model.strip():
                raise TranscriptSidecarError("Transcript sidecar model must be a nonempty string.")
        if "provenance" in raw:
            provenance = raw["provenance"]
            if not isinstance(provenance, dict):
                raise TranscriptSidecarError("Transcript sidecar provenance must be a JSON object.")
    elif isinstance(raw, list):
        segments = raw
    else:
        raise TranscriptSidecarError("Transcript sidecar root must be a segment list or an object.")

    if not isinstance(segments, list):
        raise TranscriptSidecarError("Transcript sidecar segments must be a list.")

    validated = []
    for index, segment in enumerate(segments):
        label = f"Transcript segment {index + 1}"
        if not isinstance(segment, dict) or set(segment) != {"start_sec", "end_sec", "text"}:
            raise TranscriptSidecarError(f"{label} requires only start_sec, end_sec, and text.")
        start = segment["start_sec"]
        end = segment["end_sec"]
        if not _finite_number(start) or not _finite_number(end):
            raise TranscriptSidecarError(f"{label} timestamps must be finite numbers, excluding booleans.")
        if not 0 <= start < end <= duration_sec:
            raise TranscriptSidecarError(f"{label} must satisfy 0 <= start_sec < end_sec <= video duration.")
        text = segment["text"]
        if not isinstance(text, str) or not text.strip():
            raise TranscriptSidecarError(f"{label} text must be a nonempty string.")
        validated.append(TranscriptSidecarSegment(start_sec=start, end_sec=end, text=text))

    return TranscriptSidecar(segments=tuple(validated), model=model, provenance=provenance)
