"""Event detection on real footage, as a function of frame interval.

This is where the frontier stops being an abstraction: the interval at which recall
collapses is the cheapest sampling rate that is still safe for the task, and it is
different for every task. Smoke tolerates sparse frames; a door opening does not.

**Reference caveat, stated up front.** The ``samples/epstein-cctv`` artifacts shipped
with this repository were themselves produced by a vision model. Scoring against them
measures *agreement with a dense reference run*, not agreement with human ground truth.
Every receipt this module writes carries that caveat, and the metric is named
``reference_recall`` rather than ``recall`` so nobody can quote it as the latter.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from av.bench.frames import sample_interval
from av.bench.vlm import BenchVLM, strip_code_fence
from av.core.exceptions import FFmpegError

# Event probes. Each is a label, the phrases that mark it in a reference caption, and
# the question put to the model. Keep the wording of both sides fixed: changing either
# changes the benchmark.
EVENT_PROBES: dict[str, dict] = {
    "door_activity": {
        "reference_terms": ["door", "gate", "doorway"],
        "question": "Does anyone open, close, or interact with a door or gate in these frames?",
    },
    "person_enters": {
        "reference_terms": ["enters", "entering", "walks in", "arrives"],
        "question": "Does any person enter the scene in these frames?",
    },
    "person_exits": {
        "reference_terms": ["exits", "exiting", "leaves", "walks out", "departs"],
        "question": "Does any person leave or exit the scene in these frames?",
    },
    "group_present": {
        "reference_terms": ["group", "several people", "multiple people", "three ", "four "],
        "question": "Are three or more people visible together in these frames?",
    },
    "escort": {
        "reference_terms": ["escort", "escorted", "restrain", "handcuff"],
        "question": "Is anyone being escorted, guided, or restrained by another person in these frames?",
    },
}

ANSWER_INSTRUCTION = 'Reply with JSON only: {"present": true} or {"present": false}'


@dataclass
class ReferenceEvent:
    video: Path
    start_sec: float
    end_sec: float
    probe: str
    source_text: str


def load_reference_events(
    artifacts_path: Path,
    video_dir: Path,
    *,
    probes: list[str] | None = None,
    max_per_probe: int = 10,
) -> list[ReferenceEvent]:
    """Derive event windows from a shipped artifacts JSONL.

    A window counts as containing an event when its caption text mentions one of the
    probe's reference terms. Windows whose video file is missing are skipped, since a
    benchmark that scores questions it cannot show frames for is measuring nothing.
    """
    wanted = probes or list(EVENT_PROBES)
    video_dir = Path(video_dir).expanduser()
    counts = {p: 0 for p in wanted}
    events: list[ReferenceEvent] = []

    for line in Path(artifacts_path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(row.get("text") or "")
        if not text or row.get("end_sec") in (None, ""):
            continue
        video = video_dir / str(row.get("filename") or "")
        if not video.exists():
            continue
        low = text.lower()
        for probe in wanted:
            if counts[probe] >= max_per_probe:
                continue
            terms = EVENT_PROBES[probe]["reference_terms"]
            if any(t in low for t in terms):
                events.append(
                    ReferenceEvent(
                        video=video,
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                        probe=probe,
                        source_text=text,
                    )
                )
                counts[probe] += 1
    return events


_TRUE = re.compile(r"\btrue\b|\byes\b", re.I)
_FALSE = re.compile(r"\bfalse\b|\bno\b", re.I)
# Matches the field even when a token limit cut the closing brace off.
_PRESENT_FIELD = re.compile(r'"present"\s*:\s*(true|false)', re.I)


def parse_presence(text: str) -> bool | None:
    """Read a presence reply. ``None`` means unparseable, which is not a detection.

    Kept deliberately forgiving about formatting and strict about content: a reply
    the harness cannot read is recorded separately from a reply that says "no", so a
    parsing problem never masquerades as a detection failure.
    """
    if not text:
        return None
    cleaned = strip_code_fence(text)
    m = re.search(r"\{.*\}", cleaned, re.S)
    if m:
        try:
            value = json.loads(m.group(0)).get("present")
            if isinstance(value, bool):
                return value
        except json.JSONDecodeError:
            pass
    field = _PRESENT_FIELD.search(cleaned)
    if field:
        return field.group(1).lower() == "true"
    if _TRUE.search(cleaned) and not _FALSE.search(cleaned):
        return True
    if _FALSE.search(cleaned):
        return False
    return None


@dataclass
class EventCell:
    interval_sec: float
    probe: str
    events_total: int
    events_detected: int
    unparseable: int
    frames_total: int
    tokens_in: int
    tokens_out: int
    wall_sec: float
    errors: list[str] = field(default_factory=list)

    @property
    def reference_recall(self) -> float | None:
        if not self.events_total:
            return None
        return self.events_detected / self.events_total

    def to_dict(self) -> dict:
        return {
            "interval_sec": self.interval_sec,
            "probe": self.probe,
            "events_total": self.events_total,
            "events_detected": self.events_detected,
            "reference_recall": (
                round(self.reference_recall, 4) if self.reference_recall is not None else None
            ),
            "unparseable": self.unparseable,
            "frames_total": self.frames_total,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "wall_sec": round(self.wall_sec, 3),
            "errors": self.errors[:5],
        }


def run_event_cell(
    vlm: BenchVLM,
    events: list[ReferenceEvent],
    interval_sec: float,
    *,
    probe: str,
    max_frames: int = 64,
    scale_width: int | None = 512,
    on_event=None,
) -> EventCell:
    """Ask the presence question over every reference window at one sampling interval."""
    subset = [e for e in events if e.probe == probe]
    question = EVENT_PROBES[probe]["question"] + "\n\n" + ANSWER_INSTRUCTION

    detected = unparseable = frames_total = 0
    tokens_in = tokens_out = 0
    wall = 0.0
    errors: list[str] = []

    for event in subset:
        work = Path(tempfile.mkdtemp(prefix="av_bench_event_"))
        try:
            span = max(event.end_sec - event.start_sec, interval_sec)
            try:
                frames = sample_interval(
                    event.video, interval_sec,
                    start_sec=event.start_sec, duration_sec=span,
                    max_frames=max_frames, scale_width=scale_width, out_dir=work,
                )
            except FFmpegError as e:
                # A window we cannot decode is a gap in coverage, not a reason to
                # abandon the sweep. It is recorded and the cell continues.
                errors.append(f"{event.video.name}@{event.start_sec:.0f}s: {e}")
                continue
            if not frames.paths:
                errors.append(f"no frames at {event.start_sec:.0f}s in {event.video.name}")
                continue
            frames_total += len(frames.paths)
            res = vlm.ask(frames.paths, question)
            wall += res.wall_sec
            tokens_in += res.tokens_in or 0
            tokens_out += res.tokens_out or 0
            if not res.ok:
                errors.append(res.error or "unknown error")
                unparseable += 1
                continue
            verdict = parse_presence(res.text)
            if verdict is True:
                detected += 1
            elif verdict is None:
                unparseable += 1
            if on_event:
                on_event(event, verdict, len(frames.paths))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    return EventCell(
        interval_sec=interval_sec,
        probe=probe,
        events_total=len(subset),
        events_detected=detected,
        unparseable=unparseable,
        frames_total=frames_total,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        wall_sec=wall,
        errors=errors,
    )


REFERENCE_CAVEAT = (
    "Reference windows come from a dense vision-model run shipped with this repository, "
    "not from human annotation. This metric measures agreement with that reference run "
    "and must not be quoted as recall against ground truth."
)
