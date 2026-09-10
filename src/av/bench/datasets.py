"""Adapters for public video-QA benchmarks.

No benchmark data is vendored into this repository. These adapters read an
annotation file you fetched yourself and emit the harness's own JSONL task format,
so licences stay with their owners and this Apache-2.0 tree stays clean. LVBench in
particular is CC BY-NC-SA with an explicit commercial-use prohibition — parsing it
is fine, redistributing it here would not be.

Videos are never fetched by these adapters. Both benchmarks reference YouTube ids,
which means yt-dlp, bandwidth, and link rot are the caller's problem and the caller's
decision. Each emitted row records the source id so a fetch step can be run separately.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Where the annotation files come from. Recorded here so a receipt can name a source
# rather than a vague dataset name.
SOURCES: dict[str, dict[str, str]] = {
    "minerva": {
        "name": "MINERVA",
        "annotations_url": "https://storage.googleapis.com/neptunedata/minerva.json",
        "repo": "https://github.com/google-deepmind/neptune",
        "paper": "arXiv:2505.00681",
        "annotations_licence": "CC BY 4.0",
        "video_licence": "not granted — YouTube ids only",
        "format": "5-way multiple choice",
    },
    "lvbench": {
        "name": "LVBench",
        "annotations_url": (
            "https://huggingface.co/datasets/zai-org/LVBench/resolve/main/video_info.meta.jsonl"
        ),
        "repo": "https://github.com/zai-org/LVBench",
        "paper": "arXiv:2406.08035",
        "annotations_licence": "CC BY-NC-SA 4.0 (non-commercial; see upstream)",
        "video_licence": "not granted — YouTube ids only",
        "format": "4-way multiple choice",
    },
}

_LETTERS = "ABCDEFGH"
_OPTION_LINE = re.compile(r"^\(([A-H])\)\s*(.*)$")
_TIME_RANGE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?-(\d{1,2}):(\d{2})(?::(\d{2}))?$")


@dataclass
class AdaptedRow:
    id: str
    video_id: str
    question: str
    options: list[str]
    answer: str
    start_sec: float | None = None
    end_sec: float | None = None
    meta: dict | None = None

    def to_task_row(self, video_template: str) -> dict:
        row: dict = {
            "id": self.id,
            "video": video_template.format(video_id=self.video_id, id=self.id),
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "meta": {"video_id": self.video_id, **(self.meta or {})},
        }
        if self.start_sec is not None:
            row["start_sec"] = self.start_sec
        if self.end_sec is not None:
            row["end_sec"] = self.end_sec
        return row


def _parse_timespan(value: str) -> tuple[float, float] | None:
    """LVBench's ``time_reference`` — ``MM:SS-MM:SS`` or ``HH:MM:SS-HH:MM:SS``."""
    m = _TIME_RANGE.match((value or "").strip())
    if not m:
        return None
    a1, a2, a3, b1, b2, b3 = m.groups()
    start = (int(a1) * 3600 + int(a2) * 60 + int(a3)) if a3 else (int(a1) * 60 + int(a2))
    end = (int(b1) * 3600 + int(b2) * 60 + int(b3)) if b3 else (int(b1) * 60 + int(b2))
    return (float(start), float(end)) if end > start else None


def adapt_minerva(annotations: list[dict]) -> list[AdaptedRow]:
    """MINERVA: flat list, choices in ``answer_choice_N``, gold index in ``answer_id``."""
    rows: list[AdaptedRow] = []
    for item in annotations:
        choices = []
        i = 0
        while f"answer_choice_{i}" in item:
            choices.append(str(item[f"answer_choice_{i}"]))
            i += 1
        if not choices:
            continue
        gold = item.get("answer_id")
        if not isinstance(gold, int) or not 0 <= gold < len(choices):
            continue
        rows.append(
            AdaptedRow(
                id=str(item["key"]),
                video_id=str(item["video_id"]),
                question=str(item["question"]),
                options=[f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices)],
                answer=_LETTERS[gold],
                meta={
                    "question_type": item.get("question_type"),
                    "split": item.get("split"),
                    "category": item.get("category"),
                },
            )
        )
    return rows


def adapt_lvbench(annotations: list[dict]) -> list[AdaptedRow]:
    """LVBench: one object per video, options inlined in the question text as ``(A) ...``."""
    rows: list[AdaptedRow] = []
    for video in annotations:
        video_id = str(video.get("key", ""))
        for qa in video.get("qa") or []:
            lines = str(qa.get("question", "")).splitlines()
            stem_lines: list[str] = []
            options: list[str] = []
            for line in lines:
                m = _OPTION_LINE.match(line.strip())
                if m:
                    options.append(f"{m.group(1)}. {m.group(2)}")
                elif not options:
                    stem_lines.append(line)
            if not options:
                continue
            span = _parse_timespan(str(qa.get("time_reference", "")))
            rows.append(
                AdaptedRow(
                    id=f"{video_id}:{qa.get('uid')}",
                    video_id=video_id,
                    question="\n".join(stem_lines).strip(),
                    options=options,
                    answer=str(qa.get("answer", "")).strip().upper(),
                    start_sec=span[0] if span else None,
                    end_sec=span[1] if span else None,
                    meta={
                        "question_type": qa.get("question_type"),
                        "video_type": video.get("type"),
                        "time_reference": qa.get("time_reference"),
                    },
                )
            )
    return rows


ADAPTERS = {"minerva": adapt_minerva, "lvbench": adapt_lvbench}


def load_annotations(path: Path) -> list[dict]:
    """Read either a JSON array (MINERVA) or JSONL (LVBench)."""
    text = Path(path).expanduser().read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def subset_by_video(
    rows: list[AdaptedRow], *, max_questions: int, max_videos: int | None = None
) -> list[AdaptedRow]:
    """Take a subset grouped by video, so a small run does not pull many hours of video.

    Rows are ordered by their stable id first, so the same arguments always select the
    same questions. Sampling by question instead of by video is how a 30-question run
    turns into a 40-hour download.
    """
    by_video: dict[str, list[AdaptedRow]] = {}
    for row in sorted(rows, key=lambda r: r.id):
        by_video.setdefault(row.video_id, []).append(row)

    chosen: list[AdaptedRow] = []
    for n, (_, group) in enumerate(sorted(by_video.items())):
        if max_videos is not None and n >= max_videos:
            break
        for row in group:
            if len(chosen) >= max_questions:
                return chosen
            chosen.append(row)
    return chosen
