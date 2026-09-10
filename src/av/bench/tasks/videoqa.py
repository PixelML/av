"""Video question answering under two arms: dense and agentic.

The comparison mirrors the shape of published agentic-video results so the numbers
are readable side by side:

``dense``    every frame at a fixed rate is stuffed into one request. Simple,
             expensive, and the baseline everyone reports.
``agentic``  a cheap coarse pass over widely-spaced frames decides *where to look*,
             then a second request fetches only those moments at full rate. The
             scouting pass is charged to the arm — its tokens count.

Both arms answer the same questions with the same prompt and the same scorer, so
the only difference between them is which frames the model got to see.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from av.bench.cost import CellUsage
from av.bench.frames import sample_at, sample_interval
from av.bench.vlm import BenchVLM, VLMResult, strip_code_fence

ARMS = ("dense", "agentic")

# Frame budget for the agentic arm's targeted fetch, per question.
DEFAULT_AGENTIC_BUDGET = 16
# Coarse pass spacing: one frame every N seconds. Deliberately sparse — the point
# is that a cheap look is enough to decide where the answer lives.
DEFAULT_COARSE_INTERVAL_SEC = 60.0
DEFAULT_COARSE_MAX_FRAMES = 32


@dataclass
class Question:
    id: str
    video: Path
    question: str
    options: list[str] = field(default_factory=list)
    answer: str = ""
    start_sec: float = 0.0
    end_sec: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.options)


def load_questions(path: Path, video_root: Path | None = None) -> list[Question]:
    """Load a JSONL task file.

    One object per line::

        {"id": "q1", "video": "clips/a.mp4", "question": "...",
         "options": ["A. ...", "B. ..."], "answer": "B",
         "start_sec": 0, "end_sec": 600}

    ``options`` may be omitted for open-ended questions. Relative ``video`` paths
    resolve against ``video_root`` (default: the task file's directory), which keeps
    a task file portable across machines.
    """
    path = Path(path).expanduser()
    root = Path(video_root).expanduser() if video_root else path.parent
    questions: list[Question] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON — {e}") from e

        video = Path(row["video"]).expanduser()
        if not video.is_absolute():
            video = (root / video).resolve()

        questions.append(
            Question(
                id=str(row.get("id") or f"q{lineno}"),
                video=video,
                question=row["question"],
                options=list(row.get("options") or []),
                answer=str(row.get("answer", "")),
                start_sec=float(row.get("start_sec") or 0.0),
                end_sec=float(row["end_sec"]) if row.get("end_sec") is not None else None,
                meta=row.get("meta") or {},
            )
        )
    return questions


# --- prompting ---------------------------------------------------------------

_ANSWER_INSTRUCTION_MC = (
    "Answer with the letter of the correct option only — a single character, "
    "no explanation, no punctuation."
)
_ANSWER_INSTRUCTION_OPEN = "Answer in as few words as possible. No explanation."


def build_question_prompt(q: Question, frame_timestamps: list[float]) -> str:
    stamps = ", ".join(f"{t:.1f}s" for t in frame_timestamps)
    header = (
        f"You are shown {len(frame_timestamps)} frames sampled from a video, in "
        f"chronological order, at these timestamps: {stamps}.\n\n"
    )
    body = f"Question: {q.question}\n"
    if q.is_multiple_choice:
        body += "Options:\n" + "\n".join(q.options) + "\n\n" + _ANSWER_INSTRUCTION_MC
    else:
        body += "\n" + _ANSWER_INSTRUCTION_OPEN
    return header + body


_SELECTION_INSTRUCTION = (
    "These frames are a coarse, widely-spaced preview of a longer video.\n\n"
    "Question you will later have to answer: {question}\n\n"
    "Do not answer it yet. Decide which moments of the video you need to see at a "
    "finer sampling rate to answer it confidently.\n"
    "Reply with JSON only, in exactly this form, and nothing else:\n"
    '{{"timestamps": [12.0, 13.5, 40.0]}}\n'
    "Give at most {budget} timestamps, in seconds, within 0 and {duration:.1f}."
)


def build_selection_prompt(q: Question, frame_timestamps: list[float], budget: int, duration: float) -> str:
    stamps = ", ".join(f"{t:.1f}s" for t in frame_timestamps)
    return (
        f"You are shown {len(frame_timestamps)} frames at these timestamps: {stamps}.\n\n"
        + _SELECTION_INSTRUCTION.format(question=q.question, budget=budget, duration=duration)
    )


# --- scoring -----------------------------------------------------------------

_LETTER = re.compile(r"\b([A-H])\b")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def extract_choice(text: str, options: list[str]) -> str | None:
    """Pull a choice letter out of a model reply, tolerating light chattiness."""
    if not text:
        return None
    stripped = strip_code_fence(text).strip().strip(".").strip()
    if len(stripped) == 1 and stripped.upper().isalpha():
        return stripped.upper()

    # Match the option text before the bare letter. A reply like "a red car" contains
    # a standalone "a", and reading that as choice A would score a correct answer wrong.
    norm_reply = _norm(stripped)
    for i, opt in enumerate(options):
        body = _norm(re.sub(r"^\s*[A-Ha-h][.)]\s*", "", opt))
        if len(body) > 1 and body in norm_reply:
            return chr(ord("A") + i)

    m = _LETTER.search(stripped.upper())
    return m.group(1) if m else None


def score_answer(q: Question, text: str) -> tuple[bool, str | None]:
    """Return (correct, parsed answer). Unparseable replies score as incorrect."""
    if q.is_multiple_choice:
        choice = extract_choice(text, q.options)
        return (choice is not None and choice == q.answer.strip().upper()), choice
    parsed = text.strip()
    gold = _norm(q.answer)
    return (bool(gold) and gold in _norm(parsed)), parsed or None


def parse_timestamps(text: str, duration: float, budget: int) -> list[float]:
    """Read the selection reply. Falls back to bare numbers if the JSON is malformed."""
    if not text:
        return []
    text = strip_code_fence(text)
    candidates: list[float] = []
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            candidates = [float(t) for t in (data.get("timestamps") or [])]
        except (json.JSONDecodeError, TypeError, ValueError):
            candidates = []
    if not candidates:
        candidates = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    kept = [t for t in candidates if 0.0 <= t <= max(duration, 0.0)]
    return sorted(set(kept))[:budget]


# --- arms --------------------------------------------------------------------

@dataclass
class QuestionResult:
    question_id: str
    arm: str
    correct: bool
    parsed_answer: str | None
    gold_answer: str
    frames_sent: int
    frames_extracted: int
    requests: int
    tokens_in: int | None
    tokens_out: int | None
    ttft_sec: float | None
    wall_sec: float
    ok: bool
    error: str | None = None
    selected_timestamps: list[float] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "arm": self.arm,
            "correct": self.correct,
            "parsed_answer": self.parsed_answer,
            "gold_answer": self.gold_answer,
            "frames_sent": self.frames_sent,
            "frames_extracted": self.frames_extracted,
            "requests": self.requests,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "ttft_sec": round(self.ttft_sec, 4) if self.ttft_sec is not None else None,
            "wall_sec": round(self.wall_sec, 4),
            "ok": self.ok,
            "error": self.error,
            "selected_timestamps": self.selected_timestamps,
            "raw_text": self.raw_text[:1000],
        }


def window_for(q: Question, duration: float) -> tuple[float, float]:
    start = max(q.start_sec, 0.0)
    end = q.end_sec if q.end_sec is not None else duration
    end = min(end, duration) if duration else end
    return start, max(end - start, 0.0)


def _accumulate(results: list[VLMResult]) -> tuple[int | None, int | None, float | None, float]:
    tin = sum(r.tokens_in for r in results if r.tokens_in is not None) or None
    tout = sum(r.tokens_out for r in results if r.tokens_out is not None) or None
    ttfts = [r.ttft_sec for r in results if r.ttft_sec is not None]
    wall = sum(r.wall_sec for r in results)
    return tin, tout, (ttfts[0] if ttfts else None), wall


def run_dense(
    vlm: BenchVLM,
    q: Question,
    duration: float,
    *,
    interval_sec: float = 1.0,
    max_frames: int = 1024,
    scale_width: int | None = 768,
) -> QuestionResult:
    """Sample the whole window at a fixed rate and ask once."""
    work = Path(tempfile.mkdtemp(prefix="av_bench_dense_"))
    try:
        start, span = window_for(q, duration)
        frames = sample_interval(
            q.video, interval_sec,
            start_sec=start, duration_sec=span or None,
            max_frames=max_frames, scale_width=scale_width, out_dir=work,
        )
        if not frames.paths:
            return QuestionResult(
                question_id=q.id, arm="dense", correct=False, parsed_answer=None,
                gold_answer=q.answer, frames_sent=0, frames_extracted=0, requests=0,
                tokens_in=None, tokens_out=None, ttft_sec=None, wall_sec=0.0,
                ok=False, error="no frames extracted",
            )
        prompt = build_question_prompt(q, frames.timestamps)
        res = vlm.ask(frames.paths, prompt)
        correct, parsed = score_answer(q, res.text) if res.ok else (False, None)
        tin, tout, ttft, wall = _accumulate([res])
        return QuestionResult(
            question_id=q.id, arm="dense", correct=correct, parsed_answer=parsed,
            gold_answer=q.answer, frames_sent=len(frames.paths),
            frames_extracted=len(frames.paths), requests=1,
            tokens_in=tin, tokens_out=tout, ttft_sec=ttft, wall_sec=wall,
            ok=res.ok, error=res.error, raw_text=res.text,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_agentic(
    vlm: BenchVLM,
    q: Question,
    duration: float,
    *,
    coarse_interval_sec: float = DEFAULT_COARSE_INTERVAL_SEC,
    coarse_max_frames: int = DEFAULT_COARSE_MAX_FRAMES,
    budget_frames: int = DEFAULT_AGENTIC_BUDGET,
    scale_width: int | None = 768,
    seed_timestamps: list[float] | None = None,
) -> QuestionResult:
    """Coarse look, then targeted fetch. Both requests are charged to this arm.

    ``seed_timestamps`` lets an external retriever (for example ``av``'s FTS5 search
    over already-ingested captions) propose moments before the model looks at all.
    They are merged with the model's own selection rather than replacing it.
    """
    work = Path(tempfile.mkdtemp(prefix="av_bench_agentic_"))
    try:
        start, span = window_for(q, duration)
        coarse = sample_interval(
            q.video, coarse_interval_sec,
            start_sec=start, duration_sec=span or None,
            max_frames=coarse_max_frames, scale_width=scale_width,
            out_dir=work / "coarse",
        )
        if not coarse.paths:
            return QuestionResult(
                question_id=q.id, arm="agentic", correct=False, parsed_answer=None,
                gold_answer=q.answer, frames_sent=0, frames_extracted=0, requests=0,
                tokens_in=None, tokens_out=None, ttft_sec=None, wall_sec=0.0,
                ok=False, error="no coarse frames extracted",
            )

        sel_prompt = build_selection_prompt(q, coarse.timestamps, budget_frames, start + (span or duration))
        sel = vlm.ask(coarse.paths, sel_prompt)
        chosen = parse_timestamps(sel.text, start + (span or duration), budget_frames) if sel.ok else []
        if seed_timestamps:
            chosen = sorted(set(chosen) | set(seed_timestamps))[:budget_frames]
        if not chosen:
            # The model declined to choose. Falling back to the coarse frames keeps
            # the arm answerable, and the empty selection is recorded either way.
            chosen = list(coarse.timestamps[:budget_frames])

        targeted = sample_at(q.video, chosen, scale_width=scale_width, out_dir=work / "targeted")
        if not targeted.paths:
            targeted = coarse

        ans_prompt = build_question_prompt(q, targeted.timestamps)
        ans = vlm.ask(targeted.paths, ans_prompt)
        correct, parsed = score_answer(q, ans.text) if ans.ok else (False, None)
        tin, tout, ttft, wall = _accumulate([sel, ans])
        return QuestionResult(
            question_id=q.id, arm="agentic", correct=correct, parsed_answer=parsed,
            gold_answer=q.answer,
            frames_sent=len(coarse.paths) + len(targeted.paths),
            frames_extracted=len(coarse.paths) + len(targeted.paths),
            requests=2,
            tokens_in=tin, tokens_out=tout, ttft_sec=ttft, wall_sec=wall,
            ok=sel.ok and ans.ok, error=ans.error or sel.error,
            selected_timestamps=chosen, raw_text=ans.text,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def aggregate(results: list[QuestionResult]) -> dict:
    """Arm-level summary: the two headline axes plus what they were computed over."""
    answered = [r for r in results if r.ok]
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    with_tokens = [r for r in results if r.tokens_in is not None]
    total_in = sum(r.tokens_in or 0 for r in with_tokens)
    total_out = sum(r.tokens_out or 0 for r in with_tokens)
    return {
        "questions": n,
        "answered_ok": len(answered),
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "tokens_per_query_in": round(total_in / len(with_tokens), 1) if with_tokens else None,
        "tokens_per_query_out": round(total_out / len(with_tokens), 1) if with_tokens else None,
        "tokens_per_query_total": (
            round((total_in + total_out) / len(with_tokens), 1) if with_tokens else None
        ),
        "tokens_reported_for": len(with_tokens),
        "frames_per_query": round(sum(r.frames_sent for r in results) / n, 1) if n else None,
        "wall_sec_total": round(sum(r.wall_sec for r in results), 3),
    }


def usage_for(results: list[QuestionResult]) -> CellUsage:
    return CellUsage(
        tokens_in=sum(r.tokens_in or 0 for r in results),
        tokens_out=sum(r.tokens_out or 0 for r in results),
        wall_total_sec=sum(r.wall_sec for r in results),
        ttft_sec=next((r.ttft_sec for r in results if r.ttft_sec is not None), None),
        requests=sum(r.requests for r in results),
    )
