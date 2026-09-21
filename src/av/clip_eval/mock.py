"""Deterministic labeled stand-in for the System One decision endpoint.

Answers Noul, Choice, and Score questions purely from the frozen corpus
labels with string and interval arithmetic. It exists to exercise the typed
decision pipeline and the evaluation harness offline. It is NOT a measurement
of Jev quality and must never be presented as one.
"""

from __future__ import annotations

from typing import Any

_RELEVANT_P = 0.9
_IRRELEVANT_P = 0.05
_COHERENT_P = 0.9
_INCOHERENT_P = 0.4
_CONFIDENT = 0.85


def _interval_overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


class LabeledDecisionClient:
    """Answers typed questions by looking up the labeled moments a clip hits."""

    def __init__(self, queries: list[dict]) -> None:
        self._by_topic: dict[tuple[str, str], dict] = {}
        for query in queries:
            key = (query["video_id"], query["topic"].casefold())
            self._by_topic[key] = query

    def _lookup(self, state: dict[str, Any]) -> dict | None:
        topic = str(state.get("query", "")).casefold()
        clips = state.get("clips") or state.get("surrounding_events") or {}
        for value in clips.values():
            video_id = value.get("video_id") if isinstance(value, dict) else None
            if video_id:
                return self._by_topic.get((video_id, topic))
        # Boundary state carries no video_id; fall back to a unique match.
        matches = [
            query
            for (video_id, topic_key), query in self._by_topic.items()
            if topic_key == topic
        ]
        return matches[0] if len(matches) == 1 else None

    def _moments_hit(self, label: dict | None, start: float, end: float) -> list[dict]:
        if not label:
            return []
        return [
            moment
            for moment in label.get("moments", [])
            if _interval_overlaps(start, end, float(moment["start_sec"]), float(moment["end_sec"]))
        ]

    @staticmethod
    def _clip_interval(value: dict) -> tuple[float, float]:
        start = value.get("start_sec")
        end = value.get("end_sec")
        if start is None or end is None:
            return (-1.0, -1.0)
        return (float(start), float(end))

    def ask(self, state: dict[str, Any], questions: dict[str, dict]) -> tuple[dict, dict]:
        label = self._lookup(state)
        answers: dict[str, dict] = {}
        for key, question in questions.items():
            qtype = question.get("type")
            clips = state.get("clips") or {}
            value = clips.get(key, {})
            start, end = self._clip_interval(value)
            if qtype == "noul":
                instructions = str(question.get("instructions", ""))
                hits = self._moments_hit(label, start, end)
                if "standalone" in instructions or "on its own" in instructions:
                    # Coherence: only a hard-context moment whose setup head
                    # is missing reads as incoherent.
                    p = _COHERENT_P
                    for moment in hits:
                        setup_end = moment.get("setup_end_sec")
                        if (
                            moment.get("hard_context")
                            and setup_end is not None
                            and start >= float(setup_end) - 0.5
                        ):
                            # The clip is the payoff without its setup head.
                            p = _INCOHERENT_P
                    answers[key] = {"type": "noul", "noul": p}
                elif "visual description" in instructions:
                    has_caption = bool(value.get("caption") and value["caption"] != "(none)")
                    answers[key] = {
                        "type": "noul",
                        "noul": _RELEVANT_P if (hits and has_caption) else _IRRELEVANT_P,
                    }
                else:
                    answers[key] = {
                        "type": "noul",
                        "noul": _RELEVANT_P if hits else _IRRELEVANT_P,
                    }
            elif qtype == "choice":
                answers[key] = self._choice_answer(label, state, key, question)
            elif qtype == "score":
                hits = self._moments_hit(label, start, end)
                score = hits[0].get("appeal", 0.3) if hits else 0.3
                answers[key] = {"type": "score", "score": float(score)}
            else:
                answers[key] = {"type": "error", "message": f"unsupported type {qtype}"}
        return answers, {"requests": 1, "input_tokens": 0, "output_tokens": 0}

    def _choice_answer(
        self,
        label: dict | None,
        state: dict[str, Any],
        key: str,
        question: dict,
    ) -> dict:
        candidates = question.get("criteria", {})
        surrounding = state.get("surrounding_events", {})
        best_label = "e0"
        best_distance = float("inf")
        moments = label.get("moments", []) if label else []
        target = 0.0
        if moments:
            if key == "start":
                target = min(float(m["start_sec"]) for m in moments)
            else:
                target = max(float(m["end_sec"]) for m in moments)
        for candidate_label, criterion in candidates.items():
            seconds = criterion.get("seconds") if isinstance(criterion, dict) else None
            if not seconds and candidate_label in surrounding:
                # The e0 criterion describes the hit event without seconds;
                # its interval is still available in the surrounding state.
                event = surrounding[candidate_label]
                seconds = f"{event.get('start')}-{event.get('end')}"
            if not seconds:
                continue
            start_s, _, end_s = str(seconds).partition("-")
            try:
                c_start, c_end = float(start_s), float(end_s)
            except ValueError:
                continue
            distance = abs(c_start - target) + abs(c_end - target)
            if distance < best_distance:
                best_distance = distance
                best_label = candidate_label
        if not moments:
            # Absent topics: any choice is arbitrary; keep the minimal answer.
            best_label = "e0" if "e0" in candidates else next(iter(candidates), "e0")
        return {"type": "choice", "choice": best_label, "confidence": _CONFIDENT}
