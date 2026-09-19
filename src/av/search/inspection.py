"""Bounded sampled-frame inspection for unsupported refined answers.

The inspection transport is intentionally separate from ordinary providers: it
uses only the endpoint, model, and optional credential configured for this stage.
It has no OAuth lookup, CLI fallback, SDK retry, or seed retry.
"""

from __future__ import annotations

import base64
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from av.bench.frames import sample_at
from av.core.config import AVConfig
from av.db.repository import Repository, _fmt_timestamp
from av.search.usage import new_usage, record_usage


@dataclass(frozen=True)
class InspectionWindow:
    video_id: str
    filename: str
    video_path: Path
    requested_start_sec: float
    requested_end_sec: float
    start_sec: float
    end_sec: float
    truncated: bool = False


@dataclass
class VisionResponse:
    ok: bool
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempted: bool = True


class ExplicitVisionClient:
    """One-attempt OpenAI-compatible image request with explicit credentials."""

    def __init__(self, config: AVConfig, *, session: requests.Session | None = None) -> None:
        base = config.strong_vision_api_base_url.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.api_key = config.strong_vision_api_key
        self.model = config.strong_vision_model
        self.timeout = config.api_timeout_sec
        self.session = session or requests.Session()

    @staticmethod
    def _image_url(path: Path) -> str:
        extension = path.suffix.lstrip(".").lower() or "jpeg"
        if extension == "jpg":
            extension = "jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/{extension};base64,{encoded}"

    def ask(self, images: list[Path], prompt: str) -> VisionResponse:
        try:
            content: list[dict] = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": self._image_url(image)}}
                for image in images
            )
        except OSError:
            return VisionResponse(ok=False, attempted=False)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0,
        }
        try:
            response = self.session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException:
            return VisionResponse(ok=False)
        if not response.ok:
            return VisionResponse(ok=False)
        try:
            data = response.json()
        except ValueError:
            return VisionResponse(ok=False)
        if not isinstance(data, dict):
            return VisionResponse(ok=False)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return VisionResponse(ok=False)
        message = choices[0].get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            return VisionResponse(ok=False)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        return VisionResponse(
            ok=True,
            text=text.strip(),
            input_tokens=input_tokens if isinstance(input_tokens, int) and input_tokens >= 0 else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) and output_tokens >= 0 else None,
        )


def _select_windows(
    results: list[dict],
    repo: Repository,
    config: AVConfig,
) -> tuple[list[InspectionWindow], list[str]]:
    warnings: list[str] = []
    if config.inspection_max_windows <= 0 or config.inspection_max_seconds <= 0:
        return [], ["Sampled-frame inspection budget is exhausted."]
    selected: list[InspectionWindow] = []
    seconds_left = config.inspection_max_seconds
    for result in results:
        if len(selected) >= config.inspection_max_windows or seconds_left <= 0:
            break
        if result.get("evidence_scope") == "broad":
            warnings.append("Broad summary/report evidence is not eligible for sampled-frame inspection.")
            continue
        try:
            video = repo.get_video(str(result.get("video_id") or ""))
        except Exception:
            warnings.append("A selected scene has no indexed video record for sampled-frame inspection.")
            continue
        path = Path(video.file_path)
        if not path.is_file():
            warnings.append(f"Media is unavailable for sampled-frame inspection: {video.filename}.")
            continue
        requested_start = float(result.get("timestamp_sec") or 0)
        raw_end = result.get("end_sec")
        requested_end = float(raw_end) if isinstance(raw_end, (int, float)) else requested_start
        start = max(0.0, min(requested_start, video.duration_sec))
        end = max(start, min(requested_end, video.duration_sec))
        if end <= start:
            end = min(video.duration_sec, start + 1.0)
        actual_end = min(end, start + seconds_left)
        if actual_end <= start:
            continue
        selected.append(
            InspectionWindow(
                video_id=video.id,
                filename=video.filename,
                video_path=path,
                requested_start_sec=requested_start,
                requested_end_sec=requested_end,
                start_sec=start,
                end_sec=actual_end,
                truncated=(start != requested_start or actual_end != requested_end),
            )
        )
        seconds_left -= actual_end - start
    return selected, warnings


def _window_key(window: InspectionWindow) -> str:
    return f"{window.video_id}:{window.start_sec:.3f}:{window.end_sec:.3f}"


def sample_window_timestamps(
    windows: list[InspectionWindow],
    max_frames: int,
) -> dict[str, list[float]]:
    """Allocate a hard frame budget; two frames cover both ends when affordable."""
    if not windows or max_frames <= 0:
        return {}
    counts = [0] * len(windows)
    remaining = max_frames
    for index in range(len(windows)):
        if remaining <= 0:
            break
        counts[index] += 1
        remaining -= 1
    for index in range(len(windows)):
        if remaining <= 0:
            break
        counts[index] += 1
        remaining -= 1
    cursor = 0
    while remaining > 0:
        counts[cursor % len(windows)] += 1
        cursor += 1
        remaining -= 1

    out: dict[str, list[float]] = {}
    for window, count in zip(windows, counts):
        if count <= 0:
            timestamps: list[float] = []
        elif count == 1:
            timestamps = [round((window.start_sec + window.end_sec) / 2, 3)]
        else:
            last = max(window.start_sec, window.end_sec - 0.001)
            step = (last - window.start_sec) / (count - 1)
            timestamps = [round(window.start_sec + step * index, 3) for index in range(count)]
        out[_window_key(window)] = timestamps
    return out


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _parse_result(
    text: str,
    windows: list[InspectionWindow],
    sampled_by_video: dict[str, list[float]],
) -> tuple[str, list[dict]] | None:
    try:
        data = json.loads(_strip_code_fence(text))
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("supported") is not True:
        return None
    answer = data.get("answer")
    evidence = data.get("evidence")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(evidence, list):
        return None
    allowed: dict[str, list[tuple[float, float, str]]] = {}
    for window in windows:
        allowed.setdefault(window.video_id, []).append(
            (window.start_sec, window.end_sec, window.filename)
        )
    citations: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            return None
        video_id = item.get("video_id")
        timestamp = item.get("timestamp_sec")
        description = item.get("description")
        if (
            not isinstance(video_id, str)
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or not isinstance(description, str)
            or not description.strip()
        ):
            return None
        matches = [
            entry
            for entry in allowed.get(video_id, [])
            if entry[0] <= float(timestamp) <= entry[1]
        ]
        successful = sampled_by_video.get(video_id, [])
        if not matches or not any(abs(float(timestamp) - value) <= 0.01 for value in successful):
            return None
        citations.append({
            "video_id": video_id,
            "filename": matches[0][2],
            "start_sec": float(timestamp),
            "end_sec": float(timestamp),
            "source_type": "sampled_frame_inspection",
            "text": description.strip(),
            "score": None,
        })
    if not citations:
        return None
    return answer.strip(), citations


def _attempt_budgets(config: AVConfig, window_count: int) -> list[int]:
    total = config.inspection_max_frames
    if total <= 0 or window_count <= 0:
        return []
    minimum_end_coverage = window_count * 2
    if (
        config.inspection_dense_pass
        and config.inspection_max_attempts > 1
        and total >= minimum_end_coverage * 2
    ):
        first = max(minimum_end_coverage, total // 3)
        return [first, total - first][: config.inspection_max_attempts]
    return [total]


def inspect_with_stronger_vision(
    question: str,
    initial_answer: str,
    results: list[dict],
    repo: Repository,
    config: AVConfig,
    *,
    provider_factory=ExplicitVisionClient,
) -> dict:
    usage = new_usage()
    warnings: list[str] = []
    if not config.strong_vision_api_base_url or not config.strong_vision_model:
        return {
            "status": "not_configured",
            "answer": None,
            "citations": [],
            "windows": [],
            "usage": usage,
            "warnings": ["Stronger sampled-frame inspection is not configured."],
        }
    windows, selection_warnings = _select_windows(results, repo, config)
    warnings.extend(selection_warnings)
    if not windows:
        return {
            "status": "unavailable",
            "answer": None,
            "citations": [],
            "windows": [],
            "usage": usage,
            "warnings": warnings or ["No bounded media window was available for sampled-frame inspection."],
        }
    budgets = _attempt_budgets(config, len(windows))
    if not budgets:
        return {
            "status": "budget_exhausted",
            "answer": None,
            "citations": [],
            "windows": [],
            "usage": usage,
            "warnings": warnings + ["Sampled-frame inspection frame budget is exhausted."],
        }

    try:
        provider = provider_factory(config)
    except Exception:
        return {
            "status": "unavailable",
            "answer": None,
            "citations": [],
            "windows": [],
            "usage": usage,
            "warnings": warnings + ["The stronger sampled-frame inspection provider could not be initialized."],
        }
    inspected: list[dict] = []
    for attempt, budget in enumerate(budgets, 1):
        plan = sample_window_timestamps(windows, budget)
        frame_paths: list[Path] = []
        frame_labels: list[str] = []
        sampled_by_video: dict[str, list[float]] = {}
        with tempfile.TemporaryDirectory(prefix="av_ask_inspect_") as temp_dir:
            root = Path(temp_dir)
            for index, window in enumerate(windows):
                requested = plan.get(_window_key(window), [])
                frame_set = sample_at(
                    window.video_path,
                    requested,
                    out_dir=root / f"video_{index}",
                )
                successful = list(frame_set.timestamps)
                sampled_by_video.setdefault(window.video_id, []).extend(successful)
                for path, timestamp in zip(frame_set.paths, successful):
                    frame_paths.append(path)
                    frame_labels.append(
                        f"image {len(frame_paths)}: video_id={window.video_id}, "
                        f"absolute_time={_fmt_timestamp(timestamp)} ({timestamp:.3f}s)"
                    )
                inspected.append({
                    "video_id": window.video_id,
                    "filename": window.filename,
                    "requested_start_sec": window.requested_start_sec,
                    "requested_end_sec": window.requested_end_sec,
                    "start_sec": window.start_sec,
                    "end_sec": window.end_sec,
                    "truncated": window.truncated,
                    "requested_timestamps": requested,
                    "sampled_timestamps": successful,
                    "all_requested_frames_extracted": len(successful) == len(requested),
                    "attempt": attempt,
                })
            if not frame_paths:
                warnings.append("Frame extraction returned no usable sampled frames.")
                continue
            prompt = (
                "Inspect only the supplied independent sampled frames. They do not provide native video or audio, "
                "and they do not establish what happened between sampled timestamps. Answer only when the visible "
                "frames support it. Return strict JSON with keys: supported (boolean), answer (string), evidence "
                "(array of objects with video_id, timestamp_sec, description). Every evidence timestamp must be one "
                "of the successfully sampled absolute timestamps listed below. If evidence is insufficient, set "
                "supported=false and evidence=[].\n\n"
                f"Question: {question}\nInitial answer to verify or replace: {initial_answer}\n\n"
                "Successful frame mapping:\n" + "\n".join(frame_labels)
            )
            try:
                response = provider.ask(frame_paths, prompt)
            except Exception:
                record_usage(usage, None, requests=1, ambiguous_attempts=True)
                warnings.append("The stronger sampled-frame inspection provider was unavailable.")
                continue
            if getattr(response, "attempted", True):
                record_usage(
                    usage,
                    {
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                    requests=1,
                )
            if not response.ok:
                warnings.append("The stronger sampled-frame inspection provider was unavailable.")
                continue
            parsed = _parse_result(response.text, windows, sampled_by_video)
            if parsed is not None:
                answer, citations = parsed
                return {
                    "status": "supported",
                    "answer": answer,
                    "citations": citations,
                    "windows": inspected,
                    "usage": usage,
                    "warnings": warnings,
                }
            warnings.append("The sampled-frame inspection did not return validated supporting evidence.")

    return {
        "status": "insufficient",
        "answer": None,
        "citations": [],
        "windows": inspected,
        "usage": usage,
        "warnings": warnings,
    }
