"""Three-layer captioning cascade: chunk VLM → structured summary → final report."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from av.core.config import AVConfig
from av.core.constants import (
    DEFAULT_CHUNK_DURATION_SEC,
    DEFAULT_FRAMES_PER_CHUNK,
    TOPIC_PRESETS,
    _GENERAL_TEMPLATE,
)
from av.db.models import ArtifactRecord
from av.providers.openai import OpenAICaptioner, OpenAILLM


# --- Layer 1 & 2 system prompts ---

LAYER1_SYSTEM_PROMPT = """\
You are a video analysis system producing a structured event log from chunk-level observations.

Format each event as:
START_SEC:END_SEC:EVENT

Rules:
- Merge consecutive chunks describing the same ongoing event into one entry.
- Drop trivial or purely static observations (e.g., "nothing happens", "camera is still").
- Use concrete language: who/what did what, when.
- Preserve timestamps accurately.
- If the input is empty or all chunks were STATIC, respond with: NO_EVENTS"""

LAYER2_SYSTEM_PROMPT = """\
You are producing the final consolidated video analysis report from a structured event log.

Format:
## Events
List each event with MM:SS timestamps and a clear description.

## Summary
2-3 sentence overview of the video content.

## Categories
Tag each event with relevant categories (e.g., "person_entry", "vehicle_movement", "equipment_use").

If the input indicates NO_EVENTS, produce a report stating no significant events were detected."""


def _extract_chunk_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
) -> list[tuple[Path, float]]:
    """Extract exactly N frames evenly spaced within [start_sec, end_sec] using fast-seek."""
    output_dir = Path(tempfile.mkdtemp(prefix="av_chunk_"))
    duration = end_sec - start_sec
    if duration <= 0 or num_frames <= 0:
        return []

    # Calculate timestamps for evenly spaced frames
    if num_frames == 1:
        frame_timestamps = [start_sec + duration / 2]
    else:
        step = duration / (num_frames - 1)
        frame_timestamps = [start_sec + i * step for i in range(num_frames)]

    frames: list[tuple[Path, float]] = []
    for i, ts in enumerate(frame_timestamps):
        out_path = output_dir / f"chunk_{i:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            "-y",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            frames.append((out_path, ts))

    return frames


def _build_chunk_prompt(
    topic: str,
    start_sec: float,
    end_sec: float,
    chunk_duration: int,
    frames_per_chunk: int,
) -> str:
    """Build the VLM prompt for a chunk, using topic presets or custom descriptions."""
    if topic in TOPIC_PRESETS:
        template = TOPIC_PRESETS[topic]
    else:
        # Custom topic: inject as focus area into the general template
        template = _GENERAL_TEMPLATE.replace(
            "{custom_focus}",
            f"Focus on: {topic}. ",
        )

    return template.format(
        start_sec=f"{start_sec:.1f}",
        end_sec=f"{end_sec:.1f}",
        chunk_duration=chunk_duration,
        frames_per_chunk=frames_per_chunk,
    )


def run_cascade(
    video_path: Path,
    video_id: str,
    config: AVConfig,
    duration_sec: float,
    *,
    topic: str = "general",
    chunk_duration_sec: int = DEFAULT_CHUNK_DURATION_SEC,
    frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK,
) -> tuple[list[ArtifactRecord], list[ArtifactRecord], list[ArtifactRecord]]:
    """Run the three-layer captioning cascade.

    Returns (layer0_artifacts, layer1_artifacts, layer2_artifacts).
    Each layer is best-effort: if Layer 1/2 fails, earlier layers are still returned.
    """
    layer0: list[ArtifactRecord] = []
    layer1: list[ArtifactRecord] = []
    layer2: list[ArtifactRecord] = []
    temp_dirs: list[Path] = []

    try:
        # Compute chunks
        num_chunks = max(1, math.ceil(duration_sec / chunk_duration_sec))
        captioner = OpenAICaptioner(config)

        meta_base = {
            "model": config.vision_model,
            "topic": topic,
            "layer": 0,
        }

        # --- Layer 0: Chunk-level VLM captioning ---
        print(f"  Cascade Layer 0: {num_chunks} chunk(s) × {frames_per_chunk} frames...", file=sys.stderr)
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_duration_sec
            end = min(start + chunk_duration_sec, duration_sec)

            prompt = _build_chunk_prompt(
                topic, start, end, chunk_duration_sec, frames_per_chunk,
            )

            try:
                frames = _extract_chunk_frames(video_path, start, end, frames_per_chunk)
                if frames:
                    temp_dirs.append(frames[0][0].parent)
                if not frames:
                    continue

                frame_paths = [f[0] for f in frames]
                timestamps = [f[1] for f in frames]

                text = captioner.caption_chunk(frame_paths, timestamps, prompt)
                if not text or text.strip().upper() == "STATIC":
                    continue

                layer0.append(ArtifactRecord(
                    id=str(uuid.uuid4()),
                    video_id=video_id,
                    type="caption",
                    start_sec=start,
                    end_sec=end,
                    text=text,
                    meta_json=json.dumps(meta_base),
                ))
                print(f"    Chunk {chunk_idx + 1}/{num_chunks}: {text[:80]}...", file=sys.stderr)
            except Exception as e:
                print(f"    Warning: chunk {chunk_idx + 1} failed: {e}", file=sys.stderr)

        if not layer0:
            print("  Cascade: no events detected in Layer 0.", file=sys.stderr)
            return layer0, layer1, layer2

        # --- Layer 1: Structured event log ---
        try:
            print(f"  Cascade Layer 1: summarizing {len(layer0)} chunk captions...", file=sys.stderr)
            llm = OpenAILLM(config)

            chunk_texts = []
            for art in layer0:
                chunk_texts.append(f"[{art.start_sec:.0f}s–{art.end_sec:.0f}s] {art.text}")
            user_content = "\n".join(chunk_texts)

            summary_text = llm.summarize(LAYER1_SYSTEM_PROMPT, user_content)

            if summary_text and summary_text.strip() != "NO_EVENTS":
                layer1.append(ArtifactRecord(
                    id=str(uuid.uuid4()),
                    video_id=video_id,
                    type="summary",
                    start_sec=0.0,
                    end_sec=duration_sec,
                    text=summary_text,
                    meta_json=json.dumps({
                        "model": config.chat_model,
                        "topic": topic,
                        "layer": 1,
                        "source_chunks": len(layer0),
                    }),
                ))
                print(f"  Layer 1 complete: {len(summary_text)} chars", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Layer 1 summarization failed: {e}", file=sys.stderr)

        # --- Layer 2: Final report ---
        if layer1:
            try:
                print(f"  Cascade Layer 2: generating final report...", file=sys.stderr)
                report_text = llm.summarize(LAYER2_SYSTEM_PROMPT, layer1[0].text)

                if report_text:
                    layer2.append(ArtifactRecord(
                        id=str(uuid.uuid4()),
                        video_id=video_id,
                        type="report",
                        start_sec=0.0,
                        end_sec=duration_sec,
                        text=report_text,
                        meta_json=json.dumps({
                            "model": config.chat_model,
                            "topic": topic,
                            "layer": 2,
                        }),
                    ))
                    print(f"  Layer 2 complete: {len(report_text)} chars", file=sys.stderr)
            except Exception as e:
                print(f"  Warning: Layer 2 report failed: {e}", file=sys.stderr)

    finally:
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    return layer0, layer1, layer2
