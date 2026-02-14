"""Dense caption utilities: prompt rendering and artifact exports."""

from __future__ import annotations

import json
from pathlib import Path


def render_dense_prompt(template_path: Path, principles: list[str]) -> str:
    template = template_path.read_text(encoding="utf-8")
    principles_block = "\n".join(f"- {p}" for p in principles) if principles else "- None"
    return template.replace("{principles}", principles_block)


def _infer_event_fields(text: str) -> dict:
    """Normalize caption text into structured event fields."""
    t = text.strip()
    risk = "none"
    low = t.lower()
    if any(k in low for k in ["slip", "fall", "crash", "knock", "trip", "hazard", "wet"]):
        risk = "safety_incident"

    return {
        "primary_action": t,
        "actors": [],
        "objects": [],
        "scene_context": "security_camera",
        "risk_signal": risk,
        "suggested_next_action": "flag_for_review" if risk != "none" else "observe",
    }


def export_dense_outputs(output_dir: Path, video_id: str, rows: list[dict]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{video_id}.dense.jsonl"
    md_path = output_dir / f"{video_id}.dense.md"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for r in rows:
            event = {
                "video_id": video_id,
                "timestamp_sec": float(r.get("timestamp_sec", 0)),
                "text": r.get("text", "").strip(),
                "frame_path": r.get("frame_path"),
                **_infer_event_fields(r.get("text", "")),
            }
            jf.write(json.dumps(event, ensure_ascii=False) + "\n")

    with md_path.open("w", encoding="utf-8") as mf:
        mf.write("# Dense visual caption timeline\n\n")
        for r in rows:
            ts = float(r.get("timestamp_sec", 0))
            mm = int(ts // 60)
            ss = int(ts % 60)
            mf.write(f"- [{mm:02d}:{ss:02d}] {r.get('text','').strip()}\n")

    return jsonl_path, md_path
