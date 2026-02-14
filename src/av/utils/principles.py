"""Load guiding principles for dense captioning."""

from __future__ import annotations

import json
from pathlib import Path


def load_principles(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return [
            "Prioritize observable actions over speculation.",
            "Highlight safety-relevant signals.",
            "Include objects/actors that matter for agent decisions.",
        ]

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # JSON format: ["..."] or {"principles": ["..."]}
    if path.suffix.lower() in {".json", ".jsonl"}:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            vals = data.get("principles", [])
            return [str(x).strip() for x in vals if str(x).strip()]

    # Lightweight YAML-ish parser:
    # principles:
    #   - foo
    #   - bar
    lines = [ln.rstrip() for ln in text.splitlines()]
    in_block = False
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("principles:"):
            in_block = True
            continue
        if in_block and s.startswith("-"):
            item = s[1:].strip()
            if item:
                out.append(item)
        elif not in_block:
            out.append(s)
    return out
