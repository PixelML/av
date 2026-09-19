"""Truthful per-stage token accounting helpers.

Totals stay ``None`` once any contributing request omits that token dimension or
an attempted request fails before trustworthy usage is available. Request counts
still report the actual HTTP attempts.
"""

from __future__ import annotations

from typing import Any


def new_usage() -> dict[str, int | bool | None]:
    return {
        "requests": 0,
        "input_tokens": None,
        "output_tokens": None,
        "input_tokens_complete": True,
        "output_tokens_complete": True,
    }


def record_usage(
    total: dict[str, int | bool | None],
    usage: dict[str, Any] | None,
    *,
    requests: int = 1,
    ambiguous_attempts: bool = False,
) -> None:
    total["requests"] = int(total.get("requests") or 0) + max(requests, 0)
    for token_key, complete_key in (
        ("input_tokens", "input_tokens_complete"),
        ("output_tokens", "output_tokens_complete"),
    ):
        if total.get(complete_key) is False:
            total[token_key] = None
            continue
        value = usage.get(token_key) if isinstance(usage, dict) else None
        if ambiguous_attempts or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            total[token_key] = None
            total[complete_key] = False
            continue
        total[token_key] = int(total.get(token_key) or 0) + value


def merge_usage(
    total: dict[str, int | bool | None],
    addition: dict[str, int | bool | None],
) -> None:
    total["requests"] = int(total.get("requests") or 0) + int(addition.get("requests") or 0)
    for token_key, complete_key in (
        ("input_tokens", "input_tokens_complete"),
        ("output_tokens", "output_tokens_complete"),
    ):
        if total.get(complete_key) is False or addition.get(complete_key) is False:
            total[token_key] = None
            total[complete_key] = False
            continue
        value = addition.get(token_key)
        if isinstance(value, int):
            total[token_key] = int(total.get(token_key) or 0) + value
