"""Actual request and token receipts for provider-backed ingest stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _value(source: Any, *names: str) -> Any:
    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        value = getattr(source, name, None)
        if value is not None:
            return value
    return None


def _cached_tokens(usage: Any) -> int | None:
    direct = _value(usage, "cached_input_tokens", "cached_tokens")
    if isinstance(direct, int) and direct >= 0:
        return direct
    details = _value(usage, "prompt_tokens_details", "input_tokens_details")
    cached = _value(details, "cached_tokens")
    return cached if isinstance(cached, int) and cached >= 0 else None


@dataclass
class ProviderUsage:
    requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    input_tokens_complete: bool = True
    output_tokens_complete: bool = True
    cached_input_tokens_complete: bool = True

    def _add(self, key: str, complete_key: str, value: Any) -> None:
        if not getattr(self, complete_key):
            setattr(self, key, None)
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            setattr(self, key, None)
            setattr(self, complete_key, False)
            return
        setattr(self, key, (getattr(self, key) or 0) + value)

    def record_success(self, usage: Any) -> None:
        self.requests += 1
        self.successful_requests += 1
        self._add("input_tokens", "input_tokens_complete", _value(usage, "prompt_tokens", "input_tokens"))
        self._add("output_tokens", "output_tokens_complete", _value(usage, "completion_tokens", "output_tokens"))
        self._add("cached_input_tokens", "cached_input_tokens_complete", _cached_tokens(usage))

    def record_failure(self) -> None:
        self.requests += 1
        self.failed_requests += 1
        for key, complete_key in (
            ("input_tokens", "input_tokens_complete"),
            ("output_tokens", "output_tokens_complete"),
            ("cached_input_tokens", "cached_input_tokens_complete"),
        ):
            setattr(self, key, None)
            setattr(self, complete_key, False)

    def merge(self, receipt: dict | None) -> None:
        if not isinstance(receipt, dict) or not receipt:
            return
        self.requests += int(receipt.get("requests") or 0)
        self.successful_requests += int(receipt.get("successful_requests") or 0)
        self.failed_requests += int(receipt.get("failed_requests") or 0)
        for key, complete_key in (
            ("input_tokens", "input_tokens_complete"),
            ("output_tokens", "output_tokens_complete"),
            ("cached_input_tokens", "cached_input_tokens_complete"),
        ):
            if not getattr(self, complete_key) or receipt.get(complete_key) is False:
                setattr(self, key, None)
                setattr(self, complete_key, False)
                continue
            value = receipt.get(key)
            if isinstance(value, int):
                setattr(self, key, (getattr(self, key) or 0) + value)

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "input_tokens_complete": self.input_tokens_complete,
            "output_tokens_complete": self.output_tokens_complete,
            "cached_input_tokens_complete": self.cached_input_tokens_complete,
        }
