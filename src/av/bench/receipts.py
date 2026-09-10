"""Receipts — every benchmark number traces back to one of these files.

A receipt records what was run, against which provider, under which cost model,
and with which determinism controls. Claims carried in a receipt are explicitly
labelled so a reader never has to guess whether a figure was measured here,
computed from other figures, taken from someone else's write-up, or not tested.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# Evidence labels. Anything reported must carry one of these.
MEASURED = "measured"          # produced by this run, on this machine
DERIVED = "derived"            # arithmetic over measured or documented figures
DOCUMENTED = "documented"      # stated by the vendor's own docs
COMMUNITY = "community-reported"  # someone else's number, reproduced verbatim
UNTESTED = "untested"          # asserted nowhere; we did not check

EVIDENCE_LABELS = (MEASURED, DERIVED, DOCUMENTED, COMMUNITY, UNTESTED)

RECEIPT_VERSION = 2


@dataclass
class Claim:
    """One labelled statement. ``source`` is required for non-measured labels."""

    label: str
    statement: str
    source: str | None = None

    def __post_init__(self) -> None:
        if self.label not in EVIDENCE_LABELS:
            raise ValueError(f"unknown evidence label {self.label!r}; expected one of {EVIDENCE_LABELS}")
        if self.label in (COMMUNITY, DOCUMENTED) and not self.source:
            raise ValueError(f"{self.label} claims must cite a source: {self.statement!r}")


def redact_endpoint(base_url: str | None) -> str | None:
    """Reduce a base URL to its host.

    Receipts are meant to be published. A private or tunnelled inference endpoint
    is not something a public artifact should carry, so only the host survives,
    and hosts that are plainly private are replaced with a placeholder.
    """
    if not base_url:
        return None
    try:
        host = urlparse(base_url).hostname or ""
    except ValueError:
        return "<unparseable>"
    if not host:
        return "<none>"
    private_markers = (".ts.net", ".internal", ".local", ".lan")
    if (
        host in ("localhost", "127.0.0.1", "::1")
        or host.startswith(("10.", "192.168.", "172.16.", "100."))
        or host.endswith(private_markers)
    ):
        return "<private>"
    return host


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ffmpeg_version() -> str | None:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return None
    return out.splitlines()[0].strip() if out else None


@dataclass
class Receipt:
    kind: str
    provider: dict = field(default_factory=dict)
    determinism: dict = field(default_factory=dict)
    cost_model: dict | None = None
    cells: list[dict] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def add_claim(self, label: str, statement: str, source: str | None = None) -> None:
        self.claims.append(Claim(label=label, statement=statement, source=source))

    def to_dict(self) -> dict:
        from av import __version__

        return {
            "receipt_version": RECEIPT_VERSION,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "kind": self.kind,
            "av_version": __version__,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "ffmpeg": _ffmpeg_version(),
            },
            "provider": self.provider,
            "determinism": self.determinism,
            "cost_model": self.cost_model,
            "summary": self.summary,
            "cells": self.cells,
            "claims": [asdict(c) for c in self.claims],
            "notes": self.notes,
        }


def write_receipt(receipt: Receipt, out_dir: Path) -> Path:
    """Write a receipt to ``out_dir`` and return its path."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = receipt.created_utc.replace(":", "").replace("-", "")
    path = out_dir / f"{receipt.kind}-{stamp}-{receipt.run_id[:8]}.json"
    path.write_text(json.dumps(receipt.to_dict(), indent=2, default=str) + "\n")
    return path
