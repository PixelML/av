"""Deterministic file hashing for idempotent ingest."""

from __future__ import annotations

import hashlib
from pathlib import Path

from av.core.constants import HASH_PREFIX_BYTES


def file_hash(path: Path) -> str:
    """Compute a fast deterministic hash: SHA-256 of first 64KB + file size.

    This is NOT a full-file hash — it's designed for quick idempotency
    checks on large video files. Collisions are astronomically unlikely
    when combined with file_size_bytes stored separately.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(HASH_PREFIX_BYTES))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()
