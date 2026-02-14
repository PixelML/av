"""SQLite connection management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from av.core.constants import DEFAULT_DB_PATH


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode and foreign keys enabled."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
