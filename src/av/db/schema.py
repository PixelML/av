"""Database migrations with schema_version tracking."""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[str] = [
    # Version 1: Initial schema
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL,
        applied_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS videos (
        id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,
        file_hash TEXT NOT NULL UNIQUE,
        file_size_bytes INTEGER NOT NULL,
        filename TEXT NOT NULL,
        duration_sec REAL NOT NULL,
        width INTEGER,
        height INTEGER,
        fps REAL,
        codec TEXT,
        bitrate INTEGER,
        ingested_at TEXT DEFAULT (datetime('now')),
        status TEXT DEFAULT 'pending',
        error_message TEXT,
        ingest_config_json TEXT,
        metadata_json TEXT
    );

    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        type TEXT NOT NULL,
        start_sec REAL NOT NULL,
        end_sec REAL,
        text TEXT NOT NULL,
        meta_json TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_artifacts_video ON artifacts(video_id, start_sec);
    CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(video_id, type);

    CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY,
        artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
        model TEXT NOT NULL,
        dim INTEGER NOT NULL,
        vector BLOB NOT NULL
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
        text, content=artifacts, content_rowid=rowid
    );

    -- FTS triggers to keep in sync
    CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
        INSERT INTO artifacts_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    CREATE TRIGGER IF NOT EXISTS artifacts_ad AFTER DELETE ON artifacts BEGIN
        INSERT INTO artifacts_fts(artifacts_fts, rowid, text) VALUES('delete', old.rowid, old.text);
    END;
    CREATE TRIGGER IF NOT EXISTS artifacts_au AFTER UPDATE ON artifacts BEGIN
        INSERT INTO artifacts_fts(artifacts_fts, rowid, text) VALUES('delete', old.rowid, old.text);
        INSERT INTO artifacts_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    """,
]


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version, or 0 if no schema exists."""
    try:
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        return row[0] or 0 if row else 0
    except sqlite3.OperationalError:
        return 0


def migrate(conn: sqlite3.Connection) -> int:
    """Run pending migrations. Returns the final schema version."""
    current = get_schema_version(conn)

    for i, sql in enumerate(MIGRATIONS, start=1):
        if i <= current:
            continue
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
        conn.commit()

    return len(MIGRATIONS)
