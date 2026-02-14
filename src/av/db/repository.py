"""CRUD operations + search (FTS + vector) for the av database."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import uuid
from pathlib import Path

from av.core.exceptions import DatabaseError, VideoNotFoundError
from av.db.connection import get_connection
from av.db.models import (
    ArtifactRecord,
    SearchResult,
    VideoInfo,
    VideoListItem,
    VideoRecord,
)
from av.db.schema import migrate


def _fmt_duration(secs: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_timestamp(secs: float) -> str:
    """Format seconds as HH:MM:SS for display."""
    return _fmt_duration(secs)


class Repository:
    def __init__(self, db_path: Path | None = None):
        self.conn = get_connection(db_path)
        migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    # --- Videos ---

    def insert_video(self, video: VideoRecord) -> None:
        try:
            self.conn.execute(
                """INSERT INTO videos (id, file_path, file_hash, file_size_bytes, filename,
                   duration_sec, width, height, fps, codec, bitrate, status, ingest_config_json, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video.id, video.file_path, video.file_hash, video.file_size_bytes,
                    video.filename, video.duration_sec, video.width, video.height,
                    video.fps, video.codec, video.bitrate, video.status,
                    video.ingest_config_json, video.metadata_json,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise DatabaseError(f"Video already exists: {e}") from e

    def get_video_by_hash(self, file_hash: str) -> VideoRecord | None:
        row = self.conn.execute(
            "SELECT * FROM videos WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return VideoRecord(**dict(row)) if row else None

    def get_video(self, video_id: str) -> VideoRecord:
        row = self.conn.execute(
            "SELECT * FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
        if not row:
            raise VideoNotFoundError(f"Video not found: {video_id}")
        return VideoRecord(**dict(row))

    def update_video_status(self, video_id: str, status: str, error_message: str | None = None) -> None:
        self.conn.execute(
            "UPDATE videos SET status = ?, error_message = ? WHERE id = ?",
            (status, error_message, video_id),
        )
        self.conn.commit()

    def list_videos(self) -> list[VideoListItem]:
        rows = self.conn.execute(
            """SELECT v.id, v.filename, v.duration_sec, v.status,
                      COUNT(a.id) as artifacts_count
               FROM videos v
               LEFT JOIN artifacts a ON a.video_id = v.id
               GROUP BY v.id
               ORDER BY v.ingested_at DESC"""
        ).fetchall()
        return [
            VideoListItem(
                video_id=r["id"],
                filename=r["filename"],
                duration_formatted=_fmt_duration(r["duration_sec"]),
                status=r["status"],
                artifacts_count=r["artifacts_count"],
            )
            for r in rows
        ]

    def get_video_info(self, video_id: str) -> VideoInfo:
        video = self.get_video(video_id)
        artifact_counts: dict[str, int] = {}
        for row in self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM artifacts WHERE video_id = ? GROUP BY type",
            (video_id,),
        ).fetchall():
            artifact_counts[row["type"]] = row["cnt"]

        resolution = None
        if video.width and video.height:
            resolution = f"{video.width}x{video.height}"

        return VideoInfo(
            video_id=video.id,
            filename=video.filename,
            file_path=video.file_path,
            duration_sec=video.duration_sec,
            duration_formatted=_fmt_duration(video.duration_sec),
            resolution=resolution,
            status=video.status,
            artifacts=artifact_counts,
            ingested_at=video.ingested_at,
        )

    def delete_video(self, video_id: str) -> None:
        self.conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        self.conn.commit()

    # --- Artifacts ---

    def insert_artifact(self, artifact: ArtifactRecord) -> None:
        self.conn.execute(
            """INSERT INTO artifacts (id, video_id, type, start_sec, end_sec, text, meta_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact.id, artifact.video_id, artifact.type,
                artifact.start_sec, artifact.end_sec, artifact.text,
                artifact.meta_json,
            ),
        )
        self.conn.commit()

    def insert_artifacts_batch(self, artifacts: list[ArtifactRecord]) -> None:
        self.conn.executemany(
            """INSERT INTO artifacts (id, video_id, type, start_sec, end_sec, text, meta_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (a.id, a.video_id, a.type, a.start_sec, a.end_sec, a.text, a.meta_json)
                for a in artifacts
            ],
        )
        self.conn.commit()

    def get_artifacts(self, video_id: str, artifact_type: str | None = None) -> list[ArtifactRecord]:
        if artifact_type:
            rows = self.conn.execute(
                "SELECT * FROM artifacts WHERE video_id = ? AND type = ? ORDER BY start_sec",
                (video_id, artifact_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM artifacts WHERE video_id = ? ORDER BY start_sec",
                (video_id,),
            ).fetchall()
        return [ArtifactRecord(**dict(r)) for r in rows]

    def count_artifacts(self, video_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE video_id = ?", (video_id,)
        ).fetchone()
        return row[0] if row else 0

    # --- Embeddings ---

    def insert_embedding(self, artifact_id: str, model: str, dim: int, vector: list[float]) -> None:
        blob = struct.pack(f"{len(vector)}f", *vector)
        emb_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO embeddings (id, artifact_id, model, dim, vector) VALUES (?, ?, ?, ?, ?)",
            (emb_id, artifact_id, model, dim, blob),
        )
        self.conn.commit()

    def insert_embeddings_batch(
        self, items: list[tuple[str, str, int, list[float]]]
    ) -> None:
        """Batch insert: list of (artifact_id, model, dim, vector)."""
        rows = []
        for artifact_id, model, dim, vector in items:
            blob = struct.pack(f"{len(vector)}f", *vector)
            rows.append((str(uuid.uuid4()), artifact_id, model, dim, blob))
        self.conn.executemany(
            "INSERT INTO embeddings (id, artifact_id, model, dim, vector) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    # --- Search ---

    def search_fts(
        self, query: str, limit: int = 10, video_id: str | None = None
    ) -> list[SearchResult]:
        """Full-text search using FTS5."""
        if video_id:
            rows = self.conn.execute(
                """SELECT a.*, v.filename,
                          rank as score
                   FROM artifacts a
                   JOIN videos v ON v.id = a.video_id
                   WHERE a.video_id = ?
                     AND a.rowid IN (SELECT rowid FROM artifacts_fts WHERE artifacts_fts MATCH ?)
                   ORDER BY a.start_sec
                   LIMIT ?""",
                (video_id, query, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT a.*, v.filename,
                          rank as score
                   FROM artifacts_fts
                   JOIN artifacts a ON a.rowid = artifacts_fts.rowid
                   JOIN videos v ON v.id = a.video_id
                   WHERE artifacts_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()

        results = []
        for i, r in enumerate(rows):
            results.append(
                SearchResult(
                    rank=i + 1,
                    score=round(abs(float(r["score"])), 4) if r["score"] else 0.0,
                    video_id=r["video_id"],
                    filename=r["filename"],
                    timestamp_sec=r["start_sec"],
                    timestamp_formatted=_fmt_timestamp(r["start_sec"]),
                    source_type=r["type"],
                    text=r["text"],
                    artifact_id=r["id"],
                )
            )
        return results

    def get_embeddings_for_artifacts(self, artifact_ids: list[str]) -> dict[str, list[float]]:
        """Load embedding vectors for a set of artifact IDs."""
        if not artifact_ids:
            return {}
        placeholders = ",".join("?" for _ in artifact_ids)
        rows = self.conn.execute(
            f"SELECT artifact_id, dim, vector FROM embeddings WHERE artifact_id IN ({placeholders})",
            artifact_ids,
        ).fetchall()
        result: dict[str, list[float]] = {}
        for r in rows:
            vec = list(struct.unpack(f"{r['dim']}f", r["vector"]))
            result[r["artifact_id"]] = vec
        return result

    def get_all_artifacts_with_text(
        self, video_id: str | None = None, limit: int = 100
    ) -> list[ArtifactRecord]:
        """Get artifacts with non-empty text, optionally filtered by video_id."""
        if video_id:
            rows = self.conn.execute(
                "SELECT * FROM artifacts WHERE video_id = ? AND text != '' ORDER BY start_sec LIMIT ?",
                (video_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM artifacts WHERE text != '' ORDER BY video_id, start_sec LIMIT ?",
                (limit,),
            ).fetchall()
        return [ArtifactRecord(**dict(r)) for r in rows]
