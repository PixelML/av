"""Pydantic models for database entities and API responses."""

from __future__ import annotations

from pydantic import BaseModel


class VideoRecord(BaseModel):
    id: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    filename: str
    duration_sec: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    bitrate: int | None = None
    ingested_at: str | None = None
    status: str = "pending"
    error_message: str | None = None
    ingest_config_json: str | None = None
    metadata_json: str | None = None


class ArtifactRecord(BaseModel):
    id: str
    video_id: str
    type: str  # 'transcript' | 'caption' | 'scene'
    start_sec: float
    end_sec: float | None = None
    text: str
    meta_json: str | None = None
    created_at: str | None = None


class EmbeddingRecord(BaseModel):
    id: str
    artifact_id: str
    model: str
    dim: int
    vector: bytes


class SearchResult(BaseModel):
    rank: int
    score: float
    video_id: str
    filename: str
    timestamp_sec: float
    timestamp_formatted: str
    source_type: str
    text: str
    artifact_id: str | None = None


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    file_path: str
    duration_sec: float
    duration_formatted: str
    resolution: str | None = None
    status: str
    artifacts: dict[str, int]
    ingested_at: str | None = None


class VideoListItem(BaseModel):
    video_id: str
    filename: str
    duration_formatted: str
    status: str
    artifacts_count: int
