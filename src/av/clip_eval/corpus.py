"""Load the frozen synthetic clip-evaluation corpus into an isolated AV index.

The corpus is text-only and inline: transcript segments and vision captions
with stable artifact IDs and checksums. Media is never stored in the repo; a
deterministic ffmpeg generation spec is included so export-validity checks can
render synthetic videos on demand outside the working tree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from av import clip_eval
from av.core.exceptions import AVError, VideoNotFoundError
from av.db.models import ArtifactRecord, VideoRecord
from av.db.repository import Repository

SUPPORTED_VERSIONS = {clip_eval.CONTRACT_VERSION}


class CorpusError(AVError):
    """The corpus file is missing, malformed, or fails its frozen checksums."""


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _checksum(content: dict) -> str:
    return "sha256:" + hashlib.sha256(_canonical(content)).hexdigest()


def load_corpus(path: Path) -> dict:
    """Read and verify the corpus, including per-video frozen checksums."""
    try:
        corpus = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CorpusError(f"Corpus file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f"Corpus file is not valid JSON: {path}: {exc}") from exc
    if corpus.get("contract_version") not in SUPPORTED_VERSIONS:
        raise CorpusError(
            f"Unsupported corpus contract_version: {corpus.get('contract_version')}"
        )
    checksums = corpus.get("checksums")
    if not isinstance(checksums, dict):
        raise CorpusError("Corpus is missing the frozen checksums block")
    for video in corpus.get("videos", []):
        video_id = video.get("id")
        content = video.get("content")
        if not video_id or not isinstance(content, dict):
            raise CorpusError(f"Corpus video {video_id!r} has no content block")
        expected = checksums.get(video_id)
        actual = _checksum(content)
        if expected != actual:
            raise CorpusError(
                f"Checksum mismatch for {video_id}: expected {expected}, got {actual}. "
                "The corpus was modified; this breaks the frozen evaluation contract."
            )
    if not corpus.get("videos"):
        raise CorpusError("Corpus contains no videos")
    return corpus


def load_queries(path: Path) -> list[dict]:
    """Read a labeled query set (dev or held-out)."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CorpusError(f"Query file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f"Query file is not valid JSON: {path}: {exc}") from exc
    queries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(queries, list) or not queries:
        raise CorpusError(f"Query file has no queries: {path}")
    for query in queries:
        for field in ("query_id", "topic", "video_id", "expected", "moments"):
            if field not in query:
                raise CorpusError(f"Query {query!r} is missing field {field!r}")
        if query["expected"] not in {"present", "absent"}:
            raise CorpusError(f"Query {query['query_id']} has invalid expected field")
    return queries


def materialize_corpus(corpus: dict, repo: Repository) -> dict[str, str]:
    """Insert every corpus video and artifact into an empty repository.

    Returns video_id -> source file path (may not exist until media is
    generated; selection never reads media, only export does).
    """
    paths: dict[str, str] = {}
    for video in corpus["videos"]:
        video_id = video["id"]
        content = video["content"]
        try:
            repo.get_video(video_id)
        except VideoNotFoundError:
            pass
        else:
            raise CorpusError(
                f"Target database already contains {video_id}; use a fresh database"
            )
        media = content.get("media", {})
        source_path = media.get("filename", f"{video_id}.mp4")
        repo.insert_video(
            VideoRecord(
                id=video_id,
                file_path=source_path,
                file_hash=_checksum(content),
                file_size_bytes=media.get("approx_size_bytes", 0),
                filename=Path(source_path).name,
                duration_sec=float(content["duration_sec"]),
                status="complete",
            )
        )
        for segment in content.get("transcript", []):
            repo.insert_artifact(
                ArtifactRecord(
                    id=segment["id"],
                    video_id=video_id,
                    type="transcript",
                    start_sec=float(segment["start_sec"]),
                    end_sec=float(segment["end_sec"]),
                    text=segment["text"],
                )
            )
        for caption in content.get("vision", []):
            repo.insert_artifact(
                ArtifactRecord(
                    id=caption["id"],
                    video_id=video_id,
                    type=caption.get("source_type", "caption"),
                    start_sec=float(caption["start_sec"]),
                    end_sec=float(caption["end_sec"]),
                    text=caption["text"],
                )
            )
        paths[video_id] = source_path
    return paths


def generate_media(spec: dict, out_dir: Path) -> Path:
    """Render one synthetic corpus video with ffmpeg (color source + tone).

    The generation command is recorded in the corpus so any ffmpeg build can
    reproduce media locally; media itself is never committed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / spec["filename"]
    if out_path.exists():
        return out_path
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=duration={spec['duration_sec']}:size=320x240:rate=15",
        "-f", "lavfi", "-i", f"sine=frequency={spec.get('tone_hz', 440)}:duration={spec['duration_sec']}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        "-y", str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    except FileNotFoundError as exc:
        raise AVError("ffmpeg not found; cannot generate synthetic eval media") from exc
    except subprocess.CalledProcessError as exc:
        raise AVError(f"ffmpeg failed to generate eval media: {exc.stderr}") from exc
    return out_path
