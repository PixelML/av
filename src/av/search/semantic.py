"""Semantic search: FTS5 primary + optional cosine reranking on stored BLOBs."""

from __future__ import annotations

import math
import time

from av.core.config import AVConfig
from av.db.models import SearchResult
from av.db.repository import Repository
from av.providers.openai import OpenAIEmbedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(
    query: str,
    repo: Repository,
    config: AVConfig,
    *,
    limit: int = 10,
    video_id: str | None = None,
) -> dict:
    """Search artifacts using FTS5, optionally reranked by cosine similarity."""
    start_time = time.time()

    # Step 1: FTS search (always available)
    fts_results = repo.search_fts(query, limit=limit * 3, video_id=video_id)

    if not fts_results:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "search_time_ms": elapsed_ms,
        }

    # Step 2: Try cosine reranking if embeddings exist
    artifact_ids = [r.artifact_id for r in fts_results if r.artifact_id]
    embeddings = repo.get_embeddings_for_artifacts(artifact_ids)

    if embeddings:
        # Embed the query
        try:
            embedder = OpenAIEmbedder(config)
            query_vecs = embedder.embed([query])
            if query_vecs:
                query_vec = query_vecs[0]
                # Rerank by cosine similarity
                scored: list[tuple[float, SearchResult]] = []
                for r in fts_results:
                    if r.artifact_id and r.artifact_id in embeddings:
                        sim = _cosine_similarity(query_vec, embeddings[r.artifact_id])
                        scored.append((sim, r))
                    else:
                        scored.append((0.0, r))

                scored.sort(key=lambda x: x[0], reverse=True)
                fts_results = []
                for i, (sim, r) in enumerate(scored[:limit]):
                    fts_results.append(
                        SearchResult(
                            rank=i + 1,
                            score=round(sim, 4),
                            video_id=r.video_id,
                            filename=r.filename,
                            timestamp_sec=r.timestamp_sec,
                            timestamp_formatted=r.timestamp_formatted,
                            source_type=r.source_type,
                            text=r.text,
                            artifact_id=r.artifact_id,
                        )
                    )
        except Exception:
            # Fall back to FTS-only results if embedding fails
            fts_results = fts_results[:limit]
    else:
        fts_results = fts_results[:limit]

    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "query": query,
        "results": [r.model_dump() for r in fts_results],
        "total_results": len(fts_results),
        "search_time_ms": elapsed_ms,
    }
