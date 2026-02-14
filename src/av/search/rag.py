"""RAG Q&A with citations over video artifacts."""

from __future__ import annotations

from av.core.config import AVConfig
from av.db.repository import Repository, _fmt_timestamp
from av.providers.openai import OpenAILLM
from av.search.semantic import search


def ask(
    question: str,
    repo: Repository,
    config: AVConfig,
    *,
    video_id: str | None = None,
    top_k: int = 10,
) -> dict:
    """Answer a question using RAG over video artifacts."""
    # Step 1: Retrieve relevant context
    search_result = search(
        question, repo, config, limit=top_k, video_id=video_id
    )

    results = search_result.get("results", [])
    if not results:
        return {
            "answer": "No relevant content found in the indexed videos.",
            "citations": [],
            "confidence": 0.0,
        }

    # Step 2: Build context string
    context_parts: list[str] = []
    for r in results:
        vid = r.get("video_id", "")
        fn = r.get("filename", "")
        ts = r.get("timestamp_formatted", "")
        src = r.get("source_type", "")
        text = r.get("text", "")
        context_parts.append(f"[{fn} @ {ts} ({src})] {text}")

    context = "\n\n".join(context_parts)

    # Step 3: Generate answer
    llm = OpenAILLM(config)
    answer = llm.complete(question, context)

    # Step 4: Build citations
    citations = []
    for r in results:
        citations.append({
            "video_id": r.get("video_id", ""),
            "start_sec": r.get("timestamp_sec", 0),
            "end_sec": None,
            "source_type": r.get("source_type", ""),
            "text": r.get("text", ""),
            "score": r.get("score", 0),
        })

    # Confidence is a rough heuristic based on top result score
    top_score = results[0].get("score", 0) if results else 0
    confidence = min(round(float(top_score), 2), 1.0) if top_score else 0.5

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
    }
