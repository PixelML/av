"""Text chunking for embedding generation."""

from __future__ import annotations

from av.db.models import ArtifactRecord


def chunk_artifacts(
    artifacts: list[ArtifactRecord],
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[ArtifactRecord]:
    """Chunk artifacts by merging short consecutive segments.

    Groups consecutive transcript segments into chunks of ~max_tokens words,
    preserving start/end timestamps from the constituent segments.
    This is a word-based approximation (1 token ~ 0.75 words).
    """
    if not artifacts:
        return []

    max_words = int(max_tokens * 0.75)
    chunks: list[ArtifactRecord] = []
    current_texts: list[str] = []
    current_start: float = artifacts[0].start_sec
    current_end: float = artifacts[0].end_sec or artifacts[0].start_sec
    current_video_id: str = artifacts[0].video_id
    current_type: str = artifacts[0].type
    word_count = 0

    for art in artifacts:
        words = art.text.split()
        if word_count + len(words) > max_words and current_texts:
            # Emit chunk
            chunks.append(
                ArtifactRecord(
                    id="",  # Caller assigns ID
                    video_id=current_video_id,
                    type=current_type,
                    start_sec=current_start,
                    end_sec=current_end,
                    text=" ".join(current_texts),
                )
            )
            # Start new chunk (no overlap for simplicity in MVP)
            current_texts = []
            current_start = art.start_sec
            word_count = 0

        current_texts.append(art.text)
        current_end = art.end_sec or art.start_sec
        current_video_id = art.video_id
        current_type = art.type
        word_count += len(words)

    # Emit final chunk
    if current_texts:
        chunks.append(
            ArtifactRecord(
                id="",
                video_id=current_video_id,
                type=current_type,
                start_sec=current_start,
                end_sec=current_end,
                text=" ".join(current_texts),
            )
        )

    return chunks
