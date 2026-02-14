"""Ingest pipeline orchestrator."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from av.core.config import AVConfig
from av.core.constants import LONG_VIDEO_WARN_MINUTES, MAX_AUDIO_CHUNK_BYTES
from av.core.exceptions import IngestError
from av.db.models import ArtifactRecord, VideoRecord
from av.db.repository import Repository
from av.pipeline.cascade import run_cascade
from av.pipeline.chunker import chunk_artifacts
from av.pipeline.dense_caption import export_dense_outputs, render_dense_prompt
from av.pipeline.ffmpeg import extract_audio, extract_frames, get_video_info
from av.providers.openai import OpenAICaptioner, OpenAIEmbedder, OpenAITranscriber
from av.utils.hashing import file_hash
from av.utils.principles import load_principles


def _chunk_seconds_for_audio(audio_path: Path, target_bytes: int = MAX_AUDIO_CHUNK_BYTES - (1 * 1024 * 1024)) -> int:
    size = audio_path.stat().st_size
    if size <= 0:
        return 600
    # wav from ffmpeg is roughly constant bitrate; estimate seconds per chunk
    meta = get_video_info(audio_path)
    duration = max(meta.duration_sec, 1.0)
    bytes_per_sec = size / duration
    secs = int(max(60, target_bytes / max(bytes_per_sec, 1.0)))
    return min(secs, 1200)


def _split_audio(audio_path: Path, segment_sec: int) -> list[Path]:
    out_dir = Path(tempfile.mkdtemp(prefix="av_audio_chunks_"))
    out_pattern = out_dir / "chunk_%03d.wav"
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(segment_sec),
        "-c", "copy",
        "-y",
        str(out_pattern),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    return sorted(out_dir.glob("chunk_*.wav"))


def _transcribe_with_chunking(transcriber: OpenAITranscriber, audio_path: Path) -> list:
    """Transcribe single file or chunk to stay under provider upload limits."""
    if audio_path.stat().st_size <= MAX_AUDIO_CHUNK_BYTES:
        return transcriber.transcribe(audio_path)

    seg_sec = _chunk_seconds_for_audio(audio_path)
    chunks = _split_audio(audio_path, seg_sec)
    if not chunks:
        return transcriber.transcribe(audio_path)

    all_segments = []
    offset = 0.0
    for ch in chunks:
        try:
            ch_dur = get_video_info(ch).duration_sec
        except Exception:
            ch_dur = float(seg_sec)
        try:
            ch_segments = transcriber.transcribe(ch)
            for seg in ch_segments:
                seg.start_sec += offset
                seg.end_sec += offset
                all_segments.append(seg)
        finally:
            try:
                ch.unlink(missing_ok=True)
            except Exception:
                pass
        offset += ch_dur
    try:
        shutil.rmtree(chunks[0].parent, ignore_errors=True)
    except Exception:
        pass
    return all_segments




def ingest_video(
    path: Path,
    repo: Repository,
    config: AVConfig,
    *,
    captions: bool = False,
    fps_sample: float = 0.5,
    max_frames: int = 200,
    no_embed: bool = False,
    force: bool = False,
    dry_run: bool = False,
    dense_vision: bool = False,
    principles_path: Path | None = None,
    dense_output_dir: Path | None = None,
    topic: str = "general",
    frame_captions: bool = False,
) -> dict:
    """Ingest a single video file. Returns JSON-serializable result dict."""
    start_time = time.time()
    path = path.resolve()

    if not path.exists():
        raise IngestError(f"File not found: {path}")
    if not path.is_file():
        raise IngestError(f"Not a file: {path}")

    # Step 1: Hash + idempotency check
    fhash = file_hash(path)
    existing = repo.get_video_by_hash(fhash)
    if existing and not force:
        print(f"  Skipping (already ingested): {path.name}", file=sys.stderr)
        return {
            "status": "skipped",
            "video_id": existing.id,
            "filename": existing.filename,
            "reason": "already_ingested",
        }

    if existing and force:
        print(f"  Re-ingesting (--force): {path.name}", file=sys.stderr)
        repo.delete_video(existing.id)

    # Step 2: Extract metadata
    print(f"  Probing: {path.name}...", file=sys.stderr)
    meta = get_video_info(path)

    video_id = str(uuid.uuid4())
    ingest_config = {
        "captions": captions,
        "fps_sample": fps_sample,
        "max_frames": max_frames,
        "no_embed": no_embed,
        "force": force,
        "dense_vision": dense_vision,
        "principles_path": str(principles_path) if principles_path else None,
    }

    video = VideoRecord(
        id=video_id,
        file_path=str(path),
        file_hash=fhash,
        file_size_bytes=meta.file_size_bytes,
        filename=path.name,
        duration_sec=meta.duration_sec,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        codec=meta.codec,
        bitrate=meta.bitrate,
        status="pending",
        ingest_config_json=json.dumps(ingest_config),
    )

    if dry_run:
        return {
            "status": "dry_run",
            "video_id": video_id,
            "filename": path.name,
            "duration_sec": meta.duration_sec,
            "would_caption": captions,
            "would_embed": not no_embed,
            "would_dense_vision": dense_vision,
        }

    # Step 3: Long video warning
    if meta.duration_sec > LONG_VIDEO_WARN_MINUTES * 60:
        mins = meta.duration_sec / 60
        print(
            f"  Warning: Video is {mins:.0f} min long. Consider --max-minutes flag for very long videos.",
            file=sys.stderr,
        )

    # Insert video record
    repo.insert_video(video)

    artifacts_count = 0
    audio_path: Path | None = None
    frames_dir: Path | None = None
    warnings: list[str] = []

    try:
        transcript_artifacts: list[ArtifactRecord] = []
        caption_artifacts: list[ArtifactRecord] = []
        dense_rows: list[dict] = []
        dense_artifacts: list[ArtifactRecord] = []

        # Step 4: Extract audio and transcribe (best-effort)
        can_transcribe = bool(config.transcribe_model)
        if can_transcribe:
            try:
                print(f"  Extracting audio...", file=sys.stderr)
                audio_path = extract_audio(path)

                print(f"  Transcribing...", file=sys.stderr)
                transcriber = OpenAITranscriber(config)

                audio_size = audio_path.stat().st_size
                if audio_size > MAX_AUDIO_CHUNK_BYTES:
                    print(
                        f"  Audio is {audio_size / 1024 / 1024:.0f}MB (>{MAX_AUDIO_CHUNK_BYTES / 1024 / 1024:.0f}MB). Using chunked transcription.",
                        file=sys.stderr,
                    )

                segments = _transcribe_with_chunking(transcriber, audio_path)
                print(f"  Got {len(segments)} transcript segments.", file=sys.stderr)

                for seg in segments:
                    transcript_artifacts.append(
                        ArtifactRecord(
                            id=str(uuid.uuid4()),
                            video_id=video_id,
                            type="transcript",
                            start_sec=seg.start_sec,
                            end_sec=seg.end_sec,
                            text=seg.text,
                            meta_json=json.dumps({"model": config.transcribe_model}),
                        )
                    )

                if transcript_artifacts:
                    repo.insert_artifacts_batch(transcript_artifacts)
                    artifacts_count += len(transcript_artifacts)
            except Exception as e:
                msg = f"Transcription skipped: {e}"
                warnings.append(msg)
                print(f"  Warning: {msg}", file=sys.stderr)
        else:
            msg = f"Transcription disabled (provider={config.provider or 'current'})."
            warnings.append(msg)
            print(f"  {msg}", file=sys.stderr)

        # Step 5: Captions — cascade (default) or legacy per-frame
        cascade_artifacts: list[ArtifactRecord] = []
        if captions:
            try:
                l0, l1, l2 = run_cascade(
                    path,
                    video_id,
                    config,
                    meta.duration_sec,
                    topic=topic,
                )
                cascade_artifacts = l0 + l1 + l2
                if cascade_artifacts:
                    repo.insert_artifacts_batch(cascade_artifacts)
                    artifacts_count += len(cascade_artifacts)
                    # Also track captions for embedding
                    caption_artifacts.extend(cascade_artifacts)
            except Exception as e:
                msg = f"Cascade captioning skipped: {e}"
                warnings.append(msg)
                print(f"  Warning: {msg}", file=sys.stderr)

        if frame_captions:
            try:
                print(
                    f"  Frame captioning enabled (fps={fps_sample}, max={max_frames}). This uses the vision API.",
                    file=sys.stderr,
                )
                frames = extract_frames(path, fps_sample=fps_sample, max_frames=max_frames)
                frames_dir = frames[0][0].parent if frames else None

                if frames:
                    captioner = OpenAICaptioner(config)
                    frame_paths = [f[0] for f in frames]
                    timestamps = [f[1] for f in frames]
                    caps = captioner.caption_frames(frame_paths, timestamps)

                    for cap in caps:
                        caption_artifacts.append(
                            ArtifactRecord(
                                id=str(uuid.uuid4()),
                                video_id=video_id,
                                type="caption",
                                start_sec=cap.timestamp_sec,
                                end_sec=None,
                                text=cap.text,
                                meta_json=json.dumps({"model": config.vision_model}),
                            )
                        )

                    fc_arts = [a for a in caption_artifacts if a not in cascade_artifacts]
                    if fc_arts:
                        repo.insert_artifacts_batch(fc_arts)
                        artifacts_count += len(fc_arts)
            except Exception as e:
                msg = f"Frame captioning skipped: {e}"
                warnings.append(msg)
                print(f"  Warning: {msg}", file=sys.stderr)

        # Step 5b: Dense visual captions (best-effort)
        if dense_vision:
            try:
                if not frames_dir:
                    frames = extract_frames(path, fps_sample=fps_sample, max_frames=max_frames)
                    frames_dir = frames[0][0].parent if frames else None
                else:
                    frames = sorted(
                        (p, (int(p.stem.split("_")[1]) - 1) / fps_sample)
                        for p in frames_dir.glob("frame_*.jpg")
                    )

                principles = load_principles(principles_path)
                template_path = Path(__file__).resolve().parents[3] / "prompts" / "dense_caption.md"
                prompt = render_dense_prompt(template_path, principles)

                if frames:
                    print(f"  Dense vision captioning enabled for {len(frames)} frame(s)...", file=sys.stderr)
                    captioner = OpenAICaptioner(config)
                    frame_paths = [f[0] for f in frames]
                    timestamps = [f[1] for f in frames]
                    dense_caps = captioner.caption_frames(frame_paths, timestamps, prompt=prompt)

                    for cap in dense_caps:
                        dense_rows.append(
                            {
                                "timestamp_sec": cap.timestamp_sec,
                                "text": cap.text,
                                "frame_path": cap.frame_path,
                            }
                        )
                        dense_artifacts.append(
                            ArtifactRecord(
                                id=str(uuid.uuid4()),
                                video_id=video_id,
                                type="dense_caption",
                                start_sec=cap.timestamp_sec,
                                end_sec=None,
                                text=cap.text,
                                meta_json=json.dumps(
                                    {
                                        "model": config.vision_model,
                                        "principles_path": str(principles_path) if principles_path else None,
                                        "prompt_template": str(template_path),
                                    }
                                ),
                            )
                        )

                    if dense_artifacts:
                        repo.insert_artifacts_batch(dense_artifacts)
                        artifacts_count += len(dense_artifacts)

                    out_dir = dense_output_dir or (Path("/tmp/av_dense"))
                    export_dense_outputs(out_dir, video_id, dense_rows)
            except Exception as e:
                msg = f"Dense vision skipped: {e}"
                warnings.append(msg)
                print(f"  Warning: {msg}", file=sys.stderr)

        # Step 6: Embeddings (best-effort)
        can_embed = bool(config.embed_model)
        if not no_embed and not can_embed:
            msg = f"Embeddings disabled (provider={config.provider or 'current'})."
            warnings.append(msg)
            print(f"  {msg}", file=sys.stderr)

        if not no_embed and can_embed:
            try:
                all_artifacts = transcript_artifacts + caption_artifacts + dense_artifacts
                if all_artifacts:
                    print(f"  Generating embeddings for {len(all_artifacts)} artifacts...", file=sys.stderr)
                    embedder = OpenAIEmbedder(config)
                    texts = [a.text for a in all_artifacts]
                    ids = [a.id for a in all_artifacts]

                    batch_size = 100
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i : i + batch_size]
                        batch_ids = ids[i : i + batch_size]
                        vectors = embedder.embed(batch_texts)

                        items = [(aid, config.embed_model, len(vec), vec) for aid, vec in zip(batch_ids, vectors)]
                        repo.insert_embeddings_batch(items)

                    print(f"  Embedded {len(texts)} artifacts.", file=sys.stderr)
            except Exception as e:
                msg = f"Embeddings skipped: {e}"
                warnings.append(msg)
                print(f"  Warning: {msg}", file=sys.stderr)

        # Step 7: Mark complete
        repo.update_video_status(video_id, "complete")

    except Exception as e:
        repo.update_video_status(video_id, "error", str(e))
        raise IngestError(f"Ingest failed for {path.name}: {e}") from e
    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink(missing_ok=True)
        if frames_dir and frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    out = {
        "status": "complete_with_warnings" if warnings else "complete",
        "video_id": video_id,
        "filename": path.name,
        "duration_sec": round(meta.duration_sec, 2),
        "artifacts_count": artifacts_count,
        "elapsed_sec": round(elapsed, 2),
    }
    if warnings:
        out["warnings"] = warnings
    return out
