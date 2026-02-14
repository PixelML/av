"""av ingest command."""

from __future__ import annotations

from pathlib import Path

import typer

from av.cli.output import error, output_json, progress
from av.core.config import get_config
from av.core.constants import DEFAULT_FPS_SAMPLE, DEFAULT_MAX_FRAMES
from av.core.exceptions import AVError
from av.db.repository import Repository
from av.pipeline.ingest import ingest_video
from av.utils.video import discover_videos
from av.utils.youtube import download_video, is_url


def register(app: typer.Typer) -> None:
    @app.command("ingest")
    def ingest(
        path: str = typer.Argument(..., help="Video file, directory, or supported URL (e.g., YouTube)"),
        captions: bool = typer.Option(False, "--captions", help="Enable frame captioning"),
        fps_sample: float = typer.Option(DEFAULT_FPS_SAMPLE, "--fps-sample", help="Frames per second to sample"),
        max_frames: int = typer.Option(DEFAULT_MAX_FRAMES, "--max-frames", help="Max frames to caption"),
        no_embed: bool = typer.Option(False, "--no-embed", help="Skip embedding generation"),
        force: bool = typer.Option(False, "--force", help="Re-ingest even if hash matches"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen"),
        dense_vision: bool = typer.Option(False, "--dense-vision", help="Enable dense visual captioning timeline"),
        principles: str = typer.Option(None, "--principles", help="Path to principles file (.yaml/.json/.txt)"),
        dense_output_dir: str = typer.Option(None, "--dense-output-dir", help="Directory for dense caption outputs (.jsonl/.md)"),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Ingest video file(s) into the av index."""
        videos = []

        if is_url(path):
            progress("Detected URL input. Downloading video first...")
            try:
                downloaded = download_video(path)
            except AVError as e:
                error(str(e))
                raise typer.Exit(1)
            videos = [downloaded]
        else:
            target = Path(path).expanduser().resolve()
            videos = discover_videos(target)

        if not videos:
            error(f"No video files found at: {path}")
            raise typer.Exit(1)

        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        progress(f"Found {len(videos)} video(s) to ingest.")

        results = []
        for video_path in videos:
            progress(f"\nIngesting: {video_path.name}")
            try:
                result = ingest_video(
                    video_path,
                    repo,
                    config,
                    captions=captions,
                    fps_sample=fps_sample,
                    max_frames=max_frames,
                    no_embed=no_embed,
                    force=force,
                    dry_run=dry_run,
                    dense_vision=dense_vision,
                    principles_path=Path(principles).expanduser().resolve() if principles else None,
                    dense_output_dir=Path(dense_output_dir).expanduser().resolve() if dense_output_dir else None,
                )
                results.append(result)
            except AVError as e:
                error(str(e))
                results.append({"status": "error", "filename": video_path.name, "error": str(e)})

        repo.close()

        if len(results) == 1:
            output_json(results[0])
        else:
            output_json({"results": results, "total": len(results)})
