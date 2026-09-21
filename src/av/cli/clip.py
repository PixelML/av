"""av clip command — topic-specific highlight clips from one indexed video."""

from __future__ import annotations

from pathlib import Path

import typer

from av.cli.output import error, output_json
from av.core.config import get_config
from av.db.repository import Repository
from av.pipeline.clip_export import export_clips
from av.search.clip import (
    DEFAULT_RETRIEVAL_LIMIT,
    ClipError,
    clip_video,
)


def register(app: typer.Typer) -> None:
    @app.command("clip")
    def clip_cmd(
        topic: str = typer.Argument(..., help="Topic or moment to find highlights for"),
        video_id: str = typer.Option(..., "--video-id", "-v", help="Indexed video to clip"),
        clips: int = typer.Option(3, "--clips", "-k", min=1, help="Maximum clips to return"),
        target_seconds: float = typer.Option(
            30.0, "--target-seconds", min=0.1, help="Preferred clip duration in seconds"
        ),
        min_seconds: float = typer.Option(
            10.0, "--min-seconds", min=0.1, help="Minimum clip duration in seconds"
        ),
        max_seconds: float = typer.Option(
            None, "--max-seconds", min=0.1, help="Maximum clip duration (default: target)"
        ),
        retrieval_limit: int = typer.Option(
            DEFAULT_RETRIEVAL_LIMIT, "--retrieval-limit", min=1, help="Retrieval hits to consider"
        ),
        no_decide: bool = typer.Option(
            False,
            "--no-decide",
            help="Skip Jev decisions; rank retrieval candidates deterministically",
        ),
        max_decision_requests: int = typer.Option(
            None,
            "--max-decision-requests",
            min=1,
            help="Per-run ceiling on System One HTTP attempts (default: AV_CLIP_REQUEST_CAP)",
        ),
        export_dir: str = typer.Option(
            None, "--export", help="Directory to render selected clips with ffmpeg"
        ),
        overwrite_export: bool = typer.Option(
            False, "--overwrite-export", help="Replace existing export files"
        ),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Find and optionally export topic-specific highlight clips.

        Candidates come from retrieval plus bounded temporal neighborhoods, and
        every boundary is an artifact boundary already in the index. When
        configured, Jev judges topical relevance, standalone coherence, visual
        evidence, boundaries, and highlight appeal with typed Noul, Choice,
        and Score operations under an explicit request ceiling. Absent topics
        return no clips instead of best guesses.
        """
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            result = clip_video(
                topic,
                video_id,
                repo,
                config,
                clips_wanted=clips,
                target_seconds=target_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                decide=not no_decide,
                retrieval_limit=retrieval_limit,
                max_requests=max_decision_requests,
            )
            if export_dir:
                if not result["clips"]:
                    raise ClipError("No clips were selected; nothing to export.")
                video = repo.get_video(video_id)
                receipts, elapsed_ms = export_clips(
                    Path(video.file_path),
                    result["clips"],
                    Path(export_dir),
                    overwrite=overwrite_export,
                )
                result["export"] = receipts
                result["timings"]["render_ms"] = round(elapsed_ms, 1)
            output_json(result)
        except Exception as e:  # noqa: BLE001 - CLI boundary renders provider/export failures
            error(str(e))
            raise typer.Exit(1)
        finally:
            repo.close()
