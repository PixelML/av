"""av info command."""

from __future__ import annotations

from pathlib import Path

import typer

from av.cli.output import error, output_json
from av.core.config import get_config
from av.core.exceptions import VideoNotFoundError
from av.db.repository import Repository


def register(app: typer.Typer) -> None:
    @app.command("info")
    def info_cmd(
        video_id: str = typer.Argument(..., help="Video ID to inspect"),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Show detailed info for a specific video."""
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            info = repo.get_video_info(video_id)
            output_json(info.model_dump())
        except VideoNotFoundError:
            error(f"Video not found: {video_id}")
            raise typer.Exit(1)
        finally:
            repo.close()
