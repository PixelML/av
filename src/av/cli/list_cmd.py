"""av list command."""

from __future__ import annotations

from pathlib import Path

import typer

from av.cli.output import output_json
from av.core.config import get_config
from av.db.repository import Repository


def register(app: typer.Typer) -> None:
    @app.command("list")
    def list_cmd(
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """List all indexed videos."""
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            videos = repo.list_videos()
            output_json({
                "videos": [v.model_dump() for v in videos],
                "total": len(videos),
            })
        finally:
            repo.close()
