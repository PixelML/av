"""av search command."""

from __future__ import annotations

from pathlib import Path

import typer

from av.cli.output import error, output_json
from av.core.config import get_config
from av.core.constants import DEFAULT_SEARCH_LIMIT
from av.db.repository import Repository
from av.search.semantic import search


def register(app: typer.Typer) -> None:
    @app.command("search")
    def search_cmd(
        query: str = typer.Argument(..., help="Search query"),
        limit: int = typer.Option(DEFAULT_SEARCH_LIMIT, "--limit", "-n", help="Max results"),
        video_id: str = typer.Option(None, "--video-id", "-v", help="Restrict to specific video"),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Search indexed videos by text."""
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            result = search(query, repo, config, limit=limit, video_id=video_id)
            output_json(result)
        except Exception as e:
            error(str(e))
            raise typer.Exit(1)
        finally:
            repo.close()
