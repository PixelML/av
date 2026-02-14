"""av ask command."""

from __future__ import annotations

from pathlib import Path

import typer

from av.cli.output import error, output_json
from av.core.config import get_config
from av.core.constants import DEFAULT_TOP_K
from av.db.repository import Repository
from av.search.rag import ask


def register(app: typer.Typer) -> None:
    @app.command("ask")
    def ask_cmd(
        question: str = typer.Argument(..., help="Question to ask about video content"),
        video_id: str = typer.Option(None, "--video-id", "-v", help="Restrict to specific video"),
        top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k", help="Context chunks"),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Ask a question about indexed video content (RAG)."""
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            result = ask(question, repo, config, video_id=video_id, top_k=top_k)
            output_json(result)
        except Exception as e:
            error(str(e))
            raise typer.Exit(1)
        finally:
            repo.close()
