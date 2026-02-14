"""av open command — open video at specific timestamp."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import typer

from av.cli.output import error, progress
from av.core.config import get_config
from av.core.exceptions import VideoNotFoundError
from av.db.repository import Repository


def register(app: typer.Typer) -> None:
    @app.command("open")
    def open_cmd(
        video_id: str = typer.Argument(..., help="Video ID to open"),
        at: float = typer.Option(0.0, "--at", help="Timestamp in seconds to seek to"),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Open a video file at a specific timestamp."""
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            video = repo.get_video(video_id)
            file_path = Path(video.file_path)

            if not file_path.exists():
                error(f"Video file not found on disk: {file_path}")
                raise typer.Exit(1)

            # Try mpv first if seeking is requested (supports --start)
            if at > 0:
                try:
                    subprocess.Popen(
                        ["mpv", f"--start={at}", str(file_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    progress(f"Opened {video.filename} with mpv at {at:.1f}s")
                    return
                except FileNotFoundError:
                    pass

            # Fall back to system default
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(["open", str(file_path)])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", str(file_path)])
            else:
                subprocess.Popen(["start", str(file_path)], shell=True)

            progress(f"Opened {video.filename}")
            if at > 0:
                progress(f"(Seek to {at:.1f}s manually — install mpv for auto-seek)")

        except VideoNotFoundError:
            error(f"Video not found: {video_id}")
            raise typer.Exit(1)
        finally:
            repo.close()
