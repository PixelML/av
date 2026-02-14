"""av transcript command — output transcript in VTT/SRT/text."""

from __future__ import annotations

from pathlib import Path

import typer

from av.cli.output import error, output_text
from av.core.config import get_config
from av.core.exceptions import VideoNotFoundError
from av.db.repository import Repository


def _fmt_vtt_time(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _fmt_srt_time(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def register(app: typer.Typer) -> None:
    @app.command("transcript")
    def transcript_cmd(
        video_id: str = typer.Argument(..., help="Video ID"),
        format: str = typer.Option("vtt", "--format", "-f", help="Output format: vtt, srt, text"),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Output transcript for a video in VTT, SRT, or plain text."""
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            repo.get_video(video_id)
            artifacts = repo.get_artifacts(video_id, artifact_type="transcript")

            if not artifacts:
                error(f"No transcript found for video: {video_id}")
                raise typer.Exit(1)

            if format == "vtt":
                lines = ["WEBVTT", ""]
                for art in artifacts:
                    start = _fmt_vtt_time(art.start_sec)
                    end = _fmt_vtt_time(art.end_sec or art.start_sec)
                    lines.append(f"{start} --> {end}")
                    lines.append(art.text)
                    lines.append("")
                output_text("\n".join(lines))

            elif format == "srt":
                lines: list[str] = []
                for i, art in enumerate(artifacts, 1):
                    start = _fmt_srt_time(art.start_sec)
                    end = _fmt_srt_time(art.end_sec or art.start_sec)
                    lines.append(str(i))
                    lines.append(f"{start} --> {end}")
                    lines.append(art.text)
                    lines.append("")
                output_text("\n".join(lines))

            elif format == "text":
                output_text("\n".join(art.text for art in artifacts))

            else:
                error(f"Unknown format: {format}. Use vtt, srt, or text.")
                raise typer.Exit(1)

        except VideoNotFoundError:
            error(f"Video not found: {video_id}")
            raise typer.Exit(1)
        finally:
            repo.close()
