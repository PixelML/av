"""av export command — Video Memory JSONL or subtitle formats."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from av.cli.output import error
from av.core.config import get_config
from av.db.repository import Repository, _fmt_timestamp


def _fmt_vtt_time(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _fmt_srt_time(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def register(app: typer.Typer) -> None:
    @app.command("export")
    def export_cmd(
        format: str = typer.Option("jsonl", "--format", "-f", help="Export format: jsonl, srt, vtt"),
        video_id: str = typer.Option(None, "--video-id", "-v", help="Export specific video only"),
        db: str = typer.Option(None, "--db", help="Database path override"),
    ) -> None:
        """Export video memory as JSONL or subtitle formats."""
        config = get_config(db_path=Path(db) if db else None)
        repo = Repository(config.db_path)

        try:
            videos = repo.list_videos()

            if format == "jsonl":
                for v in videos:
                    if video_id and v.video_id != video_id:
                        continue
                    artifacts = repo.get_artifacts(v.video_id)
                    for art in artifacts:
                        record = {
                            "video_id": v.video_id,
                            "filename": v.filename,
                            "type": art.type,
                            "start_sec": art.start_sec,
                            "end_sec": art.end_sec,
                            "timestamp_formatted": _fmt_timestamp(art.start_sec),
                            "text": art.text,
                        }
                        print(json.dumps(record))

            elif format == "vtt":
                for v in videos:
                    if video_id and v.video_id != video_id:
                        continue
                    artifacts = repo.get_artifacts(v.video_id, artifact_type="transcript")
                    if not artifacts:
                        continue
                    print("WEBVTT")
                    print(f"NOTE video_id={v.video_id} filename={v.filename}")
                    print()
                    for art in artifacts:
                        start = _fmt_vtt_time(art.start_sec)
                        end = _fmt_vtt_time(art.end_sec or art.start_sec)
                        print(f"{start} --> {end}")
                        print(art.text)
                        print()

            elif format == "srt":
                for v in videos:
                    if video_id and v.video_id != video_id:
                        continue
                    artifacts = repo.get_artifacts(v.video_id, artifact_type="transcript")
                    if not artifacts:
                        continue
                    for i, art in enumerate(artifacts, 1):
                        start = _fmt_srt_time(art.start_sec)
                        end = _fmt_srt_time(art.end_sec or art.start_sec)
                        print(i)
                        print(f"{start} --> {end}")
                        print(art.text)
                        print()

            else:
                error(f"Unknown format: {format}. Use jsonl, vtt, or srt.")
                raise typer.Exit(1)

        finally:
            repo.close()
