"""av sentinel — Detect surveillance events in video.

Two modes:
  Video:  av sentinel video.mp4          (fast testing)
  RTSP:   av sentinel --watch rtsp://cam (live monitoring, coming soon)

Quick start (cloud):
  export AV_API_KEY=your-gemini-key
  av sentinel video.mp4 --provider gemini

Local (free, private):
  ollama pull mistral-small3.2
  av sentinel video.mp4 --provider ollama --model mistral-small3.2
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import typer

from av.cli.output import output_json, error, progress
from av.pipeline.sentinel import (
    SentinelAgent,
    get_prompt_for_model,
    parse_vlm_response,
    Alert,
)

ALL_ALERT_TYPES = ["FALL", "LONG_QUEUE", "CROWD_GATHERING", "WHEELCHAIR_COMPLIANCE"]


def _call_vlm(
    images_b64: list[str],
    prompt: str,
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """Call VLM with retry. Supports gemini, openrouter, ollama, openai."""
    import requests

    headers = {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "ollama" else {}
    messages = [{"role": "user", "content": [
        *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64],
        {"type": "text", "text": prompt},
    ]}]

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": 4096, "temperature": 0},
                headers=headers,
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"].get("content") or ""
            if resp.status_code in (400, 401, 422):
                return f"[ERROR {resp.status_code}]"
        except Exception:
            pass
        if attempt < 2:
            time.sleep(1 * (2 ** attempt))
    return ""


def _extract_frames(video_path: Path, max_frames: int = 8,
                    start_sec: float = 0, duration_sec: float = 0) -> list[str]:
    """Extract frames as base64 JPEG strings."""
    tmp = Path(tempfile.mkdtemp(prefix="av_sentinel_"))

    if duration_sec <= 0:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        duration_sec = min(float(probe.stdout.strip() or "30"), 30)

    fps = max_frames / duration_sec if duration_sec > 0 else 1

    cmd = ["ffmpeg", "-v", "quiet"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(video_path), "-t", str(duration_sec),
            "-vf", f"fps={fps}", "-q:v", "2", "-frames:v", str(max_frames),
            str(tmp / "frame_%04d.jpg")]
    subprocess.run(cmd, capture_output=True, timeout=60)

    frames = sorted(tmp.glob("frame_*.jpg"))
    b64 = [base64.b64encode(f.read_bytes()).decode() for f in frames]
    shutil.rmtree(tmp, ignore_errors=True)
    return b64


def _resolve_provider(provider: str) -> tuple[str, str, str]:
    """Resolve provider to (api_key, base_url, default_model)."""
    if provider == "ollama":
        return "ollama", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), "mistral-small3.2"
    if provider == "gemini":
        key = os.getenv("AV_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        return key, "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "")
        return key, "https://openrouter.ai/api/v1", "google/gemma-3-12b-it"
    # openai or custom
    key = os.getenv("AV_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base = os.getenv("AV_API_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return key, base, "gpt-4.1-mini"


def _auto_detect_provider() -> str:
    """Pick best available provider automatically."""
    if os.getenv("AV_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini"  # Quick start — no signup needed for free tier
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # Check if ollama is running
    try:
        import requests
        r = requests.get("http://localhost:11434/api/version", timeout=2)
        if r.status_code == 200:
            return "ollama"
    except Exception:
        pass
    return "gemini"  # Default — most accessible


def register(app: typer.Typer) -> None:
    @app.command("sentinel")
    def sentinel_cmd(
        video: str = typer.Argument(help="Video file, directory, or URL"),
        camera: str = typer.Option("cam_default", "--camera", "-c",
                                   help="Camera ID for state tracking"),
        alerts: str = typer.Option(
            "FALL,LONG_QUEUE,CROWD_GATHERING,WHEELCHAIR_COMPLIANCE",
            "--alerts", "-a",
            help="Comma-separated alert types",
        ),
        provider: str = typer.Option(
            "auto", "--provider", "-p",
            help="VLM provider: auto, gemini, openrouter, ollama, openai",
        ),
        model: str = typer.Option("", "--model", "-m", help="Model ID (auto-detected if empty)"),
        chunk_sec: float = typer.Option(30.0, "--chunk-sec", help="Chunk duration in seconds"),
        overlap_sec: float = typer.Option(5.0, "--overlap-sec"),
        frames: int = typer.Option(8, "--frames", help="Frames per chunk"),
        max_chunks: int = typer.Option(0, "--max-chunks", help="Limit chunks (0=all)"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate without VLM calls"),
        output_file: str = typer.Option("", "--output", "-o", help="Save JSON to file"),
    ) -> None:
        """Detect surveillance events in video footage.

        Detects FALL, LONG_QUEUE, CROWD_GATHERING, WHEELCHAIR_COMPLIANCE
        using temporal reasoning over VLM observations.

        Quick start (cloud):
          export AV_API_KEY=your-gemini-key
          av sentinel video.mp4

        Local (free):
          ollama pull mistral-small3.2
          av sentinel video.mp4 --provider ollama

        Examples:
          av sentinel video.mp4
          av sentinel video.mp4 --alerts FALL
          av sentinel video.mp4 --provider ollama --model mistral-small3.2
          av sentinel videos/ --camera cam_lobby
        """
        # Validate alert types
        alert_types = [a.strip().upper() for a in alerts.split(",")]
        for at in alert_types:
            if at not in ALL_ALERT_TYPES:
                error(f"Unknown alert type: {at}. Available: {ALL_ALERT_TYPES}")
                raise typer.Exit(1)
                return

        # Resolve provider
        actual_provider = provider if provider != "auto" else _auto_detect_provider()
        api_key, base_url, default_model = _resolve_provider(actual_provider)
        actual_model = model or default_model

        if not api_key and actual_provider not in ("ollama",):
            error(f"Set AV_API_KEY for {actual_provider} or use --provider ollama (free)")
            raise typer.Exit(1)

        # Resolve video files
        video_path = Path(video).expanduser().resolve()
        if video_path.is_dir():
            video_files = sorted(p for p in video_path.iterdir()
                                 if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"))
        elif video_path.exists():
            video_files = [video_path]
        else:
            error(f"Video not found: {video}")
            raise typer.Exit(1)

        if dry_run:
            typer.echo(json.dumps({
                "dry_run": True, "provider": actual_provider, "model": actual_model,
                "videos": len(video_files), "alert_types": alert_types,
                "chunk_sec": chunk_sec, "overlap_sec": overlap_sec, "frames": frames,
            }, indent=2))
            return

        # Process
        agent = SentinelAgent(alert_types=alert_types)
        all_results = []

        for vf in video_files:
            typer.echo(f"\n{'='*60}", err=True)

            from av.pipeline.ffmpeg import get_video_info
            info = get_video_info(vf)
            duration = info.duration_sec
            step = chunk_sec - overlap_sec

            chunk_starts = []
            t = 0.0
            while t < duration:
                chunk_starts.append(t)
                t += step
            if max_chunks > 0:
                chunk_starts = chunk_starts[:max_chunks]

            typer.echo(f"  Video: {vf.name} ({duration:.0f}s, {info.width}x{info.height})", err=True)
            typer.echo(f"  Model: {actual_model} ({actual_provider})", err=True)
            typer.echo(f"  Chunks: {len(chunk_starts)}", err=True)

            video_alerts = []
            chunks = []

            for ci, start in enumerate(chunk_starts):
                end = min(start + chunk_sec, duration)
                b64 = _extract_frames(vf, frames, start, end - start)
                if not b64:
                    continue

                prompt = get_prompt_for_model(actual_model, len(b64))
                t0 = time.time()
                raw = _call_vlm(b64, prompt, provider=actual_provider, model=actual_model,
                                api_key=api_key, base_url=base_url)
                vlm_ms = (time.time() - t0) * 1000

                obs = parse_vlm_response(raw)
                chunk_alerts = agent.process(camera, obs, timestamp=time.time())

                # Text fallback for falls
                if not any(a.type == "FALL" for a in chunk_alerts) and "FALL" in alert_types:
                    for phrase in ["lying on the ground", "person fell", "collapsed", "fallen"]:
                        if phrase in (raw or "").lower():
                            chunk_alerts.append(Alert(type="FALL", evidence=f"text: {phrase}",
                                                     confidence=0.7, severity="CRITICAL",
                                                     camera_id=camera))
                            break

                video_alerts.extend(chunk_alerts)
                pc = obs.get("person_count", "?")
                alert_str = " ".join(f"[{a.type}:{a.severity}]" for a in chunk_alerts)
                typer.echo(f"  [{ci+1:>3}/{len(chunk_starts)}] {start:>5.0f}-{end:>5.0f}s "
                           f"pc={pc:<4} {vlm_ms:>6.0f}ms {alert_str}", err=True)

                chunks.append({"chunk": ci, "start": round(start, 1), "end": round(end, 1),
                               "person_count": pc, "alerts": [a.to_dict() for a in chunk_alerts]})

            if video_alerts:
                typer.echo(f"\n  ALERTS ({len(video_alerts)}):", err=True)
                for a in video_alerts:
                    typer.echo(f"    [{a.type}] {a.severity}: {a.evidence[:60]}", err=True)
            else:
                typer.echo(f"\n  No alerts.", err=True)

            state = agent.get_state(camera)
            all_results.append({
                "video": str(vf), "camera_id": camera,
                "duration_sec": round(duration, 1),
                "resolution": f"{info.width}x{info.height}",
                "chunks_total": len(chunk_starts),
                "alerts": [a.to_dict() for a in video_alerts],
                "alert_count": len(video_alerts),
                "person_count_history": state.person_counts,
                "model": actual_model, "provider": actual_provider,
                "chunks": chunks,
            })
            agent.reset(camera)

        output = {
            "videos_processed": len(all_results),
            "total_alerts": sum(r["alert_count"] for r in all_results),
            "results": all_results[0] if len(all_results) == 1 else all_results,
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            typer.echo(f"\n  Saved to {output_file}", err=True)

        output_json(output)
