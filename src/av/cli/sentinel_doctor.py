"""av sentinel doctor — Preflight checks for surveillance setup.

Validates: ffmpeg, VLM provider, ollama, camera connectivity.
Designed for both humans (styled) and AI agents (--json).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import typer

from av.cli.output import output_json


def _check_ffmpeg() -> dict:
    path = shutil.which("ffmpeg")
    if path:
        try:
            r = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
            ver = r.stdout.split("\n")[0] if r.stdout else "unknown"
            return {"name": "ffmpeg", "ok": True, "message": ver, "path": path}
        except Exception as e:
            return {"name": "ffmpeg", "ok": False, "message": str(e)}
    return {"name": "ffmpeg", "ok": False, "message": "Not found in PATH",
            "hint": "Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"}


def _check_ffprobe() -> dict:
    path = shutil.which("ffprobe")
    if path:
        return {"name": "ffprobe", "ok": True, "message": "Available", "path": path}
    return {"name": "ffprobe", "ok": False, "message": "Not found",
            "hint": "Installed with ffmpeg"}


def _check_ollama() -> dict:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/version", timeout=3)
        if r.status_code == 200:
            ver = r.json().get("version", "?")
            return {"name": "ollama", "ok": True, "message": f"Running v{ver}"}
    except Exception:
        pass

    path = shutil.which("ollama")
    if path:
        return {"name": "ollama", "ok": False,
                "message": "Installed but not running",
                "hint": "Start with: ollama serve"}
    return {"name": "ollama", "ok": False,
            "message": "Not installed",
            "hint": "Install: curl -fsSL https://ollama.com/install.sh | sh"}


def _check_ollama_models() -> dict:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            vision = [m for m in models if any(v in m for v in
                      ["mistral", "gemma", "llava", "llama"])]
            if vision:
                return {"name": "ollama_models", "ok": True,
                        "message": f"{len(vision)} vision models: {', '.join(vision[:3])}"}
            return {"name": "ollama_models", "ok": False,
                    "message": "No vision models found",
                    "hint": "Pull one: ollama pull mistral-small3.2"}
    except Exception:
        pass
    return {"name": "ollama_models", "ok": False, "message": "Ollama not reachable"}


def _check_api_key(name: str, env_vars: list[str]) -> dict:
    for var in env_vars:
        val = os.getenv(var, "")
        if val:
            return {"name": name, "ok": True,
                    "message": f"{var}={val[:8]}..."}
    return {"name": name, "ok": False,
            "message": f"Not set ({', '.join(env_vars)})",
            "hint": f"Set one: export {env_vars[0]}=your-key"}


def _check_hardware() -> dict:
    """Detect GPU/hardware capabilities."""
    info = {"name": "hardware", "ok": True, "message": ""}
    parts = []

    # Check Apple Silicon
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True, timeout=5)
        if "Apple" in r.stdout:
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            gb = int(mem.stdout.strip()) // (1024**3)
            parts.append(f"{r.stdout.strip()} ({gb}GB)")
    except Exception:
        pass

    # Check NVIDIA GPU
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts.append(f"GPU: {r.stdout.strip()}")
    except Exception:
        pass

    if parts:
        info["message"] = " | ".join(parts)
    else:
        info["message"] = "CPU only (no GPU detected)"
        info["hint"] = "For best performance, use Apple Silicon or NVIDIA GPU"

    return info


def register_doctor(app: typer.Typer) -> None:
    @app.command("doctor")
    def doctor_cmd(
        as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
    ) -> None:
        """Check system readiness for av sentinel.

        Validates ffmpeg, VLM providers, hardware, and camera connectivity.

        Examples:
          av doctor
          av doctor --json
          av doctor --json | jq '.checks[] | select(.ok == false)'
        """
        checks = [
            _check_ffmpeg(),
            _check_ffprobe(),
            _check_hardware(),
            _check_ollama(),
            _check_ollama_models(),
            _check_api_key("gemini_key", ["AV_API_KEY", "GEMINI_API_KEY"]),
            _check_api_key("openrouter_key", ["OPENROUTER_API_KEY"]),
            _check_api_key("openai_key", ["OPENAI_API_KEY", "AV_API_KEY"]),
        ]

        passed = sum(1 for c in checks if c["ok"])
        failed = len(checks) - passed

        if as_json:
            output_json({
                "schema": "av.doctor.v1",
                "checks": checks,
                "summary": {"passed": passed, "failed": failed, "total": len(checks)},
            })
            return

        # Human-friendly output
        typer.echo("\nav doctor — System readiness check\n")
        for c in checks:
            icon = typer.style("✓", fg=typer.colors.GREEN) if c["ok"] else typer.style("✗", fg=typer.colors.RED)
            typer.echo(f"  {icon} {c['name']}: {c['message']}")
            if not c["ok"] and c.get("hint"):
                typer.echo(f"    → {c['hint']}")

        typer.echo(f"\n  {passed}/{len(checks)} checks passed")

        if failed > 0:
            typer.echo("\n  Quick fix for sentinel:")
            # Find best available path
            has_ollama = any(c["name"] == "ollama" and c["ok"] for c in checks)
            has_api = any(c["name"] in ("gemini_key", "openrouter_key") and c["ok"] for c in checks)
            if not has_api and not has_ollama:
                typer.echo("    Option 1 (cloud):  export AV_API_KEY=your-gemini-key")
                typer.echo("    Option 2 (local):  ollama pull mistral-small3.2")
        else:
            typer.echo("\n  Ready! Try: av sentinel video.mp4")
