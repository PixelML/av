"""av config command — show/set configuration."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from av.cli.output import output_json, output_text
from av.core.config import AVConfig, get_config, save_config
from av.core.constants import PROVIDER_PRESETS
from av.providers.openai import _codex_oauth_token, _openclaw_oauth_token

config_app = typer.Typer()

_console = Console(stderr=True)

_PROVIDER_MENU = [
    ("openai-oauth", "OpenAI (Codex OAuth)", "free via Codex CLI auth"),
    ("openai", "OpenAI (API key)", "paste your sk-... key"),
    ("anthropic", "Anthropic (Claude)", "paste your Anthropic key"),
    ("gemini", "Google (Gemini)", "paste your Google API key"),
]


def _validate_key(provider: str, config_data: dict) -> bool:
    """Make a lightweight API call to verify the key works. Returns True on success."""
    from av.providers.openai import _client

    temp_config = AVConfig(
        provider=provider,
        api_base_url=config_data["api_base_url"],
        api_key=config_data.get("api_key", ""),
        chat_model=config_data["chat_model"],
    )
    client = _client(temp_config)
    try:
        client.chat.completions.create(
            model=config_data["chat_model"],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True
    except Exception as e:
        _console.print(f"  [red]✗[/red] Validation failed: {e}")
        return False


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    config = get_config()
    output_json({
        "provider": config.provider or "(not set)",
        "api_base_url": config.api_base_url,
        "api_key": "***" if config.api_key else "(not set)",
        "transcribe_model": config.transcribe_model or "(disabled)",
        "vision_model": config.vision_model,
        "embed_model": config.embed_model or "(disabled)",
        "chat_model": config.chat_model,
        "db_path": str(config.db_path),
    })


@config_app.command("path")
def config_path() -> None:
    """Show path to the database file."""
    config = get_config()
    output_text(str(config.db_path))


@config_app.command("setup")
def config_setup() -> None:
    """Interactive setup wizard — choose AI provider and configure API keys."""
    _console.print()
    _console.print("[bold]Choose your AI provider:[/bold]")
    _console.print()
    for i, (_, label, desc) in enumerate(_PROVIDER_MENU, 1):
        rec = " [dim](Recommended)[/dim]" if i == 1 else ""
        _console.print(f"  {i}. {label} — {desc}{rec}")
    _console.print()

    choice = IntPrompt.ask(
        "  Enter choice",
        console=_console,
        choices=["1", "2", "3", "4"],
        default=1,
    )
    provider_key = _PROVIDER_MENU[choice - 1][0]
    preset = PROVIDER_PRESETS[provider_key]

    config_data = {"provider": provider_key, **preset}

    # Resolve API key based on provider
    if provider_key == "openai-oauth":
        token = _openclaw_oauth_token() or _codex_oauth_token()
        if token:
            _console.print("  [green]✓[/green] Found Codex OAuth token")
            config_data["api_key"] = ""  # token resolved at runtime
        else:
            _console.print(
                "  [yellow]![/yellow] No Codex OAuth token found. "
                "Run [bold]codex[/bold] or [bold]openclaw[/bold] first to authenticate, "
                "or choose option 2 to use an API key instead."
            )
            _console.print()
            fallback = Prompt.ask(
                "  Enter OpenAI API key (or press Enter to continue without)",
                console=_console,
                default="",
            )
            config_data["api_key"] = fallback.strip()

    elif provider_key == "openai":
        api_key = Prompt.ask("  Enter your OpenAI API key", console=_console)
        config_data["api_key"] = api_key.strip()

    elif provider_key == "anthropic":
        api_key = Prompt.ask("  Enter your Anthropic API key", console=_console)
        config_data["api_key"] = api_key.strip()
        _console.print()
        _console.print(
            "  [yellow]Note:[/yellow] Transcription requires OpenAI. "
            "Use [bold]--no-embed[/bold] or set [bold]AV_OPENAI_API_KEY[/bold] env var for transcription."
        )

    elif provider_key == "gemini":
        api_key = Prompt.ask("  Enter your Google API key", console=_console)
        config_data["api_key"] = api_key.strip()
        _console.print()
        _console.print(
            "  [yellow]Note:[/yellow] Transcription is not supported with Gemini. "
            "Use [bold]--no-embed[/bold] or set [bold]AV_OPENAI_API_KEY[/bold] env var for transcription."
        )

    # Validate API key if one was provided
    has_key = bool(config_data.get("api_key"))
    if has_key:
        _console.print("  Validating API key...", end="")
        if _validate_key(provider_key, config_data):
            _console.print(" [green]✓[/green]")
        else:
            if not Confirm.ask("  Save anyway?", console=_console, default=False):
                _console.print("  Setup cancelled.")
                raise typer.Exit(1)

    # Save
    path = save_config(config_data)

    _console.print()
    _console.print(f"  [green]✓[/green] Config saved to {path}")
    _console.print(f"  [green]✓[/green] Ready! Try: [bold]av ingest video.mp4[/bold]")
    _console.print()
