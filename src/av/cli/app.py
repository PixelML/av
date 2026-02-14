"""Typer root app — wires all subcommands together."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from av import __version__

app = typer.Typer(
    name="av",
    help="av — Agentic Video CLI by Pixel ML. Index once, search many.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command("version")
def version_cmd() -> None:
    """Print version info as JSON."""
    print(json.dumps({"version": __version__, "package": "pixelml-av"}))


# --- Register direct commands ---

from av.cli.ingest import register as register_ingest  # noqa: E402
from av.cli.search import register as register_search  # noqa: E402
from av.cli.ask import register as register_ask  # noqa: E402
from av.cli.list_cmd import register as register_list  # noqa: E402
from av.cli.info import register as register_info  # noqa: E402
from av.cli.transcript import register as register_transcript  # noqa: E402
from av.cli.export import register as register_export  # noqa: E402
from av.cli.open_cmd import register as register_open  # noqa: E402
from av.cli.config_cmd import config_app  # noqa: E402

register_ingest(app)
register_search(app)
register_ask(app)
register_list(app)
register_info(app)
register_transcript(app)
register_export(app)
register_open(app)
app.add_typer(config_app, name="config", help="Show/set configuration")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
