"""JSON/pretty output formatting. JSON to stdout, progress to stderr."""

from __future__ import annotations

import json
import sys


def output_json(data: dict | list, pretty: bool = False) -> None:
    """Write JSON to stdout."""
    if pretty:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(json.dumps(data, default=str))


def output_text(text: str) -> None:
    """Write plain text to stdout."""
    print(text)


def error(message: str) -> None:
    """Write error message to stderr."""
    print(f"Error: {message}", file=sys.stderr)


def warn(message: str) -> None:
    """Write warning to stderr."""
    print(f"Warning: {message}", file=sys.stderr)


def progress(message: str) -> None:
    """Write progress info to stderr."""
    print(message, file=sys.stderr)
