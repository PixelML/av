"""Command-line entry: ``python -m av.clip_eval --corpus ... --queries ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from av.clip_eval.runner import run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m av.clip_eval",
        description="Run the topic-clipping evaluation contract over frozen fixtures.",
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Path to corpus.json")
    parser.add_argument("--queries", type=Path, required=True, help="Path to a labeled query set")
    parser.add_argument("--db", type=Path, required=True, help="Fresh SQLite database path (outside the repo)")
    parser.add_argument("--out", type=Path, required=True, help="Receipt JSON output path")
    parser.add_argument("--media-dir", type=Path, default=None, help="Directory for generated synthetic media (outside the repo)")
    parser.add_argument("--export-dir", type=Path, default=None, help="Directory for rendered clips (enables export-validity metrics)")
    parser.add_argument("--arms", default="deterministic,jev_mock", help="Comma-separated arms: deterministic,jev_mock")
    parser.add_argument("--clips", type=int, default=2, help="Clips requested per query")
    parser.add_argument("--target-seconds", type=float, default=30.0)
    parser.add_argument("--min-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    receipt = run_evaluation(
        args.corpus,
        args.queries,
        db_path=args.db,
        media_dir=args.media_dir,
        export_dir=args.export_dir,
        arms=tuple(arm.strip() for arm in args.arms.split(",") if arm.strip()),
        clips_wanted=args.clips,
        target_seconds=args.target_seconds,
        min_seconds=args.min_seconds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps({"receipt": str(args.out), "arms": [arm["arm"] for arm in receipt["arms"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
