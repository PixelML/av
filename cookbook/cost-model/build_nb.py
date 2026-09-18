#!/usr/bin/env python3
"""Build or check the cost notebook using only the standard library."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Edit these cells, then regenerate. The notebook does not duplicate model.py.
CELLS = [
    ("markdown", """# AV stage-cost model (offline)

This notebook calculates with the same model.py used by the command line.
The checked-in receipt scenario includes completed ASR and direct-video baseline
attempts, but no completed AV Grok+Jev comparison. Measured tokens, list-rate
estimates, unknown costs, failures, and reservations remain separate. See
README.md for provenance and limitations. These cells never call a provider or
fetch media.
"""),
    ("code", """import json
import sys
from pathlib import Path

root = Path.cwd()
recipe = root if (root / "model.py").is_file() else root / "cookbook" / "cost-model"
sys.path.insert(0, str(recipe.resolve()))
from model import experiment_report, jsonable, scenario, token_cost

input_path = recipe / "scenario.receipts.json"
inputs = json.loads(input_path.read_text())
print(json.dumps(inputs["experiment"], indent=2))
accounting = experiment_report(inputs)
print(json.dumps(jsonable({
    "measured_usage": accounting["measured_usage"],
    "known_list_estimates_usd": accounting["list_rate_estimates"]["known_subtotal_usd"],
    "unknown_costs": accounting["unknown_costs"],
    "failures": accounting["failures"],
    "reservations": accounting["reservations"],
    "cumulative_cap_usd": accounting["cumulative_cap_usd"],
    "known_plus_reserved_usd": accounting["known_plus_reserved_usd"],
    "remaining_cap_usd": accounting["remaining_cap_usd"],
}), indent=2))
"""),
    ("markdown", """## Compare query volumes without hiding unknowns

Indexing is charged once. Retrieval, judge, answer, and optional stronger
inspection are multiplied by query volume. Hosting and storage refer to the
same observation period. A complete modeled total can contain assumptions; it
is not necessarily a measured bill. This receipt scenario remains incomplete
because no completed AV Grok+Jev query exists. Repeated-query projections are
not new benchmark measurements.
"""),
    ("code", """for queries in (1, 100, 1000):
    result = scenario(inputs, queries)
    print(json.dumps(jsonable({
        "queries": queries,
        "known_subtotal_usd": result["indexed"]["known_subtotal_usd"],
        "complete_total_usd": result["indexed"]["complete_total_usd"],
        "unknown_terms": result["indexed"]["unknown_terms"],
        "uncached_to_indexed_before_unknown_costs_ratio": result["uncached_to_indexed_before_unknown_costs_ratio"],
    }), indent=2))
"""),
    ("markdown", """## Edit cache and fallback assumptions explicitly

In the input JSON, record cache policy (TTL, reuse scope, refresh count and
storage duration), hit rate, all-in cache-hit request cost, writes and storage.
Reuse can span requests and users of one application. Do not infer a universal
cache policy from one request with zero cached tokens. The stronger fallback
frequency and per-invocation price are separate inputs; a missing price stays
unknown when the fallback runs. Historical Composer inputs are not AV evidence.
"""),
    ("code", """result = scenario(inputs, 100)
print(json.dumps(jsonable({
    "cache_policy": result["cache_policy"],
    "cache_hit_rate": result["cache_hit_rate"],
    "cache_baseline": result["cache_policy_baseline"],
    "fallback": inputs["query"]["stronger_fallback"],
}), indent=2))
"""),
]


def build_notebook():
    cells = []
    namespace = {"__name__": "cost_notebook"}
    execution = 0
    old_cwd = Path.cwd()
    try:
        os.chdir(HERE)
        for index, (kind, source) in enumerate(CELLS):
            cell = {"cell_type": kind, "id": f"av-cost-{index:02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
            if kind == "code":
                execution += 1
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(source, f"cost-model-cell-{index}", "exec"), namespace)
                cell["execution_count"] = execution
                cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": output.getvalue().splitlines(keepends=True)}]
            cells.append(cell)
    finally:
        os.chdir(old_cwd)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if source or outputs are stale")
    args = parser.parse_args()
    content = json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n"
    path = HERE / "notebook.ipynb"
    if args.check:
        if not path.exists() or path.read_text() != content:
            raise SystemExit("Notebook is stale; run python3 cookbook/cost-model/build_nb.py")
        print("Notebook source and outputs are current.")
    else:
        path.write_text(content)
        print("Wrote notebook.ipynb (offline, deterministic source and outputs).")


if __name__ == "__main__":
    main()
