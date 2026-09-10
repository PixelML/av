"""Sweep axes, noise floor, and the frontier summary.

The noise floor exists because a benchmark that publishes a single run invites
readers to over-read small deltas. Repeating one unchanged cell and publishing the
spread is what makes a difference legible: any gap smaller than the spread is noise,
and the harness says so rather than leaving the reader to work it out.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

# The cost/accuracy frontier axis, in seconds between sampled frames.
# 1 fps is the dense baseline that published agentic-video comparisons use.
DEFAULT_FRAME_INTERVALS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

DEFAULT_NOISE_REPEATS = 5


@dataclass
class Spread:
    values: list[float]
    unit: str = ""

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def minimum(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def maximum(self) -> float | None:
        return max(self.values) if self.values else None

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.values) if self.values else None

    @property
    def stdev(self) -> float | None:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def range(self) -> float | None:
        if not self.values:
            return None
        return max(self.values) - min(self.values)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "unit": self.unit,
            "values": [round(v, 6) for v in self.values],
            "min": self.minimum,
            "max": self.maximum,
            "mean": round(self.mean, 6) if self.mean is not None else None,
            "stdev": round(self.stdev, 6) if self.stdev is not None else None,
            "range": round(self.range, 6) if self.range is not None else None,
        }


def noise_floor(
    run_once: Callable[[int], float | None],
    *,
    repeats: int = DEFAULT_NOISE_REPEATS,
    unit: str = "",
    on_run: Callable[[int, float | None], None] | None = None,
) -> Spread:
    """Run one unchanged cell ``repeats`` times and collect the spread.

    ``run_once`` receives the zero-based iteration index and returns the metric being
    watched, or ``None`` if that run failed. Failed runs are dropped from the spread
    and the reduced ``n`` makes the omission visible.
    """
    values: list[float] = []
    for i in range(repeats):
        value = run_once(i)
        if on_run:
            on_run(i, value)
        if value is not None:
            values.append(float(value))
    return Spread(values=values, unit=unit)


def is_saturated(spread: Spread, ceiling: float = 1.0) -> bool:
    """True when every run hit the ceiling, so the spread says nothing about variance.

    A cell the model solves perfectly every time has no headroom to vary, and quoting
    its zero spread as *the* noise floor understates variance everywhere else. The
    floor has to be measured on a cell that is actually contested.
    """
    return bool(spread.values) and spread.range == 0.0 and spread.values[0] >= ceiling


def interpret_delta(delta: float, spread: Spread) -> str:
    """Classify an observed difference against the measured noise floor."""
    if spread.range is None or spread.n < 2:
        return "no noise floor measured — treat any delta as unverified"
    if is_saturated(spread):
        return (
            "every run hit the ceiling, so this spread measures nothing. Re-measure the "
            "noise floor on a cell the model does not solve perfectly."
        )
    if abs(delta) <= spread.range:
        return (
            f"within the noise floor (spread {spread.range:.4g}{spread.unit} over "
            f"{spread.n} identical runs) — not a real difference"
        )
    return (
        f"exceeds the noise floor (spread {spread.range:.4g}{spread.unit} over "
        f"{spread.n} identical runs)"
    )


def collapse_point(
    rows: list[dict],
    *,
    score_key: str = "score",
    tolerance: float = 0.1,
) -> dict:
    """Find the widest interval whose score is still within ``tolerance`` of the densest.

    Returns the safe interval and the one after it, so a reader sees both the
    recommendation and the evidence for where it stops being safe.
    """
    scored = [r for r in sorted(rows, key=lambda c: c.get("interval_sec") or 0.0)
              if r.get(score_key) is not None]
    if not scored:
        return {"safe_interval_sec": None, "reason": "no scored cells"}

    baseline = scored[0][score_key]
    safe = scored[0]
    first_collapse = None
    for row in scored[1:]:
        if baseline - row[score_key] <= tolerance:
            safe = row
        elif first_collapse is None:
            first_collapse = row
    return {
        "baseline_interval_sec": scored[0].get("interval_sec"),
        "baseline_score": baseline,
        "safe_interval_sec": safe.get("interval_sec"),
        "safe_score": safe.get(score_key),
        "collapse_interval_sec": first_collapse.get("interval_sec") if first_collapse else None,
        "collapse_score": first_collapse.get(score_key) if first_collapse else None,
        "tolerance": tolerance,
    }
