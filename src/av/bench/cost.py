"""Cost models for benchmark cells.

Self-hosted and API economics are different shapes and must never be conflated:

- ``HourlyCost``   — you rent or own the box. Cost accrues with wall-clock time,
  regardless of how many tokens you push through it. Cheaper per token the busier
  the box is.
- ``PerTokenCost`` — a vendor bills each token. Cost is independent of wall clock,
  and an idle box costs nothing.

Reporting one in the other's units produces a number that means nothing, so the
receipt always records which model produced a figure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CellUsage:
    """Measured usage for one benchmark cell."""

    tokens_in: int = 0
    tokens_out: int = 0
    wall_total_sec: float = 0.0
    # Time to first token. On a streaming API this is the closest measurable
    # proxy for prefill; it is not prefill itself and is labelled as a proxy.
    ttft_sec: float | None = None
    requests: int = 0

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class CostModel:
    """Base class. Subclasses turn measured usage into dollars."""

    mode: str = "unknown"

    def cost_usd(self, usage: CellUsage) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict:  # pragma: no cover - trivial
        raise NotImplementedError

    def basis(self) -> str:
        """`measured` if dollars follow from measured quantities alone."""
        return "measured"


@dataclass
class PerTokenCost(CostModel):
    """Vendor API pricing, quoted per million tokens."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float | None = None
    mode: str = "per_token"

    def cost_usd(self, usage: CellUsage) -> float:
        return (
            usage.tokens_in / 1_000_000 * self.input_usd_per_mtok
            + usage.tokens_out / 1_000_000 * self.output_usd_per_mtok
        )

    def describe(self) -> dict:
        return {"mode": "per_token", **asdict(self)}


@dataclass
class HourlyCost(CostModel):
    """Self-hosted or rented hardware, billed by the hour.

    With no throughput assumption, cost is measured wall clock x hourly rate — an
    honest single-stream number that under-uses the box. Supply
    ``prefill_tok_per_s`` / ``decode_tok_per_s`` to model a saturated box instead;
    that figure is *derived*, not measured, and the receipt says so.
    """

    hourly_usd: float
    prefill_tok_per_s: float | None = None
    decode_tok_per_s: float | None = None
    mode: str = "per_hour"

    def cost_usd(self, usage: CellUsage) -> float:
        if self.prefill_tok_per_s:
            seconds = usage.tokens_in / self.prefill_tok_per_s
            if self.decode_tok_per_s and usage.tokens_out:
                seconds += usage.tokens_out / self.decode_tok_per_s
        else:
            seconds = usage.wall_total_sec
        return seconds / 3600.0 * self.hourly_usd

    def basis(self) -> str:
        return "derived" if self.prefill_tok_per_s else "measured"

    def describe(self) -> dict:
        return {"mode": "per_hour", **asdict(self)}


def video_hours_per_dollar(video_seconds: float, cost_usd: float) -> float | None:
    """The business metric: video-hours analysed per dollar spent."""
    if cost_usd <= 0:
        return None
    return (video_seconds / 3600.0) / cost_usd


def parse_cost_model(spec: str | None) -> CostModel | None:
    """Parse a ``--cost`` spec into a cost model.

    Accepted forms::

        hourly:25.0                       measured wall clock x $25/hr
        hourly:25.0:20000                 modelled at 20,000 prefill tok/s (derived)
        hourly:25.0:20000:1200            ... and 1,200 decode tok/s
        token:0.30:2.50                   $0.30/Mtok in, $2.50/Mtok out
        @/path/to/cost.json               a JSON file with the same fields

    Returns ``None`` when *spec* is empty, so callers can report token counts
    without inventing a price.
    """
    if not spec:
        return None

    spec = spec.strip()
    if spec.startswith("@"):
        data = json.loads(Path(spec[1:]).expanduser().read_text())
        mode = data.get("mode")
        if mode == "per_token":
            return PerTokenCost(
                input_usd_per_mtok=float(data["input_usd_per_mtok"]),
                output_usd_per_mtok=float(data["output_usd_per_mtok"]),
                cached_input_usd_per_mtok=(
                    float(data["cached_input_usd_per_mtok"])
                    if data.get("cached_input_usd_per_mtok") is not None
                    else None
                ),
            )
        if mode == "per_hour":
            return HourlyCost(
                hourly_usd=float(data["hourly_usd"]),
                prefill_tok_per_s=(
                    float(data["prefill_tok_per_s"]) if data.get("prefill_tok_per_s") else None
                ),
                decode_tok_per_s=(
                    float(data["decode_tok_per_s"]) if data.get("decode_tok_per_s") else None
                ),
            )
        raise ValueError(f"cost file has unknown mode: {mode!r}")

    parts = spec.split(":")
    kind = parts[0].lower()

    if kind in ("hourly", "hour", "hr"):
        if len(parts) < 2:
            raise ValueError("hourly cost needs a rate, e.g. hourly:25.0")
        return HourlyCost(
            hourly_usd=float(parts[1]),
            prefill_tok_per_s=float(parts[2]) if len(parts) > 2 and parts[2] else None,
            decode_tok_per_s=float(parts[3]) if len(parts) > 3 and parts[3] else None,
        )

    if kind in ("token", "tokens", "per-token"):
        if len(parts) < 3:
            raise ValueError("token cost needs input and output rates, e.g. token:0.30:2.50")
        return PerTokenCost(
            input_usd_per_mtok=float(parts[1]),
            output_usd_per_mtok=float(parts[2]),
            cached_input_usd_per_mtok=float(parts[3]) if len(parts) > 3 and parts[3] else None,
        )

    raise ValueError(f"unknown cost spec: {spec!r} (expected hourly:..., token:..., or @file.json)")
