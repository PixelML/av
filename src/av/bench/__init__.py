"""av bench — cost/accuracy frontier measurement for video understanding.

Two headline axes, chosen to be readable against published agentic-video results:
tokens per query and task accuracy. A third axis that API vendors cannot report is
added alongside: dollars per query and video-hours per dollar on hardware you own.

Everything here writes a receipt. No number in a report should exist without one.
"""

from __future__ import annotations

from av.bench.cost import CostModel, HourlyCost, PerTokenCost, parse_cost_model
from av.bench.receipts import Claim, Receipt, write_receipt

__all__ = [
    "Claim",
    "CostModel",
    "HourlyCost",
    "PerTokenCost",
    "Receipt",
    "parse_cost_model",
    "write_receipt",
]
