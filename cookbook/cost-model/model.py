#!/usr/bin/env python3
"""Offline stage-cost accounting. Python stdlib only; no credentials or network."""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

BASES = {"measured", "estimated", "assumed", "unknown", "not_used"}
RESERVATION_STATES = {"retained", "released_after_metering", "superseded"}
ZERO = Decimal("0")
ONE = Decimal("1")


def number(value, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{label} must be a non-negative finite number")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite() or result < ZERO:
        raise ValueError(f"{label} must be a non-negative finite number")
    return result


def fraction(value, label: str) -> Decimal:
    result = number(value, label)
    if result > ONE:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def rate_per_million(published_usd, published_token_unit: str) -> Decimal:
    """Normalize a published token-price unit without silently assuming millions."""
    denominators = {"million": Decimal("1000000"), "billion": Decimal("1000000000")}
    if published_token_unit not in denominators:
        raise ValueError("published_token_unit must be million or billion")
    return number(published_usd, "published rate") * Decimal("1000000") / denominators[published_token_unit]


def token_cost(input_tokens, output_tokens, input_usd_per_million, output_usd_per_million):
    """Return an estimated dollar amount; unreported usage remains unknown."""
    if any(v is None for v in (input_tokens, output_tokens, input_usd_per_million, output_usd_per_million)):
        return None
    for label, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        parsed = number(value, label)
        if parsed != parsed.to_integral_value():
            raise ValueError(f"{label} must be an integer")
    return (
        number(input_tokens, "input_tokens") * number(input_usd_per_million, "input rate")
        + number(output_tokens, "output_tokens") * number(output_usd_per_million, "output rate")
    ) / Decimal("1000000")


def expense(item: dict, label: str):
    basis = item.get("basis")
    if basis not in BASES:
        raise ValueError(f"{label}: invalid basis")
    if not isinstance(item.get("note"), str) or not item["note"].strip():
        raise ValueError(f"{label}: provide an evidence or assumption note")
    value = item.get("usd")
    if basis == "unknown":
        if value is not None:
            raise ValueError(f"{label}: unknown expense must have usd=null")
        return None
    if value is None:
        raise ValueError(f"{label}: known expense needs usd")
    value = number(value, label)
    if basis == "not_used" and value != ZERO:
        raise ValueError(f"{label}: not_used must have usd=0")
    return value


def term(label: str, item: dict, multiplier=ONE):
    value = expense(item, label)
    if multiplier is None:
        value = None
    elif multiplier == ZERO:
        value = ZERO
    elif value is not None:
        value *= multiplier
    result = {"name": label, "usd": value, "basis": item["basis"], "note": item["note"]}
    for name in ("category", "outcome", "receipt", "usage_ref"):
        if name in item:
            result[name] = item[name]
    return result


def account(terms: list[dict]) -> dict:
    missing = [entry["name"] for entry in terms if entry["usd"] is None]
    known = sum((entry["usd"] for entry in terms if entry["usd"] is not None), ZERO)
    return {
        "terms": terms,
        "known_subtotal_usd": known,
        "unknown_terms": missing,
        "complete_total_usd": None if missing else known,
        "complete_means": "all modeled costs supplied; estimated and assumed inputs retain their provenance",
    }


def experiment_spend(data: dict) -> dict:
    """Actual experiment ledger, independent of projected query volume.

    Include failed/preflight/trial calls as separate items. The ledger must also
    include successful calls and unknown incurred costs before it is complete.
    """
    entries = data.get("experiment_spend")
    if not isinstance(entries, list) or not entries:
        return account([{
            "name": "experiment.not_recorded", "usd": None, "basis": "unknown",
            "note": "No experiment spend ledger was supplied.",
        }])
    return account([term(f"experiment.{i}.{entry['name']}", entry) for i, entry in enumerate(entries)])


def measured_usage(data: dict) -> list[dict]:
    """Validate token meters separately from any dollar estimate."""
    entries = data.get("measured_usage", [])
    if not isinstance(entries, list):
        raise ValueError("measured_usage must be a list")
    result = []
    token_fields = ("input_tokens", "output_tokens", "thinking_tokens", "cached_tokens")
    for index, entry in enumerate(entries):
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError(f"measured_usage.{index}: name is required")
        row = {"name": entry["name"]}
        for field in token_fields:
            value = entry.get(field)
            if value is None:
                row[field] = None
                continue
            parsed = number(value, f"measured_usage.{index}.{field}")
            if parsed != parsed.to_integral_value():
                raise ValueError(f"measured_usage.{index}.{field} must be an integer")
            row[field] = int(parsed)
        for field in ("requests", "outcome", "receipt", "note"):
            if field in entry:
                row[field] = entry[field]
        result.append(row)
    return result


def reservation_account(data: dict) -> dict:
    """Track reservation history while charging only retained ceilings to the cap."""
    entries = data.get("reservations", [])
    if not isinstance(entries, list):
        raise ValueError("reservations must be a list")
    terms = []
    status_totals = {status: ZERO for status in sorted(RESERVATION_STATES)}
    for index, entry in enumerate(entries):
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError(f"reservations.{index}: name is required")
        if not isinstance(entry.get("note"), str) or not entry["note"].strip():
            raise ValueError(f"reservations.{index}: note is required")
        status = entry.get("status")
        if status not in RESERVATION_STATES:
            raise ValueError(
                f"reservations.{index}: status must be retained, "
                "released_after_metering, or superseded"
            )
        value = number(entry.get("usd"), f"reservations.{index}.usd")
        row = {
            "name": f"reservation.{index}.{entry['name']}",
            "usd": value,
            "status": status,
            "counts_against_cap": status == "retained",
            "note": entry["note"],
        }
        for field in ("receipt", "receipt_field", "superseded_by"):
            if field in entry:
                row[field] = entry[field]
        terms.append(row)
        status_totals[status] += value
    total_recorded = sum(status_totals.values(), ZERO)
    total_retained = status_totals["retained"]
    return {
        "terms": terms,
        "status_totals_usd": status_totals,
        "total_recorded_usd": total_recorded,
        "total_retained_usd": total_retained,
    }


def experiment_report(data: dict) -> dict:
    """Separate measurements, estimates, unknowns, failures, and reservations."""
    spend = experiment_spend(data)
    unknown = account([
        term(f"unknown.{i}.{entry['name']}", entry)
        for i, entry in enumerate(data.get("unknown_costs", []))
    ])
    reservations = reservation_account(data)
    cap = number(data.get("cumulative_cap_usd"), "cumulative_cap_usd")
    committed = spend["known_subtotal_usd"] + reservations["total_retained_usd"]
    if committed > cap:
        raise ValueError("known list estimates plus reservations exceed cumulative cap")
    failures = [
        entry for entry in spend["terms"] + unknown["terms"]
        if entry.get("outcome") in {"failed", "aborted", "incompatible"}
    ]
    return {
        "measured_usage": measured_usage(data),
        "list_rate_estimates": spend,
        "unknown_costs": unknown,
        "failures": failures,
        "reservations": reservations,
        "cumulative_cap_usd": cap,
        "known_estimates_plus_retained_usd": committed,
        "remaining_cap_usd": cap - committed,
        "cap_warning": (
            "Reservations are conservative guardrails, not billed or estimated spend; "
            "only retained reservations count against current headroom."
        ),
    }


def scenario(data: dict, queries: int) -> dict:
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if isinstance(queries, bool) or not isinstance(queries, int) or queries <= 0:
        raise ValueError("queries must be a positive integer")
    q = Decimal(queries)
    indexing = data["indexing"]
    if not isinstance(indexing, list) or not indexing:
        raise ValueError("indexing must list at least one cost or explicit unknown")
    indexed_terms = [term(f"indexing.{i}.{entry['name']}", entry) for i, entry in enumerate(indexing)]
    for name in ("retrieval", "judge", "answer"):
        indexed_terms.append(term(f"query.{name}", data["query"][name], q))
    fallback = data["query"]["stronger_fallback"]
    frequency = fallback.get("frequency")
    if frequency is not None:
        frequency = fraction(frequency, "stronger_fallback.frequency")
    indexed_terms.append(term("query.stronger_fallback", fallback["per_invocation"], None if frequency is None else q * frequency))
    for name in ("hosting", "storage"):
        indexed_terms.append(term(f"period.{name}", data["period"][name]))

    baseline = data["baseline"]
    cache = baseline["cache"]
    if not isinstance(cache.get("policy"), str) or not cache["policy"].strip():
        raise ValueError("cache.policy must explain disabled or configured retention/reuse policy")
    hit_rate = cache.get("hit_rate")
    if hit_rate is not None:
        hit_rate = fraction(hit_rate, "cache.hit_rate")
    if cache["policy"] == "disabled" and hit_rate != ZERO:
        raise ValueError("disabled cache requires hit_rate=0")
    uncached = account([term("baseline.uncached_queries", baseline["uncached_per_query"], q)])
    cached_terms = [
        term("baseline.cache_misses", baseline["uncached_per_query"], None if hit_rate is None else q * (ONE - hit_rate)),
        term("baseline.cache_hits", cache["hit_per_query"], None if hit_rate is None else q * hit_rate),
        term("baseline.cache_writes", cache["write_period"]),
        term("baseline.cache_storage", cache["storage_period"]),
    ]
    cached = account(cached_terms)
    indexed = account(indexed_terms)

    def ratio(candidate):
        # Suppress comparisons unless both sides have complete modeled totals.
        total = candidate["complete_total_usd"]
        denominator = indexed["complete_total_usd"]
        if total is None or denominator is None or denominator <= ZERO:
            return None
        return total / denominator

    return {
        "queries": queries,
        "indexing_charged_once": True,
        "indexed": indexed,
        "uncached_baseline": uncached,
        "cache_policy_baseline": cached,
        "cache_policy": cache["policy"],
        "cache_hit_rate": hit_rate,
        "uncached_to_indexed_before_unknown_costs_ratio": ratio(uncached),
        "cache_to_indexed_before_unknown_costs_ratio": ratio(cached),
        "warning": "Ratios are emitted only for complete modeled totals and still inherit estimates or assumptions. They do not establish quality parity or library-scale behavior.",
    }


def jsonable(value):
    """Use decimal strings to preserve exact cost arithmetic in JSON output."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path(__file__).with_name("scenario.receipts.json"))
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 100, 1000])
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    report = {
        "experiment": data["experiment"],
        "experiment_accounting": experiment_report(data),
        "scenarios": [scenario(data, q) for q in args.queries],
    }
    print(json.dumps(jsonable(report), indent=2))


if __name__ == "__main__":
    main()
