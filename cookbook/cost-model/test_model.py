"""Arithmetic and incomplete-accounting invariants; no provider calls."""
from __future__ import annotations

import json
import unittest
from decimal import Decimal
from pathlib import Path

from build_nb import build_notebook
from model import experiment_report, experiment_spend, scenario, token_cost

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE.parent / "receipts"


def cost(value, basis="assumed"):
    return {"usd": value, "basis": basis, "note": "Synthetic arithmetic test input, not a benchmark receipt."}


def fixture():
    return {
        "schema_version": 1,
        "indexing": [{"name": "synthetic", **cost("2")}],
        "query": {
            "retrieval": cost("0.01"),
            "judge": cost("0.02"),
            "answer": cost("0.03"),
            "stronger_fallback": {"frequency": 0, "per_invocation": cost(0, "not_used")},
        },
        "period": {"hosting": cost("3"), "storage": cost("4")},
        "baseline": {
            "uncached_per_query": cost("1"),
            "cache": {
                "policy": "disabled", "hit_rate": 0,
                "hit_per_query": cost(None, "unknown"),
                "write_period": cost(0, "not_used"),
                "storage_period": cost(0, "not_used"),
            },
        },
    }


class CostAccountingTests(unittest.TestCase):
    def test_one_time_index_is_not_multiplied_by_query_count(self):
        # 2 indexing + 10*(.01+.02+.03) queries + 3 hosting + 4 storage
        self.assertEqual(scenario(fixture(), 10)["indexed"]["complete_total_usd"], Decimal("9.60"))

    def test_token_cost_is_exact_and_missing_usage_stays_unknown(self):
        self.assertEqual(token_cost(1000, 100, "2", "5"), Decimal("0.0025"))
        self.assertIsNone(token_cost(None, 100, "2", "5"))
        with self.assertRaises(ValueError):
            token_cost(1.5, 0, 1, 1)

    def test_unknown_infrastructure_keeps_total_and_ratio_unknown(self):
        data = fixture()
        data["query"]["retrieval"] = cost(None, "unknown")
        data["period"] = {"hosting": cost(None, "unknown"), "storage": cost(None, "unknown")}
        result = scenario(data, 10)
        # Known costs: 2 indexing + 10*(.02 judge + .03 answer) = 2.50
        self.assertEqual(result["indexed"]["known_subtotal_usd"], Decimal("2.50"))
        self.assertIsNone(result["indexed"]["complete_total_usd"])
        self.assertIsNone(result["uncached_to_indexed_before_unknown_costs_ratio"])
        self.assertEqual(result["indexed"]["unknown_terms"], ["query.retrieval", "period.hosting", "period.storage"])

    def test_fallback_unknown_price_is_unknown_only_when_it_can_run(self):
        data = fixture()
        data["query"]["stronger_fallback"]["per_invocation"] = cost(None, "unknown")
        self.assertEqual(scenario(data, 10)["indexed"]["complete_total_usd"], Decimal("9.60"))
        data["query"]["stronger_fallback"]["frequency"] = "0.2"
        self.assertIsNone(scenario(data, 10)["indexed"]["complete_total_usd"])
        data["query"]["stronger_fallback"]["per_invocation"] = cost("0.5")
        self.assertEqual(scenario(data, 10)["indexed"]["complete_total_usd"], Decimal("10.60"))

    def test_cache_hit_rate_combines_hits_misses_writes_and_storage(self):
        data = fixture()
        data["baseline"]["cache"] = {
            "policy": "Synthetic TTL=1h, one reusable prefix, one write in period",
            "hit_rate": "0.75", "hit_per_query": cost("0.10"),
            "write_period": cost("2"), "storage_period": cost("3"),
        }
        # 10*(.25*1 + .75*.1) + 2 write + 3 storage
        self.assertEqual(scenario(data, 10)["cache_policy_baseline"]["complete_total_usd"], Decimal("8.250"))
        data["baseline"]["cache"]["hit_rate"] = None
        result = scenario(data, 10)
        self.assertIsNone(result["cache_policy_baseline"]["complete_total_usd"])
        self.assertIsNone(result["cache_to_indexed_before_unknown_costs_ratio"])

    def test_disabled_cache_matches_uncached_requests(self):
        result = scenario(fixture(), 10)
        self.assertEqual(result["cache_policy_baseline"]["complete_total_usd"], result["uncached_baseline"]["complete_total_usd"])

    def test_pending_receipt_cannot_emit_total_or_ratio(self):
        data = json.loads((HERE / "scenario.pending.json").read_text())
        result = scenario(data, 100)
        self.assertIsNone(result["indexed"]["complete_total_usd"])
        self.assertIsNone(result["uncached_to_indexed_before_unknown_costs_ratio"])
        self.assertIn("query.judge", result["indexed"]["unknown_terms"])

    def test_receipt_ledger_preserves_cap_and_separates_unknowns(self):
        data = json.loads((HERE / "scenario.receipts.json").read_text())
        report = experiment_report(data)
        self.assertEqual(report["list_rate_estimates"]["known_subtotal_usd"],
                         Decimal("0.3933575"))
        self.assertEqual(report["reservations"]["total_recorded_usd"], Decimal("3.97945825"))
        self.assertEqual(report["reservations"]["total_retained_usd"], Decimal("1.10"))
        self.assertEqual(
            report["reservations"]["status_totals_usd"]["released_after_metering"],
            Decimal("2.87945825"),
        )
        self.assertEqual(report["known_estimates_plus_retained_usd"], Decimal("1.4933575"))
        self.assertEqual(report["remaining_cap_usd"], Decimal("3.5066425"))
        self.assertIsNone(report["unknown_costs"]["complete_total_usd"])
        self.assertEqual(
            [item["outcome"] for item in report["failures"]],
            ["failed", "aborted", "incompatible"],
        )
        self.assertEqual(report["measured_usage"][0]["input_tokens"], 118228)

    def test_sanitized_receipts_are_present_and_baseline_estimate_recomputes(self):
        required = {
            "asr.json",
            "gemini38-baseline.json",
            "caption-aborted.json",
            "caption-smoke.json",
            "cap-probe-32-incompatible.json",
            "cap-probe-32-direct-incompatible.json",
        }
        self.assertTrue(required.issubset({path.name for path in RECEIPTS.glob("*.json")}))
        receipt = json.loads((RECEIPTS / "gemini38-baseline.json").read_text())
        usage = receipt["usage"]
        recomputed = token_cost(
            usage["input_tokens"],
            usage["output_tokens"] + usage["thinking_tokens"],
            "0.75",
            "3.75",
        )
        self.assertEqual(recomputed, Decimal("0.308076"))
        self.assertEqual(
            Decimal(str(receipt["cost"]["full_rate_list_estimate_usd"])),
            recomputed,
        )
        self.assertFalse(receipt["limitations"]["paired_av_grok_jev_run_completed"])

    def test_every_receipt_reservation_is_reconciled_with_an_explicit_state(self):
        data = json.loads((HERE / "scenario.receipts.json").read_text())
        reconciled = {
            (entry.get("receipt"), entry.get("receipt_field")): entry
            for entry in data["reservations"]
            if entry.get("receipt_field")
        }
        expected = {
            ("../receipts/caption-smoke.json", "reserved_upstream_list_usd"),
            ("../receipts/gemini38-baseline.json", "cost.worst_case_reserved_list_estimate_usd"),
        }
        for key in expected:
            with self.subTest(receipt=key[0], field=key[1]):
                self.assertIn(key, reconciled)
                receipt = json.loads((HERE / key[0]).resolve().read_text())
                receipt_value = receipt
                for field in key[1].split("."):
                    receipt_value = receipt_value[field]
                self.assertEqual(Decimal(reconciled[key]["usd"]), Decimal(str(receipt_value)))
                self.assertIn(
                    reconciled[key]["status"],
                    {"retained", "released_after_metering", "superseded"},
                )

    def test_only_retained_reservations_count_against_headroom(self):
        data = fixture()
        data["reservations"] = [
            {"name": "active", "usd": "1.00", "status": "retained", "note": "active"},
            {
                "name": "metered",
                "usd": "2.00",
                "status": "released_after_metering",
                "note": "metered",
            },
            {"name": "old", "usd": "3.00", "status": "superseded", "note": "replaced"},
        ]
        data["cumulative_cap_usd"] = "10"
        report = experiment_report(data)
        self.assertEqual(report["reservations"]["total_recorded_usd"], Decimal("6.00"))
        self.assertEqual(report["reservations"]["total_retained_usd"], Decimal("1.00"))
        self.assertEqual(report["known_estimates_plus_retained_usd"], Decimal("1.00"))

        data["reservations"][0]["status"] = "retained_in_cumulative_plan"
        with self.assertRaisesRegex(ValueError, "status must be"):
            experiment_report(data)

    def test_incomplete_av_side_suppresses_baseline_ratio(self):
        data = json.loads((HERE / "scenario.receipts.json").read_text())
        result = scenario(data, 1)
        self.assertEqual(result["uncached_baseline"]["complete_total_usd"],
                         Decimal("0.308076"))
        self.assertIsNone(result["indexed"]["complete_total_usd"])
        self.assertIsNone(result["uncached_to_indexed_before_unknown_costs_ratio"])

    def test_cap_rejects_estimates_plus_reservations_above_limit(self):
        data = json.loads((HERE / "scenario.receipts.json").read_text())
        data["cumulative_cap_usd"] = "1.4933574"
        with self.assertRaisesRegex(ValueError, "exceed cumulative cap"):
            experiment_report(data)

    def test_invalid_costs_and_frequencies_fail(self):
        for value in (-1, "NaN", "Infinity", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                data = fixture()
                data["query"]["answer"] = cost(value)
                scenario(data, 1)
        for rate in (-0.1, 1.1, True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                data = fixture()
                data["query"]["stronger_fallback"]["frequency"] = rate
                scenario(data, 1)
        data = fixture()
        data["query"]["answer"] = cost(0, "unknown")
        with self.assertRaises(ValueError):
            scenario(data, 1)

    def test_trial_spend_is_visible_without_changing_repeatable_projection(self):
        data = fixture()
        data["experiment_spend"] = [
            {"name": "successful_calls", **cost("2")},
            {"name": "failed_trial_calls", **cost("0.5")},
        ]
        self.assertEqual(experiment_spend(data)["complete_total_usd"], Decimal("2.5"))
        self.assertEqual(scenario(data, 10)["indexed"]["complete_total_usd"], Decimal("9.60"))
        data["experiment_spend"].append({"name": "unmetered_attempt", **cost(None, "unknown")})
        self.assertIsNone(experiment_spend(data)["complete_total_usd"])
        self.assertEqual(experiment_spend(data)["known_subtotal_usd"], Decimal("2.5"))
        self.assertIsNone(experiment_spend(fixture())["complete_total_usd"])

    def test_notebook_source_and_executed_outputs_match_generator(self):
        checked_in = json.loads((HERE / "notebook.ipynb").read_text())
        self.assertEqual(checked_in, build_notebook())


if __name__ == "__main__":
    unittest.main()
