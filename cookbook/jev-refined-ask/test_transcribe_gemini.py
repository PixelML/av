"""Offline invariants for the public Gemini transcript helper."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "transcribe_gemini", HERE / "transcribe_gemini.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TranscriptRecipeTests(unittest.TestCase):
    def test_sub_second_source_is_one_fractional_window(self):
        self.assertEqual(MODULE.windows(0.25), [(0, 0.25)])

    def test_short_tail_is_folded_into_previous_window(self):
        self.assertEqual(MODULE.windows(60.5), [(0, 60.5)])
        self.assertEqual(MODULE.windows(61), [(0, 60), (60, 61)])

    def test_manifest_binds_nonempty_directory_to_exact_source_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            manifest = {"source_sha256": "abc", "model": "example", "prompt": "v1"}
            MODULE.bind_directory(directory, manifest)
            MODULE.bind_directory(directory, manifest)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                MODULE.bind_directory(directory, {**manifest, "prompt": "v2"})

    def test_prior_and_persisted_reservations_survive_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "usage.jsonl"
            first = MODULE.Ledger(path, "5", "0.90")
            attempt = first.reserve({"stage": "asr"}, "0.20")
            first.record({**attempt, "status": "failed"})
            # External reservations are supplied cumulatively on every resume;
            # this directory contributes its persisted 0.20 reservation.
            resumed = MODULE.Ledger(path, "5", "0.91")
            self.assertEqual(str(resumed.reserved), "1.11")
            self.assertEqual(str(resumed.remaining), "3.89")

    def test_response_cache_is_bound_and_replayable_without_network(self):
        binding = {
            "manifest_sha256": "manifest",
            "source_sha256": "source",
            "model": "model",
            "chunk": 0,
            "start_sec": 0,
            "end_sec": 0.5,
        }
        response = {
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": json.dumps({
                    "text": "hello", "silence": False
                })}]},
            }],
        }
        artifact = MODULE.response_artifact(response, binding, 0.1)
        self.assertEqual(MODULE.parse_response_artifact(artifact, binding), ("hello", False))
        with self.assertRaisesRegex(ValueError, "cached response mismatch"):
            MODULE.parse_response_artifact(artifact, {**binding, "model": "other"})


if __name__ == "__main__":
    unittest.main()
