"""djev-spark decision client: Jev-compatible structured reads on self-hosted
DiffusionGemma, speaking the same documented ``/v1/systemone`` contract as
:class:`~av.search.refine.SystemOneClient`.

Upstream audited at ``mmastrac/djev-spark`` commit
``1444f3e927f83ba508e5b28a4fd4fdd9ecd0976b``. Properties of that server that
shape this adapter:

* A request's ``model`` field is ignored. Every 200 response carries the model
  the server actually served plus the engine in ``diagnostics`` — this client
  requires both and records them into usage receipts, so a djev answer is
  never presented as a Jev measurement and never carries an unknown runtime.
* An exported ``score`` is the zero-based expected level
  ``sum(i * p)`` over legend indices, in ``[0, len(levels) - 1]``
  (``decide_group`` sums one-based internally; ``jev_answer`` re-exports it
  zero-based from the same probabilities).
* A question whose ``ask_if`` condition failed is answered ``null``; that is
  the only way an asked question can be null. Mandatory decisions may not be
  skipped.
* The server may skip a question whose ``ask_if`` condition failed and report
  it as ``null``. Skipped answers are preserved; consumers reject them where a
  decision is mandatory.

No default endpoint ships with av: djev-spark is self-hosted, its upstream
deployment binds all interfaces by default, and pointing at a machine by
default would couple the public package to somebody's private box.
"""

from __future__ import annotations

import math
import time
from typing import Any

import requests

from av.bench.receipts import redact_endpoint
from av.core.config import AVConfig
from av.search.refine import _RETRYABLE_STATUS, RefinementError, SystemOneClient

# Sampled probabilities are means of per-read label distributions, so they sum
# to one only up to sampling and float noise; anything further off is a
# malformed response, not a calibration quirk.
_PROB_SUM_TOLERANCE = 0.05
_SCORE_RANGE_SLACK = 0.01
_SCORE_CONSISTENCY_TOLERANCE = 0.01


def _finite_unit_interval(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


class DjevClient:
    """Same ``ask`` protocol as SystemOneClient against a djev-spark server."""

    provider_label = "djev-spark"

    def __init__(self, config: AVConfig, *, session: requests.Session | None = None) -> None:
        if not config.djev_endpoint:
            raise RefinementError(
                "djev-spark endpoint is not configured; set AV_DJEV_ENDPOINT "
                "(self-hosted only — av ships no default endpoint)"
            )
        self.endpoint = config.djev_endpoint
        self.api_key = config.djev_api_key
        self.model = config.djev_model  # advisory: the server ignores it
        self.seed = config.djev_seed
        self.timeout = config.djev_timeout_sec
        self.max_retries = config.djev_max_retries
        self.session = session or requests.Session()
        self.served_model: str | None = None
        self.server_engine: str | None = None
        self.served_endpoint_host: str | None = redact_endpoint(self.endpoint)

    def ask(self, state: Any, questions: dict[str, dict]) -> tuple[dict[str, dict], dict]:
        payload: dict[str, Any] = {"state": state, "questions": questions, "seed": self.seed}
        if self.model:
            payload["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = "request failed"
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts += 1
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = type(exc).__name__
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                raise RefinementError(
                    f"djev-spark unavailable ({last_error})",
                    attempts=attempts,
                ) from exc
            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                time.sleep(min(0.25 * (2**attempt), 1.0))
                continue
            if not response.ok:
                raise RefinementError(
                    f"djev-spark request failed with HTTP {response.status_code}",
                    attempts=attempts,
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise RefinementError(
                    "djev-spark returned invalid JSON",
                    attempts=attempts,
                ) from exc
            if not isinstance(data, dict):
                raise RefinementError(
                    "djev-spark returned an invalid JSON document",
                    attempts=attempts,
                )
            raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            answers = data.get("answers")
            if not isinstance(answers, dict):
                raise RefinementError(
                    "djev-spark response is missing answers",
                    attempts=attempts,
                    raw_usage=raw_usage,
                )
            self._capture_identity(data)
            if self._missing_identity():
                raise RefinementError(
                    "djev-spark response did not identify the served runtime: "
                    "the pinned contract requires the served model and engine "
                    "on every success",
                    attempts=attempts,
                    raw_usage=raw_usage,
                )
            self._validate_answers(questions, answers, attempts=attempts, raw_usage=raw_usage)
            usage = {
                **raw_usage,
                "_attempts": attempts,
                "served_model": self.served_model,
                "server_engine": self.server_engine,
                "served_endpoint_host": self.served_endpoint_host,
            }
            return answers, usage
        raise RefinementError(f"djev-spark unavailable ({last_error})", attempts=attempts)

    def _missing_identity(self) -> bool:
        return not self.served_model or not self.server_engine

    def _capture_identity(self, data: dict) -> None:
        """Record the identity the server reported for itself, never the
        request's. Reset per response: an identity from an earlier call must
        not vouch for this one."""
        model = data.get("model")
        self.served_model = model if isinstance(model, str) and model else None
        diagnostics = data.get("diagnostics")
        engine = diagnostics.get("engine") if isinstance(diagnostics, dict) else None
        self.server_engine = engine if isinstance(engine, str) and engine else None

    def _validate_answers(
        self,
        questions: dict[str, dict],
        answers: dict[str, Any],
        *,
        attempts: int,
        raw_usage: dict,
    ) -> None:
        """Visibly reject responses that are malformed or incomplete for the
        questions actually asked. ``null`` is accepted only for questions that
        declare an ``ask_if`` condition — the only skip path in the pinned
        contract; a mandatory decision reported as ``null`` is rejected."""
        problems: list[str] = []
        for key in answers:
            if key not in questions:
                problems.append(f"{key}: answer for a question that was not asked")
        for qid, question in questions.items():
            if qid not in answers:
                problems.append(f"{qid}: missing answer")
                continue
            answer = answers[qid]
            if answer is None:
                if not question.get("ask_if"):
                    problems.append(f"{qid}: mandatory question was skipped")
                continue
            kind = question.get("type")
            if not isinstance(answer, dict) or answer.get("type") != kind:
                problems.append(f"{qid}: expected a {kind} answer")
                continue
            if "confidence" in answer and not _finite_unit_interval(answer["confidence"]):
                problems.append(f"{qid}: confidence out of range")
            if kind == "noul":
                if not _finite_unit_interval(answer.get("noul")):
                    problems.append(f"{qid}: noul probability out of range")
            elif kind == "choice":
                self._validate_choice(qid, question, answer, problems)
            elif kind == "score":
                self._validate_score(qid, question, answer, problems)
        if problems:
            raise RefinementError(
                "djev-spark response failed validation: " + "; ".join(problems),
                attempts=attempts,
                raw_usage=raw_usage,
            )

    def _validate_choice(
        self, qid: str, question: dict, answer: dict, problems: list[str]
    ) -> None:
        criteria = question.get("criteria")
        names = set(criteria) if isinstance(criteria, dict) else set()
        # A set membership test raises TypeError on an unhashable value (a
        # JSON array or object), so require a string before comparing.
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in names:
            problems.append(f"{qid}: choice outside the offered options")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != names:
            problems.append(f"{qid}: choice probabilities do not match the offered options")
            return
        self._validate_distribution(qid, probabilities, problems)

    def _validate_score(
        self, qid: str, question: dict, answer: dict, problems: list[str]
    ) -> None:
        levels = question.get("criteria")
        if not isinstance(levels, list) or not levels:
            problems.append(f"{qid}: score question without levels")
            return
        expected_legend = {str(index): level for index, level in enumerate(levels)}
        if answer.get("legend") != expected_legend:
            problems.append(f"{qid}: score legend does not match the offered levels")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(expected_legend):
            problems.append(f"{qid}: score probabilities do not match the legend")
            return
        self._validate_distribution(qid, probabilities, problems)
        score = answer.get("score")
        # The contract exports the zero-based expected level sum(i * p) over
        # legend indices, so a three-level question scores in [0, 2].
        highest = float(len(levels) - 1)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not -_SCORE_RANGE_SLACK <= float(score) <= highest + _SCORE_RANGE_SLACK
        ):
            problems.append(f"{qid}: score outside the legend range")
            return
        if all(_finite_unit_interval(p) for p in probabilities.values()):
            expected = sum(int(key) * float(p) for key, p in probabilities.items())
            if abs(float(score) - expected) > _SCORE_CONSISTENCY_TOLERANCE:
                problems.append(f"{qid}: score inconsistent with the legend probabilities")

    def _validate_distribution(self, qid: str, probabilities: dict, problems: list[str]) -> None:
        total = 0.0
        for label, probability in probabilities.items():
            if not _finite_unit_interval(probability):
                problems.append(f"{qid}: probability for {label!r} out of range")
                return
            total += float(probability)
        if abs(total - 1.0) > _PROB_SUM_TOLERANCE:
            problems.append(f"{qid}: probabilities sum to {total:.3f}")


def decision_provider_name(config: AVConfig) -> str | None:
    """The decision provider this configuration selects, or ``None`` when no
    decision lane is configured and refinement must fall back to retrieval."""
    if config.djev_endpoint:
        return "djev-spark"
    if config.typesafe_api_key:
        return "jev"
    return None


def open_decision_client(
    config: AVConfig, *, session: requests.Session | None = None
) -> SystemOneClient | DjevClient:
    """Construct the configured decision client.

    An explicitly configured djev endpoint selects the self-hosted lane over
    hosted System One; both speak the same ``/v1/systemone`` contract through
    the same ``ask`` protocol.
    """
    if config.djev_endpoint:
        return DjevClient(config, session=session)
    return SystemOneClient(config, session=session)
