"""RAG Q&A with citations over video artifacts."""

from __future__ import annotations

from av.core.config import AVConfig
from av.db.repository import Repository, _fmt_timestamp
from av.providers.openai import OpenAILLM
from av.search.inspection import inspect_with_stronger_vision
from av.search.refine import (
    RefinementError,
    SystemOneClient,
    judge_answer_support,
    refine_search_results,
)
from av.search.semantic import search
from av.search.usage import merge_usage, new_usage, record_usage


def _citations(results: list[dict]) -> list[dict]:
    citations = []
    provenance_keys = (
        "artifact_id",
        "chunk_start_sec",
        "chunk_end_sec",
        "scene_confidence",
        "merged_artifact_ids",
        "relevance_p",
        "evidence_scope",
    )
    for result in results:
        citation = {
            "video_id": result.get("video_id", ""),
            "start_sec": result.get("timestamp_sec", 0),
            "end_sec": result.get("end_sec"),
            "source_type": result.get("source_type", ""),
            "text": result.get("text", ""),
            "score": result.get("score", 0),
        }
        for key in provenance_keys:
            if key in result:
                citation[key] = result.get(key)
        citations.append(citation)
    return citations


def _context(results: list[dict]) -> str:
    parts: list[str] = []
    for result in results:
        start = result.get("timestamp_formatted", "")
        end_sec = result.get("end_sec")
        end = _fmt_timestamp(float(end_sec)) if isinstance(end_sec, (int, float)) else start
        parts.append(
            f"[{result.get('filename', '')} @ {start}-{end} ({result.get('source_type', '')})] "
            f"{result.get('text', '')}"
        )
    return "\n\n".join(parts)


def _heuristic_confidence(results: list[dict]) -> float:
    top_score = results[0].get("score", 0) if results else 0
    return min(round(float(top_score), 2), 1.0) if top_score else 0.5


def _ask_settings(config: AVConfig) -> dict:
    return {
        "chat_model": config.chat_model,
        "chat_max_output_tokens": config.chat_max_output_tokens,
    }


def _llm_usage_snapshot(llm: OpenAILLM | None) -> dict:
    usage = getattr(llm, "usage", None)
    snapshot = getattr(usage, "snapshot", None)
    if callable(snapshot):
        receipt = snapshot()
        if isinstance(receipt, dict):
            return receipt
    return new_usage()


def _legacy_ask(
    question: str,
    results: list[dict],
    config: AVConfig,
    embedding_usage: dict | None,
) -> dict:
    warnings: list[str] = []
    stage_usage = {
        "embedding": embedding_usage,
        "answer": new_usage(),
    }
    if not results:
        return {
            "answer": "No relevant content found in the indexed videos.",
            "citations": [],
            "confidence": 0.0,
            "confidence_basis": "no_evidence",
            "route": "legacy_no_results",
            "evidence_status": "no_retrieval_hits",
            "warnings": warnings,
            "stage_usage": stage_usage,
            "ask_settings": _ask_settings(config),
        }
    llm: OpenAILLM | None = None
    try:
        llm = OpenAILLM(config)
        completion = llm.complete_with_usage(question, _context(results))
        stage_usage["answer"] = _usage_from_completion(completion)
    except Exception:
        stage_usage["answer"] = _llm_usage_snapshot(llm)
        warnings.append("Answer generation was unavailable; no answer was produced.")
        return {
            "answer": "Answer generation was unavailable. Retrieved video moments are included as citations.",
            "citations": _citations(results),
            "confidence": 0.0,
            "confidence_basis": "unknown",
            "route": "legacy_answer_failed",
            "evidence_status": "answer_unavailable",
            "warnings": warnings,
            "stage_usage": stage_usage,
            "ask_settings": _ask_settings(config),
        }
    return {
        "answer": completion.text,
        "citations": _citations(results),
        "confidence": _heuristic_confidence(results),
        "confidence_basis": "retrieval_heuristic",
        "route": "legacy",
        "evidence_status": "raw_unjudged",
        "warnings": warnings,
        "stage_usage": stage_usage,
        "ask_settings": _ask_settings(config),
    }


def _usage_from_completion(completion) -> dict:
    if isinstance(getattr(completion, "usage", None), dict):
        return dict(completion.usage)
    usage = new_usage()
    record_usage(
        usage,
        {
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
        },
    )
    return usage


def _add_stage_usage(target: dict, source: dict) -> None:
    merge_usage(target, source)


def ask(
    question: str,
    repo: Repository,
    config: AVConfig,
    *,
    video_id: str | None = None,
    top_k: int = 10,
    refine: bool = True,
) -> dict:
    """Answer a question using RAG over video artifacts."""
    # Step 1: Retrieve relevant context
    search_result = search(
        question, repo, config, limit=top_k, video_id=video_id
    )

    raw_results = search_result.get("results", [])
    if not refine or not config.refine_enabled or not config.typesafe_api_key:
        return _legacy_ask(
            question,
            raw_results,
            config,
            search_result.get("embedding_usage"),
        )

    warnings: list[str] = []
    stage_usage: dict[str, dict | None] = {
        "relevance": new_usage(),
        "boundary": new_usage(),
        "answer": new_usage(),
        "support": new_usage(),
        "vision": new_usage(),
        "embedding": search_result.get("embedding_usage"),
    }
    if not raw_results:
        return {
            "answer": "No relevant content found in the indexed videos.",
            "citations": [],
            "confidence": 0.0,
            "confidence_basis": "no_evidence",
            "route": "refined_no_results",
            "evidence_status": "no_retrieval_hits",
            "refinement": {"status": "no_retrieval_hits", "raw_count": 0, "scene_count": 0},
            "warnings": warnings,
            "inspected_windows": [],
            "stage_usage": stage_usage,
            "ask_settings": _ask_settings(config),
        }

    client = SystemOneClient(config)
    try:
        results, refinement, refinement_usage = refine_search_results(
            question, raw_results, repo, config, client=client
        )
        stage_usage.update(refinement_usage)
    except RefinementError as exc:
        for stage, usage in exc.stage_usage.items():
            stage_usage[stage] = usage
        warnings.append("Jev refinement was unavailable; answering from raw retrieval without judged evidence confidence.")
        llm: OpenAILLM | None = None
        try:
            llm = OpenAILLM(config)
            completion = llm.complete_with_usage(question, _context(raw_results))
            stage_usage["answer"] = _usage_from_completion(completion)
        except Exception:
            stage_usage["answer"] = _llm_usage_snapshot(llm)
            warnings.append("Answer generation was unavailable; no answer was produced.")
            return {
                "answer": "Answer generation was unavailable. Retrieved video moments are included as citations.",
                "citations": _citations(raw_results),
                "confidence": 0.0,
                "confidence_basis": "unknown",
                "route": "refinement_fallback_answer_failed",
                "evidence_status": "answer_unavailable",
                "refinement": {"status": "provider_fallback", "raw_count": len(raw_results)},
                "warnings": warnings,
                "inspected_windows": [],
                "stage_usage": stage_usage,
                "ask_settings": _ask_settings(config),
            }
        return {
            "answer": completion.text,
            "citations": _citations(raw_results),
            "confidence": _heuristic_confidence(raw_results),
            "confidence_basis": "retrieval_heuristic",
            "route": "refinement_fallback",
            "evidence_status": "raw_unjudged",
            "refinement": {"status": "provider_fallback", "raw_count": len(raw_results)},
            "warnings": warnings,
            "inspected_windows": [],
            "stage_usage": stage_usage,
            "ask_settings": _ask_settings(config),
        }

    if not results:
        return {
            "answer": "No supported evidence was found for this question in the retrieved video moments.",
            "citations": [],
            "confidence": 0.0,
            "confidence_basis": "jev_relevance",
            "route": "refined_no_results",
            "evidence_status": "all_sources_irrelevant",
            "refinement": refinement,
            "warnings": warnings,
            "inspected_windows": [],
            "stage_usage": stage_usage,
            "ask_settings": _ask_settings(config),
        }

    llm = None
    try:
        llm = OpenAILLM(config)
        completion = llm.complete_with_usage(question, _context(results))
        stage_usage["answer"] = _usage_from_completion(completion)
    except Exception:
        stage_usage["answer"] = _llm_usage_snapshot(llm)
        warnings.append("Answer generation was unavailable; no answer was produced.")
        return {
            "answer": "Answer generation was unavailable. Relevant video moments are included as citations.",
            "citations": _citations(results),
            "confidence": 0.0,
            "confidence_basis": "unknown",
            "route": "refined_answer_failed",
            "evidence_status": "answer_unavailable",
            "refinement": refinement,
            "warnings": warnings,
            "inspected_windows": [],
            "stage_usage": stage_usage,
            "ask_settings": _ask_settings(config),
        }
    answer = completion.text
    citations = _citations(results)
    support_probability: float | None = None
    support_failed = False
    try:
        support_probability, support_usage = judge_answer_support(client, question, answer, results)
        stage_usage["support"] = support_usage
    except RefinementError as exc:
        if "support" in exc.stage_usage:
            stage_usage["support"] = exc.stage_usage["support"]
        support_failed = True
        warnings.append("Answer support judgment was unavailable; the answer is not marked supported.")

    if support_probability is not None and support_probability >= config.refine_support_min:
        return {
            "answer": answer,
            "citations": citations,
            "confidence": support_probability,
            "confidence_basis": "jev_answer_support",
            "route": "refined",
            "evidence_status": "supported",
            "refinement": refinement,
            "warnings": warnings,
            "inspected_windows": [],
            "stage_usage": stage_usage,
            "ask_settings": _ask_settings(config),
        }

    inspection = inspect_with_stronger_vision(question, answer, results, repo, config)
    stage_usage["vision"] = inspection["usage"]
    warnings.extend(inspection["warnings"])
    if inspection["status"] == "supported":
        inspected_answer = inspection["answer"]
        inspected_citations = inspection["citations"]
        try:
            inspected_support, inspected_usage = judge_answer_support(
                client,
                question,
                inspected_answer,
                [
                    {
                        "video_id": citation["video_id"],
                        "timestamp_sec": citation["start_sec"],
                        "end_sec": citation["end_sec"],
                        "source_type": citation["source_type"],
                        "text": citation["text"],
                    }
                    for citation in inspected_citations
                ],
            )
            _add_stage_usage(stage_usage["support"], inspected_usage)
        except RefinementError as exc:
            if "support" in exc.stage_usage:
                _add_stage_usage(stage_usage["support"], exc.stage_usage["support"])
            inspected_support = None
            warnings.append("Inspected evidence could not be independently support-judged.")
        if inspected_support is not None and inspected_support >= config.refine_support_min:
            return {
                "answer": inspected_answer,
                "citations": inspected_citations,
                "confidence": inspected_support,
                "confidence_basis": "jev_answer_support_after_sampled_frames",
                "route": "vision_inspected",
                "evidence_status": "sampled_frames_supported",
                "refinement": refinement,
                "warnings": warnings,
                "inspected_windows": inspection["windows"],
                "stage_usage": stage_usage,
                "ask_settings": _ask_settings(config),
            }

    if support_failed:
        uncertain = f"Unverified: {answer}" if answer else "The answer could not be verified from the available evidence."
        evidence_status = "support_unknown"
    else:
        uncertain = "The retrieved evidence was relevant, but it did not support a reliable answer."
        evidence_status = "unsupported"
    return {
        "answer": uncertain,
        "citations": citations,
        "confidence": support_probability or 0.0,
        "confidence_basis": "jev_answer_support" if support_probability is not None else "unknown",
        "route": "refined_uncertain",
        "evidence_status": evidence_status,
        "refinement": refinement,
        "warnings": warnings,
        "inspected_windows": inspection["windows"],
        "stage_usage": stage_usage,
        "ask_settings": _ask_settings(config),
    }
