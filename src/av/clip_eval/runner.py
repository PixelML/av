"""Run the deterministic-vs-Jev clip comparison on identical candidates.

Both arms call the same ``clip_video`` engine over the same frozen corpus and
index. The deterministic arm ranks retrieval candidates without any provider;
the Jev arm runs the typed decision stages against a supplied client (the
labeled mock offline, or a real System One client under an explicit allowance
and request ceiling). Metrics come only from the frozen labels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av import clip_eval
from av.clip_eval import contract
from av.clip_eval import corpus as corpus_mod
from av.clip_eval.mock import LabeledDecisionClient
from av.core.config import AVConfig
from av.db.repository import Repository
from av.search.clip import clip_video

_SUCCESS_STATUSES = {"ok", "no_usable_clips", "deterministic_only"}


@dataclass
class ArmConfig:
    name: str
    decide: bool
    client: Any | None


def _run_arm(
    arm: ArmConfig,
    corpus: dict,
    queries: list[dict],
    repo: Repository,
    config: AVConfig,
    *,
    clips_wanted: int,
    target_seconds: float,
    min_seconds: float,
    export_dir: Path | None,
    media_dir: Path | None,
) -> dict:
    per_query: list[dict] = []
    stage_usages: list[dict] = []
    warnings: list[str] = []
    export_receipts: list[dict] = []
    render_ms = 0.0
    for label in queries:
        result = clip_video(
            label["topic"],
            label["video_id"],
            repo,
            config,
            clips_wanted=clips_wanted,
            target_seconds=target_seconds,
            min_seconds=min_seconds,
            decide=arm.decide,
            client=arm.client,
        )
        timings = result.get("timings", {})
        export_for_query: list[dict] = []
        if export_dir and result["clips"]:
            from av.pipeline.clip_export import export_clips

            video = repo.get_video(label["video_id"])
            source = Path(video.file_path)
            if not source.exists() and media_dir is not None:
                spec = _media_spec(corpus, label["video_id"])
                source = corpus_mod.generate_media(spec, media_dir)
            export_for_query, elapsed_ms = export_clips(
                source,
                result["clips"],
                export_dir / arm.name / label["query_id"],
                overwrite=True,
            )
            render_ms += elapsed_ms
        if export_dir and not result["clips"]:
            export_for_query, elapsed_ms = [], 0.0
        export_receipts.extend(export_for_query)
        metrics = contract.query_metrics(result["clips"], label, k=clips_wanted)
        metrics["query_id"] = label["query_id"]
        metrics["topic"] = label["topic"]
        metrics["status"] = result["status"]
        metrics["timings"] = timings
        if result["status"] not in _SUCCESS_STATUSES:
            warnings.append(f"{label['query_id']}: status {result['status']}")
        per_query.append(metrics)
        stage_usages.append(result.get("stage_usage", {}))

    valid_exports = [receipt for receipt in export_receipts if receipt.get("valid")]
    usage = contract.merge_stage_usage(stage_usages)
    usage_block = usage
    timings = {
        "prepare_ms_total": round(sum(q["timings"].get("prepare_ms", 0.0) for q in per_query), 1),
        "selection_ms_total": round(sum(q["timings"].get("selection_ms", 0.0) for q in per_query), 1),
        "selection_warm_ms_total": round(
            sum(q["timings"].get("selection_warm_ms", 0.0) for q in per_query), 1
        ),
        "render_ms_total": round(render_ms, 1),
    }
    return {
        "arm": arm.name,
        "decide": arm.decide,
        "metrics_per_query": per_query,
        "metrics_total": contract.aggregate(per_query),
        "stage_usage": usage_block,
        "timings": timings,
        "export": {
            "requested": export_dir is not None,
            "clips_exported": len(export_receipts),
            "clips_valid": len(valid_exports),
        },
        "failures": warnings,
        "decision_provider": "typesafe" if arm.decide else "none",
    }


def _media_spec(corpus: dict, video_id: str) -> dict:
    for video in corpus["videos"]:
        if video["id"] == video_id:
            media = dict(video["content"].get("media", {}))
            media.setdefault("filename", f"{video_id}.mp4")
            media.setdefault("duration_sec", video["content"]["duration_sec"])
            return media
    raise KeyError(video_id)


def run_evaluation(
    corpus_path: Path,
    queries_path: Path,
    *,
    db_path: Path,
    media_dir: Path | None = None,
    export_dir: Path | None = None,
    arms: tuple[str, ...] = ("deterministic", "jev_mock"),
    clips_wanted: int = 2,
    target_seconds: float = 30.0,
    min_seconds: float = 10.0,
    config: AVConfig | None = None,
) -> dict:
    """Execute the evaluation and return the full contract-v1 receipt."""
    corpus = corpus_mod.load_corpus(corpus_path)
    queries = corpus_mod.load_queries(queries_path)
    config = config or AVConfig()

    ingest_start = time.perf_counter()
    repo = Repository(db_path)
    corpus_mod.materialize_corpus(corpus, repo)
    ingestion_ms = (time.perf_counter() - ingest_start) * 1000

    by_video = {query["video_id"] for query in queries}
    missing = by_video - {video["id"] for video in corpus["videos"]}
    if missing:
        repo.close()
        raise corpus_mod.CorpusError(f"Queries reference unknown videos: {sorted(missing)}")

    arm_configs: list[ArmConfig] = []
    for name in arms:
        if name == "deterministic":
            arm_configs.append(ArmConfig(name=name, decide=False, client=None))
        elif name == "jev_mock":
            client = LabeledDecisionClient(queries)
            arm_configs.append(ArmConfig(name=name, decide=True, client=client))
        else:
            repo.close()
            raise ValueError(
                f"Unknown arm {name!r}; supported arms: deterministic, jev_mock"
            )

    arm_receipts = [
        _run_arm(
            arm,
            corpus,
            queries,
            repo,
            config,
            clips_wanted=clips_wanted,
            target_seconds=target_seconds,
            min_seconds=min_seconds,
            export_dir=export_dir,
            media_dir=media_dir,
        )
        for arm in arm_configs
    ]
    repo.close()

    return {
        "contract_version": clip_eval.CONTRACT_VERSION,
        "corpus": str(corpus_path),
        "queries": str(queries_path),
        "query_count": len(queries),
        "ingestion_ms": round(ingestion_ms, 1),
        "arms": arm_receipts,
        "notes": [
            "Synthetic rights-cleared corpus; labels are frozen and independent of decision scores.",
            "The jev_mock arm exercises the typed decision pipeline; it is not a Jev quality measurement.",
            "Percentile timings are reported only when at least five samples support them.",
        ],
    }
