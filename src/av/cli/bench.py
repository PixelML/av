"""av bench — measure the cost/accuracy frontier for video understanding.

Headline axes, chosen so results read against published agentic-video comparisons:
**tokens per query** and **accuracy**. Alongside them sits the axis an API vendor
cannot report — **dollars per query** and **video-hours per dollar** on hardware you
own — because that is the number that decides whether in-shore deployment pays.

Subcommands, in the order you should run them:

``av bench probe``    what can this deployment actually do? Is tokens-per-frame tunable?
``av bench gate``     can the model order a handful of images at all? Run this first.
``av bench prepare``  turn a public benchmark's annotations into a task file.
``av bench run``      dense vs agentic arms over a task file.
``av bench sweep``    event recall against sampling interval, on real footage.
``av bench noise``    the spread across identical runs — the floor below which deltas are noise.
``av bench cost``     the arithmetic, with every input labelled. No API calls.

Every subcommand writes a receipt. JSON to stdout, progress to stderr.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import typer

from av.bench.cost import CellUsage, HourlyCost, parse_cost_model, video_hours_per_dollar
from av.bench.datasets import ADAPTERS, SOURCES, load_annotations, subset_by_video
from av.bench.fixtures import FIXTURE_KINDS, MAX_N, generate_fixture
from av.bench.receipts import (
    COMMUNITY,
    DERIVED,
    MEASURED,
    UNTESTED,
    Receipt,
    redact_endpoint,
    sha256_file,
    write_receipt,
)
from av.bench.runner import (
    DEFAULT_FRAME_INTERVALS,
    DEFAULT_NOISE_REPEATS,
    collapse_point,
    interpret_delta,
    is_saturated,
    noise_floor,
)
from av.bench.tasks import events as events_task
from av.bench.tasks import videoqa
from av.bench.tasks.ordering import run_gate
from av.bench.vlm import BenchVLM, probe_tokens_per_frame
from av.cli.output import error, output_json, progress
from av.core.config import get_config
from av.pipeline.ffmpeg import get_video_info

bench_app = typer.Typer(help="Measure the cost/accuracy frontier for video understanding.")

DEFAULT_RECEIPTS_DIR = Path("./bench-receipts")


# --- shared helpers ----------------------------------------------------------

def _provider_record(config, model: str | None, *, temperature: float, seed: int | None) -> dict:
    """What the receipt records about the provider. Endpoints are reduced to a host."""
    return {
        "provider": config.provider or "(unset)",
        "model": model or config.vision_model,
        "endpoint_host": redact_endpoint(config.api_base_url),
        "temperature": temperature,
        "seed": seed,
    }


def _resolve_config(provider: str, model: str, base_url: str):
    """Build a config from flags, falling back to the user's saved configuration.

    A provider override never reaches into source for an endpoint: it takes the
    preset's default, which for self-hosted providers is a local placeholder, and
    expects the endpoint from ``--base-url``, ``AV_API_BASE_URL``, or config.json.
    """
    config = get_config()
    if provider:
        from av.core.constants import PROVIDER_PRESETS

        preset = PROVIDER_PRESETS.get(provider)
        if preset is None:
            raise typer.BadParameter(
                f"unknown provider {provider!r}; known: {', '.join(sorted(PROVIDER_PRESETS))}"
            )
        config = config.model_copy(update={"provider": provider, **preset})
    if base_url:
        config = config.model_copy(update={"api_base_url": base_url})
    if model:
        config = config.model_copy(update={"vision_model": model, "chat_model": model})
    return config


def _parse_intervals(spec: str) -> tuple[float, ...]:
    if not spec:
        return DEFAULT_FRAME_INTERVALS
    return tuple(float(part) for part in spec.split(",") if part.strip())


def _parse_sizes(spec: str) -> tuple[int, ...]:
    return tuple(int(part) for part in spec.split(",") if part.strip())


def _emit(receipt: Receipt, payload: dict, receipts_dir: Path) -> None:
    path = write_receipt(receipt, receipts_dir)
    payload["receipt"] = str(path)
    payload["run_id"] = receipt.run_id
    output_json(payload)


# --- probe -------------------------------------------------------------------

@bench_app.command("probe")
def probe_cmd(
    provider: str = typer.Option("", "--provider", help="Provider preset to use"),
    model: str = typer.Option("", "--model", "-m", help="Model id override"),
    base_url: str = typer.Option("", "--base-url", help="Endpoint base URL override"),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS_DIR, "--receipts", help="Receipt output directory"),
) -> None:
    """Ask a live deployment what it can do, instead of assuming.

    Answers two questions that decide the shape of the whole sweep: does the endpoint
    accept multiple images in one request, and is the per-frame token cost tunable?
    Two candidate knobs are tested separately — the OpenAI ``detail`` hint and the
    resolution actually uploaded — because a server may honour one and silently
    ignore the other. If neither moves the count, the tokens-per-frame axis is
    dropped rather than faked.
    """
    config = _resolve_config(provider, model, base_url)
    work = Path(tempfile.mkdtemp(prefix="av_bench_probe_"))
    fixture = generate_fixture("color", 2, work)

    progress(f"  Probing {config.provider or '(unset)'} / {model or config.vision_model}...")
    tokens = probe_tokens_per_frame(config, fixture.frame_paths[0], model=model or None)

    multi = BenchVLM(config, model=model or None, stream=False, max_tokens=16).ask(
        fixture.frame_paths, "Reply with the single word: ok"
    )
    multi_supported = (
        True if multi.ok else (False if multi.multi_image_unsupported else None)
    )

    receipt = Receipt(
        kind="probe",
        provider=_provider_record(config, model, temperature=0.0, seed=0),
        determinism={
            "temperature": 0.0,
            "seed": 0,
            "fixture_ffmpeg_commands": fixture.ffmpeg_commands,
            "fixture_sha256": [sha256_file(p) for p in fixture.frame_paths],
        },
        cells=[tokens],
        summary={
            "tokens_per_frame_verdict": tokens["verdict"],
            "effective_knobs": tokens["effective_knobs"],
            "distinct_image_token_counts": tokens["distinct_image_token_counts"],
            "multi_image_supported": multi_supported,
            "multi_image_error": multi.error,
        },
    )

    if tokens["verdict"] == "tunable":
        receipt.add_claim(
            MEASURED,
            "Per-frame token count moved on this deployment via "
            f"{', '.join(tokens['effective_knobs'])}, so tokens-per-frame is a usable "
            "sweep axis here. Drive it with --scale-width.",
        )
    elif tokens["verdict"] == "fixed":
        receipt.add_claim(
            MEASURED,
            "Per-frame token count did not change across any tested detail setting or "
            "input resolution on this deployment. The tokens-per-frame axis is dropped "
            "rather than simulated.",
        )
    else:
        receipt.add_claim(
            UNTESTED,
            "Per-frame token count could not be established — the provider returned no "
            "usage, or the endpoint was unreachable.",
        )

    receipt.notes.append(
        "Measured through an OpenAI-compatible chat endpoint. A provider's native API "
        "may bill images differently from its compatibility layer, so this verdict "
        "describes the endpoint you are actually calling, not the model in general."
    )

    if multi_supported is False:
        receipt.add_claim(
            MEASURED,
            "This deployment refused a two-image request. That is a capability result, "
            "not a harness error: multi-image benchmarks cannot run against it.",
        )

    _emit(
        receipt,
        {
            "verdict": tokens["verdict"],
            "effective_knobs": tokens["effective_knobs"],
            "probe": tokens,
            "multi_image_supported": multi_supported,
        },
        receipts,
    )


# --- gate --------------------------------------------------------------------

@bench_app.command("gate")
def gate_cmd(
    kind: str = typer.Option("color", "--kind", help=f"Fixture kind: {', '.join(FIXTURE_KINDS)}"),
    sizes: str = typer.Option("2,4,8", "--sizes", help="Comma-separated frame counts"),
    provider: str = typer.Option("", "--provider", help="Provider preset to use"),
    model: str = typer.Option("", "--model", "-m", help="Model id override"),
    base_url: str = typer.Option("", "--base-url", help="Endpoint base URL override"),
    keep_fixtures: str = typer.Option("", "--keep-fixtures", help="Directory to keep fixtures in"),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS_DIR, "--receipts", help="Receipt output directory"),
) -> None:
    """Temporal-ordering capability gate — run this before anything else.

    A model that cannot report the order of a handful of images cannot be
    meaningfully scored on long-video reasoning, and any throughput number measured
    against it describes a machine doing the wrong thing quickly. The gate is cheap
    and it can save the entire sweep.
    """
    if kind not in FIXTURE_KINDS:
        error(f"unknown fixture kind {kind!r}; expected one of {', '.join(FIXTURE_KINDS)}")
        raise typer.Exit(2)

    fixture_sizes = _parse_sizes(sizes)
    too_big = [n for n in fixture_sizes if n > MAX_N[kind]]
    if too_big:
        error(f"{kind} fixtures support at most {MAX_N[kind]} frames; asked for {too_big}")
        raise typer.Exit(2)

    config = _resolve_config(provider, model, base_url)
    work = Path(keep_fixtures).expanduser() if keep_fixtures else Path(
        tempfile.mkdtemp(prefix="av_bench_gate_")
    )
    vlm = BenchVLM(config, model=model or None, temperature=0.0, seed=0, max_tokens=512)

    def report(cell) -> None:
        mark = "PASS" if cell.exact_order else "FAIL"
        progress(
            f"  n={cell.n:<3} {mark}  expected={len(cell.expected)} "
            f"reported={cell.n_reported} prefix={cell.correct_prefix}"
        )

    progress(f"  Gate: {kind} fixtures, sizes {list(fixture_sizes)}")
    cells, summary = run_gate(vlm, kind=kind, sizes=fixture_sizes, work_dir=work, on_cell=report)

    receipt = Receipt(
        kind="gate",
        provider=_provider_record(config, model, temperature=0.0, seed=0),
        determinism={
            "temperature": 0.0,
            "seed": 0,
            "fixture_version": 1,
            "ffmpeg_commands": summary.pop("ffmpeg_commands"),
        },
        cells=[c.to_dict() for c in cells],
        summary=summary,
    )

    verdict = summary["verdict"]
    if verdict == "pass":
        receipt.add_claim(MEASURED, f"Exact-order accuracy was 100% at every tested size {list(fixture_sizes)}.")
    elif verdict == "no_multi_image":
        receipt.add_claim(MEASURED, "The provider refused multi-image input. No ordering result is possible.")
    else:
        receipt.add_claim(
            MEASURED,
            f"Ordering failed at sizes {summary['failed_sizes']}. Benchmark scores for this "
            "model do not measure temporal understanding until this is fixed.",
        )

    payload = {
        "verdict": verdict,
        "largest_passing_n": summary["largest_passing_n"],
        "cells": [c.to_dict() for c in cells],
    }
    if keep_fixtures:
        payload["fixtures_dir"] = str(work)
    _emit(receipt, payload, receipts)


# --- prepare -----------------------------------------------------------------

@bench_app.command("prepare")
def prepare_cmd(
    dataset: str = typer.Argument(help=f"Benchmark name: {', '.join(sorted(ADAPTERS))}"),
    annotations: Path = typer.Argument(help="Annotation file you downloaded yourself"),
    out: Path = typer.Option(..., "--out", "-o", help="Task JSONL to write"),
    video_template: str = typer.Option(
        "videos/{video_id}.mp4", "--video-template",
        help="Path pattern for the video of each row; {video_id} is substituted",
    ),
    max_questions: int = typer.Option(50, "--max-questions", help="Cap on questions"),
    max_videos: int = typer.Option(0, "--max-videos", help="Cap on distinct videos (0 = no cap)"),
) -> None:
    """Convert a public benchmark's annotations into a task file.

    No benchmark data ships with av and no videos are downloaded here. Fetch the
    annotation file yourself, mind its licence — LVBench's is non-commercial — and
    fetch the videos separately. Subsetting is grouped by video on purpose: sampling
    by question is how a 30-question run turns into a 40-hour download.
    """
    if dataset not in ADAPTERS:
        error(f"unknown dataset {dataset!r}; known: {', '.join(sorted(ADAPTERS))}")
        raise typer.Exit(2)

    rows = ADAPTERS[dataset](load_annotations(annotations))
    chosen = subset_by_video(
        rows, max_questions=max_questions, max_videos=max_videos or None
    )

    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in chosen:
            f.write(json.dumps(row.to_task_row(video_template)) + "\n")

    source = SOURCES.get(dataset, {})
    progress(f"  Wrote {len(chosen)} questions to {out}")
    output_json({
        "dataset": dataset,
        "source": source,
        "questions_available": len(rows),
        "questions_written": len(chosen),
        "videos_referenced": len({r.video_id for r in chosen}),
        "task_file": str(out),
        "video_ids": sorted({r.video_id for r in chosen}),
        "note": (
            "Videos are not downloaded. Fetch them separately and check the upstream "
            "licence before any downstream use."
        ),
    })


# --- run ---------------------------------------------------------------------

@bench_app.command("run")
def run_cmd(
    task_file: Path = typer.Argument(help="Task JSONL (see `av bench prepare`)"),
    arms: str = typer.Option("dense,agentic", "--arms", help="Comma-separated: dense, agentic"),
    interval: float = typer.Option(1.0, "--interval", help="Dense arm sampling interval, seconds"),
    coarse_interval: float = typer.Option(
        videoqa.DEFAULT_COARSE_INTERVAL_SEC, "--coarse-interval",
        help="Agentic arm coarse-pass interval, seconds",
    ),
    budget: int = typer.Option(
        videoqa.DEFAULT_AGENTIC_BUDGET, "--budget", help="Agentic arm targeted frame budget"
    ),
    max_frames: int = typer.Option(1024, "--max-frames", help="Hard ceiling on frames per request"),
    scale_width: int = typer.Option(768, "--scale-width", help="Frame width in pixels (0 = native)"),
    limit: int = typer.Option(0, "--limit", help="Only run the first N questions (0 = all)"),
    provider: str = typer.Option("", "--provider", help="Provider preset to use"),
    model: str = typer.Option("", "--model", "-m", help="Model id override"),
    base_url: str = typer.Option("", "--base-url", help="Endpoint base URL override"),
    cost: str = typer.Option(
        "", "--cost",
        help="Cost model: hourly:25.0[:prefill_tok_s[:decode_tok_s]] | token:IN:OUT | @file.json",
    ),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS_DIR, "--receipts", help="Receipt output directory"),
) -> None:
    """Dense versus agentic on the same questions, with token and dollar cost.

    The dense arm samples the whole window at a fixed rate and asks once. The agentic
    arm takes a cheap coarse look, decides which moments it needs, then fetches only
    those — and is charged for both requests. Nothing else differs between them.
    """
    config = _resolve_config(provider, model, base_url)
    cost_model = parse_cost_model(cost)
    wanted = [a.strip() for a in arms.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in videoqa.ARMS]
    if unknown:
        error(f"unknown arm(s): {unknown}; expected any of {list(videoqa.ARMS)}")
        raise typer.Exit(2)

    questions = videoqa.load_questions(task_file)
    if limit:
        questions = questions[:limit]
    missing = [q for q in questions if not q.video.exists()]
    if missing:
        error(
            f"{len(missing)} question(s) reference videos that are not on disk, "
            f"e.g. {missing[0].video}. Fetch them first."
        )
        raise typer.Exit(1)
    if not questions:
        error("task file contains no questions")
        raise typer.Exit(1)

    vlm = BenchVLM(
        config, model=model or None, temperature=0.0, seed=0,
        max_tokens=512, stream=True,
    )
    width = scale_width or None

    arm_results: dict[str, list[videoqa.QuestionResult]] = {}
    video_seconds_total = 0.0
    for arm in wanted:
        progress(f"  Arm: {arm}")
        results: list[videoqa.QuestionResult] = []
        for i, q in enumerate(questions, 1):
            duration = get_video_info(q.video).duration_sec
            if arm == wanted[0]:
                start, span = videoqa.window_for(q, duration)
                video_seconds_total += span or duration
            if arm == "dense":
                res = videoqa.run_dense(
                    vlm, q, duration, interval_sec=interval,
                    max_frames=max_frames, scale_width=width,
                )
            else:
                res = videoqa.run_agentic(
                    vlm, q, duration, coarse_interval_sec=coarse_interval,
                    budget_frames=budget, scale_width=width,
                )
            results.append(res)
            progress(
                f"    [{i}/{len(questions)}] {q.id} {'OK ' if res.correct else 'X  '}"
                f"frames={res.frames_sent} tok_in={res.tokens_in} wall={res.wall_sec:.1f}s"
            )
        arm_results[arm] = results

    cells: list[dict] = []
    for arm, results in arm_results.items():
        summary = videoqa.aggregate(results)
        usage = videoqa.usage_for(results)
        cell = {
            "arm": arm,
            "interval_sec": interval if arm == "dense" else coarse_interval,
            **summary,
            "results": [r.to_dict() for r in results],
        }
        if cost_model:
            total_cost = cost_model.cost_usd(usage)
            cell["cost_usd_total"] = round(total_cost, 6)
            cell["cost_usd_per_query"] = round(total_cost / len(results), 6) if results else None
            cell["video_hours_per_dollar"] = video_hours_per_dollar(video_seconds_total, total_cost)
            cell["cost_basis"] = cost_model.basis()
        cells.append(cell)

    receipt = Receipt(
        kind="arms",
        provider=_provider_record(config, model, temperature=0.0, seed=0),
        determinism={
            "temperature": 0.0,
            "seed": 0,
            "dense_interval_sec": interval,
            "agentic_coarse_interval_sec": coarse_interval,
            "agentic_budget_frames": budget,
            "scale_width": width,
            "max_frames": max_frames,
            "task_file": str(task_file),
            "task_file_sha256": sha256_file(Path(task_file)),
        },
        cost_model=cost_model.describe() if cost_model else None,
        cells=cells,
        summary={
            "questions": len(questions),
            "arms": wanted,
            "video_seconds_covered": round(video_seconds_total, 1),
            "by_arm": {
                arm: {
                    k: v for k, v in videoqa.aggregate(res).items()
                    if k in ("accuracy", "tokens_per_query_total", "frames_per_query")
                }
                for arm, res in arm_results.items()
            },
        },
    )
    receipt.add_claim(MEASURED, "Accuracy and token counts are from this run; tokens are the provider's own usage figures.")
    if cost_model:
        receipt.add_claim(
            DERIVED if cost_model.basis() == "derived" else MEASURED,
            f"Dollar figures follow the supplied {cost_model.mode} cost model.",
        )
    receipt.add_claim(
        COMMUNITY,
        "Published dense-versus-agentic figures for other models were produced on "
        "different hardware with different methodology. Read them alongside these "
        "numbers as a comparison of approaches, never as a like-for-like ratio.",
        source="vendor-published benchmark charts",
    )
    receipt.notes.append(
        "This run measures one model on one deployment. It is not a head-to-head "
        "against any vendor's published result."
    )

    _emit(receipt, {"summary": receipt.summary, "cells": [
        {k: v for k, v in c.items() if k != "results"} for c in cells
    ]}, receipts)


# --- sweep -------------------------------------------------------------------

@bench_app.command("sweep")
def sweep_cmd(
    artifacts: Path = typer.Argument(help="Reference artifacts JSONL with start_sec/end_sec/text"),
    video_dir: Path = typer.Argument(help="Directory holding the referenced video files"),
    probes: str = typer.Option(
        "door_activity", "--probes",
        help=f"Comma-separated probes: {', '.join(events_task.EVENT_PROBES)}",
    ),
    intervals: str = typer.Option("", "--intervals", help="Comma-separated seconds between frames"),
    max_per_probe: int = typer.Option(10, "--max-per-probe", help="Reference windows per probe"),
    scale_width: int = typer.Option(768, "--scale-width", help="Frame width in pixels (0 = native)"),
    provider: str = typer.Option("", "--provider", help="Provider preset to use"),
    model: str = typer.Option("", "--model", "-m", help="Model id override"),
    base_url: str = typer.Option("", "--base-url", help="Endpoint base URL override"),
    cost: str = typer.Option("", "--cost", help="Cost model (see `av bench run --help`)"),
    tolerance: float = typer.Option(0.1, "--tolerance", help="Score drop still counted as safe"),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS_DIR, "--receipts", help="Receipt output directory"),
) -> None:
    """Event detection against sampling interval on real footage.

    The interval at which detection collapses is the cheapest safe sampling rate, and
    it is a per-task answer, not a global one. Note the reference caveat: the shipped
    artifacts were produced by a vision model, so this measures agreement with a dense
    reference run, not recall against human ground truth.
    """
    config = _resolve_config(provider, model, base_url)
    cost_model = parse_cost_model(cost)
    axis = _parse_intervals(intervals)
    probe_list = [p.strip() for p in probes.split(",") if p.strip()]
    unknown = [p for p in probe_list if p not in events_task.EVENT_PROBES]
    if unknown:
        error(f"unknown probe(s): {unknown}; expected any of {list(events_task.EVENT_PROBES)}")
        raise typer.Exit(2)

    reference = events_task.load_reference_events(
        artifacts, video_dir, probes=probe_list, max_per_probe=max_per_probe
    )
    if not reference:
        error(
            "no reference windows found — check that the artifacts file has "
            "start_sec/end_sec/text and that the videos exist in the given directory"
        )
        raise typer.Exit(1)

    # Generous ceiling on purpose: a reasoning model can spend most of a small budget
    # before it emits the answer, and a truncated reply would score as a miss.
    vlm = BenchVLM(config, model=model or None, temperature=0.0, seed=0, max_tokens=512)
    width = scale_width or None

    cells: list[dict] = []
    for probe in probe_list:
        windows = [e for e in reference if e.probe == probe]
        covered = sum(e.end_sec - e.start_sec for e in windows)
        for interval_sec in axis:
            progress(f"  {probe} @ 1 frame / {interval_sec:g}s over {len(windows)} windows...")
            cell = events_task.run_event_cell(
                vlm, reference, interval_sec, probe=probe, scale_width=width
            )
            row = cell.to_dict()
            row["score"] = cell.reference_recall
            if cost_model:
                usage = CellUsage(
                    tokens_in=cell.tokens_in, tokens_out=cell.tokens_out,
                    wall_total_sec=cell.wall_sec, requests=cell.events_total,
                )
                cell_cost = cost_model.cost_usd(usage)
                row["cost_usd"] = round(cell_cost, 6)
                row["video_hours_per_dollar"] = video_hours_per_dollar(covered, cell_cost)
                row["cost_basis"] = cost_model.basis()
            cells.append(row)
            progress(
                f"    recall={row['reference_recall']} frames={row['frames_total']} "
                f"tok_in={row['tokens_in']}"
            )

    frontier = {
        probe: collapse_point(
            [c for c in cells if c["probe"] == probe], score_key="score", tolerance=tolerance
        )
        for probe in probe_list
    }

    receipt = Receipt(
        kind="sweep",
        provider=_provider_record(config, model, temperature=0.0, seed=0),
        determinism={
            "temperature": 0.0,
            "seed": 0,
            "intervals_sec": list(axis),
            "scale_width": width,
            "artifacts_file": str(artifacts),
            "artifacts_sha256": sha256_file(Path(artifacts)),
            "probe_questions": {p: events_task.EVENT_PROBES[p]["question"] for p in probe_list},
        },
        cost_model=cost_model.describe() if cost_model else None,
        cells=cells,
        summary={
            "probes": probe_list,
            "intervals_sec": list(axis),
            "reference_windows": len(reference),
            "frontier": frontier,
        },
        notes=[events_task.REFERENCE_CAVEAT],
    )
    receipt.add_claim(MEASURED, "Detection rates and token counts are from this run.")
    receipt.add_claim(
        UNTESTED,
        "Precision was not measured: only windows the reference marks as containing "
        "the event were shown, so false positives on empty windows are unknown.",
    )
    _emit(receipt, {"frontier": frontier, "cells": cells, "caveat": events_task.REFERENCE_CAVEAT}, receipts)


# --- noise -------------------------------------------------------------------

@bench_app.command("noise")
def noise_cmd(
    kind: str = typer.Option("color", "--kind", help=f"Fixture kind: {', '.join(FIXTURE_KINDS)}"),
    n: int = typer.Option(6, "--n", help="Frames in the repeated fixture"),
    repeats: int = typer.Option(DEFAULT_NOISE_REPEATS, "--repeats", help="Identical runs"),
    provider: str = typer.Option("", "--provider", help="Provider preset to use"),
    model: str = typer.Option("", "--model", "-m", help="Model id override"),
    base_url: str = typer.Option("", "--base-url", help="Endpoint base URL override"),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS_DIR, "--receipts", help="Receipt output directory"),
) -> None:
    """Run one unchanged cell repeatedly and publish the spread.

    This is the number that makes every other number readable. A benchmark delta
    smaller than this spread is noise, and reporting it as a result is how a
    measurement turns into a claim it cannot support.
    """
    config = _resolve_config(provider, model, base_url)
    work = Path(tempfile.mkdtemp(prefix="av_bench_noise_"))
    fixture = generate_fixture(kind, n, work)
    vlm = BenchVLM(config, model=model or None, temperature=0.0, seed=0, max_tokens=512)

    from av.bench.tasks.ordering import run_ordering_cell

    observations: list[dict] = []

    def once(i: int) -> float | None:
        cell = run_ordering_cell(vlm, fixture)
        observations.append(cell.to_dict())
        if not cell.ok:
            return None
        return cell.correct_prefix / cell.n

    def report(i: int, value: float | None) -> None:
        progress(f"  run {i + 1}/{repeats}: score={value}")

    progress(f"  Noise floor: {repeats} identical runs of a {n}-frame {kind} fixture")
    spread = noise_floor(once, repeats=repeats, unit=" score", on_run=report)

    token_values = [o["tokens_in"] for o in observations if o["tokens_in"] is not None]
    token_spread = (max(token_values) - min(token_values)) if token_values else None

    receipt = Receipt(
        kind="noise",
        provider=_provider_record(config, model, temperature=0.0, seed=0),
        determinism={
            "temperature": 0.0,
            "seed": 0,
            "repeats": repeats,
            "fixture_kind": kind,
            "fixture_n": n,
            "ffmpeg_commands": fixture.ffmpeg_commands,
            "fixture_sha256": [sha256_file(p) for p in fixture.frame_paths],
        },
        cells=observations,
        summary={
            "score_spread": spread.to_dict(),
            "prompt_token_spread": token_spread,
            "interpretation": interpret_delta(0.0, spread),
        },
    )
    saturated = is_saturated(spread)
    receipt.summary["saturated"] = saturated
    if saturated:
        receipt.add_claim(
            UNTESTED,
            "Every run scored full marks, so this cell has no headroom to vary and its "
            "zero spread is not a usable noise floor. Re-run --n at a size the model "
            "does not solve perfectly.",
        )
    else:
        receipt.add_claim(
            MEASURED,
            f"Across {spread.n} identical runs the score spread was "
            f"{spread.range if spread.range is not None else 'unmeasured'}. Any "
            "single-run difference at or below that spread is noise.",
        )
    _emit(
        receipt,
        {
            "score_spread": spread.to_dict(),
            "prompt_token_spread": token_spread,
            "saturated": saturated,
            "interpretation": receipt.summary["interpretation"],
        },
        receipts,
    )


# --- cost --------------------------------------------------------------------

@bench_app.command("cost")
def cost_cmd(
    tokens_per_frame: int = typer.Option(..., "--tokens-per-frame", help="Tokens the model retains per frame"),
    context_tokens: int = typer.Option(0, "--context-tokens", help="Model context window in tokens"),
    prefill_tok_s: float = typer.Option(0.0, "--prefill-tok-s", help="Prefill throughput, tokens/sec"),
    hourly_usd: float = typer.Option(0.0, "--hourly-usd", help="Hardware cost, dollars per hour"),
    kv_bytes_per_token: float = typer.Option(0.0, "--kv-bytes-per-token", help="KV cache bytes per token"),
    intervals: str = typer.Option("", "--intervals", help="Comma-separated seconds between frames"),
    target_vhpd: float = typer.Option(
        100.0, "--target-vhpd", help="Target video-hours per dollar to solve for"
    ),
    source: str = typer.Option(
        "", "--source", help="Where the inputs came from, recorded in the receipt"
    ),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS_DIR, "--receipts", help="Receipt output directory"),
) -> None:
    """Work the arithmetic, with every input labelled. Makes no API calls.

    Nothing here is measured — it is all derived from the figures you pass in, which
    is exactly why the receipt records them and why ``--source`` exists. Supply your
    own numbers and the tool will tell you what sampling rate the target implies,
    how many frames fit in one request, and how much KV cache an hour of video costs.
    """
    if tokens_per_frame <= 0:
        error("--tokens-per-frame must be positive")
        raise typer.Exit(2)

    axis = _parse_intervals(intervals)
    cost_model = HourlyCost(hourly_usd=hourly_usd, prefill_tok_per_s=prefill_tok_s or None) if hourly_usd else None

    rows: list[dict] = []
    for interval_sec in axis:
        frames_per_video_hour = 3600.0 / interval_sec
        tokens_per_video_hour = frames_per_video_hour * tokens_per_frame
        row: dict = {
            "interval_sec": interval_sec,
            "frames_per_video_hour": round(frames_per_video_hour, 3),
            "tokens_per_video_hour": round(tokens_per_video_hour, 1),
        }
        if prefill_tok_s:
            gpu_seconds = tokens_per_video_hour / prefill_tok_s
            row["gpu_seconds_per_video_hour"] = round(gpu_seconds, 3)
            if hourly_usd:
                dollars = gpu_seconds / 3600.0 * hourly_usd
                row["cost_usd_per_video_hour"] = round(dollars, 6)
                row["video_hours_per_dollar"] = round(1.0 / dollars, 3) if dollars else None
        if kv_bytes_per_token:
            row["kv_gib_per_video_hour"] = round(
                tokens_per_video_hour * kv_bytes_per_token / (1024 ** 3), 4
            )
        rows.append(row)

    summary: dict = {"tokens_per_frame": tokens_per_frame}

    if context_tokens:
        frames_in_context = context_tokens // tokens_per_frame
        summary["frames_per_request_max"] = frames_in_context
        summary["single_request_minutes_at_1fps"] = round(frames_in_context / 60.0, 2)

    if prefill_tok_s and hourly_usd:
        # Solve: 1 / ((3600/I * tpf) / prefill / 3600 * hourly) = target
        # => I = target * tpf * hourly / prefill
        required_interval = target_vhpd * tokens_per_frame * hourly_usd / prefill_tok_s
        summary["target_video_hours_per_dollar"] = target_vhpd
        summary["required_interval_sec"] = round(required_interval, 2)
        summary["required_interval_description"] = (
            f"one frame every {required_interval:.0f} seconds at {tokens_per_frame} tokens/frame"
        )

    receipt = Receipt(
        kind="cost",
        provider={"note": "no provider contacted — this subcommand is arithmetic only"},
        determinism={
            "tokens_per_frame": tokens_per_frame,
            "context_tokens": context_tokens or None,
            "prefill_tok_per_s": prefill_tok_s or None,
            "hourly_usd": hourly_usd or None,
            "kv_bytes_per_token": kv_bytes_per_token or None,
            "intervals_sec": list(axis),
            "input_source": source or "(not stated by the caller)",
        },
        cost_model=cost_model.describe() if cost_model else None,
        cells=rows,
        summary=summary,
    )
    receipt.add_claim(
        DERIVED,
        "Every figure in this receipt is arithmetic over the inputs supplied on the "
        "command line. None of it was measured against a running model.",
    )
    if not source:
        receipt.add_claim(
            UNTESTED,
            "The caller did not state where these inputs came from, so their provenance "
            "is unknown. Pass --source to record it.",
        )
    _emit(receipt, {"summary": summary, "rows": rows}, receipts)


# --- plan --------------------------------------------------------------------

@bench_app.command("plan")
def plan_cmd(
    widths: str = typer.Option(
        "256,512,768,1024,1536,1920", "--widths", help="Comma-separated frame widths in pixels"
    ),
    aspect: float = typer.Option(16 / 9, "--aspect", help="Frame aspect ratio (width / height)"),
    budgets: str = typer.Option(
        "", "--budgets", help="Comma-separated token budgets to solve widths for"
    ),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS_DIR, "--receipts", help="Receipt output directory"),
) -> None:
    """Predict per-frame token cost against frame resolution. Makes no API calls.

    Useful before spending anything: it shows where the two walls are — the upscale
    floor, below which shrinking frames buys nothing, and the token ceiling, above
    which extra resolution is discarded. A tokens-per-frame sweep belongs between
    them, and ``--scale-width`` on the other subcommands is how you drive it.

    These are predictions from a published preprocessor algorithm, not measurements
    of your server. Confirm them with ``av bench probe``.
    """
    from av.providers.deepseek import (
        MAX_IMAGE_TOKENS,
        MIN_IMAGE_TOKENS,
        MIN_PIXELS,
        plan_image_tokens,
        widths_for_token_budgets,
    )

    rows = []
    for width in (int(w) for w in widths.split(",") if w.strip()):
        height = max(int(round(width / aspect)), 1)
        rows.append({"width": width, "height": height, **plan_image_tokens(width, height).to_dict()})

    solved = (
        widths_for_token_budgets([int(b) for b in budgets.split(",") if b.strip()], aspect)
        if budgets
        else {}
    )

    receipt = Receipt(
        kind="plan",
        provider={"note": "no provider contacted — this subcommand is arithmetic only"},
        determinism={"aspect_ratio": aspect, "widths": widths, "budgets": budgets or None},
        cells=rows,
        summary={
            "min_pixels_floor": MIN_PIXELS,
            "min_tokens_per_frame": MIN_IMAGE_TOKENS,
            "max_tokens_per_frame": MAX_IMAGE_TOKENS,
            "widths_for_budgets": solved,
        },
    )
    receipt.add_claim(
        DERIVED,
        "Token counts are computed from the model's published vision preprocessor "
        "algorithm and reproduce its published worked examples exactly below the "
        "token ceiling. Rows marked approximate ran a shrink search and may differ "
        "by one grid step; measure those with `av bench probe`.",
        source="model vision_config and reference image processor",
    )
    _emit(receipt, {"rows": rows, "widths_for_budgets": solved}, receipts)


def register(app: typer.Typer) -> None:
    app.add_typer(bench_app, name="bench", help="Benchmark the cost/accuracy frontier")
