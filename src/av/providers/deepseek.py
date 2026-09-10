"""DeepSeek-V4.1-Flash, served behind an OpenAI-compatible API (SGLang).

This module is deliberately thin. SGLang exposes ``/v1/chat/completions`` with the
standard ``image_url`` content format, so the existing OpenAI-compatible client in
``providers/openai.py`` already speaks to it correctly. Reimplementing that shape
would add a second code path to maintain and would quietly break the repository's
provider-agnostic contract, so what lives here is only what is genuinely different:

* endpoint resolution that reads config and environment, never source
* the model's declared capability record, so the benchmark can report what it
  assumed rather than silently assuming it
* a runtime probe, because the interesting properties of a self-hosted deployment
  (does it accept multiple images? is per-frame token count tunable?) are
  deployment facts, not model facts, and must be measured against the server you
  actually have

**No endpoint is hard-coded.** The preset default points at SGLang's own local
default port. Any other endpoint comes from ``AV_API_BASE_URL`` or
``~/.config/av/config.json``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from av.core.config import AVConfig

PROVIDER_NAME = "deepseek"
DEFAULT_MODEL = "deepseek-v4.1-flash"

# SGLang's documented default bind address. A placeholder, not a deployment.
DEFAULT_BASE_URL = "http://localhost:30000/v1"

# Environment variables consulted for credentials, in order.
API_KEY_ENV_VARS = ("AV_API_KEY", "DEEPSEEK_API_KEY", "SGLANG_API_KEY")

# A self-hosted server usually needs no key; SGLang accepts any bearer token unless
# started with --api-key. This placeholder keeps the OpenAI SDK from refusing to send.
PLACEHOLDER_KEY = "no-key"


@dataclass
class CapabilityRecord:
    """What the harness believes about a deployment, and on what evidence.

    Fields default to ``None`` — meaning *not established* — rather than to a
    plausible value. A benchmark that guesses its own assumptions is not a benchmark.
    """

    model: str = DEFAULT_MODEL
    context_tokens: int | None = None
    image_tokens_per_frame: int | None = None
    image_tokens_tunable: bool | None = None
    kv_bytes_per_token: float | None = None
    multi_image_supported: bool | None = None
    evidence: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "context_tokens": self.context_tokens,
            "image_tokens_per_frame": self.image_tokens_per_frame,
            "image_tokens_tunable": self.image_tokens_tunable,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "multi_image_supported": self.multi_image_supported,
            "evidence": self.evidence,
        }

    def max_frames_per_request(self, prompt_overhead_tokens: int = 512) -> int | None:
        """How many frames fit in one request, given the context and per-frame cost.

        This is the constraint that caps a single-request video window, and it is the
        reason dense processing of a long video is not merely expensive but impossible
        past a certain duration. Returns ``None`` when either input is unestablished,
        because the alternative is inventing a limit.
        """
        if not self.context_tokens or not self.image_tokens_per_frame:
            return None
        usable = self.context_tokens - prompt_overhead_tokens
        return max(usable // self.image_tokens_per_frame, 0)


def resolve_api_key(config: AVConfig | None = None) -> str:
    """First non-empty of the configured key, then the environment, then a placeholder."""
    if config is not None:
        configured = (config.api_key or "").strip()
        if configured and configured.lower() != PLACEHOLDER_KEY:
            return configured
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return PLACEHOLDER_KEY


def resolve_base_url(config: AVConfig | None = None) -> str:
    """Endpoint from env, then config, then the local SGLang default."""
    env = os.environ.get("AV_API_BASE_URL", "").strip()
    if env:
        return env
    if config is not None and (config.api_base_url or "").strip():
        return config.api_base_url.strip()
    return DEFAULT_BASE_URL


def make_config(
    base_config: AVConfig | None = None,
    *,
    model: str | None = None,
) -> AVConfig:
    """Build an ``AVConfig`` pointed at a DeepSeek-V4.1-Flash deployment."""
    chosen = model or (base_config.vision_model if base_config else None) or DEFAULT_MODEL
    return AVConfig(
        provider=PROVIDER_NAME,
        api_base_url=resolve_base_url(base_config),
        api_key=resolve_api_key(base_config),
        transcribe_model="",   # not served by this deployment
        embed_model="",        # not served by this deployment
        vision_model=chosen,
        chat_model=chosen,
    )


def probe(
    config: AVConfig,
    image: Path,
    *,
    model: str | None = None,
    details: tuple[str | None, ...] = (None, "low", "high"),
) -> CapabilityRecord:
    """Measure a live deployment's image-token behaviour and multi-image support.

    Nothing here is assumed from documentation. If the server is unreachable the
    record comes back with ``None`` fields and the error recorded as evidence, which
    is the honest outcome for an endpoint that is scaled to zero.
    """
    from av.bench.vlm import BenchVLM, probe_tokens_per_frame

    record = CapabilityRecord(model=model or config.vision_model or DEFAULT_MODEL)

    tokens = probe_tokens_per_frame(config, image, model=model, details=details)
    record.evidence["image_tokens"] = f"probe verdict: {tokens['verdict']}"
    counts = tokens.get("distinct_image_token_counts") or []
    if tokens["verdict"] == "tunable":
        record.image_tokens_tunable = True
        record.image_tokens_per_frame = max(counts) if counts else None
    elif tokens["verdict"] == "fixed":
        record.image_tokens_tunable = False
        record.image_tokens_per_frame = counts[0] if counts else None
    else:
        record.evidence["image_tokens"] = (
            "probe inconclusive: "
            + (tokens.get("baseline_error") or "provider returned no usage")
        )

    two_frames = BenchVLM(config, model=model, stream=False, max_tokens=16)
    multi = two_frames.ask([image, image], "Reply with the single word: ok")
    if multi.ok:
        record.multi_image_supported = True
        record.evidence["multi_image"] = "two-image request accepted"
    elif multi.multi_image_unsupported:
        record.multi_image_supported = False
        record.evidence["multi_image"] = f"provider refused multiple images: {multi.error}"
    else:
        record.evidence["multi_image"] = f"inconclusive: {multi.error}"

    return record


# --- image token planning ----------------------------------------------------
#
# The vision tower is a single aspect-preserving grid, not LLaVA AnyRes tiling and
# not Qwen2-VL min/max pixels. Frames are resized, cut into `patch_size` patches,
# then a 3x3 aligner collapses each 42x42 pixel cell into one LLM token, with one
# newline token per grid row plus a start and end token.
#
# The consequence that matters for video: **per-frame token cost is a function of
# input resolution, and the client controls it.** Resizing before upload is the
# tuning knob. The OpenAI `detail` field is not — SGLang parses it and no
# multimodal processor reads it back, so it is inert against a self-hosted server
# even though the vendor's own hosted API honours it.
#
# These constants mirror the published `vision_config`. They are DOCUMENTED, not
# measured here, and `probe()` exists precisely to check them against a live server.

PATCH_SIZE = 14
ALIGNER_DOWNSAMPLE = 3
PIXELS_PER_LLM_TOKEN = PATCH_SIZE * ALIGNER_DOWNSAMPLE  # 42
MAX_IMAGE_TOKENS = 1024
MIN_PIXELS = 295_936  # 544 x 544 — smaller frames are upscaled to meet this floor
MIN_IMAGE_TOKENS = 184  # what that floor costs: a 13x13 grid


@dataclass
class ImageTokenPlan:
    """What one frame will cost, and at what resolution it will be processed."""

    source_width: int
    source_height: int
    processed_width: int
    processed_height: int
    grid_rows: int
    grid_cols: int
    tokens: int
    upscaled: bool
    downscaled: bool

    @property
    def approximate(self) -> bool:
        """True when a shrink search ran, where we may differ by one grid step.

        The published algorithm reproduces exactly for frames at or below the
        token ceiling. Above it, the upstream resize solver can land one grid row
        away from this one, so a downscaled estimate is a plan, not a promise —
        measure the real count with ``av bench probe``.
        """
        return self.downscaled

    def to_dict(self) -> dict:
        return {
            "source": [self.source_width, self.source_height],
            "processed": [self.processed_width, self.processed_height],
            "grid": [self.grid_rows, self.grid_cols],
            "tokens": self.tokens,
            "upscaled": self.upscaled,
            "downscaled": self.downscaled,
            "approximate": self.approximate,
        }


def _align_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _grid_for(width: int, height: int) -> tuple[int, int]:
    rows = math.ceil(math.ceil(height / PATCH_SIZE) / ALIGNER_DOWNSAMPLE)
    cols = math.ceil(math.ceil(width / PATCH_SIZE) / ALIGNER_DOWNSAMPLE)
    return rows, cols


def tokens_for_grid(rows: int, cols: int) -> int:
    """One token per cell, one newline per row, plus image start and end."""
    return rows * (cols + 1) + 2


def plan_image_tokens(
    width: int,
    height: int,
    *,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
    min_pixels: int = MIN_PIXELS,
) -> ImageTokenPlan:
    """Predict the token cost of one frame at a given source resolution.

    Two behaviours are worth knowing before choosing a frame size:

    * Below ``min_pixels`` the frame is **upscaled**, so a 64x64 thumbnail costs
      exactly what a 544x544 frame costs. Shrinking past that floor buys nothing.
    * Above the token ceiling the frame is shrunk until it fits, so beyond roughly
      1300x1300-equivalent area, extra resolution is discarded rather than charged.

    The useful range is therefore between those two walls, and that is where a
    tokens-per-frame sweep should place its samples.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    ratio = 1.0
    upscaled = False
    if width * height < min_pixels:
        ratio = math.sqrt(min_pixels / (width * height))
        upscaled = True

    downscaled = False
    processed_w = _align_up(int(width * ratio), PATCH_SIZE)
    processed_h = _align_up(int(height * ratio), PATCH_SIZE)
    rows, cols = _grid_for(processed_w, processed_h)

    # Shrink one patch step at a time on the long edge until the grid fits.
    guard = 0
    while tokens_for_grid(rows, cols) > max_image_tokens and guard < 10_000:
        downscaled = True
        guard += 1
        ratio *= 0.99
        processed_w = _align_up(max(int(width * ratio), PATCH_SIZE), PATCH_SIZE)
        processed_h = _align_up(max(int(height * ratio), PATCH_SIZE), PATCH_SIZE)
        rows, cols = _grid_for(processed_w, processed_h)

    return ImageTokenPlan(
        source_width=width,
        source_height=height,
        processed_width=processed_w,
        processed_height=processed_h,
        grid_rows=rows,
        grid_cols=cols,
        tokens=tokens_for_grid(rows, cols),
        upscaled=upscaled,
        downscaled=downscaled,
    )


def widths_for_token_budgets(
    budgets: list[int], aspect_ratio: float = 16 / 9
) -> dict[int, int]:
    """Largest frame width whose predicted cost stays within each token budget.

    This is what turns tokens-per-frame into an actual sweep axis: pass the widths
    to ``--scale-width`` and the frames arrive at the intended cost. Budgets below
    the upscale floor map to the floor width, since nothing cheaper exists.
    """
    out: dict[int, int] = {}
    for budget in budgets:
        best = 0
        for width in range(PIXELS_PER_LLM_TOKEN, 2400, PATCH_SIZE):
            height = max(int(round(width / aspect_ratio)), PATCH_SIZE)
            plan = plan_image_tokens(width, height)
            # Skip widths past the ceiling: they all collapse to the same processed
            # size, so reporting the largest of them would suggest a resolution the
            # server will silently discard.
            if plan.downscaled:
                break
            if plan.tokens <= budget:
                best = width
        out[budget] = best or PIXELS_PER_LLM_TOKEN
    return out


# Documented and community-reported figures are intentionally NOT baked in as
# defaults. Supply them explicitly via `av bench cost --context-tokens ... etc` so
# that every receipt records where the number came from instead of inheriting it
# from a constant nobody re-checked.
