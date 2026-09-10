"""Provider-agnostic multi-image calls with usage accounting.

The benchmark's headline axis is tokens per query, so every call here reports the
token usage the provider itself returned. Nothing is estimated: when a provider
omits ``usage``, the receipt records ``null`` rather than a guess.

Time to first token is captured by streaming. It is the closest measurable proxy
for prefill on a chat-completions API and is labelled as a proxy everywhere it
appears — it is not a prefill measurement.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path

from av.core.config import AVConfig
from av.providers.openai import _client

# Providers differ on whether an unsupported request field is ignored or rejected.
# These fragments mark a rejection we should retry without the field.
_UNSUPPORTED_MARKERS = (
    "unknown field",
    "unknown name",
    "unrecognized",
    "unexpected keyword",
    "not supported",
    "unsupported",
    "invalid_request_error",
    "invalid_argument",
    "extra fields not permitted",
    "does not support",
)

# Fragments that mark the provider refusing multi-image input outright. That is a
# capability result, not a bug in the harness.
_MULTI_IMAGE_MARKERS = (
    "only one image",
    "single image",
    "at most 1 image",
    "too many images",
    "image count",
    "multiple images",
)


def strip_code_fence(text: str) -> str:
    """Drop a markdown code fence around a reply.

    Models routinely wrap JSON in ```json fences, and a fence that gets truncated by a
    token limit leaves the payload unterminated. Both are formatting, not content, so
    parsers strip them before trying to read an answer.
    """
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def encode_image(path: Path) -> str:
    """Base64 data URL for an image file, in the OpenAI ``image_url`` format."""
    ext = path.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{ext};base64,{data}"


@dataclass
class VLMResult:
    text: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    ttft_sec: float | None = None
    wall_sec: float = 0.0
    ok: bool = True
    error: str | None = None
    # True when the provider refused the request because of the image count.
    multi_image_unsupported: bool = False
    detail_accepted: bool | None = None
    # False when the provider rejected `seed` and the call was retried without it.
    seed_accepted: bool = True
    raw_usage: dict = field(default_factory=dict)

    @property
    def tokens_total(self) -> int | None:
        if self.tokens_in is None and self.tokens_out is None:
            return None
        return (self.tokens_in or 0) + (self.tokens_out or 0)


class BenchVLM:
    """A thin, deterministic wrapper over any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        config: AVConfig,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        detail: str | None = None,
        max_tokens: int = 1024,
        stream: bool = True,
        timeout: float = 600.0,
    ) -> None:
        self.config = config
        self.model = model or config.vision_model
        self.temperature = temperature
        self.seed = seed
        self.detail = detail
        self.max_tokens = max_tokens
        self.stream = stream
        self.timeout = timeout
        self.client = _client(config)

    # -- request construction -------------------------------------------------

    def _content(self, images: list[Path], prompt: str) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            image_url: dict = {"url": encode_image(img)}
            if self.detail:
                image_url["detail"] = self.detail
            content.append({"type": "image_url", "image_url": image_url})
        return content

    def _kwargs(self, images: list[Path], prompt: str, *, with_seed: bool) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": self._content(images, prompt)}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if with_seed and self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    # -- calling --------------------------------------------------------------

    def ask(self, images: list[Path], prompt: str) -> VLMResult:
        """One question over ``images``. Never raises for provider-side refusals.

        Determinism is requested, not assumed. Providers that reject ``seed`` outright
        get one retry without it, and the result records that the run was unseeded so
        a reader knows the noise floor is doing more work on this provider.
        """
        result = self._attempt(images, prompt, with_seed=True)
        if result.ok or not result.error or self.seed is None:
            return result
        low = result.error.lower()
        if "seed" in low or any(m in low for m in _UNSUPPORTED_MARKERS):
            retry = self._attempt(images, prompt, with_seed=False)
            retry.seed_accepted = False
            return retry
        return result

    def _attempt(self, images: list[Path], prompt: str, *, with_seed: bool) -> VLMResult:
        kwargs = self._kwargs(images, prompt, with_seed=with_seed)
        started = time.perf_counter()
        try:
            if self.stream:
                return self._stream(kwargs, started)
            return self._blocking(kwargs, started)
        except Exception as e:  # provider-side failure is data, not a crash
            msg = str(e)
            low = msg.lower()
            return VLMResult(
                ok=False,
                error=msg,
                wall_sec=time.perf_counter() - started,
                multi_image_unsupported=len(images) > 1 and any(m in low for m in _MULTI_IMAGE_MARKERS),
                detail_accepted=None if self.detail is None else False,
            )

    def _blocking(self, kwargs: dict, started: float) -> VLMResult:
        response = self.client.chat.completions.create(timeout=self.timeout, **kwargs)
        wall = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        return VLMResult(
            text=(response.choices[0].message.content or "").strip(),
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            ttft_sec=None,
            wall_sec=wall,
            raw_usage=_usage_dict(usage),
            detail_accepted=None if self.detail is None else True,
        )

    def _stream(self, kwargs: dict, started: float) -> VLMResult:
        stream = self.client.chat.completions.create(
            stream=True,
            stream_options={"include_usage": True},
            timeout=self.timeout,
            **kwargs,
        )
        chunks: list[str] = []
        ttft: float | None = None
        usage = None
        for event in stream:
            if getattr(event, "usage", None):
                usage = event.usage
            for choice in getattr(event, "choices", None) or []:
                piece = getattr(choice.delta, "content", None)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    chunks.append(piece)
        wall = time.perf_counter() - started
        return VLMResult(
            text="".join(chunks).strip(),
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            ttft_sec=ttft,
            wall_sec=wall,
            raw_usage=_usage_dict(usage),
            detail_accepted=None if self.detail is None else True,
        )


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    return {
        k: getattr(usage, k)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(usage, k, None) is not None
    }


DEFAULT_PROBE_WIDTHS: tuple[int, ...] = (256, 512, 768, 1024, 1536)


def _resized_copy(image: Path, width: int, out_dir: Path) -> Path | None:
    """Aspect-preserving resize via ffmpeg, so the probe controls the one knob that
    actually moves per-frame token cost on most vision stacks: input resolution."""
    import subprocess

    out_path = out_dir / f"probe_w{width}{image.suffix or '.jpg'}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(image), "-vf", f"scale={width}:-2", "-q:v", "2", str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    except Exception:
        return None
    return out_path if out_path.exists() and out_path.stat().st_size else None


def probe_tokens_per_frame(
    config: AVConfig,
    image: Path,
    *,
    model: str | None = None,
    details: tuple[str | None, ...] = (None, "low", "high"),
    widths: tuple[int, ...] = DEFAULT_PROBE_WIDTHS,
    baseline_prompt: str = "Reply with the single word: ok",
) -> dict:
    """Measure whether tokens-per-frame is tunable on this deployment, and by which knob.

    Two candidate levers are tested separately, because they are not equivalent and
    a provider may honour one and ignore the other:

    ``detail``      the OpenAI ``image_url`` hint. Some hosted APIs implement it;
                    some self-hosted servers parse it and never read it back, in
                    which case it is inert and must not be reported as a knob.
    ``resolution``  the pixels actually uploaded. Where a vision tower maps a frame
                    onto a grid, this is the real lever and the client owns it.

    Per-image cost is the prompt-token count minus a text-only baseline. When every
    setting on both axes yields the same count, the axis is genuinely fixed for this
    deployment and the sweep should drop it rather than fake it.
    """
    import shutil
    import tempfile

    text_only = BenchVLM(config, model=model, stream=False, max_tokens=16)
    base = text_only.ask([], baseline_prompt)

    def attributable(res: VLMResult) -> int | None:
        if res.tokens_in is None or base.tokens_in is None:
            return None
        return res.tokens_in - base.tokens_in

    detail_obs: list[dict] = []
    for detail in details:
        vlm = BenchVLM(config, model=model, detail=detail, stream=False, max_tokens=16)
        res = vlm.ask([image], baseline_prompt)
        detail_obs.append(
            {
                "detail": detail or "(unset)",
                "ok": res.ok,
                "error": res.error,
                "prompt_tokens": res.tokens_in,
                "tokens_attributable_to_image": attributable(res),
            }
        )

    width_obs: list[dict] = []
    work = Path(tempfile.mkdtemp(prefix="av_bench_probe_widths_"))
    try:
        for width in widths:
            resized = _resized_copy(image, width, work)
            if resized is None:
                width_obs.append({"width": width, "ok": False, "error": "resize failed"})
                continue
            vlm = BenchVLM(config, model=model, stream=False, max_tokens=16)
            res = vlm.ask([resized], baseline_prompt)
            width_obs.append(
                {
                    "width": width,
                    "ok": res.ok,
                    "error": res.error,
                    "prompt_tokens": res.tokens_in,
                    "tokens_attributable_to_image": attributable(res),
                }
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    def distinct(observations: list[dict]) -> list[int]:
        return sorted(
            {
                o["tokens_attributable_to_image"]
                for o in observations
                if o.get("ok") and o.get("tokens_attributable_to_image") is not None
            }
        )

    detail_counts = distinct(detail_obs)
    width_counts = distinct(width_obs)
    all_counts = sorted(set(detail_counts) | set(width_counts))

    detail_is_knob = len(detail_counts) > 1
    width_is_knob = len(width_counts) > 1

    if width_is_knob or detail_is_knob:
        verdict = "tunable"
    elif all_counts:
        verdict = "fixed"
    else:
        verdict = "unknown"

    knobs = [k for k, on in (("detail", detail_is_knob), ("resolution", width_is_knob)) if on]

    return {
        "baseline_text_only_prompt_tokens": base.tokens_in,
        "baseline_ok": base.ok,
        "baseline_error": base.error,
        "detail_observations": detail_obs,
        "resolution_observations": width_obs,
        "distinct_image_token_counts": all_counts,
        "effective_knobs": knobs,
        "verdict": verdict,
    }
