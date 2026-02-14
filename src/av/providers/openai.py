"""OpenAI-compatible provider implementations."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from openai import OpenAI

from av.core.config import AVConfig
from av.core.exceptions import APIError
from av.providers.base import (
    Caption,
    CaptionerProvider,
    ChunkCaption,
    EmbedderProvider,
    LLMProvider,
    TranscriberProvider,
    TranscriptSegment,
)


def _token_from_codex_auth_file(auth_path: Path) -> str | None:
    if not auth_path.exists():
        return None
    try:
        data = json.loads(auth_path.read_text())
    except Exception:
        return None
    token = (data.get("tokens") or {}).get("access_token")
    return token if isinstance(token, str) and token.strip() else None


def _codex_oauth_token() -> str | None:
    """Read OAuth access token from Codex CLI auth cache, if available."""
    return _token_from_codex_auth_file(Path.home() / ".codex" / "auth.json")


def _openclaw_oauth_token() -> str | None:
    """Read OpenAI Codex OAuth token from OpenClaw auth profiles if present."""
    agents_dir = Path.home() / ".openclaw" / "agents"
    if not agents_dir.exists():
        return None

    best: tuple[int, str] | None = None
    for profile in agents_dir.glob("*/agent/auth-profiles.json"):
        try:
            data = json.loads(profile.read_text())
        except Exception:
            continue

        profiles = data.get("profiles") or {}
        usage = data.get("usageStats") or {}

        for profile_id, cfg in profiles.items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("provider") != "openai-codex" or cfg.get("type") != "oauth":
                continue
            token = cfg.get("access")
            if not isinstance(token, str) or not token.strip():
                continue
            last_used = 0
            stats = usage.get(profile_id)
            if isinstance(stats, dict):
                lu = stats.get("lastUsed")
                if isinstance(lu, (int, float)):
                    last_used = int(lu)
            if best is None or last_used > best[0]:
                best = (last_used, token)

    return best[1] if best else None


def _resolve_api_key(config: AVConfig) -> str:
    """Resolve API key, preferring explicit AV_API_KEY then Codex OAuth token."""
    configured = (config.api_key or "").strip()

    # Preserve existing behavior for real keys, but allow placeholder to fallback.
    if configured and configured.lower() != "no-key":
        return configured

    # Prefer OpenClaw auth-profile OAuth (often fresher), then Codex CLI cache.
    oauth = _openclaw_oauth_token() or _codex_oauth_token()
    if oauth:
        return oauth

    # Final fallback maintains previous explicit failure behavior.
    return configured or "no-key"


def _client(config: AVConfig) -> OpenAI:
    kwargs: dict = {
        "base_url": config.api_base_url,
        "api_key": _resolve_api_key(config),
    }
    # Anthropic's OpenAI-compatible endpoint requires anthropic-version header
    if config.provider == "anthropic":
        kwargs["default_headers"] = {"anthropic-version": "2023-06-01"}
    return OpenAI(**kwargs)


class OpenAITranscriber(TranscriberProvider):
    def __init__(self, config: AVConfig):
        self.config = config
        self.client = _client(config)

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        try:
            with open(audio_path, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    model=self.config.transcribe_model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
        except Exception as e:
            raise APIError(f"Transcription failed: {e}", provider="openai") from e

        segments: list[TranscriptSegment] = []
        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segments.append(
                    TranscriptSegment(
                        start_sec=seg.get("start", seg.start) if isinstance(seg, dict) else seg.start,
                        end_sec=seg.get("end", seg.end) if isinstance(seg, dict) else seg.end,
                        text=(seg.get("text", "") if isinstance(seg, dict) else seg.text).strip(),
                    )
                )
        elif hasattr(response, "text") and response.text:
            segments.append(TranscriptSegment(start_sec=0.0, end_sec=0.0, text=response.text.strip()))
        return segments


class OpenAICaptioner(CaptionerProvider):
    def __init__(self, config: AVConfig):
        self.config = config
        self.client = _client(config)

    def caption_frames(
        self, frame_paths: list[Path], timestamps: list[float], prompt: str | None = None
    ) -> list[Caption]:
        captions: list[Caption] = []
        total = len(frame_paths)
        for i, (fp, ts) in enumerate(zip(frame_paths, timestamps)):
            print(f"  Captioning frame {i + 1}/{total}...", file=sys.stderr, end="\r")
            try:
                img_data = base64.b64encode(fp.read_bytes()).decode()
                ext = fp.suffix.lstrip(".").lower()
                if ext == "jpg":
                    ext = "jpeg"
                response = self.client.chat.completions.create(
                    model=self.config.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt or "Describe this video frame in one detailed sentence. Focus on actions, objects, and scene context.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{ext};base64,{img_data}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                )
                text = response.choices[0].message.content or ""
                captions.append(Caption(timestamp_sec=ts, text=text.strip(), frame_path=str(fp)))
            except Exception as e:
                print(f"\n  Warning: caption failed for frame at {ts:.1f}s: {e}", file=sys.stderr)
        if total > 0:
            print(file=sys.stderr)  # newline after \r progress
        return captions


    def caption_chunk(
        self, frame_paths: list[Path], timestamps: list[float], prompt: str
    ) -> str:
        """Caption multiple frames as a single temporal chunk via one multi-image API call."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for fp in frame_paths:
            img_data = base64.b64encode(fp.read_bytes()).decode()
            ext = fp.suffix.lstrip(".").lower()
            if ext == "jpg":
                ext = "jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{img_data}"},
            })

        response = self.client.chat.completions.create(
            model=self.config.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
        )
        return (response.choices[0].message.content or "").strip()


class OpenAIEmbedder(EmbedderProvider):
    def __init__(self, config: AVConfig):
        self.config = config
        self.client = _client(config)
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.config.embed_model,
                input=texts,
            )
            vecs = [item.embedding for item in response.data]
            if vecs and self._dim is None:
                self._dim = len(vecs[0])
            return vecs
        except Exception as e:
            raise APIError(f"Embedding failed: {e}", provider="openai") from e

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe with a dummy embedding to discover dimensionality
            vecs = self.embed(["dim probe"])
            if vecs:
                self._dim = len(vecs[0])
            else:
                return 1536  # fallback
        return self._dim


class OpenAILLM(LLMProvider):
    def __init__(self, config: AVConfig):
        self.config = config
        self.client = _client(config)

    def complete(self, prompt: str, context: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": VIDEO_QA_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Context from video analysis:\n\n{context}\n\nQuestion: {prompt}",
                    },
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise APIError(f"Chat completion failed: {e}", provider="openai") from e

    def summarize(self, system_prompt: str, user_content: str) -> str:
        """Generic system/user LLM call for cascade summarization."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise APIError(f"Summarization failed: {e}", provider="openai") from e


# --- System prompts ---

VIDEO_QA_SYSTEM_PROMPT = """\
You are a helpful AI assistant that helps users search, explore, and understand video content. \
You have access to transcripts, dense captions, and scene descriptions extracted from indexed videos.

When answering questions:
- Use the provided context from video transcripts and captions to answer accurately
- Provide specific timestamps when referencing video content (e.g., "at 02:15")
- Summarize key information clearly and concisely
- If the context contains dense captions, use them to describe visual actions and scene details
- Distinguish between what was said (transcript) and what was seen (captions) when both are present
- If the context is insufficient to answer confidently, say so rather than speculating
- Ask clarifying questions if the user's request is ambiguous
"""
