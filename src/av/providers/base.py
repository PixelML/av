"""Abstract provider interfaces — all OpenAI-compatible, swappable via base_url."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class Caption:
    timestamp_sec: float
    text: str
    frame_path: str | None = None


@dataclass
class ChunkCaption:
    start_sec: float
    end_sec: float
    text: str
    frame_count: int


class TranscriberProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        ...


class CaptionerProvider(ABC):
    @abstractmethod
    def caption_frames(
        self, frame_paths: list[Path], timestamps: list[float]
    ) -> list[Caption]:
        ...

    def caption_chunk(
        self, frame_paths: list[Path], timestamps: list[float], prompt: str
    ) -> str:
        """Caption multiple frames as a single temporal chunk. Returns raw VLM text."""
        raise NotImplementedError("caption_chunk not implemented for this provider")


class EmbedderProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, context: str) -> str:
        ...
