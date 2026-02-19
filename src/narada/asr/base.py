from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class EngineUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionRequest:
    pcm_bytes: bytes
    sample_rate_hz: int
    languages: tuple[str, ...]
    model: str
    compute: str


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    confidence: float
    start_s: float
    end_s: float
    is_final: bool = True


class AsrEngine(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def transcribe(self, request: TranscriptionRequest) -> Sequence[TranscriptSegment]: ...


def build_engine(engine_name: str) -> AsrEngine:
    normalized = engine_name.strip().lower()
    if normalized == "faster-whisper":
        from narada.asr.faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine()
    if normalized == "whisper-cpp":
        from narada.asr.whisper_cpp_engine import WhisperCppEngine

        return WhisperCppEngine()
    raise EngineUnavailableError(f"Unsupported ASR engine '{engine_name}'.")
