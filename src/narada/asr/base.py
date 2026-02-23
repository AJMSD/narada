from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
    asr_preset: str = "balanced"


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


def build_engine(
    engine_name: str,
    *,
    faster_whisper_model_dir: Path | None = None,
    whisper_cpp_model_dir: Path | None = None,
) -> AsrEngine:
    normalized = engine_name.strip().lower()
    if normalized == "faster-whisper":
        from narada.asr.faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(model_directory=faster_whisper_model_dir)
    if normalized == "whisper-cpp":
        from narada.asr.whisper_cpp_engine import WhisperCppEngine

        return WhisperCppEngine(model_directory=whisper_cpp_model_dir)
    raise EngineUnavailableError(f"Unsupported ASR engine '{engine_name}'.")
