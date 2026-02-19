from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec

from narada.asr.base import EngineUnavailableError, TranscriptionRequest, TranscriptSegment


class FasterWhisperEngine:
    name = "faster-whisper"

    def is_available(self) -> bool:
        return find_spec("faster_whisper") is not None

    def transcribe(self, request: TranscriptionRequest) -> Sequence[TranscriptSegment]:
        if not self.is_available():
            raise EngineUnavailableError(
                "faster-whisper is not installed. Install optional dependency group: narada[asr]."
            )
        if request.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        return []
