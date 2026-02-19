from __future__ import annotations

import shutil
from collections.abc import Sequence
from importlib.util import find_spec

from narada.asr.base import EngineUnavailableError, TranscriptionRequest, TranscriptSegment


class WhisperCppEngine:
    name = "whisper-cpp"

    def is_available(self) -> bool:
        has_python_binding = find_spec("whispercpp") is not None
        has_cli_binary = shutil.which("whisper-cli") is not None
        return has_python_binding or has_cli_binary

    def transcribe(self, request: TranscriptionRequest) -> Sequence[TranscriptSegment]:
        if not self.is_available():
            raise EngineUnavailableError(
                "whisper.cpp runtime is not available. Install whispercpp "
                "or provide whisper-cli binary."
            )
        if request.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        return []
