from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RuntimePerformance:
    total_audio_seconds: float = 0.0
    total_processing_seconds: float = 0.0
    committed_segments: int = 0
    _commit_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=512))

    def record_transcription(self, *, audio_seconds: float, processing_seconds: float) -> None:
        if audio_seconds < 0:
            raise ValueError("audio_seconds must be non-negative.")
        if processing_seconds < 0:
            raise ValueError("processing_seconds must be non-negative.")
        self.total_audio_seconds += audio_seconds
        self.total_processing_seconds += processing_seconds

    def record_commit_latency(self, *, elapsed_seconds: float) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative.")
        self.committed_segments += 1
        self._commit_latency_ms.append(elapsed_seconds * 1000.0)

    @property
    def realtime_factor(self) -> float | None:
        if self.total_audio_seconds <= 0:
            return None
        return self.total_processing_seconds / self.total_audio_seconds

    @property
    def average_commit_latency_ms(self) -> float | None:
        if not self._commit_latency_ms:
            return None
        return sum(self._commit_latency_ms) / len(self._commit_latency_ms)

    def status_fragment(self) -> str:
        rtf = self.realtime_factor
        commit_latency = self.average_commit_latency_ms
        rtf_text = "n/a" if rtf is None else f"{rtf:.2f}"
        commit_text = "n/a" if commit_latency is None else f"{commit_latency:.0f}ms"
        return f"rtf={rtf_text} | commit={commit_text}"
