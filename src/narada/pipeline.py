from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from narada.asr.base import TranscriptSegment


@dataclass(frozen=True)
class CommittedLine:
    text: str
    confidence: float


class ConfidenceGate:
    def __init__(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        self.threshold = threshold
        self._pending: list[TranscriptSegment] = []

    def ingest(self, segments: Sequence[TranscriptSegment]) -> list[CommittedLine]:
        committed: list[CommittedLine] = []
        for segment in segments:
            cleaned = segment.text.strip()
            if not cleaned:
                continue
            if segment.confidence >= self.threshold or segment.is_final:
                committed.append(CommittedLine(text=cleaned, confidence=segment.confidence))
                continue
            self._pending.append(segment)
        return committed

    def drain_pending(self) -> list[CommittedLine]:
        committed = [
            CommittedLine(text=item.text.strip(), confidence=item.confidence)
            for item in self._pending
            if item.text.strip()
        ]
        self._pending.clear()
        return committed
