from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from narada.asr.base import TranscriptSegment


@dataclass(frozen=True)
class CommittedLine:
    text: str
    confidence: float


@dataclass(frozen=True)
class AudioChunkWindow:
    pcm_bytes: bytes
    sample_rate_hz: int
    channels: int


class OverlapChunker:
    def __init__(
        self,
        chunk_duration_s: float = 6.0,
        overlap_duration_s: float = 1.5,
        min_flush_duration_s: float = 1.0,
    ) -> None:
        if chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be positive.")
        if overlap_duration_s < 0:
            raise ValueError("overlap_duration_s cannot be negative.")
        if overlap_duration_s >= chunk_duration_s:
            raise ValueError("overlap_duration_s must be smaller than chunk_duration_s.")
        if min_flush_duration_s < 0:
            raise ValueError("min_flush_duration_s cannot be negative.")

        self.chunk_duration_s = chunk_duration_s
        self.overlap_duration_s = overlap_duration_s
        self.min_flush_duration_s = min_flush_duration_s
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._channels: int | None = None

    def ingest(
        self,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        channels: int,
    ) -> list[AudioChunkWindow]:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if channels <= 0:
            raise ValueError("channels must be positive.")
        frame_size = channels * 2
        if len(pcm_bytes) % frame_size != 0:
            raise ValueError("PCM payload must align with channels and 16-bit sample width.")

        emitted: list[AudioChunkWindow] = []
        if self._sample_rate_hz is None or self._channels is None:
            self._sample_rate_hz = sample_rate_hz
            self._channels = channels
        elif self._sample_rate_hz != sample_rate_hz or self._channels != channels:
            emitted.extend(self.flush(force=True))
            self._sample_rate_hz = sample_rate_hz
            self._channels = channels

        self._buffer.extend(pcm_bytes)
        emitted.extend(self._drain_ready_windows())
        return emitted

    def flush(self, force: bool = False) -> list[AudioChunkWindow]:
        if self._sample_rate_hz is None or self._channels is None:
            return []
        if not self._buffer:
            return []
        if not force and self.pending_duration_s() < self.min_flush_duration_s:
            return []

        window = AudioChunkWindow(
            pcm_bytes=bytes(self._buffer),
            sample_rate_hz=self._sample_rate_hz,
            channels=self._channels,
        )
        self._buffer.clear()
        return [window]

    def pending_duration_s(self) -> float:
        if self._sample_rate_hz is None or self._channels is None:
            return 0.0
        bytes_per_second = self._sample_rate_hz * self._channels * 2
        if bytes_per_second == 0:
            return 0.0
        return len(self._buffer) / bytes_per_second

    def _drain_ready_windows(self) -> list[AudioChunkWindow]:
        if self._sample_rate_hz is None or self._channels is None:
            return []
        chunk_bytes = int(self.chunk_duration_s * self._sample_rate_hz) * self._channels * 2
        overlap_bytes = int(self.overlap_duration_s * self._sample_rate_hz) * self._channels * 2
        stride = chunk_bytes - overlap_bytes
        if stride <= 0:
            raise ValueError("Invalid chunker stride; overlap must be smaller than chunk duration.")

        windows: list[AudioChunkWindow] = []
        while len(self._buffer) >= chunk_bytes:
            window_bytes = bytes(self._buffer[:chunk_bytes])
            windows.append(
                AudioChunkWindow(
                    pcm_bytes=window_bytes,
                    sample_rate_hz=self._sample_rate_hz,
                    channels=self._channels,
                )
            )
            del self._buffer[:stride]
        return windows


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


class StabilizedConfidenceGate:
    def __init__(
        self,
        threshold: float,
        holdback_windows: int = 1,
        recent_window_size: int = 256,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if holdback_windows < 0:
            raise ValueError("holdback_windows must be non-negative.")
        if recent_window_size <= 0:
            raise ValueError("recent_window_size must be positive.")
        self.threshold = threshold
        self.holdback_windows = holdback_windows
        self._window_holdback: deque[list[CommittedLine]] = deque()
        self._pending_low_conf: list[TranscriptSegment] = []
        self._recent_emitted: deque[str] = deque(maxlen=recent_window_size)

    @staticmethod
    def _normalize_for_dedupe(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    def _dedupe(self, lines: Sequence[CommittedLine]) -> list[CommittedLine]:
        committed: list[CommittedLine] = []
        for line in lines:
            normalized = self._normalize_for_dedupe(line.text)
            if not normalized:
                continue
            if normalized in self._recent_emitted:
                continue
            self._recent_emitted.append(normalized)
            committed.append(line)
        return committed

    def _emit_ready_windows(self) -> list[CommittedLine]:
        committed: list[CommittedLine] = []
        while len(self._window_holdback) > self.holdback_windows:
            window_lines = self._window_holdback.popleft()
            committed.extend(self._dedupe(window_lines))
        return committed

    def _emit_all_windows(self) -> list[CommittedLine]:
        committed: list[CommittedLine] = []
        while self._window_holdback:
            committed.extend(self._dedupe(self._window_holdback.popleft()))
        return committed

    def ingest(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        is_final_window: bool = False,
    ) -> list[CommittedLine]:
        current_window: list[CommittedLine] = []
        for segment in segments:
            cleaned = segment.text.strip()
            if not cleaned:
                continue
            if segment.confidence >= self.threshold or segment.is_final:
                current_window.append(CommittedLine(text=cleaned, confidence=segment.confidence))
                continue
            self._pending_low_conf.append(segment)
        self._window_holdback.append(current_window)
        if is_final_window:
            return self._emit_all_windows()
        return self._emit_ready_windows()

    def drain_pending(self, *, force_low_conf: bool = False) -> list[CommittedLine]:
        committed = self._emit_all_windows()
        if force_low_conf:
            committed.extend(
                self._dedupe(
                    [
                        CommittedLine(text=item.text.strip(), confidence=item.confidence)
                        for item in self._pending_low_conf
                        if item.text.strip()
                    ]
                )
            )
            self._pending_low_conf.clear()
        return committed
