from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from narada.asr.base import TranscriptSegment


@dataclass(frozen=True)
class SpoolRecord:
    start_byte: int
    end_byte: int
    sample_rate_hz: int
    channels: int

    @property
    def byte_length(self) -> int:
        return self.end_byte - self.start_byte

    @property
    def duration_s(self) -> float:
        bytes_per_second = self.sample_rate_hz * self.channels * 2
        if bytes_per_second <= 0:
            return 0.0
        return self.byte_length / bytes_per_second


@dataclass(frozen=True)
class AsrTask:
    task_id: int
    start_byte: int
    end_byte: int
    sample_rate_hz: int
    channels: int
    is_final: bool
    label: str
    created_monotonic: float

    @property
    def audio_seconds(self) -> float:
        bytes_per_second = self.sample_rate_hz * self.channels * 2
        if bytes_per_second <= 0:
            return 0.0
        return (self.end_byte - self.start_byte) / bytes_per_second


@dataclass(frozen=True)
class AsrResult:
    task: AsrTask
    segments: tuple[TranscriptSegment, ...]
    audio_seconds: float
    processing_seconds: float
    error: str | None = None


class SessionSpool:
    def __init__(
        self,
        *,
        base_dir: Path,
        prefix: str = "narada-spool",
        flush_interval_seconds: float = 0.25,
        flush_bytes: int = 65536,
    ) -> None:
        if flush_interval_seconds < 0.0:
            raise ValueError("flush_interval_seconds must be >= 0.0.")
        if flush_bytes < 0:
            raise ValueError("flush_bytes must be >= 0.")
        session_dir = tempfile.mkdtemp(prefix=f"{prefix}-", dir=str(base_dir))
        self.directory = Path(session_dir)
        self.data_path = self.directory / "audio.pcm16le"
        self.index_path = self.directory / "audio.idx"
        self._data_handle = self.data_path.open("wb")
        self._index_handle = self.index_path.open("w", encoding="utf-8", newline="\n")
        self._index_handle.write("start_byte\tend_byte\tsample_rate_hz\tchannels\n")
        self._index_handle.flush()
        self._cursor = 0
        self._records: list[SpoolRecord] = []
        self._lock = threading.Lock()
        self._closed = False
        self._flush_interval_seconds = flush_interval_seconds
        self._flush_bytes = flush_bytes
        self._bytes_since_flush = 0
        self._last_flush_monotonic = time.monotonic()

    def _flush_pending_handles(self, *, now_monotonic: float | None = None) -> None:
        self._data_handle.flush()
        self._index_handle.flush()
        self._bytes_since_flush = 0
        self._last_flush_monotonic = (
            now_monotonic if now_monotonic is not None else time.monotonic()
        )

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._cursor

    def records_snapshot(self) -> tuple[SpoolRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def append_frame(self, *, pcm_bytes: bytes, sample_rate_hz: int, channels: int) -> SpoolRecord:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if channels <= 0:
            raise ValueError("channels must be positive.")
        frame_bytes = channels * 2
        if frame_bytes <= 0 or len(pcm_bytes) % frame_bytes != 0:
            raise ValueError("PCM payload must align to channels and 16-bit sample width.")

        with self._lock:
            if self._closed:
                raise RuntimeError("SessionSpool is closed.")
            start = self._cursor
            self._data_handle.write(pcm_bytes)
            end = start + len(pcm_bytes)
            self._cursor = end
            record = SpoolRecord(
                start_byte=start,
                end_byte=end,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            )
            self._records.append(record)
            self._index_handle.write(f"{start}\t{end}\t{sample_rate_hz}\t{channels}\n")
            self._bytes_since_flush += len(pcm_bytes)
            now_monotonic = time.monotonic()
            if self._flush_interval_seconds == 0.0 and self._flush_bytes == 0:
                self._flush_pending_handles(now_monotonic=now_monotonic)
            else:
                should_flush = False
                if self._flush_bytes > 0 and self._bytes_since_flush >= self._flush_bytes:
                    should_flush = True
                if (
                    self._flush_interval_seconds > 0.0
                    and now_monotonic - self._last_flush_monotonic >= self._flush_interval_seconds
                ):
                    should_flush = True
                if should_flush:
                    self._flush_pending_handles(now_monotonic=now_monotonic)
            return record

    def read_range(self, *, start_byte: int, end_byte: int) -> bytes:
        if start_byte < 0:
            raise ValueError("start_byte must be >= 0.")
        if end_byte < start_byte:
            raise ValueError("end_byte must be >= start_byte.")
        with self._lock:
            if end_byte > self._cursor:
                raise ValueError("Requested range exceeds spool size.")
            self._data_handle.flush()
            with self.data_path.open("rb") as handle:
                handle.seek(start_byte)
                return handle.read(end_byte - start_byte)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._data_handle.flush()
            os.fsync(self._data_handle.fileno())
            self._index_handle.flush()
            os.fsync(self._index_handle.fileno())
            self._data_handle.close()
            self._index_handle.close()
            self._closed = True

    def cleanup(self, *, keep_files: bool) -> None:
        self.close()
        if keep_files:
            return
        shutil.rmtree(self.directory, ignore_errors=True)


class IntervalPlanner:
    def __init__(self, *, interval_seconds: float, overlap_seconds: float) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive.")
        if overlap_seconds < 0.0:
            raise ValueError("overlap_seconds must be non-negative.")
        if overlap_seconds >= interval_seconds:
            raise ValueError("overlap_seconds must be smaller than interval_seconds.")

        self.interval_seconds = interval_seconds
        self.overlap_seconds = overlap_seconds
        self._segment_start_byte: int | None = None
        self._segment_end_byte: int | None = None
        self._sample_rate_hz: int | None = None
        self._channels: int | None = None
        self._chunk_bytes: int | None = None
        self._stride_bytes: int | None = None
        self._bytes_per_second: int | None = None
        self._next_window_start_rel_byte = 0
        self._pending_forced_tasks: deque[AsrTask] = deque()
        self._next_task_id = 1

    def _derive_window_bytes(self, *, sample_rate_hz: int, channels: int) -> tuple[int, int, int]:
        frame_bytes = channels * 2
        chunk_frames = max(1, int(round(self.interval_seconds * sample_rate_hz)))
        overlap_frames = max(0, int(round(self.overlap_seconds * sample_rate_hz)))
        if overlap_frames >= chunk_frames:
            overlap_frames = chunk_frames - 1
        chunk_bytes = chunk_frames * frame_bytes
        stride_bytes = (chunk_frames - overlap_frames) * frame_bytes
        bytes_per_second = sample_rate_hz * frame_bytes
        return chunk_bytes, stride_bytes, bytes_per_second

    def _begin_segment(self, record: SpoolRecord) -> None:
        chunk_bytes, stride_bytes, bytes_per_second = self._derive_window_bytes(
            sample_rate_hz=record.sample_rate_hz,
            channels=record.channels,
        )
        self._segment_start_byte = record.start_byte
        self._segment_end_byte = record.end_byte
        self._sample_rate_hz = record.sample_rate_hz
        self._channels = record.channels
        self._chunk_bytes = chunk_bytes
        self._stride_bytes = stride_bytes
        self._bytes_per_second = bytes_per_second
        self._next_window_start_rel_byte = 0

    def _new_task(
        self,
        *,
        start_byte: int,
        end_byte: int,
        is_final: bool,
        label: str,
        created_monotonic: float,
    ) -> AsrTask:
        if self._sample_rate_hz is None or self._channels is None:
            raise RuntimeError("IntervalPlanner segment format is not initialized.")
        task = AsrTask(
            task_id=self._next_task_id,
            start_byte=start_byte,
            end_byte=end_byte,
            sample_rate_hz=self._sample_rate_hz,
            channels=self._channels,
            is_final=is_final,
            label=label,
            created_monotonic=created_monotonic,
        )
        self._next_task_id += 1
        return task

    def _build_tail_task(self, *, label: str, created_monotonic: float) -> AsrTask | None:
        if self._segment_start_byte is None or self._segment_end_byte is None:
            return None
        start_byte = self._segment_start_byte + self._next_window_start_rel_byte
        end_byte = self._segment_end_byte
        if start_byte >= end_byte:
            return None
        return self._new_task(
            start_byte=start_byte,
            end_byte=end_byte,
            is_final=True,
            label=label,
            created_monotonic=created_monotonic,
        )

    def ingest_record(self, record: SpoolRecord, *, now_monotonic: float | None = None) -> None:
        created_at = now_monotonic if now_monotonic is not None else time.monotonic()
        if self._segment_start_byte is None:
            self._begin_segment(record)
            return

        assert self._sample_rate_hz is not None and self._channels is not None
        if record.sample_rate_hz != self._sample_rate_hz or record.channels != self._channels:
            tail = self._build_tail_task(label="format-tail", created_monotonic=created_at)
            if tail is not None:
                self._pending_forced_tasks.append(tail)
            self._begin_segment(record)
            return

        self._segment_end_byte = record.end_byte

    def pop_next_ready_task(self, *, now_monotonic: float | None = None) -> AsrTask | None:
        created_at = now_monotonic if now_monotonic is not None else time.monotonic()
        if self._pending_forced_tasks:
            return self._pending_forced_tasks.popleft()
        if (
            self._segment_start_byte is None
            or self._segment_end_byte is None
            or self._chunk_bytes is None
            or self._stride_bytes is None
        ):
            return None
        available_bytes = self._segment_end_byte - self._segment_start_byte
        if self._next_window_start_rel_byte + self._chunk_bytes > available_bytes:
            return None
        start_byte = self._segment_start_byte + self._next_window_start_rel_byte
        end_byte = start_byte + self._chunk_bytes
        task = self._new_task(
            start_byte=start_byte,
            end_byte=end_byte,
            is_final=False,
            label="interval",
            created_monotonic=created_at,
        )
        self._next_window_start_rel_byte += self._stride_bytes
        return task

    def pending_backlog_seconds(self) -> float:
        if (
            self._segment_start_byte is None
            or self._segment_end_byte is None
            or self._bytes_per_second is None
            or self._bytes_per_second <= 0
        ):
            return 0.0
        pending_bytes = self._segment_end_byte - (
            self._segment_start_byte + self._next_window_start_rel_byte
        )
        if pending_bytes <= 0:
            return 0.0
        return pending_bytes / self._bytes_per_second

    def build_final_tasks(self, *, now_monotonic: float | None = None) -> list[AsrTask]:
        created_at = now_monotonic if now_monotonic is not None else time.monotonic()
        tasks = list(self._pending_forced_tasks)
        self._pending_forced_tasks.clear()
        tail = self._build_tail_task(label="final-tail", created_monotonic=created_at)
        if tail is not None:
            tasks.append(tail)
        return tasks
