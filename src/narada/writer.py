from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import TextIO


class TranscriptWriter:
    def __init__(
        self,
        path: Path,
        *,
        fsync_mode: str = "line",
        fsync_lines: int = 20,
        fsync_seconds: float = 1.0,
    ) -> None:
        normalized_mode = fsync_mode.strip().lower()
        if normalized_mode not in {"line", "periodic"}:
            raise ValueError("fsync_mode must be 'line' or 'periodic'.")
        if fsync_lines < 0:
            raise ValueError("fsync_lines must be >= 0.")
        if fsync_seconds < 0:
            raise ValueError("fsync_seconds must be >= 0.")
        if normalized_mode == "periodic" and fsync_lines == 0 and fsync_seconds == 0.0:
            raise ValueError("periodic fsync mode requires fsync_lines > 0 or fsync_seconds > 0.")

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._closed = False
        self._fsync_mode = normalized_mode
        self._fsync_lines = fsync_lines
        self._fsync_seconds = fsync_seconds
        self._lines_since_fsync = 0
        self._last_fsync_monotonic = time.monotonic()

    def _sync_to_disk(self, *, now_monotonic: float | None = None) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._lines_since_fsync = 0
        self._last_fsync_monotonic = now_monotonic if now_monotonic is not None else time.monotonic()

    def append_line(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot append to a closed TranscriptWriter.")
            self._handle.write(f"{cleaned}\n")
            self._handle.flush()
            if self._fsync_mode == "line":
                self._sync_to_disk()
                return

            self._lines_since_fsync += 1
            now_monotonic = time.monotonic()
            should_sync = False
            if self._fsync_lines > 0 and self._lines_since_fsync >= self._fsync_lines:
                should_sync = True
            if (
                self._fsync_seconds > 0.0
                and now_monotonic - self._last_fsync_monotonic >= self._fsync_seconds
            ):
                should_sync = True
            if should_sync:
                self._sync_to_disk(now_monotonic=now_monotonic)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._sync_to_disk()
            self._handle.close()
            self._closed = True

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
