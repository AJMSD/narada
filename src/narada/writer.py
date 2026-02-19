from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TextIO


class TranscriptWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._closed = False

    def append_line(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot append to a closed TranscriptWriter.")
            self._handle.write(f"{cleaned}\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
