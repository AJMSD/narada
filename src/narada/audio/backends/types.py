from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendProbe:
    backend: str
    supports_mic_capture: bool
    supports_system_capture: bool
    summary: str
