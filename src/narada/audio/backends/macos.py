from __future__ import annotations

from collections.abc import Sequence

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice


def probe(devices: Sequence[AudioDevice]) -> BackendProbe:
    has_input = any(device.type == "input" for device in devices)
    has_blackhole = any("blackhole" in device.name.lower() for device in devices)
    has_loopback = any(device.type in {"monitor", "loopback"} for device in devices)
    supports_system = has_blackhole or has_loopback
    if supports_system:
        summary = "System capture devices detected (virtual loopback available)."
    else:
        summary = "Install and route audio through a virtual device such as BlackHole."
    return BackendProbe(
        backend="macos",
        supports_mic_capture=has_input,
        supports_system_capture=supports_system,
        summary=summary,
    )
