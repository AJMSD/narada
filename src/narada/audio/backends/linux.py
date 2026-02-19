from __future__ import annotations

from collections.abc import Sequence

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice


def probe(devices: Sequence[AudioDevice]) -> BackendProbe:
    has_input = any(device.type == "input" for device in devices)
    has_monitor = any(device.type in {"monitor", "loopback"} for device in devices)
    has_output = any(device.type == "output" for device in devices)
    supports_system = has_monitor or has_output
    if supports_system:
        summary = "System capture may use PulseAudio/PipeWire monitor sources."
    else:
        summary = "No monitor-capable source detected for system capture."
    return BackendProbe(
        backend="linux",
        supports_mic_capture=has_input,
        supports_system_capture=supports_system,
        summary=summary,
    )
