from __future__ import annotations

from collections.abc import Sequence

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice


def probe(devices: Sequence[AudioDevice]) -> BackendProbe:
    has_input = any(device.type == "input" for device in devices)
    has_output = any(device.type in {"loopback", "monitor", "output"} for device in devices)
    if has_output:
        summary = "System capture may use WASAPI loopback devices."
    else:
        summary = "No output or loopback devices detected for system capture."
    return BackendProbe(
        backend="windows",
        supports_mic_capture=has_input,
        supports_system_capture=has_output,
        summary=summary,
    )
