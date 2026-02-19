from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice, DeviceResolutionError


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


def resolve_system_capture_device(
    selected_device: AudioDevice, devices: Sequence[AudioDevice]
) -> AudioDevice:
    if selected_device.type not in {"output", "loopback", "monitor"}:
        raise DeviceResolutionError(
            f"Windows system capture requires output/loopback device, got: {selected_device.type}"
        )
    available = {item.id for item in devices}
    if selected_device.id not in available:
        raise DeviceResolutionError(f"Selected device {selected_device.id} is not available.")
    return selected_device


def build_loopback_settings() -> Any | None:
    try:
        import sounddevice as sd
    except ImportError:
        return None

    wasapi_class = getattr(sd, "WasapiSettings", None)
    if wasapi_class is None:
        return None
    try:
        return wasapi_class(loopback=True)
    except TypeError:
        return wasapi_class()
