from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice, DeviceResolutionError


def probe(devices: Sequence[AudioDevice]) -> BackendProbe:
    has_input = any(device.type == "input" for device in devices)
    has_wasapi_output = any(
        device.type in {"loopback", "monitor", "output"}
        and "wasapi" in (device.hostapi or "").lower()
        for device in devices
    )
    loopback_issue = loopback_support_error()
    supports_system_capture = has_wasapi_output and loopback_issue is None
    if supports_system_capture:
        summary = "System capture may use WASAPI loopback devices."
    elif not has_wasapi_output:
        summary = "No WASAPI output or loopback devices detected for system capture."
    else:
        summary = loopback_issue or "WASAPI loopback configuration is unavailable."
    return BackendProbe(
        backend="windows",
        supports_mic_capture=has_input,
        supports_system_capture=supports_system_capture,
        summary=summary,
    )


def resolve_system_capture_device(
    selected_device: AudioDevice, devices: Sequence[AudioDevice]
) -> AudioDevice:
    if selected_device.type not in {"output", "loopback", "monitor"}:
        raise DeviceResolutionError(
            f"Windows system capture requires output/loopback device, got: {selected_device.type}"
        )
    candidates = [
        item
        for item in devices
        if item.id == selected_device.id and item.type in {"output", "loopback", "monitor"}
    ]
    if not candidates:
        raise DeviceResolutionError(f"Selected device {selected_device.id} is not available.")
    resolved = candidates[0]
    hostapi = resolved.hostapi or "unknown"
    if "wasapi" not in hostapi.lower():
        raise DeviceResolutionError(
            "Windows system capture requires a WASAPI output device. "
            f"Selected '{resolved.name}' uses host API '{hostapi}'."
        )
    return resolved


def loopback_support_error() -> str | None:
    try:
        import sounddevice as sd
    except ImportError:
        return "sounddevice is required for live audio capture."

    wasapi_class = getattr(sd, "WasapiSettings", None)
    version = getattr(sd, "__version__", "unknown")
    if wasapi_class is None:
        return (
            f"Installed sounddevice ({version}) does not expose WasapiSettings; "
            "WASAPI loopback capture is unavailable."
        )
    try:
        parameters = inspect.signature(wasapi_class).parameters
    except (TypeError, ValueError):
        return (
            f"Installed sounddevice ({version}) does not expose a compatible "
            "WasapiSettings signature for loopback capture."
        )
    if "loopback" not in parameters:
        return (
            f"Installed sounddevice ({version}) does not support "
            "WasapiSettings(loopback=True). Use a compatible build for system mode "
            "or select 'Stereo Mix' if your driver provides it."
        )
    return None


def build_loopback_settings() -> Any | None:
    support_error = loopback_support_error()
    if support_error is not None:
        return None

    import sounddevice as sd

    wasapi_class = getattr(sd, "WasapiSettings", None)
    if wasapi_class is None:
        return None
    try:
        parameters = inspect.signature(wasapi_class).parameters
    except (TypeError, ValueError):
        return None
    kwargs: dict[str, Any] = {"loopback": True}
    if "auto_convert" in parameters:
        kwargs["auto_convert"] = True
    try:
        return wasapi_class(**kwargs)
    except TypeError:
        return None
