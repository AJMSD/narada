from __future__ import annotations

from collections.abc import Sequence

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice, DeviceResolutionError


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


def _is_virtual_capture_device(device: AudioDevice) -> bool:
    lowered = device.name.lower()
    return (
        device.type in {"monitor", "loopback"}
        or "blackhole" in lowered
        or "loopback" in lowered
        or "virtual" in lowered
    )


def resolve_system_capture_device(
    selected_device: AudioDevice, devices: Sequence[AudioDevice]
) -> AudioDevice:
    if _is_virtual_capture_device(selected_device):
        return selected_device

    candidates = [item for item in devices if _is_virtual_capture_device(item)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        options = ", ".join(f"{item.id}:{item.name}" for item in candidates)
        raise DeviceResolutionError(
            "Selected macOS system device is not virtual. "
            f"Choose one of these virtual devices: {options}"
        )
    raise DeviceResolutionError(
        "macOS system capture requires a virtual loopback device (for example BlackHole)."
    )
