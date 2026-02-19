from __future__ import annotations

import re
from collections.abc import Sequence

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice, DeviceResolutionError


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


def _tokenize(name: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", name.lower()) if len(token) > 2}


def resolve_system_capture_device(
    selected_device: AudioDevice, devices: Sequence[AudioDevice]
) -> AudioDevice:
    if selected_device.type in {"monitor", "loopback"}:
        return selected_device
    if selected_device.type != "output":
        raise DeviceResolutionError(
            f"Linux system capture requires output or monitor device, got: {selected_device.type}"
        )

    monitors = [item for item in devices if item.type in {"monitor", "loopback"}]
    if not monitors:
        raise DeviceResolutionError(
            "No monitor/loopback source found. Configure PulseAudio/PipeWire monitor source."
        )

    selected_tokens = _tokenize(selected_device.name)
    scored: list[tuple[int, AudioDevice]] = []
    for monitor in monitors:
        monitor_tokens = _tokenize(monitor.name)
        score = len(selected_tokens & monitor_tokens)
        scored.append((score, monitor))

    scored.sort(key=lambda item: (item[0], -item[1].id), reverse=True)
    best_score, best = scored[0]
    if best_score == 0:
        raise DeviceResolutionError(
            f"Could not map output '{selected_device.name}' to a monitor source. "
            "Select a monitor/loopback device directly using --system."
        )
    return best
