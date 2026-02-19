from __future__ import annotations

import json
import platform
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

DeviceType = Literal["input", "output", "loopback", "monitor"]
DEVICE_TYPES: tuple[DeviceType, ...] = ("input", "output", "loopback", "monitor")

_TYPE_ORDER: dict[DeviceType, int] = {
    "input": 0,
    "output": 1,
    "loopback": 2,
    "monitor": 3,
}

_WINDOWS_ALIAS_NAMES = {
    "microsoft sound mapper - input",
    "microsoft sound mapper - output",
    "primary sound capture driver",
    "primary sound driver",
}

_HOSTAPI_PRIORITY_BY_OS: dict[str, tuple[str, ...]] = {
    "windows": ("wasapi", "asio", "wdm-ks", "directsound", "mme"),
    "linux": ("pipewire", "pulse", "alsa", "jack"),
    "darwin": ("core audio",),
}


class DeviceResolutionError(ValueError):
    pass


class AmbiguousDeviceError(DeviceResolutionError):
    def __init__(self, selector: str, matches: Sequence[AudioDevice]) -> None:
        self.selector = selector
        self.matches = tuple(matches)
        names = ", ".join(f"{device.id}:{device.name}" for device in matches)
        super().__init__(f"Device selector '{selector}' is ambiguous. Matches: {names}")


@dataclass(frozen=True)
class AudioDevice:
    id: int
    name: str
    type: DeviceType
    is_default: bool = False
    hostapi: str | None = None


def _output_device_type(name: str) -> DeviceType:
    lowered = name.lower()
    if "loopback" in lowered:
        return "loopback"
    if "monitor" in lowered:
        return "monitor"
    return "output"


def _compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sanitize_device_name(raw_name: str, fallback_index: int) -> str:
    compact = _compact_whitespace(raw_name)
    if compact:
        return compact
    return f"Device {fallback_index}"


def _canonical_device_name(name: str) -> str:
    normalized = _compact_whitespace(name).lower()
    normalized = re.sub(r"\b(input|output|headphones|speakers)\s+\d+(?=\s*\()", r"\1", normalized)
    if ";(" in normalized and normalized.endswith(")"):
        normalized = normalized.split(";(", 1)[1][:-1].strip()
    return normalized


def _is_alias_or_placeholder_name(name: str, *, os_name: str) -> bool:
    canonical = _canonical_device_name(name)
    if os_name == "windows" and canonical in _WINDOWS_ALIAS_NAMES:
        return True
    if canonical.endswith("()"):
        return True
    return canonical in {"input ()", "output ()", "headphones ()", "speakers ()"}


def _hostapi_rank(hostapi: str | None, *, os_name: str) -> int:
    if hostapi is None:
        return 100
    normalized = hostapi.lower()
    ordered = _HOSTAPI_PRIORITY_BY_OS.get(os_name, ())
    for rank, item in enumerate(ordered):
        if item in normalized:
            return rank
    return len(ordered) + 10


def _device_preference_key(device: AudioDevice, *, os_name: str) -> tuple[int, int, int]:
    return (
        0 if device.is_default else 1,
        _hostapi_rank(device.hostapi, os_name=os_name),
        device.id,
    )


def _device_sort_key(device: AudioDevice) -> tuple[int, int]:
    return (device.id, _TYPE_ORDER[device.type])


def curate_devices(
    devices: Sequence[AudioDevice], *, os_name: str | None = None
) -> list[AudioDevice]:
    normalized_os = (os_name or platform.system()).strip().lower()
    curated_by_key: dict[tuple[str, DeviceType], AudioDevice] = {}

    for device in devices:
        if _is_alias_or_placeholder_name(device.name, os_name=normalized_os):
            continue
        key = (_canonical_device_name(device.name), device.type)
        existing = curated_by_key.get(key)
        if existing is None:
            curated_by_key[key] = device
            continue
        if _device_preference_key(device, os_name=normalized_os) < _device_preference_key(
            existing,
            os_name=normalized_os,
        ):
            curated_by_key[key] = device

    if curated_by_key:
        curated = list(curated_by_key.values())
        curated.sort(key=_device_sort_key)
        return curated

    fallback = list(devices)
    fallback.sort(key=_device_sort_key)
    return fallback


def _query_hostapi_names(sd: Any) -> dict[int, str]:
    try:
        raw_hostapis = sd.query_hostapis()
    except Exception:
        return {}

    if isinstance(raw_hostapis, dict):
        hostapis: Sequence[Any] = [raw_hostapis]
    else:
        hostapis = raw_hostapis

    names: dict[int, str] = {}
    for idx, item in enumerate(hostapis):
        if isinstance(item, dict):
            names[idx] = _compact_whitespace(str(item.get("name", f"Host API {idx}")))
        else:
            names[idx] = _compact_whitespace(str(item))
    return names


def _default_device_indices(sd: Any) -> tuple[int, int]:
    try:
        raw_default = tuple(sd.default.device)
    except Exception:
        return (-1, -1)
    if len(raw_default) < 2:
        return (-1, -1)
    input_idx = int(raw_default[0]) if raw_default[0] is not None else -1
    output_idx = int(raw_default[1]) if raw_default[1] is not None else -1
    return (input_idx, output_idx)


def _enumerate_raw_devices() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except ImportError:
        return []

    raw_devices = sd.query_devices()
    default_input_idx, default_output_idx = _default_device_indices(sd)
    hostapi_names = _query_hostapi_names(sd)
    devices: list[AudioDevice] = []

    for idx, raw in enumerate(raw_devices):
        name = _sanitize_device_name(str(raw.get("name", f"Device {idx}")), idx)
        input_channels = int(raw.get("max_input_channels", 0))
        output_channels = int(raw.get("max_output_channels", 0))
        hostapi_index = int(raw.get("hostapi", -1))
        hostapi_name = hostapi_names.get(hostapi_index)

        if input_channels > 0:
            devices.append(
                AudioDevice(
                    id=idx,
                    name=name,
                    type="input",
                    is_default=idx == default_input_idx,
                    hostapi=hostapi_name,
                )
            )
        if output_channels > 0:
            devices.append(
                AudioDevice(
                    id=idx,
                    name=name,
                    type=_output_device_type(name),
                    is_default=idx == default_output_idx,
                    hostapi=hostapi_name,
                )
            )

    devices.sort(key=_device_sort_key)
    return devices


def enumerate_devices(*, include_all: bool = False) -> list[AudioDevice]:
    raw = _enumerate_raw_devices()
    if include_all:
        return raw
    return curate_devices(raw)


def filter_devices(
    devices: Sequence[AudioDevice],
    device_type: DeviceType | None = None,
    search: str | None = None,
) -> list[AudioDevice]:
    filtered = [device for device in devices if device_type is None or device.type == device_type]
    if search:
        lowered = search.lower().strip()
        filtered = [device for device in filtered if lowered in device.name.lower()]
    return filtered


def resolve_device(
    selector: str,
    devices: Sequence[AudioDevice],
    allowed_types: set[DeviceType],
) -> AudioDevice:
    candidates = [device for device in devices if device.type in allowed_types]
    cleaned_selector = selector.strip()
    if not cleaned_selector:
        raise DeviceResolutionError("Device selector cannot be empty.")

    if cleaned_selector.isdigit():
        target_id = int(cleaned_selector)
        exact_by_id = [device for device in candidates if device.id == target_id]
        if len(exact_by_id) == 1:
            return exact_by_id[0]
        if len(exact_by_id) > 1:
            raise AmbiguousDeviceError(selector, exact_by_id)

    exact_by_name = [
        device for device in candidates if device.name.lower() == cleaned_selector.lower()
    ]
    if len(exact_by_name) == 1:
        return exact_by_name[0]
    if len(exact_by_name) > 1:
        raise AmbiguousDeviceError(selector, exact_by_name)

    fuzzy = [device for device in candidates if cleaned_selector.lower() in device.name.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        raise AmbiguousDeviceError(selector, fuzzy)

    available = ", ".join(f"{device.id}:{device.name}" for device in candidates)
    raise DeviceResolutionError(
        f"No matching device for selector '{selector}'. Available options: {available}"
    )


def format_devices_table(devices: Sequence[AudioDevice]) -> str:
    if not devices:
        return "No audio devices found."

    show_hostapi = any(device.hostapi for device in devices)
    header: tuple[str, ...]
    rows: list[tuple[str, ...]]
    if show_hostapi:
        header = ("ID", "Name", "Type", "Host API", "Default")
        rows = [
            (
                str(device.id),
                device.name,
                device.type,
                device.hostapi or "-",
                "*" if device.is_default else "",
            )
            for device in devices
        ]
    else:
        header = ("ID", "Name", "Type", "Default")
        rows = [
            (str(device.id), device.name, device.type, "*" if device.is_default else "")
            for device in devices
        ]
    widths = [len(header[idx]) for idx in range(len(header))]
    for row in rows:
        for idx, column in enumerate(row):
            widths[idx] = max(widths[idx], len(column))

    lines: list[str] = []
    lines.append("  ".join(f"{column:<{widths[idx]}}" for idx, column in enumerate(header)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(f"{column:<{widths[idx]}}" for idx, column in enumerate(row)))
    return "\n".join(lines)


def devices_to_json(devices: Sequence[AudioDevice]) -> str:
    payload = [asdict(device) for device in devices]
    return json.dumps(payload, indent=2)
