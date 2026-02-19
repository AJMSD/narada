from __future__ import annotations

import json
import platform
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

EndpointType = Literal["input", "output", "loopback", "monitor"]
DeviceType = Literal["input", "output", "loopback", "monitor", "input/output"]
SystemDeviceType = Literal["output", "loopback", "monitor"]
DEVICE_TYPES: tuple[EndpointType, ...] = ("input", "output", "loopback", "monitor")

_DISPLAY_TYPE_ORDER: dict[DeviceType, int] = {
    "input": 0,
    "output": 1,
    "input/output": 2,
    "loopback": 3,
    "monitor": 4,
}

_OUTPUT_ENDPOINT_TYPES: tuple[SystemDeviceType, ...] = ("output", "loopback", "monitor")

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
    input_device_id: int | None = None
    system_device_id: int | None = None
    system_device_type: SystemDeviceType | None = None


def _output_device_type(name: str) -> SystemDeviceType:
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


def _bridge_key(name: str) -> str:
    canonical = _canonical_device_name(name)
    parenthetical = re.search(r"\(([^()]*)\)\s*$", canonical)
    anchor = parenthetical.group(1) if parenthetical else canonical
    normalized = re.sub(
        r"\b(microphone|mic|headphones?|speakers?|headset|input|output|device)\b",
        " ",
        anchor,
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = _compact_whitespace(normalized)
    if normalized:
        return normalized
    return canonical


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
    return (device.id, _DISPLAY_TYPE_ORDER[device.type])


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
                    input_device_id=idx,
                )
            )
        if output_channels > 0:
            endpoint_type = _output_device_type(name)
            devices.append(
                AudioDevice(
                    id=idx,
                    name=name,
                    type=endpoint_type,
                    is_default=idx == default_output_idx,
                    hostapi=hostapi_name,
                    system_device_id=idx,
                    system_device_type=endpoint_type,
                )
            )

    devices.sort(key=_device_sort_key)
    return devices


def _dedupe_endpoints(
    devices: Sequence[AudioDevice],
    *,
    os_name: str,
) -> list[AudioDevice]:
    deduped_by_key: dict[tuple[str, DeviceType], AudioDevice] = {}

    for device in devices:
        if _is_alias_or_placeholder_name(device.name, os_name=os_name):
            continue
        key = (_canonical_device_name(device.name), device.type)
        existing = deduped_by_key.get(key)
        if existing is None:
            deduped_by_key[key] = device
            continue
        if _device_preference_key(device, os_name=os_name) < _device_preference_key(
            existing,
            os_name=os_name,
        ):
            deduped_by_key[key] = device

    if deduped_by_key:
        deduped = list(deduped_by_key.values())
        deduped.sort(key=_device_sort_key)
        return deduped

    fallback = list(devices)
    fallback.sort(key=_device_sort_key)
    return fallback


def _choose_logical_name(input_name: str, output_name: str) -> str:
    candidates = [input_name, output_name]
    filtered = [name for name in candidates if "@system32" not in name.lower()]
    preferred = filtered if filtered else candidates
    preferred.sort(key=lambda item: (len(item), item.lower()))
    return preferred[0]


def _build_logical_devices(endpoints: Sequence[AudioDevice]) -> list[AudioDevice]:
    inputs: dict[str, list[AudioDevice]] = {}
    outputs: dict[str, list[AudioDevice]] = {}

    for endpoint in endpoints:
        key = _bridge_key(endpoint.name)
        if endpoint.type == "input":
            inputs.setdefault(key, []).append(endpoint)
        elif endpoint.type in _OUTPUT_ENDPOINT_TYPES:
            outputs.setdefault(key, []).append(endpoint)

    paired_input_ids: set[int] = set()
    paired_output_ids: set[int] = set()
    logical: list[AudioDevice] = []

    for key in sorted(set(inputs) | set(outputs)):
        input_candidates = inputs.get(key, [])
        output_candidates = outputs.get(key, [])
        if len(input_candidates) != 1 or len(output_candidates) != 1:
            continue

        input_endpoint = input_candidates[0]
        output_endpoint = output_candidates[0]
        paired_input_ids.add(input_endpoint.id)
        paired_output_ids.add(output_endpoint.id)

        logical.append(
            AudioDevice(
                id=min(input_endpoint.id, output_endpoint.id),
                name=_choose_logical_name(input_endpoint.name, output_endpoint.name),
                type="input/output",
                is_default=input_endpoint.is_default or output_endpoint.is_default,
                input_device_id=input_endpoint.input_device_id,
                system_device_id=output_endpoint.system_device_id,
                system_device_type=output_endpoint.system_device_type,
            )
        )

    for endpoint in endpoints:
        if endpoint.type == "input" and endpoint.id in paired_input_ids:
            continue
        if endpoint.type in _OUTPUT_ENDPOINT_TYPES and endpoint.id in paired_output_ids:
            continue

        if endpoint.type == "input":
            logical_type: DeviceType = "input"
        else:
            logical_type = "output"

        logical.append(
            AudioDevice(
                id=endpoint.id,
                name=endpoint.name,
                type=logical_type,
                is_default=endpoint.is_default,
                input_device_id=endpoint.input_device_id,
                system_device_id=endpoint.system_device_id,
                system_device_type=endpoint.system_device_type,
            )
        )

    if logical:
        logical.sort(key=_device_sort_key)
        return logical

    fallback = list(endpoints)
    fallback.sort(key=_device_sort_key)
    return fallback


def curate_devices(
    devices: Sequence[AudioDevice],
    *,
    os_name: str | None = None,
) -> list[AudioDevice]:
    normalized_os = (os_name or platform.system()).strip().lower()
    deduped_endpoints = _dedupe_endpoints(devices, os_name=normalized_os)
    return _build_logical_devices(deduped_endpoints)


def enumerate_devices(*, include_all: bool = False) -> list[AudioDevice]:
    raw = _enumerate_raw_devices()
    if include_all:
        return raw
    return curate_devices(raw)


def _matches_type_filter(device: AudioDevice, device_type: EndpointType) -> bool:
    if device_type == "input":
        return device.type in {"input", "input/output"}
    if device_type == "output":
        return device.type in {"output", "input/output", "loopback", "monitor"}
    return device.type == device_type


def filter_devices(
    devices: Sequence[AudioDevice],
    device_type: EndpointType | None = None,
    search: str | None = None,
) -> list[AudioDevice]:
    filtered = [
        device
        for device in devices
        if device_type is None or _matches_type_filter(device, device_type)
    ]
    if search:
        lowered = search.lower().strip()
        filtered = [device for device in filtered if lowered in device.name.lower()]
    return filtered


def _supports_allowed_types(device: AudioDevice, allowed_types: set[EndpointType]) -> bool:
    wants_input = "input" in allowed_types
    wants_system = bool(set(_OUTPUT_ENDPOINT_TYPES).intersection(allowed_types))
    if wants_input and device.input_device_id is not None:
        return True
    if wants_system and device.system_device_id is not None:
        return True
    return False


def _materialize_selection(
    device: AudioDevice,
    *,
    allowed_types: set[EndpointType],
) -> AudioDevice:
    wants_input = "input" in allowed_types
    wants_system = bool(set(_OUTPUT_ENDPOINT_TYPES).intersection(allowed_types))

    if wants_input and device.input_device_id is not None:
        return AudioDevice(
            id=device.input_device_id,
            name=device.name,
            type="input",
            is_default=device.is_default,
            input_device_id=device.input_device_id,
        )

    if wants_system and device.system_device_id is not None:
        system_type: DeviceType = device.system_device_type or "output"
        return AudioDevice(
            id=device.system_device_id,
            name=device.name,
            type=system_type,
            is_default=device.is_default,
            system_device_id=device.system_device_id,
            system_device_type=cast(SystemDeviceType, system_type),
        )

    allowed = ", ".join(sorted(allowed_types))
    raise DeviceResolutionError(
        f"Selected device '{device.name}' does not support required type(s): {allowed}"
    )


def resolve_device(
    selector: str,
    devices: Sequence[AudioDevice],
    allowed_types: set[EndpointType],
) -> AudioDevice:
    candidates = [device for device in devices if _supports_allowed_types(device, allowed_types)]
    cleaned_selector = selector.strip()
    if not cleaned_selector:
        raise DeviceResolutionError("Device selector cannot be empty.")

    if cleaned_selector.isdigit():
        target_id = int(cleaned_selector)
        exact_by_id = [device for device in candidates if device.id == target_id]
        if len(exact_by_id) == 1:
            return _materialize_selection(exact_by_id[0], allowed_types=allowed_types)
        if len(exact_by_id) > 1:
            raise AmbiguousDeviceError(selector, exact_by_id)

    exact_by_name = [
        device for device in candidates if device.name.lower() == cleaned_selector.lower()
    ]
    if len(exact_by_name) == 1:
        return _materialize_selection(exact_by_name[0], allowed_types=allowed_types)
    if len(exact_by_name) > 1:
        raise AmbiguousDeviceError(selector, exact_by_name)

    fuzzy = [device for device in candidates if cleaned_selector.lower() in device.name.lower()]
    if len(fuzzy) == 1:
        return _materialize_selection(fuzzy[0], allowed_types=allowed_types)
    if len(fuzzy) > 1:
        raise AmbiguousDeviceError(selector, fuzzy)

    available = ", ".join(f"{device.id}:{device.name}" for device in candidates)
    raise DeviceResolutionError(
        f"No matching device for selector '{selector}'. Available options: {available}"
    )


def format_devices_table(devices: Sequence[AudioDevice]) -> str:
    if not devices:
        return "No audio devices found."

    header = ("ID", "Name", "Type")
    rows: list[tuple[str, str, str]] = [
        (str(device.id), device.name, device.type) for device in devices
    ]

    id_width = max(len(header[0]), max(len(row[0]) for row in rows))
    name_width = max(len(header[1]), max(len(row[1]) for row in rows))
    type_width = max(len(header[2]), max(len(row[2]) for row in rows))

    lines = [
        f"{header[0]:<{id_width}}  {header[1]:<{name_width}}  {header[2]:<{type_width}}",
        f"{'-' * id_width}  {'-' * name_width}  {'-' * type_width}",
    ]
    for row in rows:
        lines.append(f"{row[0]:<{id_width}}  {row[1]:<{name_width}}  {row[2]:<{type_width}}")
    return "\n".join(lines)


def devices_to_json(devices: Sequence[AudioDevice]) -> str:
    payload = [asdict(device) for device in devices]
    return json.dumps(payload, indent=2)
