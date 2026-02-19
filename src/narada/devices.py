from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

DeviceType = Literal["input", "output", "loopback", "monitor"]
DEVICE_TYPES: tuple[DeviceType, ...] = ("input", "output", "loopback", "monitor")


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


def _output_device_type(name: str) -> DeviceType:
    lowered = name.lower()
    if "loopback" in lowered:
        return "loopback"
    if "monitor" in lowered:
        return "monitor"
    return "output"


def enumerate_devices() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except ImportError:
        return []

    raw_devices = sd.query_devices()
    default_input_idx, default_output_idx = tuple(sd.default.device)
    devices: list[AudioDevice] = []

    for idx, raw in enumerate(raw_devices):
        name = str(raw.get("name", f"Device {idx}"))
        input_channels = int(raw.get("max_input_channels", 0))
        output_channels = int(raw.get("max_output_channels", 0))

        if input_channels > 0:
            devices.append(
                AudioDevice(
                    id=idx,
                    name=name,
                    type="input",
                    is_default=idx == default_input_idx,
                )
            )
        if output_channels > 0:
            devices.append(
                AudioDevice(
                    id=idx,
                    name=name,
                    type=_output_device_type(name),
                    is_default=idx == default_output_idx,
                )
            )

    devices.sort(key=lambda item: (item.id, item.type))
    return devices


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

    rows: list[tuple[str, str, str, str]] = [
        (str(device.id), device.name, device.type, "*" if device.is_default else "")
        for device in devices
    ]
    header = ("ID", "Name", "Type", "Default")
    widths = [len(header[idx]) for idx in range(4)]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    lines: list[str] = []
    lines.append(
        f"{header[0]:<{widths[0]}}  {header[1]:<{widths[1]}}  {header[2]:<{widths[2]}}  {header[3]}"
    )
    lines.append(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * widths[3]}")
    for row in rows:
        lines.append(
            f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]:<{widths[2]}}  {row[3]}"
        )
    return "\n".join(lines)


def devices_to_json(devices: Sequence[AudioDevice]) -> str:
    payload = [asdict(device) for device in devices]
    return json.dumps(payload, indent=2)
