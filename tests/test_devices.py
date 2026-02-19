import pytest

import narada.devices as devices_module
from narada.devices import (
    AmbiguousDeviceError,
    AudioDevice,
    DeviceResolutionError,
    curate_devices,
    enumerate_devices,
    resolve_device,
)


@pytest.fixture
def sample_devices() -> list[AudioDevice]:
    return [
        AudioDevice(id=1, name="Blue Yeti", type="input"),
        AudioDevice(id=2, name="Speakers (loopback)", type="loopback"),
        AudioDevice(id=3, name="Yeti Stereo", type="input"),
    ]


def test_resolve_by_id(sample_devices: list[AudioDevice]) -> None:
    resolved = resolve_device("2", sample_devices, {"loopback", "output", "monitor"})
    assert resolved.name == "Speakers (loopback)"


def test_resolve_by_exact_name(sample_devices: list[AudioDevice]) -> None:
    resolved = resolve_device("Blue Yeti", sample_devices, {"input"})
    assert resolved.id == 1


def test_fuzzy_ambiguity_raises_error(sample_devices: list[AudioDevice]) -> None:
    with pytest.raises(AmbiguousDeviceError):
        resolve_device("yeti", sample_devices, {"input"})


def test_no_match_raises_error(sample_devices: list[AudioDevice]) -> None:
    with pytest.raises(DeviceResolutionError):
        resolve_device("missing", sample_devices, {"input"})


def test_curate_prefers_default_for_duplicate_names() -> None:
    devices = [
        AudioDevice(id=10, name="Microphone (USB)", type="input", hostapi="MME"),
        AudioDevice(
            id=5,
            name="Microphone (USB)",
            type="input",
            is_default=True,
            hostapi="Windows WASAPI",
        ),
    ]
    curated = curate_devices(devices, os_name="windows")
    assert len(curated) == 1
    assert curated[0].id == 5
    assert curated[0].is_default


def test_curate_prefers_hostapi_priority_when_defaults_match() -> None:
    devices = [
        AudioDevice(id=11, name="Speakers (Realtek)", type="output", hostapi="MME"),
        AudioDevice(id=21, name="Speakers (Realtek)", type="output", hostapi="Windows WASAPI"),
    ]
    curated = curate_devices(devices, os_name="windows")
    assert len(curated) == 1
    assert curated[0].id == 21


def test_curate_filters_windows_alias_devices() -> None:
    devices = [
        AudioDevice(id=0, name="Microsoft Sound Mapper - Input", type="input", hostapi="MME"),
        AudioDevice(id=1, name="Primary Sound Driver", type="output", hostapi="MME"),
        AudioDevice(id=2, name="Microphone (USB)", type="input", hostapi="Windows WASAPI"),
    ]
    curated = curate_devices(devices, os_name="windows")
    assert [item.id for item in curated] == [2]


def test_curate_falls_back_to_raw_when_all_entries_filtered() -> None:
    devices = [
        AudioDevice(id=0, name="Microsoft Sound Mapper - Input", type="input", hostapi="MME"),
        AudioDevice(id=1, name="Primary Sound Driver", type="output", hostapi="MME"),
    ]
    curated = curate_devices(devices, os_name="windows")
    assert [item.id for item in curated] == [0, 1]


def test_enumerate_devices_include_all_returns_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = [
        AudioDevice(id=0, name="Microsoft Sound Mapper - Input", type="input", hostapi="MME"),
        AudioDevice(id=2, name="Microphone (USB)", type="input", hostapi="Windows WASAPI"),
        AudioDevice(id=3, name="Microphone (USB)", type="input", hostapi="MME"),
    ]
    monkeypatch.setattr(devices_module, "_enumerate_raw_devices", lambda: raw)

    curated = enumerate_devices()
    all_items = enumerate_devices(include_all=True)

    assert [item.id for item in curated] == [2]
    assert [item.id for item in all_items] == [0, 2, 3]
