import pytest

import narada.devices as devices_module
from narada.devices import (
    AmbiguousDeviceError,
    AudioDevice,
    DeviceResolutionError,
    curate_devices,
    enumerate_devices,
    filter_devices,
    format_devices_table,
    resolve_device,
)


@pytest.fixture
def sample_devices() -> list[AudioDevice]:
    return [
        AudioDevice(id=1, name="Blue Yeti", type="input", input_device_id=11),
        AudioDevice(
            id=2,
            name="Conference Headset",
            type="input/output",
            input_device_id=22,
            system_device_id=32,
            system_device_type="output",
        ),
        AudioDevice(
            id=3,
            name="Loopback Cable",
            type="output",
            system_device_id=43,
            system_device_type="loopback",
        ),
        AudioDevice(id=4, name="Yeti Stereo", type="input", input_device_id=14),
    ]


def test_resolve_combo_device_for_input_chooses_input_endpoint(
    sample_devices: list[AudioDevice],
) -> None:
    resolved = resolve_device("2", sample_devices, {"input"})
    assert resolved.id == 22
    assert resolved.type == "input"


def test_resolve_combo_device_for_output_chooses_system_endpoint(
    sample_devices: list[AudioDevice],
) -> None:
    resolved = resolve_device(
        "Conference Headset", sample_devices, {"output", "loopback", "monitor"}
    )
    assert resolved.id == 32
    assert resolved.type == "output"


def test_resolve_output_only_for_input_rejected(sample_devices: list[AudioDevice]) -> None:
    with pytest.raises(DeviceResolutionError):
        resolve_device("Loopback Cable", sample_devices, {"input"})


def test_fuzzy_ambiguity_raises_error(sample_devices: list[AudioDevice]) -> None:
    with pytest.raises(AmbiguousDeviceError):
        resolve_device("yeti", sample_devices, {"input"})


def test_curate_dedupes_cross_api_and_combines_input_output() -> None:
    raw = [
        AudioDevice(
            id=10,
            name="Microphone (USB Device)",
            type="input",
            hostapi="MME",
            input_device_id=10,
        ),
        AudioDevice(
            id=11,
            name="Microphone (USB Device)",
            type="input",
            is_default=True,
            hostapi="Windows WASAPI",
            input_device_id=11,
        ),
        AudioDevice(
            id=20,
            name="Speakers (USB Device)",
            type="output",
            hostapi="MME",
            system_device_id=20,
            system_device_type="output",
        ),
        AudioDevice(
            id=21,
            name="Speakers (USB Device)",
            type="output",
            is_default=True,
            hostapi="Windows WASAPI",
            system_device_id=21,
            system_device_type="output",
        ),
    ]
    curated = curate_devices(raw, os_name="windows")
    assert len(curated) == 1
    device = curated[0]
    assert device.type == "input/output"
    assert device.id == 11
    assert device.input_device_id == 11
    assert device.system_device_id == 21


def test_curate_wasapi_beats_mme_default_on_windows() -> None:
    """WASAPI endpoint must win deduplication even when the MME endpoint is the OS default.

    This is the real-world regression: sounddevice reports the Windows system
    default as an MME device index.  The curated list must expose the WASAPI
    ID so that WasapiSettings(loopback=True) is valid for system capture.
    """
    raw = [
        AudioDevice(
            id=7,
            name="Speakers (2- Realtek(R) Audio)",
            type="output",
            is_default=True,  # Windows system default — but it is MME
            hostapi="MME",
            system_device_id=7,
            system_device_type="output",
        ),
        AudioDevice(
            id=19,
            name="Speakers (2- Realtek(R) Audio)",
            type="output",
            hostapi="Windows DirectSound",
            system_device_id=19,
            system_device_type="output",
        ),
        AudioDevice(
            id=22,
            name="Speakers (2- Realtek(R) Audio)",
            type="output",
            hostapi="Windows WASAPI",
            system_device_id=22,
            system_device_type="output",
        ),
    ]
    curated = curate_devices(raw, os_name="windows")
    assert len(curated) == 1
    assert curated[0].id == 22
    assert curated[0].system_device_id == 22
    assert curated[0].type == "output"


def test_curate_wasapi_default_still_wins_over_mme_default() -> None:
    """When both MME and WASAPI endpoints are marked default, WASAPI still wins."""
    raw = [
        AudioDevice(
            id=5,
            name="Headphones (Realtek Audio)",
            type="output",
            is_default=True,
            hostapi="MME",
            system_device_id=5,
            system_device_type="output",
        ),
        AudioDevice(
            id=20,
            name="Headphones (Realtek Audio)",
            type="output",
            is_default=True,
            hostapi="Windows WASAPI",
            system_device_id=20,
            system_device_type="output",
        ),
    ]
    curated = curate_devices(raw, os_name="windows")
    assert len(curated) == 1
    assert curated[0].id == 20


def test_curate_prefers_wasapi_over_directsound_over_mme() -> None:
    """Host API ranking: WASAPI (0) > DirectSound (2) > MME (4) on Windows."""
    raw = [
        AudioDevice(
            id=1,
            name="Microphone (Fifine)",
            type="input",
            hostapi="MME",
            input_device_id=1,
        ),
        AudioDevice(
            id=15,
            name="Microphone (Fifine)",
            type="input",
            hostapi="Windows DirectSound",
            input_device_id=15,
        ),
        AudioDevice(
            id=26,
            name="Microphone (Fifine)",
            type="input",
            hostapi="Windows WASAPI",
            input_device_id=26,
        ),
    ]
    curated = curate_devices(raw, os_name="windows")
    assert len(curated) == 1
    assert curated[0].id == 26


def test_curate_is_default_breaks_ties_within_same_api() -> None:
    """When two endpoints share the same host API, the default one is preferred."""
    raw = [
        AudioDevice(
            id=20,
            name="Headphones (Realtek)",
            type="output",
            is_default=False,
            hostapi="Windows WASAPI",
            system_device_id=20,
            system_device_type="output",
        ),
        AudioDevice(
            id=21,
            name="Headphones (Realtek)",
            type="output",
            is_default=True,
            hostapi="Windows WASAPI",
            system_device_id=21,
            system_device_type="output",
        ),
    ]
    curated = curate_devices(raw, os_name="windows")
    assert len(curated) == 1
    assert curated[0].id == 21


def test_curate_does_not_combine_ambiguous_bridge_groups() -> None:
    raw = [
        AudioDevice(
            id=1,
            name="Microphone (Realtek Audio)",
            type="input",
            input_device_id=1,
        ),
        AudioDevice(
            id=2,
            name="Microphone Array (Realtek Audio)",
            type="input",
            input_device_id=2,
        ),
        AudioDevice(
            id=3,
            name="Speakers (Realtek Audio)",
            type="output",
            system_device_id=3,
            system_device_type="output",
        ),
    ]
    curated = curate_devices(raw, os_name="windows")
    assert [item.id for item in curated] == [1, 2, 3]
    assert [item.type for item in curated] == ["input", "input", "output"]


def test_curate_filters_alias_devices_but_keeps_fallback_if_all_filtered() -> None:
    raw = [
        AudioDevice(
            id=0,
            name="Microsoft Sound Mapper - Input",
            type="input",
            hostapi="MME",
            input_device_id=0,
        ),
        AudioDevice(
            id=1,
            name="Primary Sound Driver",
            type="output",
            hostapi="MME",
            system_device_id=1,
            system_device_type="output",
        ),
    ]
    curated = curate_devices(raw, os_name="windows")
    assert [item.id for item in curated] == [0, 1]


def test_enumerate_devices_include_all_returns_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = [
        AudioDevice(
            id=0,
            name="Microphone (USB Device)",
            type="input",
            hostapi="MME",
            input_device_id=0,
        ),
        AudioDevice(
            id=1,
            name="Microphone (USB Device)",
            type="input",
            hostapi="Windows WASAPI",
            input_device_id=1,
        ),
        AudioDevice(
            id=2,
            name="Speakers (USB Device)",
            type="output",
            hostapi="Windows WASAPI",
            system_device_id=2,
            system_device_type="output",
        ),
    ]
    monkeypatch.setattr(devices_module, "_enumerate_raw_devices", lambda: raw)

    curated = enumerate_devices()
    all_items = enumerate_devices(include_all=True)

    assert [item.id for item in curated] == [1]
    assert curated[0].type == "input/output"
    assert curated[0].system_device_id == 2
    assert [item.id for item in all_items] == [0, 1, 2]


def test_filter_devices_includes_input_output_for_input_and_output_filters() -> None:
    devices = [
        AudioDevice(id=1, name="Mic", type="input", input_device_id=11),
        AudioDevice(
            id=2,
            name="Combo",
            type="input/output",
            input_device_id=22,
            system_device_id=32,
            system_device_type="output",
        ),
        AudioDevice(
            id=3, name="Out", type="output", system_device_id=33, system_device_type="output"
        ),
        AudioDevice(
            id=4,
            name="Loop",
            type="loopback",
            system_device_id=44,
            system_device_type="loopback",
        ),
    ]

    input_filtered = filter_devices(devices, device_type="input")
    output_filtered = filter_devices(devices, device_type="output")
    loopback_filtered = filter_devices(devices, device_type="loopback")

    assert [item.id for item in input_filtered] == [1, 2]
    assert [item.id for item in output_filtered] == [2, 3, 4]
    assert [item.id for item in loopback_filtered] == [4]


def test_format_devices_table_has_id_name_type_only() -> None:
    table = format_devices_table(
        [
            AudioDevice(id=2, name="Conference Headset", type="input/output"),
            AudioDevice(id=3, name="Loopback Cable", type="output"),
        ]
    )
    lines = table.splitlines()
    assert lines[0].strip() == "ID  Name                Type"
    assert "Host API" not in table
    assert "Default" not in table
