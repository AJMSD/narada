import sys

import pytest

from narada.audio.backends import linux, macos, windows
from narada.devices import AudioDevice, DeviceResolutionError


def test_windows_system_capture_rejects_input_type() -> None:
    device = AudioDevice(id=1, name="Mic", type="input")
    with pytest.raises(DeviceResolutionError):
        windows.resolve_system_capture_device(device, [device])


def test_windows_system_capture_rejects_non_wasapi_output() -> None:
    selected = AudioDevice(id=8, name="Headphones", type="output")
    mme_output = AudioDevice(id=8, name="Headphones", type="output", hostapi="MME")
    with pytest.raises(DeviceResolutionError, match="requires a WASAPI output device"):
        windows.resolve_system_capture_device(selected, [mme_output])


def test_windows_system_capture_accepts_wasapi_output() -> None:
    selected = AudioDevice(id=22, name="Speakers", type="output")
    wasapi_output = AudioDevice(id=22, name="Speakers", type="output", hostapi="Windows WASAPI")
    resolved = windows.resolve_system_capture_device(selected, [wasapi_output])
    assert resolved == wasapi_output


def test_windows_build_loopback_settings_returns_none_without_loopback_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WasapiSettings:
        def __init__(
            self,
            exclusive: bool = False,
            auto_convert: bool = False,
            explicit_sample_format: bool = False,
        ) -> None:
            self.exclusive = exclusive
            self.auto_convert = auto_convert
            self.explicit_sample_format = explicit_sample_format

    fake_sd = type(
        "FakeSoundDevice",
        (),
        {"WasapiSettings": _WasapiSettings, "__version__": "0.5.5"},
    )()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert windows.build_loopback_settings() is None


def test_windows_build_loopback_settings_uses_loopback_and_auto_convert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {}

    class _WasapiSettings:
        def __init__(self, loopback: bool = False, auto_convert: bool = False) -> None:
            captured["loopback"] = loopback
            captured["auto_convert"] = auto_convert

    fake_sd = type(
        "FakeSoundDevice",
        (),
        {"WasapiSettings": _WasapiSettings, "__version__": "0.6.0"},
    )()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    settings = windows.build_loopback_settings()
    assert settings is not None
    assert captured["loopback"] is True
    assert captured["auto_convert"] is True


def test_windows_probe_requires_wasapi_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("narada.audio.backends.windows.loopback_support_error", lambda: None)
    devices = [AudioDevice(id=34, name="Speakers", type="output", hostapi="Windows WDM-KS")]
    probe = windows.probe(devices)
    assert not probe.supports_system_capture
    assert "No WASAPI output" in probe.summary


def test_windows_probe_surfaces_loopback_support_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "narada.audio.backends.windows.loopback_support_error",
        lambda: "Installed sounddevice does not support WasapiSettings(loopback=True).",
    )
    devices = [AudioDevice(id=22, name="Speakers", type="output", hostapi="Windows WASAPI")]
    probe = windows.probe(devices)
    assert not probe.supports_system_capture
    assert "does not support WasapiSettings(loopback=True)" in probe.summary


def test_linux_maps_output_to_monitor_source() -> None:
    output = AudioDevice(id=10, name="Built-in Audio Output", type="output")
    monitor = AudioDevice(id=11, name="Built-in Audio Monitor", type="monitor")
    resolved = linux.resolve_system_capture_device(output, [output, monitor])
    assert resolved == monitor


def test_linux_requires_monitor_source_when_output_selected() -> None:
    output = AudioDevice(id=10, name="Built-in Audio Output", type="output")
    with pytest.raises(DeviceResolutionError):
        linux.resolve_system_capture_device(output, [output])


def test_macos_auto_selects_single_virtual_device() -> None:
    selected = AudioDevice(id=1, name="MacBook Speakers", type="output")
    virtual = AudioDevice(id=2, name="BlackHole 2ch", type="output")
    resolved = macos.resolve_system_capture_device(selected, [selected, virtual])
    assert resolved == virtual


def test_macos_rejects_when_multiple_virtual_candidates_exist() -> None:
    selected = AudioDevice(id=1, name="MacBook Speakers", type="output")
    v1 = AudioDevice(id=2, name="BlackHole 2ch", type="output")
    v2 = AudioDevice(id=3, name="Loopback Audio", type="output")
    with pytest.raises(DeviceResolutionError):
        macos.resolve_system_capture_device(selected, [selected, v1, v2])
