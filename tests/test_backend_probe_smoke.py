from narada.audio.backends import linux, macos, windows
from narada.devices import AudioDevice


def test_windows_probe_smoke_reports_mic_and_system_support(
    monkeypatch,
) -> None:
    monkeypatch.setattr("narada.audio.backends.windows.loopback_support_error", lambda: None)
    devices = [
        AudioDevice(id=1, name="Mic", type="input"),
        AudioDevice(id=2, name="Speaker", type="output", hostapi="Windows WASAPI"),
    ]
    probe = windows.probe(devices)
    assert probe.backend == "windows"
    assert probe.supports_mic_capture
    assert probe.supports_system_capture


def test_linux_probe_smoke_reports_monitor_support() -> None:
    devices = [
        AudioDevice(id=1, name="USB Mic", type="input"),
        AudioDevice(id=2, name="PipeWire Monitor", type="monitor"),
    ]
    probe = linux.probe(devices)
    assert probe.backend == "linux"
    assert probe.supports_mic_capture
    assert probe.supports_system_capture


def test_macos_probe_smoke_requires_virtual_loopback_for_system_capture() -> None:
    missing_virtual = [
        AudioDevice(id=1, name="Built-in Mic", type="input"),
        AudioDevice(id=2, name="MacBook Speakers", type="output"),
    ]
    probe_missing = macos.probe(missing_virtual)
    assert probe_missing.backend == "macos"
    assert probe_missing.supports_mic_capture
    assert not probe_missing.supports_system_capture

    with_virtual = missing_virtual + [AudioDevice(id=3, name="BlackHole 2ch", type="output")]
    probe_present = macos.probe(with_virtual)
    assert probe_present.supports_system_capture
