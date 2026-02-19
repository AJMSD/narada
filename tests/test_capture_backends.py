import pytest

from narada.audio.backends import linux, macos, windows
from narada.devices import AudioDevice, DeviceResolutionError


def test_windows_system_capture_rejects_input_type() -> None:
    device = AudioDevice(id=1, name="Mic", type="input")
    with pytest.raises(DeviceResolutionError):
        windows.resolve_system_capture_device(device, [device])


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
