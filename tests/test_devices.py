import pytest

from narada.devices import AmbiguousDeviceError, AudioDevice, DeviceResolutionError, resolve_device


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
