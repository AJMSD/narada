import sys

import pytest

from narada.audio.backends import linux, macos, windows
from narada.devices import AudioDevice, DeviceResolutionError


class _FakeStream:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start_stream(self) -> None:
        self.started += 1

    def stop_stream(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1

    def read(self, frames: int, *, exception_on_overflow: bool = False) -> bytes:
        assert exception_on_overflow is False
        return b"\x00\x01" * frames


class _FakePyAudio:
    def __init__(
        self,
        *,
        device_infos: dict[int, dict[str, object]],
        loopback_infos: list[dict[str, object]] | None = None,
        open_succeeds_on: set[tuple[int, int, int]] | None = None,
    ) -> None:
        self._device_infos = device_infos
        self._loopback_infos = loopback_infos or []
        self._open_succeeds_on = open_succeeds_on or set()
        self.terminated = 0

    def get_host_api_count(self) -> int:
        return 1

    def get_host_api_info_by_index(self, index: int) -> dict[str, object]:
        assert index == 0
        return {"name": "Windows WASAPI"}

    def get_default_input_device_info(self) -> dict[str, object]:
        return {"index": 1}

    def get_default_output_device_info(self) -> dict[str, object]:
        return {"index": 2}

    def get_loopback_device_info_generator(self):  # type: ignore[no-untyped-def]
        yield from self._loopback_infos

    def get_device_info_by_index(self, index: int) -> dict[str, object]:
        return self._device_infos[index]

    def open(  # type: ignore[no-untyped-def]
        self,
        *,
        format: int,
        channels: int,
        rate: int,
        input: bool,
        input_device_index: int,
        frames_per_buffer: int,
        start: bool,
    ) -> _FakeStream:
        del format, input, frames_per_buffer, start
        if (input_device_index, rate, channels) in self._open_succeeds_on:
            return _FakeStream()
        raise OSError(f"open failed: device={input_device_index} rate={rate} channels={channels}")

    def terminate(self) -> None:
        self.terminated += 1


class _FakePyAudioModule:
    paInt16 = 8

    def __init__(self, instance: _FakePyAudio) -> None:
        self._instance = instance

    def PyAudio(self) -> _FakePyAudio:
        return self._instance


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


def test_windows_loopback_support_error_when_pyaudio_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", None)  # type: ignore[arg-type]
    issue = windows.loopback_support_error()
    assert issue is not None
    assert "PyAudioWPatch is required" in issue


def test_windows_loopback_support_error_when_no_loopbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePyAudio(device_infos={1: {"index": 1}}, loopback_infos=[])
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", _FakePyAudioModule(fake))
    issue = windows.loopback_support_error()
    assert issue is not None
    assert "No WASAPI loopback sources" in issue


def test_windows_loopback_support_ok_with_loopback_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePyAudio(
        device_infos={1: {"index": 1}},
        loopback_infos=[
            {
                "index": 9,
                "name": "Speakers (USB Device) [Loopback]",
                "hostApi": 0,
                "maxInputChannels": 2,
            }
        ],
    )
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", _FakePyAudioModule(fake))
    assert windows.loopback_support_error() is None


def test_windows_probe_requires_pyaudiowpatch_for_mic_and_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("narada.audio.backends.windows._dependency_error", lambda: "missing")
    devices = [
        AudioDevice(id=1, name="Mic", type="input"),
        AudioDevice(id=2, name="Speakers", type="output", hostapi="Windows WASAPI"),
    ]
    probe = windows.probe(devices)
    assert not probe.supports_mic_capture
    assert not probe.supports_system_capture
    assert probe.summary == "missing"


def test_windows_probe_requires_wasapi_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("narada.audio.backends.windows._dependency_error", lambda: None)
    monkeypatch.setattr("narada.audio.backends.windows.loopback_support_error", lambda: None)
    devices = [AudioDevice(id=34, name="Speakers", type="output", hostapi="Windows WDM-KS")]
    probe = windows.probe(devices)
    assert not probe.supports_system_capture
    assert "No WASAPI output devices detected" in probe.summary


def test_windows_probe_surfaces_loopback_support_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("narada.audio.backends.windows._dependency_error", lambda: None)
    monkeypatch.setattr(
        "narada.audio.backends.windows.loopback_support_error",
        lambda: "No WASAPI loopback sources detected.",
    )
    devices = [AudioDevice(id=22, name="Speakers", type="output", hostapi="Windows WASAPI")]
    probe = windows.probe(devices)
    assert not probe.supports_system_capture
    assert "No WASAPI loopback sources detected" in probe.summary


def test_windows_loopback_mapping_exact_match() -> None:
    loopback = windows._resolve_output_loopback_device(
        output_device_name="Speakers (USB Device)",
        output_device_id=2,
        loopback_devices=[
            {"index": 9, "name": "Speakers (USB Device) [Loopback]"},
            {"index": 10, "name": "Headphones (USB Device) [Loopback]"},
        ],
    )
    assert int(loopback["index"]) == 9


def test_windows_loopback_mapping_contains_match_fallback() -> None:
    loopback = windows._resolve_output_loopback_device(
        output_device_name="Realtek(R) Audio",
        output_device_id=2,
        loopback_devices=[
            {"index": 9, "name": "Speakers (2- Realtek(R) Audio) [Loopback]"},
        ],
    )
    assert int(loopback["index"]) == 9


def test_windows_loopback_mapping_ambiguous_raises() -> None:
    with pytest.raises(DeviceResolutionError, match="maps to multiple loopback devices"):
        windows._resolve_output_loopback_device(
            output_device_name="Realtek(R) Audio",
            output_device_id=2,
            loopback_devices=[
                {"index": 9, "name": "Speakers (2- Realtek(R) Audio) [Loopback]"},
                {"index": 10, "name": "Headphones (2- Realtek(R) Audio) [Loopback]"},
            ],
        )


def test_windows_open_system_stream_falls_back_sample_rate_and_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePyAudio(
        device_infos={
            22: {"index": 22, "name": "Speakers", "hostApi": 0},
        },
        loopback_infos=[
            {
                "index": 122,
                "name": "Speakers [Loopback]",
                "hostApi": 0,
                "defaultSampleRate": 48000,
                "maxInputChannels": 6,
            }
        ],
        open_succeeds_on={(122, 48000, 2)},
    )
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", _FakePyAudioModule(fake))

    stream, opened_rate_hz, opened_channels, resolved_name = windows.open_windows_system_stream(
        output_device_id=22,
        output_device_name="Speakers",
        sample_rate_hz=16000,
        blocksize=1600,
    )
    data, overflowed = stream.read(2)
    stream.close()

    assert opened_rate_hz == 48000
    assert opened_channels == 2
    assert resolved_name == "Speakers [Loopback]"
    assert data == b"\x00\x01\x00\x01"
    assert overflowed is False
    assert fake.terminated == 1


def test_windows_open_mic_stream_falls_back_sample_rate_and_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakePyAudio(
        device_infos={
            7: {
                "index": 7,
                "name": "USB Mic",
                "hostApi": 0,
                "defaultSampleRate": 48000,
                "maxInputChannels": 4,
            },
        },
        open_succeeds_on={(7, 48000, 4)},
    )
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", _FakePyAudioModule(fake))

    stream, opened_rate_hz, opened_channels = windows.open_windows_mic_stream(
        device_id=7,
        device_name="USB Mic",
        sample_rate_hz=16000,
        channels=1,
        blocksize=1600,
    )
    stream.close()

    assert opened_rate_hz == 48000
    assert opened_channels == 4
    assert fake.terminated == 1


def test_windows_stream_wrapper_close_is_idempotent() -> None:
    fake_stream = _FakeStream()
    fake_audio = _FakePyAudio(device_infos={})
    wrapped = windows._PyAudioInputStream(fake_stream, fake_audio)
    wrapped.close()
    wrapped.close()
    assert fake_stream.closed == 1
    assert fake_audio.terminated == 1


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
