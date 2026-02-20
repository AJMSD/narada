import struct
import sys
from unittest.mock import MagicMock

import pytest

from narada.audio.capture import (
    _PA_INVALID_CHANNELS,
    CaptureError,
    CaptureHandle,
    DeviceDisconnectedError,
    _downmix_pcm16le_to_mono,
    _open_loopback_stream,
    _query_native_channels,
    _raise_loopback_error,
    open_mic_capture,
    open_system_capture,
    pcm16le_to_float32,
)
from narada.devices import AudioDevice, DeviceResolutionError


def _pack_s16le(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


class _FakeStream:
    def __init__(
        self, frames: list[tuple[bytes, bool]] | None = None, fail_on_read: bool = False
    ) -> None:
        self._frames = frames or []
        self._idx = 0
        self.fail_on_read = fail_on_read
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        return

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def read(self, frames: int) -> tuple[bytes, bool]:
        if self.fail_on_read:
            raise OSError("device gone")
        if self._idx >= len(self._frames):
            return b"", False
        item = self._frames[self._idx]
        self._idx += 1
        return item


# ---------------------------------------------------------------------------
# CaptureHandle — basic behaviour
# ---------------------------------------------------------------------------


def test_capture_handle_reads_frame() -> None:
    stream = _FakeStream(frames=[(b"\x01\x00\xff\x7f", False)])
    handle = CaptureHandle(
        stream=stream,
        sample_rate_hz=16000,
        channels=1,
        blocksize=2,
        device_name="FakeMic",
    )
    frame = handle.read_frame()
    assert frame is not None
    assert frame.sample_rate_hz == 16000
    assert frame.pcm_bytes == b"\x01\x00\xff\x7f"


def test_capture_handle_returns_none_on_overflow() -> None:
    stream = _FakeStream(frames=[(b"\x01\x00", True)])
    handle = CaptureHandle(
        stream=stream, sample_rate_hz=16000, channels=1, blocksize=1, device_name="x"
    )
    assert handle.read_frame() is None


def test_capture_handle_raises_disconnected_error() -> None:
    stream = _FakeStream(fail_on_read=True)
    handle = CaptureHandle(
        stream=stream, sample_rate_hz=16000, channels=1, blocksize=1, device_name="x"
    )
    with pytest.raises(DeviceDisconnectedError):
        handle.read_frame()
    assert stream.stopped
    assert stream.closed


def test_capture_handle_native_channels_defaults_to_channels() -> None:
    stream = _FakeStream(frames=[(b"\x01\x00\xff\x7f", False)])
    handle = CaptureHandle(
        stream=stream,
        sample_rate_hz=16000,
        channels=1,
        blocksize=2,
        device_name="Mic",
    )
    assert handle._native_channels == 1
    frame = handle.read_frame()
    assert frame is not None
    assert frame.pcm_bytes == b"\x01\x00\xff\x7f"


def test_capture_handle_downmixes_stereo_to_mono() -> None:
    # left=1000, right=3000 → mono=2000
    stereo_pcm = _pack_s16le(1000, 3000)
    stream = _FakeStream(frames=[(stereo_pcm, False)])
    handle = CaptureHandle(
        stream=stream,
        sample_rate_hz=16000,
        channels=1,
        blocksize=2,
        device_name="LoopbackDevice",
        native_channels=2,
    )
    frame = handle.read_frame()
    assert frame is not None
    assert frame.channels == 1
    (mono_sample,) = struct.unpack("<h", frame.pcm_bytes)
    assert mono_sample == 2000


def test_capture_handle_downmix_produces_correct_frame_count() -> None:
    # Two stereo frames → two mono frames
    stereo_pcm = _pack_s16le(100, 200, 300, 400)
    stream = _FakeStream(frames=[(stereo_pcm, False)])
    handle = CaptureHandle(
        stream=stream,
        sample_rate_hz=16000,
        channels=1,
        blocksize=4,
        device_name="Loopback",
        native_channels=2,
    )
    frame = handle.read_frame()
    assert frame is not None
    samples = struct.unpack("<2h", frame.pcm_bytes)
    assert samples == (150, 350)


# ---------------------------------------------------------------------------
# _downmix_pcm16le_to_mono
# ---------------------------------------------------------------------------


def test_downmix_mono_passthrough() -> None:
    data = _pack_s16le(100, 200, 300)
    assert _downmix_pcm16le_to_mono(data, 1) is data


def test_downmix_empty_bytes_returns_empty() -> None:
    assert _downmix_pcm16le_to_mono(b"", 2) == b""


def test_downmix_stereo_averages_channels() -> None:
    # left=200, right=400 → mono=300
    result = _downmix_pcm16le_to_mono(_pack_s16le(200, 400), 2)
    assert struct.unpack("<h", result) == (300,)


def test_downmix_stereo_out_of_phase_cancels() -> None:
    # left=1000, right=-1000 → mono=0
    result = _downmix_pcm16le_to_mono(_pack_s16le(1000, -1000), 2)
    assert struct.unpack("<h", result) == (0,)


def test_downmix_three_channels() -> None:
    # 300+600+900=1800 // 3 = 600
    result = _downmix_pcm16le_to_mono(_pack_s16le(300, 600, 900), 3)
    assert struct.unpack("<h", result) == (600,)


def test_downmix_multiple_stereo_frames() -> None:
    data = _pack_s16le(100, 200, 300, 600)
    result = _downmix_pcm16le_to_mono(data, 2)
    assert struct.unpack("<2h", result) == (150, 450)


def test_downmix_misaligned_raises_capture_error() -> None:
    with pytest.raises(CaptureError, match="not a multiple"):
        _downmix_pcm16le_to_mono(b"\x00\x01\x02", 2)


# ---------------------------------------------------------------------------
# _query_native_channels
# ---------------------------------------------------------------------------


def test_query_native_channels_loopback_fallback_when_sounddevice_missing() -> None:
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sounddevice", None)  # type: ignore[arg-type]
        result = _query_native_channels(1, loopback=True)
    assert result == 2


def test_query_native_channels_input_fallback_when_sounddevice_missing() -> None:
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sounddevice", None)  # type: ignore[arg-type]
        result = _query_native_channels(1, loopback=False)
    assert result == 1


def test_query_native_channels_loopback_reads_max_output_channels() -> None:
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = {"max_output_channels": 6, "max_input_channels": 1}
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sounddevice", mock_sd)
        result = _query_native_channels(5, loopback=True)
    mock_sd.query_devices.assert_called_once_with(5)
    assert result == 6


def test_query_native_channels_input_reads_max_input_channels() -> None:
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = {"max_output_channels": 6, "max_input_channels": 1}
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sounddevice", mock_sd)
        result = _query_native_channels(3, loopback=False)
    assert result == 1


def test_query_native_channels_clamps_zero_to_one() -> None:
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = {"max_output_channels": 0}
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sounddevice", mock_sd)
        result = _query_native_channels(1, loopback=True)
    assert result == 1


def test_query_native_channels_query_exception_returns_fallback() -> None:
    mock_sd = MagicMock()
    mock_sd.query_devices.side_effect = Exception("device not found")
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "sounddevice", mock_sd)
        result = _query_native_channels(99, loopback=True)
    assert result == 2


# ---------------------------------------------------------------------------
# open_system_capture — integration with mocked helpers
# ---------------------------------------------------------------------------


def test_open_system_capture_windows_uses_windows_backend_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = AudioDevice(id=5, name="Speakers", type="output", hostapi="Windows WASAPI")
    resolved = AudioDevice(id=5, name="Speakers", type="output", hostapi="Windows WASAPI")
    fake_stream = _FakeStream(frames=[(_pack_s16le(500, 1500), False)])

    monkeypatch.setattr(
        "narada.audio.capture.windows.resolve_system_capture_device",
        lambda *_args, **_kwargs: resolved,
    )
    monkeypatch.setattr(
        "narada.audio.capture.windows.open_windows_system_stream",
        lambda **_kwargs: (fake_stream, 48000, 2, "Speakers [Loopback]"),
    )

    handle = open_system_capture(device=selected, all_devices=[selected], os_name="windows")
    frame = handle.read_frame()

    assert handle.sample_rate_hz == 48000
    assert handle.channels == 1
    assert handle._native_channels == 2
    assert handle.device_name == "Speakers [Loopback]"
    assert frame is not None
    assert struct.unpack("<h", frame.pcm_bytes) == (1000,)


def test_open_system_capture_mono_device_no_downmix_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = AudioDevice(id=3, name="Mono Output", type="output")
    mock_stream = _FakeStream(frames=[(_pack_s16le(999), False)])

    monkeypatch.setattr(
        "narada.audio.capture._resolve_system_backend_device",
        lambda *a, **kw: (device, None),
    )
    monkeypatch.setattr(
        "narada.audio.capture._query_native_channels",
        lambda *a, **kw: 1,
    )
    opened_with: dict[str, int] = {}

    def fake_open_loopback(**kwargs: int) -> tuple[_FakeStream, int]:
        opened_with["device_id"] = kwargs["device_id"]
        opened_with["native_channels"] = kwargs["native_channels"]
        return mock_stream, kwargs["native_channels"]

    monkeypatch.setattr("narada.audio.capture._open_loopback_stream", fake_open_loopback)

    handle = open_system_capture(device=device, all_devices=[device], os_name="linux")
    assert opened_with["device_id"] == 3
    assert opened_with["native_channels"] == 1
    assert handle._native_channels == 1
    frame = handle.read_frame()
    assert frame is not None
    assert frame.pcm_bytes == _pack_s16le(999)


def test_open_system_capture_windows_wraps_device_resolution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = AudioDevice(id=5, name="Speakers", type="output", hostapi="MME")
    monkeypatch.setattr(
        "narada.audio.capture.windows.resolve_system_capture_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DeviceResolutionError("requires a WASAPI output device")
        ),
    )
    with pytest.raises(CaptureError, match="requires a WASAPI output device"):
        open_system_capture(device=selected, all_devices=[selected], os_name="windows")


def test_open_mic_capture_windows_uses_windows_backend_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = AudioDevice(id=7, name="USB Mic", type="input")
    fake_stream = _FakeStream(frames=[(_pack_s16le(200, 400), False)])
    monkeypatch.setattr(
        "narada.audio.capture.windows.open_windows_mic_stream",
        lambda **_kwargs: (fake_stream, 48000, 2),
    )
    monkeypatch.setattr("narada.audio.capture.platform.system", lambda: "Windows")

    handle = open_mic_capture(device=device)
    frame = handle.read_frame()

    assert handle.sample_rate_hz == 48000
    assert handle.channels == 1
    assert handle._native_channels == 2
    assert frame is not None
    assert struct.unpack("<h", frame.pcm_bytes) == (300,)


# ---------------------------------------------------------------------------
# pcm16le_to_float32
# ---------------------------------------------------------------------------


def test_pcm16le_to_float32_handles_boundaries() -> None:
    samples = pcm16le_to_float32(b"\x00\x80\xff\x7f")
    assert pytest.approx(samples[0], rel=1e-4) == -1.0
    assert samples[1] > 0.99


def test_pcm16le_to_float32_rejects_odd_bytes() -> None:
    with pytest.raises(ValueError):
        pcm16le_to_float32(b"\x00")


# ---------------------------------------------------------------------------
# _raise_loopback_error
# ---------------------------------------------------------------------------


def test_raise_loopback_error_generic_message() -> None:
    cause = RuntimeError("pa error")
    with pytest.raises(CaptureError, match="Could not open system capture"):
        _raise_loopback_error("Speakers (Realtek)", cause)


def test_raise_loopback_error_includes_device_name() -> None:
    cause = RuntimeError("pa error")
    with pytest.raises(CaptureError, match="Speakers \\(Realtek\\)"):
        _raise_loopback_error("Speakers (Realtek)", cause)


def test_raise_loopback_error_bt_hfp_detected_by_driver_hint() -> None:
    cause = RuntimeError("pa error")
    bt_name = "Output 1 (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0)"
    with pytest.raises(CaptureError, match="Bluetooth HFP"):
        _raise_loopback_error(bt_name, cause)


def test_raise_loopback_error_bt_hint_case_insensitive() -> None:
    cause = RuntimeError("pa error")
    with pytest.raises(CaptureError, match="Bluetooth HFP"):
        _raise_loopback_error("Device (BTHHFENUM driver)", cause)


def test_raise_loopback_error_chains_cause() -> None:
    cause = RuntimeError("original")
    with pytest.raises(CaptureError) as exc_info:
        _raise_loopback_error("Speakers", cause)
    assert exc_info.value.__cause__ is cause


# ---------------------------------------------------------------------------
# _open_loopback_stream
# ---------------------------------------------------------------------------


class _FakePortAudioError(Exception):
    """Minimal stand-in for sounddevice.PortAudioError."""

    def __init__(self, msg: str, code: int) -> None:
        super().__init__(msg, code)


def _make_sd_mock(
    *,
    succeed_on: int | None = None,
    fail_code: int = _PA_INVALID_CHANNELS,
) -> MagicMock:
    """Return a mock sounddevice module whose PortAudioError matches *fail_code*.

    ``succeed_on`` – channel count that opens successfully; None means all fail.
    """
    mock_sd = MagicMock()
    mock_sd.PortAudioError = _FakePortAudioError

    def fake_raw_input_stream(**kwargs: object) -> MagicMock:
        channels = kwargs["channels"]
        if succeed_on is not None and channels == succeed_on:
            return MagicMock()
        raise _FakePortAudioError(
            f"Invalid number of channels [PaErrorCode {fail_code}]",
            fail_code,
        )

    mock_sd.RawInputStream = fake_raw_input_stream
    return mock_sd


def test_open_loopback_stream_succeeds_first_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_sd = _make_sd_mock(succeed_on=2)
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    stream, opened = _open_loopback_stream(
        device_id=1,
        device_name="Speakers",
        sample_rate_hz=16000,
        native_channels=2,
        blocksize=1600,
        extra_settings=None,
    )
    assert opened == 2


def test_open_loopback_stream_falls_back_to_stereo(monkeypatch: pytest.MonkeyPatch) -> None:
    """native_channels=4 fails, fallback 2 succeeds."""
    mock_sd = _make_sd_mock(succeed_on=2)
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    _, opened = _open_loopback_stream(
        device_id=1,
        device_name="Speakers",
        sample_rate_hz=16000,
        native_channels=4,
        blocksize=1600,
        extra_settings=None,
    )
    assert opened == 2


def test_open_loopback_stream_falls_back_to_mono(monkeypatch: pytest.MonkeyPatch) -> None:
    """native=2 fails, 2 is already tried, fallback 1 succeeds."""
    mock_sd = _make_sd_mock(succeed_on=1)
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    _, opened = _open_loopback_stream(
        device_id=1,
        device_name="Speakers",
        sample_rate_hz=16000,
        native_channels=2,
        blocksize=1600,
        extra_settings=None,
    )
    assert opened == 1


def test_open_loopback_stream_no_duplicates_when_native_is_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native=1 means candidates=[1, 2]; if 1 fails, 2 is tried."""
    mock_sd = _make_sd_mock(succeed_on=2)
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    _, opened = _open_loopback_stream(
        device_id=1,
        device_name="Speakers",
        sample_rate_hz=16000,
        native_channels=1,
        blocksize=1600,
        extra_settings=None,
    )
    assert opened == 2


def test_open_loopback_stream_all_fail_raises_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_sd = _make_sd_mock(succeed_on=None)
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    with pytest.raises(CaptureError, match="Could not open system capture"):
        _open_loopback_stream(
            device_id=1,
            device_name="Speakers (Realtek)",
            sample_rate_hz=16000,
            native_channels=2,
            blocksize=1600,
            extra_settings=None,
        )


def test_open_loopback_stream_bt_device_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_sd = _make_sd_mock(succeed_on=None)
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    bt_name = "Output 1 (@System32\\drivers\\bthhfenum.sys)"
    with pytest.raises(CaptureError, match="Bluetooth HFP"):
        _open_loopback_stream(
            device_id=1,
            device_name=bt_name,
            sample_rate_hz=16000,
            native_channels=2,
            blocksize=1600,
            extra_settings=None,
        )


def test_open_loopback_stream_non_channel_pa_error_surfaces_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PortAudioError that is NOT about invalid channels should not trigger retry."""
    mock_sd = MagicMock()
    mock_sd.PortAudioError = _FakePortAudioError

    call_count: list[int] = [0]

    def fail_with_other_code(**kwargs: object) -> MagicMock:
        call_count[0] += 1
        raise _FakePortAudioError("Device unavailable [PaErrorCode -9985]", -9985)

    mock_sd.RawInputStream = fail_with_other_code
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    with pytest.raises(CaptureError):
        _open_loopback_stream(
            device_id=1,
            device_name="Speakers",
            sample_rate_hz=16000,
            native_channels=2,
            blocksize=1600,
            extra_settings=None,
        )
    # Must not retry on non-channel errors.
    assert call_count[0] == 1


def test_open_loopback_stream_sounddevice_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # type: ignore[arg-type]
    with pytest.raises(CaptureError, match="sounddevice is required"):
        _open_loopback_stream(
            device_id=1,
            device_name="Speakers",
            sample_rate_hz=16000,
            native_channels=2,
            blocksize=1600,
            extra_settings=None,
        )
