import struct
import sys
from unittest.mock import MagicMock

import pytest

from narada.audio.capture import (
    CaptureError,
    CaptureHandle,
    DeviceDisconnectedError,
    _downmix_pcm16le_to_mono,
    _query_native_channels,
    open_system_capture,
    pcm16le_to_float32,
)
from narada.devices import AudioDevice


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


def test_open_system_capture_opens_stream_with_native_channel_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = AudioDevice(id=5, name="Speakers", type="output")
    mock_stream = _FakeStream()
    opened_with: dict[str, int] = {}

    def fake_resolve(
        selected_device: AudioDevice,
        all_devices: list[AudioDevice],
        os_name: str,
    ) -> tuple[AudioDevice, None]:
        return device, None

    def fake_query(device_id: int, *, loopback: bool = False) -> int:
        return 2

    def fake_open_raw(
        *,
        device_id: int,
        sample_rate_hz: int,
        channels: int,
        blocksize: int,
        extra_settings: object = None,
    ) -> _FakeStream:
        opened_with["channels"] = channels
        return mock_stream

    monkeypatch.setattr("narada.audio.capture._resolve_system_backend_device", fake_resolve)
    monkeypatch.setattr("narada.audio.capture._query_native_channels", fake_query)
    monkeypatch.setattr("narada.audio.capture._open_raw_input_stream", fake_open_raw)

    handle = open_system_capture(device=device, all_devices=[device], os_name="windows")

    assert opened_with["channels"] == 2  # stream opened with hardware channel count
    assert handle.channels == 1  # callers always see mono
    assert handle._native_channels == 2


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
    monkeypatch.setattr(
        "narada.audio.capture._open_raw_input_stream",
        lambda **kw: mock_stream,
    )

    handle = open_system_capture(device=device, all_devices=[device], os_name="linux")
    assert handle._native_channels == 1
    frame = handle.read_frame()
    assert frame is not None
    assert frame.pcm_bytes == _pack_s16le(999)


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
