import pytest

from narada.audio.capture import CaptureHandle, DeviceDisconnectedError, pcm16le_to_float32


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


def test_pcm16le_to_float32_handles_boundaries() -> None:
    samples = pcm16le_to_float32(b"\x00\x80\xff\x7f")
    assert pytest.approx(samples[0], rel=1e-4) == -1.0
    assert samples[1] > 0.99


def test_pcm16le_to_float32_rejects_odd_bytes() -> None:
    with pytest.raises(ValueError):
        pcm16le_to_float32(b"\x00")
