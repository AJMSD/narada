from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Protocol, cast

from narada.audio.backends import linux, macos, windows
from narada.devices import AudioDevice, DeviceResolutionError


class CaptureError(RuntimeError):
    pass


class DeviceDisconnectedError(CaptureError):
    pass


class StreamProtocol(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def read(self, frames: int) -> tuple[bytes, bool]: ...


@dataclass(frozen=True)
class CapturedFrame:
    pcm_bytes: bytes
    sample_rate_hz: int
    channels: int


class CaptureHandle:
    def __init__(
        self,
        stream: StreamProtocol,
        sample_rate_hz: int,
        channels: int,
        blocksize: int,
        device_name: str,
    ) -> None:
        self._stream = stream
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.blocksize = blocksize
        self.device_name = device_name
        self._closed = False

    def read_frame(self) -> CapturedFrame | None:
        if self._closed:
            return None
        try:
            data, overflowed = self._stream.read(self.blocksize)
        except Exception as exc:  # pragma: no cover - backend exceptions vary by OS
            self.close()
            raise DeviceDisconnectedError(
                f"Audio device disconnected or unavailable: {self.device_name}"
            ) from exc

        if overflowed:
            return None
        if not data:
            return None
        return CapturedFrame(
            pcm_bytes=bytes(data),
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._stream.stop()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass
        self._closed = True

    def __enter__(self) -> CaptureHandle:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def pcm16le_to_float32(pcm_bytes: bytes) -> tuple[float, ...]:
    if len(pcm_bytes) % 2 != 0:
        raise ValueError("PCM16 input must have an even byte-length.")
    samples: list[float] = []
    for idx in range(0, len(pcm_bytes), 2):
        sample = int.from_bytes(pcm_bytes[idx : idx + 2], byteorder="little", signed=True)
        samples.append(sample / 32768.0)
    return tuple(samples)


def _open_raw_input_stream(
    *,
    device_id: int,
    sample_rate_hz: int,
    channels: int,
    blocksize: int,
    extra_settings: Any | None = None,
) -> StreamProtocol:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise CaptureError("sounddevice is required for live audio capture.") from exc

    stream = sd.RawInputStream(
        samplerate=sample_rate_hz,
        device=device_id,
        channels=channels,
        dtype="int16",
        blocksize=blocksize,
        extra_settings=extra_settings,
    )
    stream.start()
    return cast(StreamProtocol, stream)


def open_mic_capture(
    *,
    device: AudioDevice,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    blocksize: int = 1600,
) -> CaptureHandle:
    if device.type != "input":
        raise DeviceResolutionError(f"Selected microphone device is not an input: {device.name}")
    stream = _open_raw_input_stream(
        device_id=device.id,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        blocksize=blocksize,
    )
    return CaptureHandle(
        stream=stream,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        blocksize=blocksize,
        device_name=device.name,
    )


def _resolve_system_backend_device(
    selected_device: AudioDevice,
    all_devices: list[AudioDevice],
    os_name: str,
) -> tuple[AudioDevice, Any | None]:
    normalized_os = os_name.lower()
    if normalized_os == "windows":
        resolved = windows.resolve_system_capture_device(selected_device, all_devices)
        return resolved, windows.build_loopback_settings()
    if normalized_os == "linux":
        resolved = linux.resolve_system_capture_device(selected_device, all_devices)
        return resolved, None
    if normalized_os in {"darwin", "macos"}:
        resolved = macos.resolve_system_capture_device(selected_device, all_devices)
        return resolved, None
    raise CaptureError(f"Unsupported OS for system capture: {os_name}")


def open_system_capture(
    *,
    device: AudioDevice,
    all_devices: list[AudioDevice],
    sample_rate_hz: int = 16000,
    channels: int = 1,
    blocksize: int = 1600,
    os_name: str | None = None,
) -> CaptureHandle:
    resolved_device, extra_settings = _resolve_system_backend_device(
        selected_device=device,
        all_devices=all_devices,
        os_name=os_name or platform.system().lower(),
    )
    stream = _open_raw_input_stream(
        device_id=resolved_device.id,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        blocksize=blocksize,
        extra_settings=extra_settings,
    )
    return CaptureHandle(
        stream=stream,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        blocksize=blocksize,
        device_name=resolved_device.name,
    )
