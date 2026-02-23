from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NoReturn

from narada.audio.backends.types import BackendProbe
from narada.devices import AudioDevice, DeviceResolutionError

if TYPE_CHECKING:
    from narada.audio.capture import StreamProtocol

_WASAPI_HOSTAPI_TOKEN = "wasapi"
_LOOPBACK_SUFFIX_RE = re.compile(r"\s*\[\s*loopback\s*\]\s*$", re.IGNORECASE)


def _raise_capture_error(message: str, cause: Exception | None = None) -> NoReturn:
    from narada.audio.capture import CaptureError

    if cause is None:
        raise CaptureError(message)
    raise CaptureError(message) from cause


def _compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_loopback_name(value: str) -> str:
    normalized = _compact_whitespace(value).lower()
    normalized = _LOOPBACK_SUFFIX_RE.sub("", normalized)
    return normalized


def _import_pyaudio() -> Any:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        _raise_capture_error(
            "PyAudioWPatch is required for Windows live capture. "
            "Install with: pip install PyAudioWPatch",
            exc,
        )
    return pyaudio


def _hostapi_name_map(audio: Any) -> dict[int, str]:
    names: dict[int, str] = {}
    try:
        count = int(audio.get_host_api_count())
    except Exception:
        return {}
    for index in range(count):
        try:
            info = audio.get_host_api_info_by_index(index)
        except Exception:
            continue
        names[index] = _compact_whitespace(str(info.get("name", f"Host API {index}")))
    return names


def _is_wasapi_device(device_info: dict[str, Any], hostapi_names: dict[int, str]) -> bool:
    hostapi_index = int(device_info.get("hostApi", -1))
    hostapi_name = hostapi_names.get(hostapi_index, "")
    return _WASAPI_HOSTAPI_TOKEN in hostapi_name.lower()


def _dependency_error() -> str | None:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return (
            "PyAudioWPatch is required for Windows live capture. "
            "Install with: pip install PyAudioWPatch"
        )

    try:
        audio = pyaudio.PyAudio()
    except Exception as exc:
        return f"PyAudioWPatch is installed but unusable: {exc}"
    try:
        audio.get_host_api_count()
    except Exception as exc:
        return f"PyAudioWPatch is installed but unusable: {exc}"
    finally:
        try:
            audio.terminate()
        except Exception:
            pass
    return None


def probe(devices: Sequence[AudioDevice]) -> BackendProbe:
    dependency_error = _dependency_error()
    has_input = any(device.type == "input" for device in devices)
    has_wasapi_output = any(
        device.type in {"loopback", "monitor", "output"}
        and _WASAPI_HOSTAPI_TOKEN in (device.hostapi or "").lower()
        for device in devices
    )
    loopback_issue = loopback_support_error() if dependency_error is None else dependency_error
    supports_mic_capture = has_input and dependency_error is None
    supports_system_capture = has_wasapi_output and loopback_issue is None

    if dependency_error is not None:
        summary = dependency_error
    elif supports_system_capture:
        summary = "System capture uses WASAPI loopback via PyAudioWPatch."
    elif not has_wasapi_output:
        summary = "No WASAPI output devices detected for system capture."
    else:
        summary = loopback_issue or "WASAPI loopback sources are unavailable."

    return BackendProbe(
        backend="windows",
        supports_mic_capture=supports_mic_capture,
        supports_system_capture=supports_system_capture,
        summary=summary,
    )


def resolve_system_capture_device(
    selected_device: AudioDevice, devices: Sequence[AudioDevice]
) -> AudioDevice:
    if selected_device.type not in {"output", "loopback", "monitor"}:
        raise DeviceResolutionError(
            f"Windows system capture requires output/loopback device, got: {selected_device.type}"
        )
    candidates = [
        item
        for item in devices
        if item.id == selected_device.id and item.type in {"output", "loopback", "monitor"}
    ]
    if not candidates:
        raise DeviceResolutionError(f"Selected device {selected_device.id} is not available.")
    resolved = candidates[0]
    hostapi = resolved.hostapi or "unknown"
    if _WASAPI_HOSTAPI_TOKEN not in hostapi.lower():
        raise DeviceResolutionError(
            "Windows system capture requires a WASAPI output device. "
            f"Selected '{resolved.name}' uses host API '{hostapi}'."
        )
    return resolved


def _list_loopback_sources() -> tuple[list[dict[str, Any]], str | None]:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return (
            [],
            "PyAudioWPatch is required for Windows live capture. "
            "Install with: pip install PyAudioWPatch",
        )

    try:
        audio = pyaudio.PyAudio()
    except Exception as exc:
        return [], f"Unable to initialize PyAudioWPatch: {exc}"
    try:
        loopbacks = [dict(item) for item in audio.get_loopback_device_info_generator()]
        return loopbacks, None
    except Exception as exc:
        return [], f"Unable to query WASAPI loopback sources via PyAudioWPatch: {exc}"
    finally:
        try:
            audio.terminate()
        except Exception:
            pass


def loopback_support_error() -> str | None:
    loopbacks, issue = _list_loopback_sources()
    if issue is not None:
        return issue
    if not loopbacks:
        return (
            "No WASAPI loopback sources detected. Ensure a playback device is enabled "
            "and supports loopback capture."
        )
    return None


def _candidate_values(*values: int) -> list[int]:
    candidates: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value < 1:
            continue
        if value in seen:
            continue
        seen.add(value)
        candidates.append(value)
    return candidates


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


class _PyAudioInputStream:
    def __init__(self, stream: Any, audio: Any) -> None:
        self._stream = stream
        self._audio = audio
        self._closed = False

    def start(self) -> None:
        if self._closed:
            return
        try:
            self._stream.start_stream()
        except Exception:
            return

    def stop(self) -> None:
        if self._closed:
            return
        try:
            self._stream.stop_stream()
        except Exception:
            return

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._stream.stop_stream()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass
        try:
            self._audio.terminate()
        except Exception:
            pass
        self._closed = True

    def read(self, frames: int) -> tuple[bytes, bool]:
        data = self._stream.read(frames, exception_on_overflow=False)
        return bytes(data), False


def _open_stream_with_fallback(
    *,
    audio: Any,
    pyaudio_module: Any,
    device_index: int,
    sample_rate_hz: int,
    default_sample_rate_hz: int,
    channel_candidates: Sequence[int],
    blocksize: int,
    open_description: str,
) -> tuple[StreamProtocol, int, int]:
    sample_rate_candidates = _candidate_values(sample_rate_hz, default_sample_rate_hz)
    channel_values = _candidate_values(*channel_candidates)
    if not sample_rate_candidates:
        sample_rate_candidates = [sample_rate_hz]
    if not channel_values:
        channel_values = [1]

    last_error: Exception | None = None
    for rate in sample_rate_candidates:
        for channels in channel_values:
            try:
                stream = audio.open(
                    format=pyaudio_module.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=blocksize,
                    start=False,
                )
                wrapped = _PyAudioInputStream(stream, audio)
                wrapped.start()
                return wrapped, rate, channels
            except Exception as exc:
                last_error = exc

    if last_error is not None:
        _raise_capture_error(
            f"Could not open {open_description}. Last backend error: {last_error}",
            last_error,
        )
    _raise_capture_error(f"Could not open {open_description}.")


def _get_device_info(audio: Any, device_id: int) -> dict[str, Any]:
    try:
        info = audio.get_device_info_by_index(device_id)
    except Exception as exc:
        _raise_capture_error(f"Could not query audio device {device_id}: {exc}", exc)
    return dict(info)


def _resolve_output_loopback_device(
    *,
    output_device_name: str,
    output_device_id: int,
    loopback_devices: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    normalized_output = _normalize_loopback_name(output_device_name)
    matches_exact = [
        item
        for item in loopback_devices
        if _normalize_loopback_name(str(item.get("name", ""))) == normalized_output
    ]
    if len(matches_exact) == 1:
        return matches_exact[0]
    if len(matches_exact) > 1:
        details = ", ".join(
            f"{int(item.get('index', -1))}:{item.get('name', 'Unknown')}" for item in matches_exact
        )
        raise DeviceResolutionError(
            f"Output '{output_device_name}' maps to multiple loopback devices: {details}"
        )

    matches_contains = []
    for item in loopback_devices:
        normalized_loopback = _normalize_loopback_name(str(item.get("name", "")))
        if normalized_output in normalized_loopback or normalized_loopback in normalized_output:
            matches_contains.append(item)

    if len(matches_contains) == 1:
        return matches_contains[0]
    if len(matches_contains) > 1:
        details = ", ".join(
            f"{int(item.get('index', -1))}:{item.get('name', 'Unknown')}"
            for item in matches_contains
        )
        raise DeviceResolutionError(
            f"Output '{output_device_name}' maps to multiple loopback devices: {details}"
        )

    available = ", ".join(
        f"{int(item.get('index', -1))}:{item.get('name', 'Unknown')}" for item in loopback_devices
    )
    raise DeviceResolutionError(
        f"Could not map output '{output_device_name}' (ID {output_device_id}) to a WASAPI "
        f"loopback source. Available loopback devices: {available}"
    )


def open_windows_mic_stream(
    *,
    device_id: int,
    device_name: str,
    sample_rate_hz: int,
    channels: int,
    blocksize: int,
) -> tuple[StreamProtocol, int, int]:
    pyaudio_module = _import_pyaudio()
    audio = pyaudio_module.PyAudio()
    try:
        device_info = _get_device_info(audio, device_id)
        default_sample_rate_hz = _safe_int(device_info.get("defaultSampleRate"), sample_rate_hz)
        max_input_channels = _safe_int(device_info.get("maxInputChannels"), 1)
        channel_candidates = _candidate_values(channels, max_input_channels, 1)
        return _open_stream_with_fallback(
            audio=audio,
            pyaudio_module=pyaudio_module,
            device_index=device_id,
            sample_rate_hz=sample_rate_hz,
            default_sample_rate_hz=default_sample_rate_hz,
            channel_candidates=channel_candidates,
            blocksize=blocksize,
            open_description=f"microphone stream for '{device_name}'",
        )
    except Exception:
        try:
            audio.terminate()
        except Exception:
            pass
        raise


def open_windows_system_stream(
    *,
    output_device_id: int,
    output_device_name: str,
    sample_rate_hz: int,
    blocksize: int,
) -> tuple[StreamProtocol, int, int, str]:
    pyaudio_module = _import_pyaudio()
    audio = pyaudio_module.PyAudio()
    try:
        hostapi_names = _hostapi_name_map(audio)
        output_info = _get_device_info(audio, output_device_id)
        if not _is_wasapi_device(output_info, hostapi_names):
            hostapi_name = hostapi_names.get(int(output_info.get("hostApi", -1)), "unknown")
            raise DeviceResolutionError(
                "Windows system capture requires a WASAPI output device. "
                f"Selected '{output_device_name}' uses host API '{hostapi_name}'."
            )

        loopbacks = [dict(item) for item in audio.get_loopback_device_info_generator()]
        loopback_candidates = [item for item in loopbacks if _is_wasapi_device(item, hostapi_names)]
        if not loopback_candidates:
            _raise_capture_error(
                "No WASAPI loopback sources detected. Ensure a playback device is enabled "
                "and supports loopback capture."
            )

        loopback_device = _resolve_output_loopback_device(
            output_device_name=output_device_name,
            output_device_id=output_device_id,
            loopback_devices=loopback_candidates,
        )
        loopback_id = int(loopback_device.get("index", -1))
        if loopback_id < 0:
            _raise_capture_error(
                f"Loopback source for '{output_device_name}' has an invalid device index."
            )

        default_sample_rate_hz = _safe_int(loopback_device.get("defaultSampleRate"), sample_rate_hz)
        max_loopback_channels = _safe_int(loopback_device.get("maxInputChannels"), 2)
        channel_candidates = _candidate_values(max_loopback_channels, 2, 1)
        stream, opened_rate_hz, opened_channels = _open_stream_with_fallback(
            audio=audio,
            pyaudio_module=pyaudio_module,
            device_index=loopback_id,
            sample_rate_hz=sample_rate_hz,
            default_sample_rate_hz=default_sample_rate_hz,
            channel_candidates=channel_candidates,
            blocksize=blocksize,
            open_description=f"system loopback stream for '{output_device_name}'",
        )
        resolved_name = _compact_whitespace(str(loopback_device.get("name", output_device_name)))
        return stream, opened_rate_hz, opened_channels, resolved_name
    except Exception:
        try:
            audio.terminate()
        except Exception:
            pass
        raise
