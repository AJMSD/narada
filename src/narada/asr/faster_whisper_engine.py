from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import platform
import queue
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import numpy as np

from narada.asr.base import EngineUnavailableError, TranscriptionRequest, TranscriptSegment
from narada.asr.model_discovery import resolve_faster_whisper_model_path

logger = logging.getLogger("narada.asr.faster_whisper")
ModelFactory = Callable[[str, str, str], Any]


class _GpuWorkerError(RuntimeError):
    pass


class _GpuWorkerTimeoutError(_GpuWorkerError):
    pass


class _GpuWorkerRuntimeError(_GpuWorkerError):
    pass


class _GpuWorkerExitError(_GpuWorkerError):
    pass


@dataclass
class _GpuWorkerHandle:
    key: tuple[str, str, str]
    process: mp.Process
    request_queue: Any
    response_queue: Any
    sample_rate_hz: int
    next_request_id: int = 1


@dataclass(frozen=True)
class _DecodePreset:
    beam_size: int
    vad_filter: bool
    condition_on_previous_text: bool


_DECODE_PRESETS: dict[str, _DecodePreset] = {
    "fast": _DecodePreset(beam_size=1, vad_filter=False, condition_on_previous_text=False),
    "balanced": _DecodePreset(beam_size=5, vad_filter=True, condition_on_previous_text=False),
    "accurate": _DecodePreset(beam_size=5, vad_filter=True, condition_on_previous_text=True),
}


def _resolve_decode_preset(asr_preset: str | None) -> _DecodePreset:
    if asr_preset is None:
        return _DECODE_PRESETS["balanced"]
    normalized = asr_preset.strip().lower()
    return _DECODE_PRESETS.get(normalized, _DECODE_PRESETS["balanced"])


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _confidence_from_scores(
    avg_logprob: float | None,
    no_speech_prob: float | None,
) -> float:
    if avg_logprob is not None:
        bounded = min(0.0, avg_logprob)
        return max(0.0, min(1.0, math.exp(bounded)))
    if no_speech_prob is not None:
        return max(0.0, min(1.0, 1.0 - no_speech_prob))
    return 0.7


def _resolve_model_sample_rate_hz(model: Any) -> int:
    feature_extractor = getattr(model, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", None)
    try:
        value = int(sampling_rate)
    except (TypeError, ValueError):
        return 16000
    if value <= 0:
        return 16000
    return value


def _resample_audio_if_needed(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    *,
    source_rate_hz: int,
    target_rate_hz: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if source_rate_hz <= 0:
        raise ValueError("source_rate_hz must be positive.")
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive.")
    if audio.size == 0 or source_rate_hz == target_rate_hz:
        return audio

    target_length = max(1, int(round(audio.shape[0] * target_rate_hz / source_rate_hz)))
    if target_length == audio.shape[0]:
        return audio

    source_positions = np.arange(audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(
        0.0,
        float(audio.shape[0] - 1),
        num=target_length,
        dtype=np.float64,
    )
    resampled = np.interp(
        target_positions,
        source_positions,
        audio.astype(np.float64, copy=False),
    )
    return resampled.astype(np.float32, copy=False)


def _call_model_transcribe(
    *,
    model: Any,
    audio: np.ndarray[Any, np.dtype[np.float32]],
    language: str | None,
    multilingual: bool,
    beam_size: int,
    vad_filter: bool,
    condition_on_previous_text: bool,
) -> Sequence[Any]:
    try:
        segments_iter, _ = model.transcribe(
            audio,
            language=language,
            multilingual=multilingual,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
        )
    except TypeError:
        segments_iter, _ = model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
        )
    return segments_iter


def _gpu_transcribe_worker_main(
    request_queue: Any,
    response_queue: Any,
    model_reference: str,
    device: str,
    compute_type: str,
) -> None:
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_reference, device=device, compute_type=compute_type)
        response_queue.put(
            {
                "kind": "ready",
                "sample_rate_hz": _resolve_model_sample_rate_hz(model),
            }
        )
    except Exception as exc:
        response_queue.put({"kind": "startup_error", "error": str(exc)})
        return

    while True:
        request = request_queue.get()
        if not isinstance(request, dict):
            continue
        kind = request.get("kind")
        if kind == "shutdown":
            return
        if kind != "transcribe":
            continue

        request_id = int(request.get("id", 0))
        probe = bool(request.get("probe", False))
        language = request.get("language")
        multilingual = bool(request.get("multilingual", False))
        asr_preset = request.get("asr_preset")
        audio = np.frombuffer(bytes(request.get("audio", b"")), dtype=np.float32).copy()

        try:
            if probe:
                segments_iter = _call_model_transcribe(
                    model=model,
                    audio=audio,
                    language="en",
                    multilingual=False,
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                for _ in segments_iter:
                    break
                response_queue.put({"kind": "result", "id": request_id, "segments": []})
                continue

            decode_preset = _resolve_decode_preset(
                asr_preset if isinstance(asr_preset, str) else None
            )
            segments_iter = _call_model_transcribe(
                model=model,
                audio=audio,
                language=language if isinstance(language, str) else None,
                multilingual=multilingual,
                beam_size=decode_preset.beam_size,
                vad_filter=decode_preset.vad_filter,
                condition_on_previous_text=decode_preset.condition_on_previous_text,
            )
            parsed_payload: list[dict[str, Any]] = []
            for segment in segments_iter:
                text = str(getattr(segment, "text", "")).strip()
                if not text:
                    continue
                start_s = float(getattr(segment, "start", 0.0))
                end_s = float(getattr(segment, "end", start_s))
                avg_logprob = _maybe_float(getattr(segment, "avg_logprob", None))
                no_speech_prob = _maybe_float(getattr(segment, "no_speech_prob", None))
                parsed_payload.append(
                    {
                        "text": text,
                        "start_s": start_s,
                        "end_s": end_s,
                        "confidence": _confidence_from_scores(avg_logprob, no_speech_prob),
                    }
                )
            response_queue.put(
                {
                    "kind": "result",
                    "id": request_id,
                    "segments": parsed_payload,
                }
            )
        except Exception as exc:
            response_queue.put({"kind": "error", "id": request_id, "error": str(exc)})


class FasterWhisperEngine:
    name = "faster-whisper"
    _model_cache: ClassVar[dict[tuple[str, str, str], Any]] = {}
    _warmed_models: ClassVar[set[tuple[str, str, str]]] = set()
    _cache_lock: ClassVar[Lock] = Lock()
    _GPU_PROBE_TIMEOUT_S: ClassVar[float] = 4.0
    _GPU_TRANSCRIBE_TIMEOUT_S: ClassVar[float] = 12.0

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        availability_probe: Callable[[], bool] | None = None,
        model_directory: Path | None = None,
    ) -> None:
        self._model_factory = model_factory or self._default_model_factory
        self._availability_probe = availability_probe or self._default_availability_probe
        self._model_directory = model_directory
        self._gpu_worker: _GpuWorkerHandle | None = None
        self._gpu_probe_complete = False
        self._gpu_disabled_reason: str | None = None

    def __del__(self) -> None:
        try:
            self._shutdown_gpu_worker()
        except Exception:
            return

    def is_available(self) -> bool:
        return self._availability_probe()

    def transcribe(self, request: TranscriptionRequest) -> Sequence[TranscriptSegment]:
        if not self.is_available():
            raise EngineUnavailableError(
                "faster-whisper is not installed. Install optional dependency group: narada[asr]."
            )
        if request.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if len(request.pcm_bytes) % 2 != 0:
            raise ValueError("PCM input must have an even byte length.")
        if not request.pcm_bytes:
            return []

        source_audio = self._pcm16le_to_float_array(request.pcm_bytes)
        language = self._choose_language(request.languages)
        multilingual = len([lang for lang in request.languages if lang != "auto"]) > 1
        model_reference = self._resolve_model_reference(request.model)
        device, compute_type = self.resolve_compute_backend(request.compute)

        if device in {"auto", "cuda"}:
            guarded = self._transcribe_with_gpu_guard(
                request=request,
                model_reference=model_reference,
                device=device,
                compute_type=compute_type,
                source_audio=source_audio,
                source_rate_hz=request.sample_rate_hz,
                language=language,
                multilingual=multilingual,
                asr_preset=request.asr_preset,
            )
            if guarded is not None:
                return guarded
            device, compute_type = "cpu", "int8"

        return self._transcribe_in_process(
            request=request,
            model_reference=model_reference,
            device=device,
            compute_type=compute_type,
            source_audio=source_audio,
            source_rate_hz=request.sample_rate_hz,
            language=language,
            multilingual=multilingual,
            asr_preset=request.asr_preset,
        )

    def _transcribe_with_gpu_guard(
        self,
        *,
        request: TranscriptionRequest,
        model_reference: str,
        device: str,
        compute_type: str,
        source_audio: np.ndarray[Any, np.dtype[np.float32]],
        source_rate_hz: int,
        language: str | None,
        multilingual: bool,
        asr_preset: str,
    ) -> list[TranscriptSegment] | None:
        if self._gpu_disabled_reason is not None:
            return None

        try:
            payload = self._transcribe_gpu_guarded(
                model_reference=model_reference,
                device=device,
                compute_type=compute_type,
                source_audio=source_audio,
                source_rate_hz=source_rate_hz,
                language=language,
                multilingual=multilingual,
                asr_preset=asr_preset,
            )
            return self._segments_from_worker_payload(payload)
        except _GpuWorkerError as exc:
            if isinstance(exc, (_GpuWorkerTimeoutError, _GpuWorkerExitError)):
                self._disable_gpu_for_session(
                    reason=str(exc),
                    device=device,
                    compute_type=compute_type,
                )
                return None
            if isinstance(exc, _GpuWorkerRuntimeError) and self._should_retry_on_cpu(
                exc,
                device=device,
            ):
                self._disable_gpu_for_session(
                    reason=str(exc),
                    device=device,
                    compute_type=compute_type,
                )
                return None
            raise EngineUnavailableError(
                f"faster-whisper failed to transcribe with model '{request.model}': {exc}"
            ) from exc

    def _transcribe_gpu_guarded(
        self,
        *,
        model_reference: str,
        device: str,
        compute_type: str,
        source_audio: np.ndarray[Any, np.dtype[np.float32]],
        source_rate_hz: int,
        language: str | None,
        multilingual: bool,
        asr_preset: str,
    ) -> list[dict[str, Any]]:
        worker = self._ensure_gpu_worker((model_reference, device, compute_type))

        if not self._gpu_probe_complete:
            probe_samples = max(worker.sample_rate_hz // 4, 4000)
            probe_audio = np.zeros(probe_samples, dtype=np.float32)
            self._run_gpu_worker_request(
                worker=worker,
                audio=probe_audio,
                language="en",
                multilingual=False,
                asr_preset="fast",
                timeout_s=self._GPU_PROBE_TIMEOUT_S,
                probe=True,
            )
            self._gpu_probe_complete = True

        worker_audio = self._resample_audio_if_needed(
            source_audio,
            source_rate_hz=source_rate_hz,
            target_rate_hz=worker.sample_rate_hz,
        )
        return self._run_gpu_worker_request(
            worker=worker,
            audio=worker_audio,
            language=language,
            multilingual=multilingual,
            asr_preset=asr_preset,
            timeout_s=self._GPU_TRANSCRIBE_TIMEOUT_S,
            probe=False,
        )

    def _ensure_gpu_worker(self, key: tuple[str, str, str]) -> _GpuWorkerHandle:
        if (
            self._gpu_worker is not None
            and self._gpu_worker.key == key
            and self._gpu_worker.process.is_alive()
        ):
            return self._gpu_worker

        self._shutdown_gpu_worker()
        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue()
        response_queue = ctx.Queue()
        process = ctx.Process(
            target=_gpu_transcribe_worker_main,
            args=(request_queue, response_queue, key[0], key[1], key[2]),
            daemon=True,
            name="narada-faster-whisper-gpu",
        )
        process.start()

        try:
            ready = response_queue.get(timeout=self._GPU_PROBE_TIMEOUT_S)
        except queue.Empty as exc:
            self._terminate_worker_process(process, request_queue, response_queue)
            raise _GpuWorkerTimeoutError(
                "Timed out while starting faster-whisper GPU worker."
            ) from exc

        if not isinstance(ready, dict):
            self._terminate_worker_process(process, request_queue, response_queue)
            raise _GpuWorkerRuntimeError("GPU worker returned an invalid startup response.")

        kind = str(ready.get("kind", ""))
        if kind == "startup_error":
            self._terminate_worker_process(process, request_queue, response_queue)
            error = str(ready.get("error", "unknown startup error"))
            raise _GpuWorkerRuntimeError(error)
        if kind != "ready":
            self._terminate_worker_process(process, request_queue, response_queue)
            raise _GpuWorkerRuntimeError("GPU worker did not signal readiness.")

        sample_rate_hz = ready.get("sample_rate_hz")
        if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            self._terminate_worker_process(process, request_queue, response_queue)
            raise _GpuWorkerRuntimeError("GPU worker reported an invalid model sample rate.")

        worker = _GpuWorkerHandle(
            key=key,
            process=process,
            request_queue=request_queue,
            response_queue=response_queue,
            sample_rate_hz=sample_rate_hz,
        )
        self._gpu_worker = worker
        self._gpu_probe_complete = False
        return worker

    def _run_gpu_worker_request(
        self,
        *,
        worker: _GpuWorkerHandle,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        language: str | None,
        multilingual: bool,
        asr_preset: str,
        timeout_s: float,
        probe: bool,
    ) -> list[dict[str, Any]]:
        request_id = worker.next_request_id
        worker.next_request_id += 1
        request_payload = {
            "kind": "transcribe",
            "id": request_id,
            "audio": audio.astype(np.float32, copy=False).tobytes(),
            "language": language,
            "multilingual": multilingual,
            "asr_preset": asr_preset,
            "probe": probe,
        }
        try:
            worker.request_queue.put(request_payload)
        except Exception as exc:
            self._shutdown_gpu_worker()
            raise _GpuWorkerExitError("GPU worker is unavailable.") from exc

        try:
            response = worker.response_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            alive = worker.process.is_alive()
            self._shutdown_gpu_worker()
            if alive:
                raise _GpuWorkerTimeoutError(
                    f"GPU worker exceeded timeout ({timeout_s:.1f}s)."
                ) from exc
            raise _GpuWorkerExitError("GPU worker exited before responding.") from exc

        if not isinstance(response, dict):
            self._shutdown_gpu_worker()
            raise _GpuWorkerRuntimeError("GPU worker returned an invalid response.")

        if int(response.get("id", request_id)) != request_id:
            self._shutdown_gpu_worker()
            raise _GpuWorkerRuntimeError("GPU worker response ID mismatch.")

        kind = str(response.get("kind", ""))
        if kind == "result":
            raw_segments = response.get("segments", [])
            if isinstance(raw_segments, list):
                return [item for item in raw_segments if isinstance(item, dict)]
            return []
        if kind == "error":
            message = str(response.get("error", "unknown GPU transcription error"))
            raise _GpuWorkerRuntimeError(message)

        self._shutdown_gpu_worker()
        raise _GpuWorkerRuntimeError(f"Unexpected GPU worker response kind '{kind}'.")

    def _shutdown_gpu_worker(self) -> None:
        worker = self._gpu_worker
        self._gpu_worker = None
        self._gpu_probe_complete = False
        if worker is None:
            return
        self._terminate_worker_process(
            worker.process,
            worker.request_queue,
            worker.response_queue,
        )

    @staticmethod
    def _terminate_worker_process(
        process: mp.Process,
        request_queue: Any,
        response_queue: Any,
    ) -> None:
        try:
            request_queue.put({"kind": "shutdown"})
        except Exception:
            pass
        try:
            process.join(timeout=0.5)
        except Exception:
            pass
        if process.is_alive():
            try:
                process.terminate()
                process.join(timeout=1.0)
            except Exception:
                pass
        for queue_obj in (request_queue, response_queue):
            try:
                queue_obj.close()
            except Exception:
                pass
            try:
                queue_obj.join_thread()
            except Exception:
                pass

    def _disable_gpu_for_session(self, *, reason: str, device: str, compute_type: str) -> None:
        if self._gpu_disabled_reason is None:
            self._gpu_disabled_reason = reason
            logger.warning(
                "faster-whisper %s/%s failed (%s); disabling GPU for this session "
                "and using cpu/int8.",
                device,
                compute_type,
                reason,
            )
        self._shutdown_gpu_worker()

    def _transcribe_in_process(
        self,
        *,
        request: TranscriptionRequest,
        model_reference: str,
        device: str,
        compute_type: str,
        source_audio: np.ndarray[Any, np.dtype[np.float32]],
        source_rate_hz: int,
        language: str | None,
        multilingual: bool,
        asr_preset: str,
    ) -> list[TranscriptSegment]:
        cache_key = (model_reference, device, compute_type)
        model = self._load_model(
            model_name=model_reference,
            device=device,
            compute_type=compute_type,
            cache_key=cache_key,
        )
        self._warmup_model(model=model, cache_key=cache_key, request=request)
        model_sample_rate_hz = self._resolve_model_sample_rate_hz(model)
        audio = self._resample_audio_if_needed(
            source_audio,
            source_rate_hz=source_rate_hz,
            target_rate_hz=model_sample_rate_hz,
        )
        try:
            segments_iter = self._transcribe_model(
                model=model,
                audio=audio,
                language=language,
                multilingual=multilingual,
                asr_preset=asr_preset,
            )
            return self._parse_segments(segments_iter)
        except Exception as exc:
            raise EngineUnavailableError(
                f"faster-whisper failed to transcribe with model '{request.model}': {exc}"
            ) from exc

    @staticmethod
    def _segments_from_worker_payload(
        payload: Sequence[dict[str, Any]],
    ) -> list[TranscriptSegment]:
        parsed: list[TranscriptSegment] = []
        for item in payload:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start_s = float(item.get("start_s", 0.0))
            end_s = float(item.get("end_s", start_s))
            confidence_raw = item.get("confidence", 0.7)
            confidence = float(confidence_raw) if isinstance(confidence_raw, (float, int)) else 0.7
            parsed.append(
                TranscriptSegment(
                    text=text,
                    confidence=max(0.0, min(1.0, confidence)),
                    start_s=start_s,
                    end_s=end_s,
                    is_final=True,
                )
            )
        return parsed

    @staticmethod
    def _parse_segments(segments_iter: Sequence[Any]) -> list[TranscriptSegment]:
        parsed: list[TranscriptSegment] = []
        for segment in segments_iter:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            start_s = float(getattr(segment, "start", 0.0))
            end_s = float(getattr(segment, "end", start_s))
            confidence = FasterWhisperEngine._confidence_from_segment(segment)
            parsed.append(
                TranscriptSegment(
                    text=text,
                    confidence=confidence,
                    start_s=start_s,
                    end_s=end_s,
                    is_final=True,
                )
            )
        return parsed

    def _resolve_model_reference(self, model_name: str) -> str:
        explicit_dir = self._model_directory
        env_dir = os.environ.get("NARADA_MODEL_DIR_FASTER_WHISPER")
        if explicit_dir is None and env_dir:
            explicit_dir = Path(env_dir)
        resolved = resolve_faster_whisper_model_path(model_name, explicit_dir)
        if resolved.exists():
            return str(resolved)
        return model_name

    @classmethod
    def clear_cache_for_tests(cls) -> None:
        with cls._cache_lock:
            cls._model_cache.clear()
            cls._warmed_models.clear()

    @staticmethod
    def _default_availability_probe() -> bool:
        return find_spec("faster_whisper") is not None

    @staticmethod
    def _default_model_factory(model_name: str, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel

        return WhisperModel(model_name, device=device, compute_type=compute_type)

    @staticmethod
    def _choose_language(languages: tuple[str, ...]) -> str | None:
        explicit = [lang for lang in languages if lang != "auto"]
        if len(explicit) == 1:
            return explicit[0]
        return None

    @staticmethod
    def _pcm16le_to_float_array(pcm_bytes: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return (samples.astype(np.float32) / 32768.0).copy()

    @staticmethod
    def _confidence_from_segment(segment: Any) -> float:
        avg_logprob = _maybe_float(getattr(segment, "avg_logprob", None))
        no_speech_prob = _maybe_float(getattr(segment, "no_speech_prob", None))
        return _confidence_from_scores(avg_logprob, no_speech_prob)

    @staticmethod
    def _resolve_model_sample_rate_hz(model: Any) -> int:
        return _resolve_model_sample_rate_hz(model)

    @staticmethod
    def _resample_audio_if_needed(
        audio: np.ndarray[Any, np.dtype[np.float32]],
        *,
        source_rate_hz: int,
        target_rate_hz: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return _resample_audio_if_needed(
            audio,
            source_rate_hz=source_rate_hz,
            target_rate_hz=target_rate_hz,
        )

    @staticmethod
    def _transcribe_model(
        *,
        model: Any,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        language: str | None,
        multilingual: bool,
        asr_preset: str,
    ) -> Sequence[Any]:
        decode_preset = _resolve_decode_preset(asr_preset)
        return _call_model_transcribe(
            model=model,
            audio=audio,
            language=language,
            multilingual=multilingual,
            beam_size=decode_preset.beam_size,
            vad_filter=decode_preset.vad_filter,
            condition_on_previous_text=decode_preset.condition_on_previous_text,
        )

    @staticmethod
    def _should_retry_on_cpu(exc: Exception, *, device: str) -> bool:
        if device not in {"auto", "cuda"}:
            return False
        message = str(exc).lower()
        cuda_runtime_hints = (
            "cublas",
            "cudnn",
            "cuda",
            "cannot be loaded",
            "cuda runtime",
            "cuda driver",
        )
        return any(hint in message for hint in cuda_runtime_hints)

    @staticmethod
    def resolve_compute_backend(compute: str) -> tuple[str, str]:
        normalized = compute.strip().lower()
        if normalized == "auto":
            return "auto", "int8"
        if normalized == "cpu":
            return "cpu", "int8"
        if normalized == "cuda":
            return "cuda", "float16"
        if normalized == "metal":
            if platform.system().lower() == "darwin":
                logger.warning(
                    "faster-whisper does not expose a dedicated 'metal' device; using auto."
                )
                return "auto", "int8"
            raise EngineUnavailableError("compute=metal is only valid on macOS.")
        raise EngineUnavailableError(f"Unsupported compute backend '{compute}' for faster-whisper.")

    def _load_model(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        cache_key: tuple[str, str, str],
    ) -> Any:
        with self._cache_lock:
            existing = self._model_cache.get(cache_key)
            if existing is not None:
                return existing

        try:
            loaded = self._model_factory(model_name, device, compute_type)
        except Exception as exc:
            raise EngineUnavailableError(
                f"Failed to load faster-whisper model '{model_name}' "
                f"on {device}/{compute_type}: {exc}"
            ) from exc

        with self._cache_lock:
            self._model_cache[cache_key] = loaded
        return loaded

    def _warmup_model(
        self,
        *,
        model: Any,
        cache_key: tuple[str, str, str],
        request: TranscriptionRequest,
    ) -> None:
        with self._cache_lock:
            if cache_key in self._warmed_models:
                return

        model_sample_rate_hz = self._resolve_model_sample_rate_hz(model)
        silence = np.zeros(min(max(model_sample_rate_hz // 2, 4000), 32000), dtype=np.float32)
        language = self._choose_language(request.languages)
        try:
            segments = _call_model_transcribe(
                model=model,
                audio=silence,
                language=language,
                multilingual=False,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            for _ in segments:
                break
        except Exception as exc:
            logger.debug("faster-whisper warmup failed: %s", exc)

        with self._cache_lock:
            self._warmed_models.add(cache_key)
