from __future__ import annotations

import array
import logging
import math
import os
import platform
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

from narada.asr.base import EngineUnavailableError, TranscriptionRequest, TranscriptSegment
from narada.asr.model_discovery import resolve_faster_whisper_model_path

logger = logging.getLogger("narada.asr.faster_whisper")
ModelFactory = Callable[[str, str, str], Any]


class FasterWhisperEngine:
    name = "faster-whisper"
    _model_cache: ClassVar[dict[tuple[str, str, str], Any]] = {}
    _warmed_models: ClassVar[set[tuple[str, str, str]]] = set()
    _cache_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        availability_probe: Callable[[], bool] | None = None,
        model_directory: Path | None = None,
    ) -> None:
        self._model_factory = model_factory or self._default_model_factory
        self._availability_probe = availability_probe or self._default_availability_probe
        self._model_directory = model_directory

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

        device, compute_type = self.resolve_compute_backend(request.compute)
        model_reference = self._resolve_model_reference(request.model)
        cache_key = (model_reference, device, compute_type)
        model = self._load_model(
            model_name=model_reference,
            device=device,
            compute_type=compute_type,
            cache_key=cache_key,
        )
        self._warmup_model(model=model, cache_key=cache_key, request=request)

        audio = self._pcm16le_to_float_list(request.pcm_bytes)
        language = self._choose_language(request.languages)
        multilingual = len([lang for lang in request.languages if lang != "auto"]) > 1

        segments_iter: Sequence[Any]
        try:
            segments_iter, _ = model.transcribe(
                audio,
                language=language,
                multilingual=multilingual,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
        except TypeError:
            segments_iter, _ = model.transcribe(
                audio,
                language=language,
                beam_size=5,
            )
        except Exception as exc:
            raise EngineUnavailableError(
                f"faster-whisper failed to transcribe with model '{request.model}': {exc}"
            ) from exc

        parsed: list[TranscriptSegment] = []
        for segment in segments_iter:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            start_s = float(getattr(segment, "start", 0.0))
            end_s = float(getattr(segment, "end", start_s))
            confidence = self._confidence_from_segment(segment)
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
    def _pcm16le_to_float_list(pcm_bytes: bytes) -> list[float]:
        samples = array.array("h")
        samples.frombytes(pcm_bytes)
        return [sample / 32768.0 for sample in samples]

    @staticmethod
    def _confidence_from_segment(segment: Any) -> float:
        avg_logprob = getattr(segment, "avg_logprob", None)
        if isinstance(avg_logprob, (float, int)):
            bounded = min(0.0, float(avg_logprob))
            return max(0.0, min(1.0, math.exp(bounded)))
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        if isinstance(no_speech_prob, (float, int)):
            return max(0.0, min(1.0, 1.0 - float(no_speech_prob)))
        return 0.7

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

        silence = [0.0] * min(max(request.sample_rate_hz // 2, 4000), 32000)
        language = self._choose_language(request.languages)
        try:
            segments, _ = model.transcribe(
                silence,
                language=language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            for _ in segments:
                break
        except TypeError:
            try:
                segments, _ = model.transcribe(silence, language=language, beam_size=1)
                for _ in segments:
                    break
            except Exception as exc:
                logger.debug("faster-whisper warmup fallback failed: %s", exc)
        except Exception as exc:
            logger.debug("faster-whisper warmup failed: %s", exc)

        with self._cache_lock:
            self._warmed_models.add(cache_key)
