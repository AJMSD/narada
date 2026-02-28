from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import ClassVar

from narada.asr.base import EngineUnavailableError, TranscriptionRequest, TranscriptSegment
from narada.asr.model_discovery import resolve_whisper_cpp_model_path

logger = logging.getLogger("narada.asr.whisper_cpp")

_WHISPER_CPP_TIMEOUT_PER_AUDIO_SECOND_GPU: dict[str, float] = {
    "fast": 0.80,
    "balanced": 1.10,
    "accurate": 1.40,
}

_WHISPER_CPP_TIMEOUT_PER_AUDIO_SECOND_CPU: dict[str, float] = {
    "fast": 1.20,
    "balanced": 1.90,
    "accurate": 2.60,
}


@dataclass(frozen=True)
class WhisperCliCapabilities:
    no_gpu_flag: str | None
    gpu_layers_flag: str | None
    backend_hints: tuple[str, ...]


@dataclass(frozen=True)
class WhisperCppRuntime:
    cli_path: str
    model_path: Path
    requested_compute: str
    compute_args: tuple[str, ...]
    capabilities: WhisperCliCapabilities


class WhisperCppEngine:
    name = "whisper-cpp"
    _runtime_cache: ClassVar[dict[tuple[str, str], WhisperCppRuntime]] = {}
    _warmed: ClassVar[set[tuple[str, str]]] = set()
    _capability_cache: ClassVar[dict[str, WhisperCliCapabilities]] = {}
    _cache_lock: ClassVar[Lock] = Lock()
    _TRANSCRIBE_TIMEOUT_MIN_S: ClassVar[float] = 20.0
    _TRANSCRIBE_TIMEOUT_BASE_GPU_S: ClassVar[float] = 15.0
    _TRANSCRIBE_TIMEOUT_BASE_CPU_S: ClassVar[float] = 30.0
    _TRANSCRIBE_TIMEOUT_MAX_GPU_S: ClassVar[float] = 600.0
    _TRANSCRIBE_TIMEOUT_MAX_CPU_S: ClassVar[float] = 1800.0

    def __init__(
        self,
        which_fn: Callable[[str], str | None] | None = None,
        run_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        model_directory: Path | None = None,
    ) -> None:
        self._which_fn = which_fn or shutil.which
        self._run_fn = run_fn or subprocess.run
        self._model_directory = model_directory

    def is_available(self) -> bool:
        return self._resolve_cli_path() is not None

    def transcribe(self, request: TranscriptionRequest) -> Sequence[TranscriptSegment]:
        if not self.is_available():
            raise EngineUnavailableError(
                "whisper.cpp runtime is not available. Install whisper.cpp and ensure "
                "'whisper-cli' or 'whisper-cpp' is available on PATH."
            )
        if request.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if len(request.pcm_bytes) % 2 != 0:
            raise ValueError("PCM input must have an even byte length.")
        if not request.pcm_bytes:
            return []

        runtime = self._resolve_runtime(request.model, request.compute)
        self._warmup(runtime)

        with tempfile.TemporaryDirectory(prefix="narada-whispercpp-") as tmp_dir:
            temp_root = Path(tmp_dir)
            input_wav = temp_root / "input.wav"
            output_base = temp_root / "result"
            self._write_pcm_to_wav(
                output_path=input_wav,
                pcm_bytes=request.pcm_bytes,
                sample_rate_hz=request.sample_rate_hz,
            )
            cmd = [
                runtime.cli_path,
                "-m",
                str(runtime.model_path),
                "-f",
                str(input_wav),
                "-of",
                str(output_base),
                "-otxt",
                "-np",
            ]
            language = self._choose_language(request.languages)
            duration_s = len(request.pcm_bytes) / 2.0 / request.sample_rate_hz
            retry_runtime = runtime
            if runtime.requested_compute in {"auto", "cuda", "metal"}:
                cpu_runtime = self._resolve_runtime(request.model, "cpu")
                if cpu_runtime.compute_args != runtime.compute_args:
                    retry_runtime = cpu_runtime
            result: subprocess.CompletedProcess[str] | None = None
            for attempt_idx in range(2):
                selected_runtime = runtime if attempt_idx == 0 else retry_runtime
                self._warmup(selected_runtime)
                attempt_cmd = list(cmd)
                attempt_cmd.extend(selected_runtime.compute_args)
                if language is not None:
                    attempt_cmd.extend(["-l", language])
                timeout_s = self._compute_transcribe_timeout_s(
                    audio_seconds=duration_s,
                    compute=selected_runtime.requested_compute,
                    asr_preset=request.asr_preset,
                )
                try:
                    result = self._run_fn(
                        attempt_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=timeout_s,
                    )
                    if result.returncode != 0:
                        raise EngineUnavailableError(
                            "whisper.cpp transcription failed: "
                            f"{(result.stderr or result.stdout).strip()}"
                        )
                    break
                except subprocess.TimeoutExpired as exc:
                    if attempt_idx == 0:
                        logger.warning(
                            "whisper.cpp timed out after %.1fs on %s; retrying once on %s.",
                            timeout_s,
                            runtime.requested_compute,
                            retry_runtime.requested_compute,
                        )
                        continue
                    raise EngineUnavailableError(
                        "whisper.cpp transcription timed out after retry."
                    ) from exc
            if result is None:
                raise EngineUnavailableError("whisper.cpp transcription did not produce a result.")

            text_output = output_base.with_suffix(".txt")
            if not text_output.exists():
                return []
            text = text_output.read_text(encoding="utf-8").strip()
            if not text:
                return []

            return [
                TranscriptSegment(
                    text=text,
                    confidence=0.7,
                    start_s=0.0,
                    end_s=duration_s,
                    is_final=True,
                )
            ]

    @classmethod
    def _compute_transcribe_timeout_s(
        cls,
        *,
        audio_seconds: float,
        compute: str,
        asr_preset: str,
    ) -> float:
        normalized_compute = compute.strip().lower()
        normalized_preset = asr_preset.strip().lower()
        if normalized_compute == "cpu":
            factor = _WHISPER_CPP_TIMEOUT_PER_AUDIO_SECOND_CPU.get(
                normalized_preset,
                _WHISPER_CPP_TIMEOUT_PER_AUDIO_SECOND_CPU["balanced"],
            )
            raw_timeout_s = cls._TRANSCRIBE_TIMEOUT_BASE_CPU_S + (max(0.0, audio_seconds) * factor)
            return min(
                cls._TRANSCRIBE_TIMEOUT_MAX_CPU_S,
                max(cls._TRANSCRIBE_TIMEOUT_MIN_S, raw_timeout_s),
            )
        factor = _WHISPER_CPP_TIMEOUT_PER_AUDIO_SECOND_GPU.get(
            normalized_preset,
            _WHISPER_CPP_TIMEOUT_PER_AUDIO_SECOND_GPU["balanced"],
        )
        raw_timeout_s = cls._TRANSCRIBE_TIMEOUT_BASE_GPU_S + (max(0.0, audio_seconds) * factor)
        return min(
            cls._TRANSCRIBE_TIMEOUT_MAX_GPU_S,
            max(cls._TRANSCRIBE_TIMEOUT_MIN_S, raw_timeout_s),
        )

    @classmethod
    def clear_cache_for_tests(cls) -> None:
        with cls._cache_lock:
            cls._runtime_cache.clear()
            cls._warmed.clear()
            cls._capability_cache.clear()

    @staticmethod
    def _choose_language(languages: tuple[str, ...]) -> str | None:
        explicit = [lang for lang in languages if lang != "auto"]
        if len(explicit) == 1:
            return explicit[0]
        return None

    def _resolve_model_path(self, model_name: str) -> Path:
        path = resolve_whisper_cpp_model_path(model_name, self._resolve_model_directory())
        if not path.exists():
            raise EngineUnavailableError(
                f"whisper.cpp model missing: {path}. "
                "Download from https://huggingface.co/ggerganov/whisper.cpp"
            )
        return path

    def _resolve_model_directory(self) -> Path | None:
        if self._model_directory is not None:
            return self._model_directory
        modern = os.environ.get("NARADA_MODEL_DIR_WHISPER_CPP")
        if modern:
            return Path(modern)
        legacy = os.environ.get("NARADA_WHISPER_CPP_MODEL_DIR")
        if legacy:
            return Path(legacy)
        return None

    @staticmethod
    def _normalize_compute(compute: str) -> str:
        normalized = compute.strip().lower()
        if normalized in {"auto", "cpu", "cuda", "metal"}:
            return normalized
        raise EngineUnavailableError(f"Unsupported compute backend '{compute}' for whisper.cpp.")

    @staticmethod
    def _detect_backend_hint(help_text: str) -> tuple[str, ...]:
        normalized = help_text.lower()
        ordered_hints = ("cuda", "metal", "vulkan", "opencl", "hip", "blas")
        detected = [hint for hint in ordered_hints if hint in normalized]
        return tuple(detected)

    def probe_cli_capabilities(self) -> WhisperCliCapabilities:
        cli_path = self._resolve_cli_path()
        if cli_path is None:
            raise EngineUnavailableError(
                "whisper.cpp CLI binary not found on PATH. "
                "Install whisper.cpp CLI for transcription."
            )
        with self._cache_lock:
            cached = self._capability_cache.get(cli_path)
            if cached is not None:
                return cached

        help_text = ""
        for help_arg in ("-h", "--help"):
            result = self._run_fn(
                [cli_path, help_arg],
                capture_output=True,
                text=True,
                check=False,
            )
            help_text = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
            if help_text:
                break

        if not help_text:
            logger.warning(
                "Could not read whisper-cli help output; "
                "compute compatibility detection is limited."
            )

        no_gpu_flag: str | None = None
        if "--no-gpu" in help_text:
            no_gpu_flag = "--no-gpu"
        elif re.search(r"(?<![\w-])-ng(?![\w-])", help_text):
            no_gpu_flag = "-ng"

        gpu_layers_flag: str | None = None
        if "--gpu-layers" in help_text:
            gpu_layers_flag = "--gpu-layers"
        elif re.search(r"(?<![\w-])-ngl(?![\w-])", help_text):
            gpu_layers_flag = "-ngl"

        capabilities = WhisperCliCapabilities(
            no_gpu_flag=no_gpu_flag,
            gpu_layers_flag=gpu_layers_flag,
            backend_hints=self._detect_backend_hint(help_text),
        )
        with self._cache_lock:
            self._capability_cache[cli_path] = capabilities
        return capabilities

    def _build_compute_args(
        self, compute: str, capabilities: WhisperCliCapabilities
    ) -> tuple[str, ...]:
        normalized = self._normalize_compute(compute)
        if normalized == "cpu":
            if capabilities.no_gpu_flag is not None:
                return (capabilities.no_gpu_flag,)
            logger.warning(
                "whisper.cpp CLI does not advertise a no-GPU flag; compute=cpu "
                "cannot be strictly enforced for this runtime."
            )
            return ()
        return ()

    def _resolve_runtime(self, model_name: str, compute: str) -> WhisperCppRuntime:
        normalized_compute = self._normalize_compute(compute)
        cache_key = (model_name, normalized_compute)
        with self._cache_lock:
            cached = self._runtime_cache.get(cache_key)
            if cached is not None:
                return cached

        cli_path = self._resolve_cli_path()
        if cli_path is None:
            raise EngineUnavailableError(
                "whisper.cpp CLI binary not found on PATH. "
                "Install whisper.cpp CLI for transcription."
            )
        model_path = self._resolve_model_path(model_name)
        capabilities = self.probe_cli_capabilities()
        compute_args = self._build_compute_args(normalized_compute, capabilities)
        runtime = WhisperCppRuntime(
            cli_path=cli_path,
            model_path=model_path,
            requested_compute=normalized_compute,
            compute_args=compute_args,
            capabilities=capabilities,
        )
        hint_text = (
            ", ".join(capabilities.backend_hints) if capabilities.backend_hints else "unknown"
        )
        args_text = " ".join(compute_args) if compute_args else "<none>"
        logger.info(
            "whisper.cpp compute resolved: request=%s args=%s backend_hints=%s",
            normalized_compute,
            args_text,
            hint_text,
        )

        with self._cache_lock:
            self._runtime_cache[cache_key] = runtime
        return runtime

    @classmethod
    def _warmup(cls, runtime: WhisperCppRuntime) -> None:
        key = (str(runtime.model_path), " ".join(runtime.compute_args))
        with cls._cache_lock:
            if key in cls._warmed:
                return
        # Warmup is intentionally lightweight: verify model file readability once.
        with runtime.model_path.open("rb") as handle:
            handle.read(4096)
        with cls._cache_lock:
            cls._warmed.add(key)

    def _resolve_cli_path(self) -> str | None:
        for candidate in ("whisper-cli", "whisper-cpp"):
            resolved = self._which_fn(candidate)
            if resolved is not None:
                return resolved
        return None

    @staticmethod
    def _write_pcm_to_wav(output_path: Path, pcm_bytes: bytes, sample_rate_hz: int) -> None:
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate_hz)
            wav.writeframes(pcm_bytes)
