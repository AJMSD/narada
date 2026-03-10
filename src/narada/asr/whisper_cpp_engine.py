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


@dataclass(frozen=True)
class WhisperCppComputeResolution:
    requested_compute: str
    effective_compute: str
    warning: str | None = None


class WhisperCppEngine:
    name = "whisper-cpp"
    _runtime_cache: ClassVar[dict[tuple[str, str], WhisperCppRuntime]] = {}
    _warmed: ClassVar[set[tuple[str, str]]] = set()
    _capability_cache: ClassVar[dict[str, WhisperCliCapabilities]] = {}
    _cache_lock: ClassVar[Lock] = Lock()
    _spawn_lock: ClassVar[Lock] = Lock()
    _TRANSCRIBE_TIMEOUT_MIN_S: ClassVar[float] = 20.0
    _TRANSCRIBE_TIMEOUT_BASE_GPU_S: ClassVar[float] = 15.0
    _TRANSCRIBE_TIMEOUT_BASE_CPU_S: ClassVar[float] = 30.0
    _TRANSCRIBE_TIMEOUT_MAX_GPU_S: ClassVar[float] = 600.0
    _TRANSCRIBE_TIMEOUT_MAX_CPU_S: ClassVar[float] = 1800.0
    _ACCELERATED_CHUNK_THRESHOLD_S: ClassVar[float] = 120.0
    _ACCELERATED_CHUNK_DURATION_S: ClassVar[float] = 60.0
    _ACCELERATED_CHUNK_OVERLAP_S: ClassVar[float] = 1.5
    _CHUNK_MERGE_MIN_OVERLAP_TOKENS: ClassVar[int] = 4

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

        compute_resolution = self.resolve_requested_compute(request.compute)
        runtime = self._resolve_runtime(request.model, compute_resolution.effective_compute)
        self._warmup(runtime)
        retry_runtime = self._resolve_retry_runtime(runtime=runtime, model_name=request.model)
        duration_s = self._pcm_duration_s(
            pcm_bytes=request.pcm_bytes,
            sample_rate_hz=request.sample_rate_hz,
        )

        with tempfile.TemporaryDirectory(prefix="narada-whispercpp-") as tmp_dir:
            temp_root = Path(tmp_dir)
            if self._should_chunk_accelerated_request(
                audio_seconds=duration_s,
                compute=runtime.requested_compute,
            ):
                text = self._transcribe_chunked_request(
                    pcm_bytes=request.pcm_bytes,
                    sample_rate_hz=request.sample_rate_hz,
                    languages=request.languages,
                    asr_preset=request.asr_preset,
                    runtime=runtime,
                    retry_runtime=retry_runtime,
                    temp_root=temp_root,
                )
            else:
                text = self._transcribe_request_text(
                    pcm_bytes=request.pcm_bytes,
                    sample_rate_hz=request.sample_rate_hz,
                    languages=request.languages,
                    asr_preset=request.asr_preset,
                    runtime=runtime,
                    retry_runtime=retry_runtime,
                    temp_root=temp_root,
                    stem="result",
                )
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
    def _pcm_duration_s(
        cls,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
    ) -> float:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        return len(pcm_bytes) / 2.0 / sample_rate_hz

    def _resolve_retry_runtime(
        self,
        *,
        runtime: WhisperCppRuntime,
        model_name: str,
    ) -> WhisperCppRuntime:
        retry_runtime = runtime
        if runtime.requested_compute in {"auto", "cuda", "metal"}:
            cpu_runtime = self._resolve_runtime(model_name, "cpu")
            if cpu_runtime.compute_args != runtime.compute_args:
                retry_runtime = cpu_runtime
        return retry_runtime

    @classmethod
    def _should_chunk_accelerated_request(
        cls,
        *,
        audio_seconds: float,
        compute: str,
    ) -> bool:
        normalized_compute = compute.strip().lower()
        return (
            normalized_compute in {"auto", "cuda", "metal"}
            and audio_seconds > cls._ACCELERATED_CHUNK_THRESHOLD_S
        )

    def _transcribe_chunked_request(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        languages: tuple[str, ...],
        asr_preset: str,
        runtime: WhisperCppRuntime,
        retry_runtime: WhisperCppRuntime,
        temp_root: Path,
    ) -> str:
        chunk_texts: list[str] = []
        for chunk_index, (chunk_pcm_bytes, _chunk_offset_s) in enumerate(
            self._iter_accelerated_chunks(
                pcm_bytes=pcm_bytes,
                sample_rate_hz=sample_rate_hz,
            )
        ):
            chunk_text = self._transcribe_request_text(
                pcm_bytes=chunk_pcm_bytes,
                sample_rate_hz=sample_rate_hz,
                languages=languages,
                asr_preset=asr_preset,
                runtime=runtime,
                retry_runtime=retry_runtime,
                temp_root=temp_root,
                stem=f"chunk-{chunk_index:03d}",
            )
            if chunk_text:
                chunk_texts.append(chunk_text)
        return self._merge_chunk_texts(chunk_texts)

    def _transcribe_request_text(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        languages: tuple[str, ...],
        asr_preset: str,
        runtime: WhisperCppRuntime,
        retry_runtime: WhisperCppRuntime,
        temp_root: Path,
        stem: str,
    ) -> str:
        input_wav = temp_root / f"{stem}.wav"
        output_base = temp_root / stem
        self._write_pcm_to_wav(
            output_path=input_wav,
            pcm_bytes=pcm_bytes,
            sample_rate_hz=sample_rate_hz,
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
        language = self._choose_language(languages)
        duration_s = self._pcm_duration_s(
            pcm_bytes=pcm_bytes,
            sample_rate_hz=sample_rate_hz,
        )
        self._run_transcribe_attempts(
            cmd=cmd,
            runtime=runtime,
            retry_runtime=retry_runtime,
            audio_seconds=duration_s,
            language=language,
            asr_preset=asr_preset,
        )
        text_output = output_base.with_suffix(".txt")
        if not text_output.exists():
            return ""
        return text_output.read_text(encoding="utf-8").strip()

    def _run_transcribe_attempts(
        self,
        *,
        cmd: Sequence[str],
        runtime: WhisperCppRuntime,
        retry_runtime: WhisperCppRuntime,
        audio_seconds: float,
        language: str | None,
        asr_preset: str,
    ) -> None:
        result: subprocess.CompletedProcess[str] | None = None
        for attempt_idx in range(2):
            selected_runtime = runtime if attempt_idx == 0 else retry_runtime
            self._warmup(selected_runtime)
            attempt_cmd = list(cmd)
            attempt_cmd.extend(selected_runtime.compute_args)
            if language is not None:
                attempt_cmd.extend(["-l", language])
            timeout_s = self._compute_transcribe_timeout_s(
                audio_seconds=audio_seconds,
                compute=selected_runtime.requested_compute,
                asr_preset=asr_preset,
            )
            try:
                result = self._run_command(attempt_cmd, timeout=timeout_s)
                if result.returncode != 0:
                    raise EngineUnavailableError(self._format_process_failure(result))
                return
            except subprocess.TimeoutExpired as exc:
                if attempt_idx == 0:
                    logger.debug(
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

    @classmethod
    def _iter_accelerated_chunks(
        cls,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
    ) -> list[tuple[bytes, float]]:
        sample_count = len(pcm_bytes) // 2
        if sample_count <= 0:
            return []
        chunk_samples = max(1, int(round(cls._ACCELERATED_CHUNK_DURATION_S * sample_rate_hz)))
        overlap_samples = max(0, int(round(cls._ACCELERATED_CHUNK_OVERLAP_S * sample_rate_hz)))
        stride_samples = max(1, chunk_samples - overlap_samples)

        chunks: list[tuple[bytes, float]] = []
        start_sample = 0
        while start_sample < sample_count:
            end_sample = min(sample_count, start_sample + chunk_samples)
            start_byte = start_sample * 2
            end_byte = end_sample * 2
            chunks.append((pcm_bytes[start_byte:end_byte], start_sample / sample_rate_hz))
            if end_sample >= sample_count:
                break
            start_sample += stride_samples
        return chunks

    @classmethod
    def _merge_chunk_texts(cls, texts: Sequence[str]) -> str:
        merged = ""
        for text in texts:
            chunk_text = text.strip()
            if not chunk_text:
                continue
            if not merged:
                merged = chunk_text
                continue
            merged = cls._merge_adjacent_chunk_texts(merged, chunk_text)
        return merged

    @classmethod
    def _merge_adjacent_chunk_texts(cls, left: str, right: str) -> str:
        trimmed_right = cls._trim_chunk_overlap_prefix(left, right)
        if trimmed_right is None:
            return cls._join_chunk_texts(left, right)
        return cls._join_chunk_texts(left, trimmed_right)

    @classmethod
    def _trim_chunk_overlap_prefix(cls, left: str, right: str) -> str | None:
        left_tokens, _ = cls._tokenize_chunk_text(left)
        right_tokens, right_spans = cls._tokenize_chunk_text(right)
        max_overlap = min(len(left_tokens), len(right_tokens))
        for overlap_size in range(max_overlap, cls._CHUNK_MERGE_MIN_OVERLAP_TOKENS - 1, -1):
            if left_tokens[-overlap_size:] != right_tokens[:overlap_size]:
                continue
            if overlap_size >= len(right_spans):
                return ""
            trim_start = right_spans[overlap_size][0]
            return right[trim_start:].lstrip()
        return None

    @staticmethod
    def _join_chunk_texts(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if left.endswith((" ", "\n", "\t")):
            return f"{left}{right.lstrip()}"
        return f"{left} {right}"

    @staticmethod
    def _tokenize_chunk_text(text: str) -> tuple[list[str], list[tuple[int, int]]]:
        tokens: list[str] = []
        spans: list[tuple[int, int]] = []
        for match in re.finditer(r"[A-Za-z0-9']+", text):
            tokens.append(match.group(0).lower())
            spans.append(match.span())
        return tokens, spans

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

    def resolve_requested_compute(self, compute: str) -> WhisperCppComputeResolution:
        normalized_compute = self._normalize_compute(compute)
        if normalized_compute in {"auto", "cpu"}:
            return WhisperCppComputeResolution(
                requested_compute=normalized_compute,
                effective_compute=normalized_compute,
            )

        capabilities = self.probe_cli_capabilities()
        if normalized_compute in capabilities.backend_hints:
            return WhisperCppComputeResolution(
                requested_compute=normalized_compute,
                effective_compute=normalized_compute,
            )

        hints_text = ", ".join(capabilities.backend_hints)
        hint_suffix = f" Advertised backends: {hints_text}." if hints_text else ""
        return WhisperCppComputeResolution(
            requested_compute=normalized_compute,
            effective_compute="auto",
            warning=(
                f"whisper-cli does not advertise support for compute={normalized_compute}; "
                f"using compute=auto for this session.{hint_suffix}"
            ),
        )

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
            result = self._run_command([cli_path, help_arg])
            help_text = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
            if help_text:
                break

        if not help_text:
            logger.debug(
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
            logger.debug(
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
        logger.debug(
            "whisper.cpp compute resolved: request=%s args=%s backend_hints=%s",
            normalized_compute,
            args_text,
            hint_text,
        )

        with self._cache_lock:
            self._runtime_cache[cache_key] = runtime
        return runtime

    def _run_command(
        self,
        cmd: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if self._run_fn is not subprocess.run:
            return self._run_fn(
                list(cmd),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        return self._run_subprocess(cmd, timeout=timeout)

    @classmethod
    def _run_subprocess(
        cls,
        cmd: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(cmd)
        if not cls._is_windows():
            return subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )

        with cls._spawn_lock:
            ignore_enabled = cls._set_windows_console_ctrl_handling(ignore=True)
            try:
                process = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            finally:
                if ignore_enabled:
                    cls._set_windows_console_ctrl_handling(ignore=False)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                cmd=exc.cmd,
                timeout=exc.timeout,
                output=stdout,
                stderr=stderr,
            ) from exc

        return subprocess.CompletedProcess(
            argv,
            int(process.returncode or 0),
            stdout or "",
            stderr or "",
        )

    @staticmethod
    def _is_windows() -> bool:
        return os.name == "nt"

    @staticmethod
    def _set_windows_console_ctrl_handling(*, ignore: bool) -> bool:
        if not WhisperCppEngine._is_windows():
            return False
        try:
            import ctypes

            windll = getattr(ctypes, "windll", None)
            if windll is None:
                return False
            kernel32 = getattr(windll, "kernel32", None)
            if kernel32 is None:
                return False
            set_console_ctrl_handler = getattr(kernel32, "SetConsoleCtrlHandler", None)
            if set_console_ctrl_handler is None:
                return False
            return int(set_console_ctrl_handler(None, bool(ignore))) != 0
        except Exception:
            return False

    @staticmethod
    def _format_process_failure(result: subprocess.CompletedProcess[str]) -> str:
        details = (result.stderr or result.stdout).strip()
        if details:
            return f"whisper.cpp transcription failed (exit {result.returncode}): {details}"
        return f"whisper.cpp transcription failed (exit {result.returncode})."

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
