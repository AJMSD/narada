from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import ClassVar

from narada.asr.base import EngineUnavailableError, TranscriptionRequest, TranscriptSegment
from narada.asr.model_discovery import resolve_whisper_cpp_model_path

logger = logging.getLogger("narada.asr.whisper_cpp")


@dataclass(frozen=True)
class WhisperCppRuntime:
    cli_path: str
    model_path: Path
    gpu_layers: int


class WhisperCppEngine:
    name = "whisper-cpp"
    _runtime_cache: ClassVar[dict[tuple[str, str], WhisperCppRuntime]] = {}
    _warmed: ClassVar[set[tuple[str, str]]] = set()
    _cache_lock: ClassVar[Lock] = Lock()

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
        has_python_binding = find_spec("whispercpp") is not None
        has_cli_binary = self._which_fn("whisper-cli") is not None
        return has_python_binding or has_cli_binary

    def transcribe(self, request: TranscriptionRequest) -> Sequence[TranscriptSegment]:
        if not self.is_available():
            raise EngineUnavailableError(
                "whisper.cpp runtime is not available. Install whispercpp "
                "or provide whisper-cli binary."
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
                "-ng",
                str(runtime.gpu_layers),
            ]
            language = self._choose_language(request.languages)
            if language is not None:
                cmd.extend(["-l", language])

            result = self._run_fn(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise EngineUnavailableError(
                    f"whisper.cpp transcription failed: {(result.stderr or result.stdout).strip()}"
                )

            text_output = output_base.with_suffix(".txt")
            if not text_output.exists():
                return []
            text = text_output.read_text(encoding="utf-8").strip()
            if not text:
                return []

            duration_s = len(request.pcm_bytes) / 2.0 / request.sample_rate_hz
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
    def clear_cache_for_tests(cls) -> None:
        with cls._cache_lock:
            cls._runtime_cache.clear()
            cls._warmed.clear()

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
    def _resolve_compute(compute: str) -> int:
        normalized = compute.strip().lower()
        if normalized in {"auto", "cpu"}:
            return 0
        if normalized == "cuda":
            return 99
        if normalized == "metal":
            return 99
        raise EngineUnavailableError(f"Unsupported compute backend '{compute}' for whisper.cpp.")

    def _resolve_runtime(self, model_name: str, compute: str) -> WhisperCppRuntime:
        cache_key = (model_name, compute.lower())
        with self._cache_lock:
            cached = self._runtime_cache.get(cache_key)
            if cached is not None:
                return cached

        cli_path = self._which_fn("whisper-cli")
        if cli_path is None:
            raise EngineUnavailableError(
                "whisper-cli binary not found on PATH. Install whisper.cpp CLI for transcription."
            )
        model_path = self._resolve_model_path(model_name)
        gpu_layers = self._resolve_compute(compute)
        runtime = WhisperCppRuntime(
            cli_path=cli_path,
            model_path=model_path,
            gpu_layers=gpu_layers,
        )

        with self._cache_lock:
            self._runtime_cache[cache_key] = runtime
        return runtime

    @classmethod
    def _warmup(cls, runtime: WhisperCppRuntime) -> None:
        key = (str(runtime.model_path), str(runtime.gpu_layers))
        with cls._cache_lock:
            if key in cls._warmed:
                return
        # Warmup is intentionally lightweight: verify model file readability once.
        with runtime.model_path.open("rb") as handle:
            handle.read(4096)
        with cls._cache_lock:
            cls._warmed.add(key)

    @staticmethod
    def _write_pcm_to_wav(output_path: Path, pcm_bytes: bytes, sample_rate_hz: int) -> None:
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate_hz)
            wav.writeframes(pcm_bytes)
