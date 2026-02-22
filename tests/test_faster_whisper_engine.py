import numpy as np
import pytest

from narada.asr.base import EngineUnavailableError, TranscriptionRequest
from narada.asr.faster_whisper_engine import (
    FasterWhisperEngine,
    _GpuWorkerRuntimeError,
    _GpuWorkerTimeoutError,
)


class _FakeSegment:
    def __init__(
        self,
        text: str,
        start: float = 0.0,
        end: float = 1.0,
        avg_logprob: float = -0.2,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.avg_logprob = avg_logprob


class _FakeModel:
    def __init__(self, model_sample_rate_hz: int = 16000) -> None:
        self.calls: list[dict[str, object]] = []
        self.audio_inputs: list[np.ndarray] = []
        self.feature_extractor = type(
            "FeatureExtractor",
            (),
            {"sampling_rate": model_sample_rate_hz},
        )()

    def transcribe(
        self, audio: object, **kwargs: object
    ) -> tuple[list[_FakeSegment], dict[str, object]]:
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        self.audio_inputs.append(audio)
        self.calls.append(dict(kwargs))
        if kwargs.get("beam_size") == 1:
            return ([], {})
        return ([_FakeSegment("hello world")], {})


class _CudaFailingModel:
    def __init__(self) -> None:
        self.feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self, audio: object, **kwargs: object
    ) -> tuple[list[_FakeSegment], dict[str, object]]:
        assert isinstance(audio, np.ndarray)
        self.calls.append(dict(kwargs))
        if kwargs.get("beam_size") == 1:
            return ([], {})
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


def _main_audio_inputs(model: _FakeModel) -> list[np.ndarray]:
    return [
        audio
        for audio, call in zip(model.audio_inputs, model.calls, strict=True)
        if call.get("beam_size") != 1
    ]


def test_faster_whisper_caches_and_warms_once() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    fake_model = _FakeModel()
    factory_calls: list[tuple[str, str, str]] = []

    def factory(model_name: str, device: str, compute_type: str) -> _FakeModel:
        factory_calls.append((model_name, device, compute_type))
        return fake_model

    engine = FasterWhisperEngine(model_factory=factory, availability_probe=lambda: True)
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("en",),
        model="small",
        compute="cpu",
    )
    first = engine.transcribe(request)
    second = engine.transcribe(request)

    assert len(factory_calls) == 1
    warmup_calls = [call for call in fake_model.calls if call.get("beam_size") == 1]
    assert len(warmup_calls) == 1
    assert len(fake_model.audio_inputs) == len(fake_model.calls)
    assert all(audio.dtype == np.float32 for audio in fake_model.audio_inputs)
    assert first[0].text == "hello world"
    assert second[0].text == "hello world"


def test_faster_whisper_pcm_conversion_returns_numpy_float32() -> None:
    pcm = b"\x00\x00\xff\x7f\x00\x80"
    audio = FasterWhisperEngine._pcm16le_to_float_array(pcm)  # noqa: SLF001

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape == (3,)


def test_faster_whisper_rejects_odd_pcm_length() -> None:
    engine = FasterWhisperEngine(
        model_factory=lambda *_: _FakeModel(), availability_probe=lambda: True
    )
    request = TranscriptionRequest(
        pcm_bytes=b"\x01",
        sample_rate_hz=16000,
        languages=("en",),
        model="small",
        compute="cpu",
    )
    with pytest.raises(ValueError):
        engine.transcribe(request)


def test_faster_whisper_resamples_non_16k_audio_before_transcribe() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    fake_model = _FakeModel(model_sample_rate_hz=16000)
    engine = FasterWhisperEngine(
        model_factory=lambda *_args: fake_model,
        availability_probe=lambda: True,
    )
    seconds = 1
    sample_rate_hz = 48000
    pcm = (np.zeros(sample_rate_hz * seconds, dtype=np.int16)).tobytes()
    request = TranscriptionRequest(
        pcm_bytes=pcm,
        sample_rate_hz=sample_rate_hz,
        languages=("en",),
        model="small",
        compute="cpu",
    )

    _ = engine.transcribe(request)

    main_inputs = _main_audio_inputs(fake_model)
    assert len(main_inputs) == 1
    assert main_inputs[0].dtype == np.float32
    assert main_inputs[0].shape == (16000,)


def test_faster_whisper_keeps_audio_unchanged_at_model_rate() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    fake_model = _FakeModel(model_sample_rate_hz=16000)
    engine = FasterWhisperEngine(
        model_factory=lambda *_args: fake_model,
        availability_probe=lambda: True,
    )
    sample_rate_hz = 16000
    pcm = (np.zeros(sample_rate_hz, dtype=np.int16)).tobytes()
    request = TranscriptionRequest(
        pcm_bytes=pcm,
        sample_rate_hz=sample_rate_hz,
        languages=("en",),
        model="small",
        compute="cpu",
    )

    _ = engine.transcribe(request)

    main_inputs = _main_audio_inputs(fake_model)
    assert len(main_inputs) == 1
    assert main_inputs[0].shape == (16000,)


def test_faster_whisper_auto_retries_on_cpu_when_cuda_runtime_fails() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    cpu_model = _FakeModel(model_sample_rate_hz=16000)
    gpu_attempts = {"count": 0}

    def factory(model_name: str, device: str, compute_type: str) -> object:
        if device == "cpu":
            return cpu_model
        raise AssertionError(f"Unexpected device: {device}")

    engine = FasterWhisperEngine(model_factory=factory, availability_probe=lambda: True)
    original_guarded = engine._transcribe_gpu_guarded

    def fake_guarded(**kwargs: object) -> list[dict[str, object]]:
        gpu_attempts["count"] += 1
        raise _GpuWorkerRuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    engine._transcribe_gpu_guarded = fake_guarded  # type: ignore[method-assign]
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("auto",),
        model="small",
        compute="auto",
    )

    result = engine.transcribe(request)
    engine._transcribe_gpu_guarded = original_guarded  # type: ignore[method-assign]

    assert result[0].text == "hello world"
    assert gpu_attempts["count"] == 1
    assert engine._gpu_disabled_reason is not None


def test_faster_whisper_gpu_timeout_falls_back_and_disables_gpu() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    cpu_model = _FakeModel(model_sample_rate_hz=16000)
    gpu_attempts = {"count": 0}

    engine = FasterWhisperEngine(
        model_factory=lambda *_args: cpu_model,
        availability_probe=lambda: True,
    )
    original_guarded = engine._transcribe_gpu_guarded

    def fake_guarded(**kwargs: object) -> list[dict[str, object]]:
        gpu_attempts["count"] += 1
        raise _GpuWorkerTimeoutError("GPU worker exceeded timeout (12.0s).")

    engine._transcribe_gpu_guarded = fake_guarded  # type: ignore[method-assign]
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("auto",),
        model="small",
        compute="auto",
    )

    result = engine.transcribe(request)
    second = engine.transcribe(request)
    engine._transcribe_gpu_guarded = original_guarded  # type: ignore[method-assign]

    assert result[0].text == "hello world"
    assert second[0].text == "hello world"
    assert gpu_attempts["count"] == 1
    assert engine._gpu_disabled_reason is not None


def test_faster_whisper_gpu_materialization_error_falls_back_to_cpu() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    cpu_model = _FakeModel(model_sample_rate_hz=16000)

    engine = FasterWhisperEngine(
        model_factory=lambda *_args: cpu_model,
        availability_probe=lambda: True,
    )
    original_guarded = engine._transcribe_gpu_guarded

    def fake_guarded(**kwargs: object) -> list[dict[str, object]]:
        raise _GpuWorkerRuntimeError(
            "CUDA runtime failure while materializing segments: cublas cannot be loaded"
        )

    engine._transcribe_gpu_guarded = fake_guarded  # type: ignore[method-assign]
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("auto",),
        model="small",
        compute="auto",
    )

    result = engine.transcribe(request)
    engine._transcribe_gpu_guarded = original_guarded  # type: ignore[method-assign]
    assert result[0].text == "hello world"
    assert engine._gpu_disabled_reason is not None


def test_faster_whisper_compute_metal_invalid_on_non_macos() -> None:
    with pytest.raises(EngineUnavailableError):
        FasterWhisperEngine.resolve_compute_backend("metal")
