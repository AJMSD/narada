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
        # Treat the first beam_size=1 invocation as warmup.
        if kwargs.get("beam_size") == 1 and len(self.calls) == 1:
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
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: True)
    try:

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
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_gpu_timeout_falls_back_and_disables_gpu() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    cpu_model = _FakeModel(model_sample_rate_hz=16000)
    gpu_attempts = {"count": 0}
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: True)
    try:
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
        assert gpu_attempts["count"] == 2
        assert engine._gpu_disabled_reason is not None
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_gpu_timeout_retry_succeeds_without_disabling_gpu() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: True)
    try:
        engine = FasterWhisperEngine(
            model_factory=lambda *_args: _FakeModel(),
            availability_probe=lambda: True,
        )
        attempts = {"count": 0}
        original_guarded = engine._transcribe_gpu_guarded

        def fake_guarded(**_kwargs: object) -> list[dict[str, object]]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _GpuWorkerTimeoutError("GPU worker exceeded timeout (12.0s).")
            return [{"text": "gpu retry success", "start_s": 0.0, "end_s": 1.0, "confidence": 0.9}]

        engine._transcribe_gpu_guarded = fake_guarded  # type: ignore[method-assign]
        request = TranscriptionRequest(
            pcm_bytes=b"\x00\x00\x10\x00",
            sample_rate_hz=16000,
            languages=("en",),
            model="small",
            compute="auto",
        )
        result = engine.transcribe(request)
        engine._transcribe_gpu_guarded = original_guarded  # type: ignore[method-assign]

        assert result[0].text == "gpu retry success"
        assert attempts["count"] == 2
        assert engine._gpu_disabled_reason is None
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_gpu_materialization_error_falls_back_to_cpu() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    cpu_model = _FakeModel(model_sample_rate_hz=16000)
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: True)
    try:
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
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_unknown_gpu_runtime_error_is_raised() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: True)
    try:
        engine = FasterWhisperEngine(
            model_factory=lambda *_args: _FakeModel(),
            availability_probe=lambda: True,
        )
        original_guarded = engine._transcribe_gpu_guarded

        def fake_guarded(**_kwargs: object) -> list[dict[str, object]]:
            raise _GpuWorkerRuntimeError("unexpected worker parse failure")

        engine._transcribe_gpu_guarded = fake_guarded  # type: ignore[method-assign]
        request = TranscriptionRequest(
            pcm_bytes=b"\x00\x00\x10\x00",
            sample_rate_hz=16000,
            languages=("en",),
            model="small",
            compute="auto",
        )
        with pytest.raises(EngineUnavailableError):
            engine.transcribe(request)
        engine._transcribe_gpu_guarded = original_guarded  # type: ignore[method-assign]
        assert engine._gpu_disabled_reason is None
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_fast_preset_uses_fast_decode_in_process() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    fake_model = _FakeModel(model_sample_rate_hz=16000)
    engine = FasterWhisperEngine(
        model_factory=lambda *_args: fake_model,
        availability_probe=lambda: True,
    )
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("en",),
        model="small",
        compute="cpu",
        asr_preset="fast",
    )

    result = engine.transcribe(request)

    assert result[0].text == "hello world"
    main_call = fake_model.calls[-1]
    assert main_call["beam_size"] == 1
    assert main_call["vad_filter"] is False
    assert main_call["condition_on_previous_text"] is False


def test_faster_whisper_accurate_preset_enables_condition_on_previous_text() -> None:
    FasterWhisperEngine.clear_cache_for_tests()
    fake_model = _FakeModel(model_sample_rate_hz=16000)
    engine = FasterWhisperEngine(
        model_factory=lambda *_args: fake_model,
        availability_probe=lambda: True,
    )
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("en",),
        model="small",
        compute="cpu",
        asr_preset="accurate",
    )

    result = engine.transcribe(request)

    assert result[0].text == "hello world"
    main_call = fake_model.calls[-1]
    assert main_call["beam_size"] == 5
    assert main_call["vad_filter"] is True
    assert main_call["condition_on_previous_text"] is True


def test_faster_whisper_passes_asr_preset_to_gpu_path() -> None:
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: True)
    try:
        engine = FasterWhisperEngine(
            model_factory=lambda *_args: _FakeModel(),
            availability_probe=lambda: True,
        )
        seen: dict[str, object] = {}

        def fake_guarded(**kwargs: object) -> list[dict[str, object]]:
            seen.update(kwargs)
            return [{"text": "gpu result", "start_s": 0.0, "end_s": 1.0, "confidence": 0.9}]

        original_guarded = engine._transcribe_gpu_guarded
        engine._transcribe_gpu_guarded = fake_guarded  # type: ignore[method-assign]
        request = TranscriptionRequest(
            pcm_bytes=b"\x00\x00\x10\x00",
            sample_rate_hz=16000,
            languages=("en",),
            model="small",
            compute="auto",
            asr_preset="accurate",
        )

        result = engine.transcribe(request)
        engine._transcribe_gpu_guarded = original_guarded  # type: ignore[method-assign]

        assert result[0].text == "gpu result"
        assert seen["asr_preset"] == "accurate"
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_compute_auto_prefers_cuda_when_available() -> None:
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: True)
    try:
        assert FasterWhisperEngine.resolve_compute_backend("auto") == ("cuda", "float16")
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_compute_auto_uses_cpu_when_no_cuda() -> None:
    original_has_cuda = FasterWhisperEngine._has_cuda_device
    FasterWhisperEngine._has_cuda_device = staticmethod(lambda: False)
    try:
        assert FasterWhisperEngine.resolve_compute_backend("auto") == ("cpu", "int8")
    finally:
        FasterWhisperEngine._has_cuda_device = original_has_cuda


def test_faster_whisper_compute_metal_invalid_on_non_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "narada.asr.faster_whisper_engine.platform.system",
        lambda: "Windows",
    )
    with pytest.raises(EngineUnavailableError):
        FasterWhisperEngine.resolve_compute_backend("metal")


def test_faster_whisper_gpu_timeout_policy_floor_scale_and_cap() -> None:
    floor_timeout = FasterWhisperEngine._compute_gpu_transcribe_timeout_s(  # noqa: SLF001
        audio_seconds=1.0,
        asr_preset="fast",
    )
    balanced_timeout = FasterWhisperEngine._compute_gpu_transcribe_timeout_s(  # noqa: SLF001
        audio_seconds=20.0,
        asr_preset="balanced",
    )
    accurate_timeout = FasterWhisperEngine._compute_gpu_transcribe_timeout_s(  # noqa: SLF001
        audio_seconds=20.0,
        asr_preset="accurate",
    )
    capped_timeout = FasterWhisperEngine._compute_gpu_transcribe_timeout_s(  # noqa: SLF001
        audio_seconds=10_000.0,
        asr_preset="accurate",
    )

    assert floor_timeout == pytest.approx(12.0)
    assert balanced_timeout == pytest.approx(19.0)
    assert accurate_timeout == pytest.approx(23.0)
    assert capped_timeout == pytest.approx(90.0)


def test_faster_whisper_long_window_chunking_covers_all_audio_without_drop() -> None:
    sample_rate_hz = 16000
    # 125s should trigger long-window chunking.
    audio = np.zeros(sample_rate_hz * 125, dtype=np.float32)
    chunks = FasterWhisperEngine._iter_gpu_audio_chunks(  # noqa: SLF001
        audio=audio,
        sample_rate_hz=sample_rate_hz,
    )

    assert FasterWhisperEngine._should_chunk_gpu_audio(audio_seconds=125.0)  # noqa: SLF001
    assert not FasterWhisperEngine._should_chunk_gpu_audio(audio_seconds=120.0)  # noqa: SLF001
    assert chunks
    assert chunks[0][1] == pytest.approx(0.0)
    last_chunk, last_offset_s = chunks[-1]
    assert last_offset_s + (last_chunk.shape[0] / sample_rate_hz) == pytest.approx(125.0)


def test_faster_whisper_compute_cuda_on_macos_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "narada.asr.faster_whisper_engine.platform.system",
        lambda: "Darwin",
    )
    assert FasterWhisperEngine.resolve_compute_backend("cuda") == ("cpu", "int8")


def test_faster_whisper_compute_metal_on_macos_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "narada.asr.faster_whisper_engine.platform.system",
        lambda: "Darwin",
    )
    assert FasterWhisperEngine.resolve_compute_backend("metal") == ("cpu", "int8")
