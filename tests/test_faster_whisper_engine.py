import numpy as np
import pytest

from narada.asr.base import EngineUnavailableError, TranscriptionRequest
from narada.asr.faster_whisper_engine import FasterWhisperEngine


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
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.audio_inputs: list[np.ndarray] = []

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


def test_faster_whisper_compute_metal_invalid_on_non_macos() -> None:
    with pytest.raises(EngineUnavailableError):
        FasterWhisperEngine.resolve_compute_backend("metal")
