import pytest

from narada.audio.mixer import (
    AudioChunk,
    downmix_to_mono,
    mix_audio_chunks,
    normalize_for_mix,
)


def test_downmix_to_mono_averages_channels() -> None:
    stereo = [1.0, 0.0, 0.5, -0.5]
    mono = downmix_to_mono(stereo, channels=2)
    assert mono == [0.5, 0.0]


def test_downmix_to_mono_rejects_invalid_frame_length() -> None:
    with pytest.raises(ValueError):
        downmix_to_mono([0.1, 0.2, 0.3], channels=2)


def test_normalize_for_mix_resamples_to_target_rate() -> None:
    chunk = AudioChunk(samples=(0.2, 0.4, 0.2, 0.0), sample_rate_hz=2, channels=1)
    normalized = normalize_for_mix(chunk, target_rate_hz=4)
    assert len(normalized) == 8
    assert max(abs(sample) for sample in normalized) <= 0.9


def test_mix_audio_chunks_handles_different_rates_and_channels() -> None:
    mic = AudioChunk(samples=(0.2, 0.2, 0.1, 0.1), sample_rate_hz=16000, channels=2)
    system = AudioChunk(samples=(0.8, -0.8), sample_rate_hz=8000, channels=1)
    mixed, sample_rate = mix_audio_chunks(mic, system)
    assert sample_rate == 16000
    assert len(mixed) > 0
    assert all(-1.0 <= sample <= 1.0 for sample in mixed)
