import pytest

from narada.audio.mixer import (
    AudioChunk,
    DriftResyncState,
    downmix_to_mono,
    mix_audio_chunks,
    normalize_for_mix,
    resync_streams,
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


def test_resync_streams_aligns_shifted_sequences() -> None:
    mic = [0.0, 0.0, 0.2, 0.7, 0.1]
    system = [0.2, 0.7, 0.1]
    aligned_mic, aligned_system = resync_streams(mic, system, sample_rate_hz=1000)
    assert len(aligned_mic) == len(aligned_system)
    naive_mic = mic[: len(system)]
    naive_error = sum(abs(naive_mic[idx] - system[idx]) for idx in range(len(system)))
    aligned_error = sum(
        abs(aligned_mic[idx] - aligned_system[idx]) for idx in range(len(aligned_system))
    )
    assert aligned_error < naive_error


def test_mix_audio_chunks_with_resync_state_reduces_drift() -> None:
    state = DriftResyncState(max_drift_ms=200)
    mic = AudioChunk(samples=(0.0, 0.0, 1.0, 1.0, 1.0, 1.0), sample_rate_hz=1000, channels=1)
    system = AudioChunk(samples=(1.0, 1.0, 1.0, 1.0), sample_rate_hz=1000, channels=1)
    mixed, _ = mix_audio_chunks(mic, system, resync_state=state, headroom=0.8)
    assert mixed
    assert mixed[0] == pytest.approx(0.72, rel=1e-3)
