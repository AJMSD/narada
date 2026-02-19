import pytest

from narada.performance import RuntimePerformance


def test_runtime_performance_computes_rtf_and_commit_latency() -> None:
    stats = RuntimePerformance()
    stats.record_transcription(audio_seconds=10.0, processing_seconds=2.5)
    stats.record_transcription(audio_seconds=2.0, processing_seconds=1.0)
    stats.record_commit_latency(elapsed_seconds=0.5)
    stats.record_commit_latency(elapsed_seconds=0.3)

    assert stats.realtime_factor == pytest.approx(3.5 / 12.0)
    assert stats.average_commit_latency_ms == pytest.approx(400.0)
    assert stats.committed_segments == 2
    assert stats.status_fragment() == "rtf=0.29 | commit=400ms"


def test_runtime_performance_handles_empty_state() -> None:
    stats = RuntimePerformance()
    assert stats.realtime_factor is None
    assert stats.average_commit_latency_ms is None
    assert stats.status_fragment() == "rtf=n/a | commit=n/a"


def test_runtime_performance_rejects_negative_values() -> None:
    stats = RuntimePerformance()
    with pytest.raises(ValueError):
        stats.record_transcription(audio_seconds=-0.1, processing_seconds=1.0)
    with pytest.raises(ValueError):
        stats.record_transcription(audio_seconds=1.0, processing_seconds=-0.1)
    with pytest.raises(ValueError):
        stats.record_commit_latency(elapsed_seconds=-0.1)
