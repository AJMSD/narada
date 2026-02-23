from __future__ import annotations

from pathlib import Path

import pytest

from narada.cli import (
    _build_live_status_lines,
    _estimate_asr_backlog_seconds,
    _estimate_capture_backlog_seconds,
    _estimate_shutdown_eta_seconds,
    _maybe_warn_asr_backlog,
    _maybe_warn_capture_backlog,
)
from narada.config import RuntimeConfig
from narada.performance import RuntimePerformance


def _runtime_config_stub() -> RuntimeConfig:
    return RuntimeConfig(
        mode="mic",
        mic="1",
        system=None,
        out=Path("transcripts/test.txt"),
        model="small",
        compute="cpu",
        engine="faster-whisper",
        languages=("en",),
        allow_multilingual=False,
        redact=False,
        noise_suppress="off",
        agc=False,
        gate=False,
        gate_threshold_db=-35.0,
        confidence_threshold=0.65,
        wall_flush_seconds=60.0,
        capture_queue_warn_seconds=120.0,
        notes_interval_seconds=12.0,
        notes_overlap_seconds=1.5,
        notes_commit_holdback_windows=1,
        asr_backlog_warn_seconds=45.0,
        keep_spool=False,
        spool_flush_interval_seconds=0.25,
        spool_flush_bytes=65536,
        writer_fsync_mode="line",
        writer_fsync_lines=20,
        writer_fsync_seconds=1.0,
        asr_preset="balanced",
        serve_token=None,
        bind="127.0.0.1",
        port=8787,
        model_dir_faster_whisper=None,
        model_dir_whisper_cpp=None,
    )


def test_estimate_capture_backlog_seconds_uses_queue_depth_and_frame_duration() -> None:
    backlog = _estimate_capture_backlog_seconds(
        queued_frames=10,
        blocksize=1600,
        sample_rate_hz=16000,
    )
    assert backlog == pytest.approx(1.0)


def test_estimate_capture_backlog_seconds_handles_invalid_values() -> None:
    assert (
        _estimate_capture_backlog_seconds(
            queued_frames=0,
            blocksize=1600,
            sample_rate_hz=16000,
        )
        == 0
    )
    assert (
        _estimate_capture_backlog_seconds(
            queued_frames=5,
            blocksize=0,
            sample_rate_hz=16000,
        )
        == 0
    )
    assert (
        _estimate_capture_backlog_seconds(
            queued_frames=5,
            blocksize=1600,
            sample_rate_hz=0,
        )
        == 0
    )


def test_maybe_warn_capture_backlog_warns_and_throttles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    echoed: list[str] = []
    monkeypatch.setattr("narada.cli._safe_echo", lambda message, **_kwargs: echoed.append(message))

    first_warn = _maybe_warn_capture_backlog(
        source_name="system",
        queued_frames=200,
        blocksize=1600,
        sample_rate_hz=16000,
        warn_threshold_s=10.0,
        now_monotonic=100.0,
        last_warned_at=None,
    )
    assert first_warn == 100.0
    assert echoed

    second_warn = _maybe_warn_capture_backlog(
        source_name="system",
        queued_frames=200,
        blocksize=1600,
        sample_rate_hz=16000,
        warn_threshold_s=10.0,
        now_monotonic=110.0,
        last_warned_at=first_warn,
    )
    assert second_warn == first_warn
    assert len(echoed) == 1

    third_warn = _maybe_warn_capture_backlog(
        source_name="system",
        queued_frames=200,
        blocksize=1600,
        sample_rate_hz=16000,
        warn_threshold_s=10.0,
        now_monotonic=131.0,
        last_warned_at=second_warn,
    )
    assert third_warn == 131.0
    assert len(echoed) == 2


def test_maybe_warn_capture_backlog_ignores_values_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    echoed: list[str] = []
    monkeypatch.setattr("narada.cli._safe_echo", lambda message, **_kwargs: echoed.append(message))
    warn_at = _maybe_warn_capture_backlog(
        source_name="mic",
        queued_frames=5,
        blocksize=1600,
        sample_rate_hz=16000,
        warn_threshold_s=2.0,
        now_monotonic=50.0,
        last_warned_at=None,
    )
    assert warn_at is None
    assert echoed == []


def test_estimate_asr_backlog_seconds_combines_planner_and_queue() -> None:
    backlog = _estimate_asr_backlog_seconds(
        planner_backlog_s=5.0,
        queued_tasks=2,
        interval_s=12.0,
    )
    assert backlog == pytest.approx(29.0)


def test_maybe_warn_asr_backlog_warns_and_throttles(monkeypatch: pytest.MonkeyPatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr("narada.cli.logger.warning", lambda message: warned.append(message))

    first_warn = _maybe_warn_asr_backlog(
        backlog_s=55.0,
        warn_threshold_s=45.0,
        now_monotonic=100.0,
        last_warned_at=None,
    )
    assert first_warn == 100.0
    assert warned

    second_warn = _maybe_warn_asr_backlog(
        backlog_s=60.0,
        warn_threshold_s=45.0,
        now_monotonic=110.0,
        last_warned_at=first_warn,
    )
    assert second_warn == first_warn
    assert len(warned) == 1

    third_warn = _maybe_warn_asr_backlog(
        backlog_s=61.0,
        warn_threshold_s=45.0,
        now_monotonic=131.0,
        last_warned_at=second_warn,
    )
    assert third_warn == 131.0
    assert len(warned) == 2


def test_estimate_shutdown_eta_uses_realtime_factor() -> None:
    performance = RuntimePerformance()
    performance.record_transcription(audio_seconds=10.0, processing_seconds=5.0)
    eta = _estimate_shutdown_eta_seconds(asr_backlog_s=20.0, performance=performance)
    assert eta == pytest.approx(10.0)


def test_build_live_status_lines_includes_warning_and_shutdown() -> None:
    cfg = _runtime_config_stub()
    performance = RuntimePerformance(
        total_audio_seconds=10.0,
        total_processing_seconds=5.0,
        capture_backlog_s=0.3,
        asr_backlog_s=9.7,
        dropped_frames=0,
    )
    lines = _build_live_status_lines(
        config=cfg,
        started_at=0.0,
        performance=performance,
        asr_backlog_warning_s=9.7,
        shutdown_eta_s=4.8,
        shutdown_reason="Ctrl+C",
    )
    assert len(lines) == 4
    assert "mode=mic" in lines[0]
    assert "drop=0" in lines[1]
    assert "Warning: ASR backlog is 9.7s." in lines[2]
    assert "Application stopping. ASR completing first." in lines[3]
    assert "Will stop in about ~4.8s" in lines[3]
