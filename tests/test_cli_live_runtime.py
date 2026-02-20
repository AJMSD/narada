from __future__ import annotations

import pytest

from narada.cli import _estimate_capture_backlog_seconds, _maybe_warn_capture_backlog


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
