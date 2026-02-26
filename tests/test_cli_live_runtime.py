from __future__ import annotations

import queue
from collections import deque
from pathlib import Path

import pytest

from narada.audio.capture import CapturedFrame
from narada.cli import (
    _build_live_status_lines,
    _drain_capture_queue_to_pending,
    _estimate_asr_backlog_seconds,
    _estimate_asr_remaining_seconds,
    _estimate_capture_backlog_seconds,
    _estimate_shutdown_eta_seconds,
    _fit_status_line_to_terminal,
    _format_elapsed_seconds,
    _LiveStatusRenderer,
    _maybe_warn_asr_backlog,
    _maybe_warn_capture_backlog,
    _resolve_terminal_columns,
    _ShutdownSignalController,
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


def test_drain_capture_queue_to_pending_respects_max_items_and_order() -> None:
    source: queue.Queue[CapturedFrame] = queue.Queue()
    first = CapturedFrame(pcm_bytes=b"a", sample_rate_hz=16000, channels=1)
    second = CapturedFrame(pcm_bytes=b"b", sample_rate_hz=16000, channels=1)
    third = CapturedFrame(pcm_bytes=b"c", sample_rate_hz=16000, channels=1)
    source.put(first)
    source.put(second)
    source.put(third)
    pending: deque[CapturedFrame] = deque()

    drained = _drain_capture_queue_to_pending(
        source_queue=source,
        target_pending=pending,
        max_items=2,
    )

    assert drained == 2
    assert [frame.pcm_bytes for frame in pending] == [b"a", b"b"]
    assert source.qsize() == 1


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


def test_estimate_asr_remaining_seconds_combines_planner_and_pending_audio() -> None:
    backlog = _estimate_asr_remaining_seconds(
        planner_backlog_s=8.5,
        pending_asr_audio_s=11.5,
    )
    assert backlog == pytest.approx(20.0)


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


def test_format_elapsed_seconds_converts_to_hh_mm_ss() -> None:
    assert _format_elapsed_seconds(0.0) == "00:00:00"
    assert _format_elapsed_seconds(65.2) == "00:01:05"
    assert _format_elapsed_seconds(3661.8) == "01:01:01"
    assert _format_elapsed_seconds(-12.0) == "00:00:00"


def test_live_status_renderer_single_line_ansi_updates_and_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStream:
        def __init__(self) -> None:
            self.output = ""

        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            self.output += text
            return len(text)

        def flush(self) -> None:
            return

    stream = _FakeStream()
    monkeypatch.setattr("narada.cli.sys.stdout", stream)
    renderer = _LiveStatusRenderer()

    renderer.render_single_line("first")
    renderer.render_single_line("second")
    renderer.break_single_line()
    renderer.break_single_line()

    assert "\rfirst" in stream.output
    assert "\rsecond" in stream.output
    assert stream.output.count("\n") == 1


def test_live_status_renderer_single_line_falls_back_to_safe_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStream:
        def isatty(self) -> bool:
            return False

        def write(self, _text: str) -> int:
            return 0

        def flush(self) -> None:
            return

    echoed: list[str] = []
    monkeypatch.setattr("narada.cli.sys.stdout", _FakeStream())
    monkeypatch.setattr("narada.cli._safe_echo", lambda message, **_kwargs: echoed.append(message))

    renderer = _LiveStatusRenderer()
    renderer.render_single_line("fallback status")
    renderer.break_single_line()

    assert echoed == ["fallback status"]


def test_live_status_renderer_single_line_clears_tail_when_line_shrinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStream:
        def __init__(self) -> None:
            self.output = ""

        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            self.output += text
            return len(text)

        def flush(self) -> None:
            return

    stream = _FakeStream()
    monkeypatch.setattr("narada.cli.sys.stdout", stream)
    renderer = _LiveStatusRenderer()

    renderer.render_single_line("longer-line")
    renderer.render_single_line("short")

    assert "short      " in stream.output


def test_resolve_terminal_columns_uses_fallback_for_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "narada.cli.shutil.get_terminal_size",
        lambda fallback: type("Size", (), {"columns": 0, "lines": fallback[1]})(),
    )
    assert _resolve_terminal_columns(default_columns=97) == 97


def test_fit_status_line_to_terminal_preserves_priority_fields() -> None:
    line = _fit_status_line_to_terminal(
        required_fields=[
            "REC 00:00:42",
            "mode=system",
            "model=small",
            "asr=45.0s",
            "state=capturing",
        ],
        optional_fields=[
            "rtf=1.20",
            "commit=9ms",
            "cap=0.0s",
            "drop=0",
            "taskq=2",
            "pend=36.0s",
            "plan=9.0s",
        ],
        terminal_columns=72,
    )
    assert len(line) <= 72
    assert "REC 00:00:42" in line
    assert "mode=system" in line
    assert "model=small" in line
    assert "asr=45.0s" in line
    assert "state=capturing" in line
    assert "plan=9.0s" not in line


def test_maybe_warn_capture_backlog_breaks_single_line_when_renderer_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Renderer:
        def __init__(self) -> None:
            self.break_calls = 0

        def break_single_line(self) -> None:
            self.break_calls += 1

    echoed: list[str] = []
    monkeypatch.setattr("narada.cli._safe_echo", lambda message, **_kwargs: echoed.append(message))
    renderer = _Renderer()

    _ = _maybe_warn_capture_backlog(
        source_name="system",
        queued_frames=320,
        blocksize=1600,
        sample_rate_hz=16000,
        warn_threshold_s=10.0,
        now_monotonic=50.0,
        last_warned_at=None,
        status_renderer=renderer,  # type: ignore[arg-type]
    )

    assert renderer.break_calls == 1
    assert echoed


def test_maybe_warn_asr_backlog_breaks_single_line_when_renderer_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Renderer:
        def __init__(self) -> None:
            self.break_calls = 0

        def break_single_line(self) -> None:
            self.break_calls += 1

    warned: list[str] = []
    monkeypatch.setattr("narada.cli.logger.warning", lambda message: warned.append(message))
    renderer = _Renderer()

    _ = _maybe_warn_asr_backlog(
        backlog_s=60.0,
        warn_threshold_s=45.0,
        now_monotonic=10.0,
        last_warned_at=None,
        status_renderer=renderer,  # type: ignore[arg-type]
    )

    assert renderer.break_calls == 1
    assert warned


def test_shutdown_signal_controller_dedupes_duplicate_signal_while_handler_interrupt_pending() -> (
    None
):
    controller = _ShutdownSignalController()
    controller.note_signal(signal_kind="sigint", now_monotonic=1.0)
    controller.note_signal(signal_kind="sigint", now_monotonic=1.2)
    assert not controller.force_exit_requested
    assert controller.shutdown_reason == "Ctrl+C"
    assert controller.interrupt_count == 1

    controller.note_keyboard_interrupt(now_monotonic=1.25)
    controller.note_signal(signal_kind="sigint", now_monotonic=1.8)
    assert controller.force_exit_requested
    assert controller.force_exit_code == 130


def test_shutdown_signal_controller_ignores_fallback_keyboard_interrupt_within_dedupe_window() -> (
    None
):
    controller = _ShutdownSignalController()
    controller.note_signal(signal_kind="sigint", now_monotonic=10.0)
    controller.note_keyboard_interrupt(now_monotonic=10.1)
    controller.note_keyboard_interrupt(now_monotonic=10.2)
    assert controller.interrupt_count == 1
    assert not controller.force_exit_requested

    controller.note_keyboard_interrupt(now_monotonic=10.7)
    assert controller.force_exit_requested
    assert controller.force_exit_code == 130


def test_shutdown_signal_controller_prefers_sigterm_exit_code_when_sigterm_first() -> None:
    controller = _ShutdownSignalController()
    controller.note_signal(signal_kind="sigterm", now_monotonic=20.0)
    controller.note_keyboard_interrupt(now_monotonic=20.1)
    assert controller.interrupt_count == 1
    assert controller.shutdown_reason == "SIGTERM"

    controller.note_keyboard_interrupt(now_monotonic=20.8)
    assert controller.force_exit_requested
    assert controller.force_exit_code == 143
