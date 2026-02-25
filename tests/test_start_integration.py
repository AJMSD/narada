from __future__ import annotations

import io
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import narada.cli as cli_module
from narada.asr.base import AsrEngine, TranscriptionRequest, TranscriptSegment
from narada.asr.model_discovery import StartModelPreflight
from narada.audio.capture import CapturedFrame, CaptureError, DeviceDisconnectedError
from narada.cli import app, start_command
from narada.config import RuntimeConfig
from narada.devices import AudioDevice
from narada.live_notes import SessionSpool as LiveSessionSpool


class _TTYStdin:
    def isatty(self) -> bool:
        return True


@dataclass
class _FakeCapture:
    frames: list[CapturedFrame]
    sample_rate_hz: int = 16000
    blocksize: int = 1600
    closed: bool = False

    def read_frame(self) -> CapturedFrame | None:
        if not self.frames:
            return None
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True

    def stats_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(dropped_frames=0)


@dataclass
class _FailingCapture:
    error: Exception
    sample_rate_hz: int = 16000
    blocksize: int = 1600
    closed: bool = False

    def read_frame(self) -> CapturedFrame | None:
        raise self.error

    def close(self) -> None:
        self.closed = True

    def stats_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(dropped_frames=0)


@dataclass
class _FakeRunningServer:
    access_url: str = "http://127.0.0.1:8787"
    stopped: bool = False

    def stop(self) -> None:
        self.stopped = True


class _FakeEngine(AsrEngine):
    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def transcribe(self, request: TranscriptionRequest) -> list[TranscriptSegment]:
        self.calls += 1
        assert request.sample_rate_hz > 0
        return [
            TranscriptSegment(
                text=self.text,
                confidence=0.95,
                start_s=0.0,
                end_s=0.5,
                is_final=True,
            )
        ]


class _RecordingSampleRateEngine(_FakeEngine):
    def __init__(self, text: str) -> None:
        super().__init__(text=text)
        self.sample_rates: list[int] = []
        self.asr_presets: list[str] = []

    def transcribe(self, request: TranscriptionRequest) -> list[TranscriptSegment]:
        self.sample_rates.append(request.sample_rate_hz)
        self.asr_presets.append(request.asr_preset)
        return super().transcribe(request)


class _SlowEngine(_FakeEngine):
    def __init__(self, text: str, delay_s: float) -> None:
        super().__init__(text=text)
        self.delay_s = delay_s

    def transcribe(self, request: TranscriptionRequest) -> list[TranscriptSegment]:
        threading.Event().wait(self.delay_s)
        return super().transcribe(request)


class _AlwaysFailingEngine(AsrEngine):
    name = "failing"

    def is_available(self) -> bool:
        return True

    def transcribe(self, request: TranscriptionRequest) -> list[TranscriptSegment]:
        raise RuntimeError("simulated asr failure")


def _runtime_config(
    mode: str,
    out_path: Path,
    *,
    wall_flush_seconds: float = 60.0,
    capture_queue_warn_seconds: float = 120.0,
    notes_interval_seconds: float = 12.0,
    notes_overlap_seconds: float = 1.5,
    notes_commit_holdback_windows: int = 1,
    asr_backlog_warn_seconds: float = 45.0,
    keep_spool: bool = False,
) -> RuntimeConfig:
    return RuntimeConfig(
        mode=mode,  # type: ignore[arg-type]
        mic="1" if mode in {"mic", "mixed"} else None,
        system="2" if mode in {"system", "mixed"} else None,
        out=out_path,
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
        wall_flush_seconds=wall_flush_seconds,
        capture_queue_warn_seconds=capture_queue_warn_seconds,
        notes_interval_seconds=notes_interval_seconds,
        notes_overlap_seconds=notes_overlap_seconds,
        notes_commit_holdback_windows=notes_commit_holdback_windows,
        asr_backlog_warn_seconds=asr_backlog_warn_seconds,
        keep_spool=keep_spool,
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


def _frame() -> CapturedFrame:
    return CapturedFrame(pcm_bytes=b"\x00\x00\x10\x00", sample_rate_hz=16000, channels=1)


def _interrupt_after_sleep_calls(limit: int = 4):
    state = {"calls": 0, "raised": False}

    def _sleep(_seconds: float) -> None:
        state["calls"] += 1
        if not state["raised"] and state["calls"] >= limit:
            state["raised"] = True
            raise KeyboardInterrupt

    return _sleep


def _run_start_for_tests(**kwargs: Any) -> None:
    defaults: dict[str, Any] = {
        "serve": False,
        "bind": None,
        "port": None,
        "qr": False,
        "serve_token": None,
    }
    defaults.update(kwargs)
    start_command(**defaults)


def test_start_integration_mic_mode(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "mic.txt"
    cfg = _runtime_config("mic", out_path)
    fake_engine = _FakeEngine("mic transcript")
    mic_capture = _FakeCapture(frames=[_frame()])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls())

    _run_start_for_tests()
    assert "mic transcript" in out_path.read_text(encoding="utf-8")
    assert mic_capture.closed
    assert fake_engine.calls >= 1


def test_start_wires_spool_flush_thresholds_into_session_spool(
    monkeypatch: Any, tmp_path: Path
) -> None:
    out_path = tmp_path / "spool-thresholds.txt"
    cfg = _runtime_config("mic", out_path)
    cfg = RuntimeConfig(
        **{
            **cfg.__dict__,
            "spool_flush_interval_seconds": 1.5,
            "spool_flush_bytes": 12345,
        }
    )
    fake_engine = _FakeEngine("spool threshold transcript")
    mic_capture = _FakeCapture(frames=[_frame()])
    seen: dict[str, Any] = {}

    def _session_spool_factory(
        *,
        base_dir: Path,
        prefix: str,
        flush_interval_seconds: float,
        flush_bytes: int,
    ) -> LiveSessionSpool:
        seen["flush_interval_seconds"] = flush_interval_seconds
        seen["flush_bytes"] = flush_bytes
        return LiveSessionSpool(
            base_dir=base_dir,
            prefix=prefix,
            flush_interval_seconds=flush_interval_seconds,
            flush_bytes=flush_bytes,
        )

    monkeypatch.setattr("narada.cli.SessionSpool", _session_spool_factory)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls())

    _run_start_for_tests()

    assert seen["flush_interval_seconds"] == pytest.approx(1.5)
    assert seen["flush_bytes"] == 12345
    assert "spool threshold transcript" in out_path.read_text(encoding="utf-8")


def test_start_integration_system_mode(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "system.txt"
    cfg = _runtime_config("system", out_path)
    fake_engine = _FakeEngine("system transcript")
    system_capture = _FakeCapture(frames=[_frame()])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (None, AudioDevice(2, "Loopback", "loopback"), []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_system_capture", lambda *_args, **_kwargs: system_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls())

    _run_start_for_tests()
    assert "system transcript" in out_path.read_text(encoding="utf-8")
    assert system_capture.closed
    assert fake_engine.calls >= 1


def test_start_integration_mixed_mode(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "mixed.txt"
    cfg = _runtime_config("mixed", out_path)
    fake_engine = _FakeEngine("mixed transcript")
    mic_capture = _FakeCapture(frames=[_frame()])
    system_capture = _FakeCapture(frames=[_frame()])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (
            AudioDevice(1, "Mic", "input"),
            AudioDevice(2, "Loopback", "loopback"),
            [],
        ),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.open_system_capture", lambda *_args, **_kwargs: system_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls())

    _run_start_for_tests()
    assert "mixed transcript" in out_path.read_text(encoding="utf-8")
    assert mic_capture.closed
    assert system_capture.closed
    assert fake_engine.calls >= 1


def test_notes_first_mixed_large_backlog_still_commits_during_runtime_cycle(
    monkeypatch: Any, tmp_path: Path
) -> None:
    out_path = tmp_path / "mixed-large-backlog.txt"
    cfg = _runtime_config(
        "mixed",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("mixed backlog transcript", delay_s=0.02)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(5000)])
    system_capture = _FakeCapture(frames=[_frame() for _ in range(5000)])
    drain_calls = {"count": 0}
    original_drain = cli_module._drain_asr_results

    def _counting_drain(**kwargs: Any):
        drain_calls["count"] += 1
        return original_drain(**kwargs)

    monkeypatch.setattr("narada.cli._drain_asr_results", _counting_drain)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (
            AudioDevice(1, "Mic", "input"),
            AudioDevice(2, "Loopback", "loopback"),
            [],
        ),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.open_system_capture", lambda *_args, **_kwargs: system_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=6))

    _run_start_for_tests()

    assert drain_calls["count"] >= 4
    assert "mixed backlog transcript" in out_path.read_text(encoding="utf-8")


def test_shutdown_drain_batches_progressively_with_large_pending_capture(
    monkeypatch: Any, tmp_path: Path
) -> None:
    out_path = tmp_path / "shutdown-batched-drain.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("shutdown batch transcript", delay_s=0.03)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(6000)])
    seen_max_items: list[int | None] = []
    original_drain_capture = cli_module._drain_capture_queue_to_pending

    def _record_drain_capture(**kwargs: Any) -> int:
        seen_max_items.append(kwargs.get("max_items"))
        return original_drain_capture(**kwargs)

    monkeypatch.setattr("narada.cli._drain_capture_queue_to_pending", _record_drain_capture)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=5))

    _run_start_for_tests()

    shutdown_calls = [
        value
        for value in seen_max_items
        if value == cli_module._SHUTDOWN_CAPTURE_DRAIN_MAX_FRAMES_PER_CYCLE
    ]
    assert len(shutdown_calls) >= 2
    assert "shutdown batch transcript" in out_path.read_text(encoding="utf-8")


def test_zero_commit_with_asr_errors_emits_summary_warning(
    monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "zero-commit-warning.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    failing_engine = _AlwaysFailingEngine()
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(64)])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: failing_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=4))

    _run_start_for_tests()

    captured = capsys.readouterr()
    assert "No transcript lines were committed before shutdown." in captured.err
    assert out_path.read_text(encoding="utf-8").strip() == ""


def test_start_notes_first_slow_engine_still_commits_tail(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "slow-tail.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("slow transcript", delay_s=0.03)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(24)])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=5))

    _run_start_for_tests()

    assert "slow transcript" in out_path.read_text(encoding="utf-8")
    assert mic_capture.closed
    assert slow_engine.calls >= 1


def test_start_interrupt_displays_asr_shutdown_notice(
    monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "shutdown-notice.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("shutdown notice transcript", delay_s=0.02)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(20)])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=4))

    _run_start_for_tests()

    captured = capsys.readouterr()
    assert "Application stopping. ASR completing first." in captured.out
    assert "Will stop in about" in captured.out
    assert "shutdown notice transcript" in out_path.read_text(encoding="utf-8")


def test_start_sigterm_displays_sigterm_shutdown_reason(
    monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "shutdown-notice-sigterm.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("sigterm transcript", delay_s=0.02)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(20)])

    @contextmanager
    def _fake_signal_handlers(shutdown_signals: Any):
        shutdown_signals.note_signal(signal_kind="sigterm")
        yield

    monkeypatch.setattr("narada.cli._install_start_signal_handlers", _fake_signal_handlers)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=4))

    _run_start_for_tests()

    captured = capsys.readouterr()
    assert "(SIGTERM)" in captured.out
    assert "sigterm transcript" in out_path.read_text(encoding="utf-8")


def test_start_shutdown_queue_full_keeps_final_tail_bounded(
    monkeypatch: Any, tmp_path: Path
) -> None:
    out_path = tmp_path / "bounded-final-tail.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("bounded tail transcript", delay_s=0.03)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(160)])
    captured_tail_seconds: dict[str, list[float]] = {"values": []}

    from narada.live_notes import IntervalPlanner as RealIntervalPlanner

    class _RecordingPlanner(RealIntervalPlanner):
        def build_final_tasks(self, *, now_monotonic: float | None = None) -> list[Any]:
            tasks = super().build_final_tasks(now_monotonic=now_monotonic)
            captured_tail_seconds["values"] = [
                task.audio_seconds for task in tasks if task.label == "final-tail"
            ]
            return tasks

    def _planner_factory(*args: object, **kwargs: object) -> RealIntervalPlanner:
        return _RecordingPlanner(*args, **kwargs)

    monkeypatch.setattr("narada.cli.IntervalPlanner", _planner_factory)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=4))

    _run_start_for_tests()

    assert captured_tail_seconds["values"]
    assert max(captured_tail_seconds["values"]) <= 1.0
    assert "bounded tail transcript" in out_path.read_text(encoding="utf-8")


def test_start_shutdown_forces_exit_on_second_ctrl_c_during_drain(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "repeated-ctrlc-drain.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("repeated ctrlc transcript", delay_s=0.02)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(64)])
    state = {"calls": 0}
    controller_ref: dict[str, Any] = {}

    @contextmanager
    def _fake_signal_handlers(shutdown_signals: Any):
        controller_ref["value"] = shutdown_signals
        shutdown_signals.note_signal(signal_kind="sigint")
        yield

    def _sleep_with_interrupt(_seconds: float) -> None:
        state["calls"] += 1
        if state["calls"] == 3:
            controller = controller_ref.get("value")
            if controller is not None:
                controller.note_signal(signal_kind="sigint")
            raise KeyboardInterrupt

    monkeypatch.setattr("narada.cli._install_start_signal_handlers", _fake_signal_handlers)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _sleep_with_interrupt)

    with pytest.raises(typer.Exit) as exc_info:
        _run_start_for_tests()

    assert state["calls"] >= 3
    assert exc_info.value.exit_code == 130


def test_start_second_signal_after_sigterm_forces_sigterm_exit_code(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "sigterm-second-force-exit.txt"
    cfg = _runtime_config(
        "mic",
        out_path,
        notes_interval_seconds=0.2,
        notes_overlap_seconds=0.0,
    )
    slow_engine = _SlowEngine("sigterm force transcript", delay_s=0.02)
    mic_capture = _FakeCapture(frames=[_frame() for _ in range(64)])
    state = {"calls": 0}

    @contextmanager
    def _fake_signal_handlers(shutdown_signals: Any):
        shutdown_signals.note_signal(signal_kind="sigterm")
        yield

    def _sleep_with_repeated_interrupts(_seconds: float) -> None:
        state["calls"] += 1
        if state["calls"] in {3, 4}:
            raise KeyboardInterrupt

    monkeypatch.setattr("narada.cli._install_start_signal_handlers", _fake_signal_handlers)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: slow_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _sleep_with_repeated_interrupts)

    with pytest.raises(typer.Exit) as exc_info:
        _run_start_for_tests()

    assert state["calls"] >= 4
    assert exc_info.value.exit_code == 143


def test_start_falls_back_to_recommended_engine(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "fallback.txt"
    cfg = _runtime_config("mic", out_path)
    fake_engine = _FakeEngine("fallback transcript")
    mic_capture = _FakeCapture(frames=[_frame()])
    selected_engine_name: dict[str, str] = {}

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=False,
            recommended_engine="whisper-cpp",
            messages=("Detected whisper-cpp model files on this device.",),
        ),
    )

    def _build_engine(engine_name: str, *_args: Any, **_kwargs: Any) -> AsrEngine:
        selected_engine_name["value"] = engine_name
        return fake_engine

    monkeypatch.setattr("narada.cli.build_engine", _build_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls())

    _run_start_for_tests()

    assert selected_engine_name["value"] == "whisper-cpp"
    assert "fallback transcript" in out_path.read_text(encoding="utf-8")


def test_start_with_serve_launches_and_stops_server(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "served.txt"
    cfg = _runtime_config("mic", out_path)
    fake_engine = _FakeEngine("served transcript")
    mic_capture = _FakeCapture(frames=[_frame()])
    fake_server = _FakeRunningServer()
    calls: dict[str, Any] = {}

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls())

    def _start_server(
        transcript_path: Path, bind: str, port: int, serve_token: str | None = None
    ) -> _FakeRunningServer:
        calls["transcript_path"] = transcript_path
        calls["bind"] = bind
        calls["port"] = port
        calls["serve_token"] = serve_token
        return fake_server

    monkeypatch.setattr("narada.cli.start_transcript_server", _start_server)
    monkeypatch.setattr("narada.cli.render_ascii_qr", lambda _url: "QR")

    _run_start_for_tests(serve=True, qr=True)
    assert calls["transcript_path"] == out_path
    assert calls["bind"] == cfg.bind
    assert calls["port"] == cfg.port
    assert calls["serve_token"] is None
    assert fake_server.stopped


def test_start_with_serve_passes_serve_token(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "served-token.txt"
    cfg = _runtime_config("mic", out_path)
    cfg = RuntimeConfig(**{**cfg.__dict__, "serve_token": "topsecret"})
    fake_engine = _FakeEngine("served transcript")
    mic_capture = _FakeCapture(frames=[_frame()])
    fake_server = _FakeRunningServer()
    calls: dict[str, Any] = {}

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls())

    def _start_server(
        transcript_path: Path, bind: str, port: int, serve_token: str | None = None
    ) -> _FakeRunningServer:
        calls["transcript_path"] = transcript_path
        calls["bind"] = bind
        calls["port"] = port
        calls["serve_token"] = serve_token
        return fake_server

    monkeypatch.setattr("narada.cli.start_transcript_server", _start_server)

    _run_start_for_tests(serve=True, serve_token="topsecret")
    assert calls["transcript_path"] == out_path
    assert calls["bind"] == cfg.bind
    assert calls["port"] == cfg.port
    assert calls["serve_token"] == "topsecret"
    assert fake_server.stopped


def test_start_capture_error_falls_back_when_stderr_handle_is_invalid(
    monkeypatch: Any, tmp_path: Path
) -> None:
    out_path = tmp_path / "capture-error.txt"
    cfg = _runtime_config("system", out_path)
    fake_engine = _FakeEngine("unused")
    fallback_err = io.StringIO()

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (None, AudioDevice(22, "Speakers", "output"), []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr(
        "narada.cli.open_system_capture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CaptureError("boom")),
    )
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.sys.__stderr__", fallback_err)

    original_echo = typer.echo

    def fake_echo(message: object, *args: object, **kwargs: object) -> None:
        if kwargs.get("err"):
            raise OSError("Windows error 6")
        original_echo(message, *args, **kwargs)

    monkeypatch.setattr("narada.cli.typer.echo", fake_echo)

    with pytest.raises(typer.Exit) as exc_info:
        _run_start_for_tests()

    assert exc_info.value.exit_code == 2
    assert "Audio capture error: boom" in fallback_err.getvalue()


def test_start_wall_flush_commits_without_waiting_for_chunk_duration(
    monkeypatch: Any, tmp_path: Path
) -> None:
    out_path = tmp_path / "wall-flush.txt"
    cfg = _runtime_config("system", out_path, wall_flush_seconds=0.5)
    fake_engine = _FakeEngine("forced flush transcript")
    system_capture = _FakeCapture(frames=[_frame()])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (None, AudioDevice(2, "Loopback", "loopback"), []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_system_capture", lambda *_args, **_kwargs: system_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=3))

    clock = {"value": 0.0}

    def _fake_monotonic() -> float:
        clock["value"] += 1.0
        return clock["value"]

    monkeypatch.setattr("narada.cli.time.monotonic", _fake_monotonic)

    _run_start_for_tests()

    assert "forced flush transcript" in out_path.read_text(encoding="utf-8")
    assert fake_engine.calls >= 1


def test_start_uses_notes_interval_planner_defaults(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "planner-defaults.txt"
    cfg = _runtime_config("mic", out_path)
    fake_engine = _FakeEngine("planner defaults transcript")
    mic_capture = _FakeCapture(frames=[_frame()])
    captured: dict[str, float] = {}

    from narada.live_notes import IntervalPlanner as RealIntervalPlanner

    def fake_planner(*args: object, **kwargs: object) -> RealIntervalPlanner:
        captured["interval_seconds"] = float(kwargs.get("interval_seconds", -1.0))
        captured["overlap_seconds"] = float(kwargs.get("overlap_seconds", -1.0))
        return RealIntervalPlanner(*args, **kwargs)

    monkeypatch.setattr("narada.cli.IntervalPlanner", fake_planner)
    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=3))

    _run_start_for_tests()

    assert captured["interval_seconds"] == 12.0
    assert captured["overlap_seconds"] == 1.5


def test_start_preserves_live_frame_sample_rate_in_transcription_request(
    monkeypatch: Any, tmp_path: Path
) -> None:
    out_path = tmp_path / "sample-rate-pass-through.txt"
    cfg = _runtime_config("mic", out_path, wall_flush_seconds=0.5)
    recording_engine = _RecordingSampleRateEngine("sample rate transcript")
    high_rate_frame = CapturedFrame(pcm_bytes=b"\x00\x00\x10\x00", sample_rate_hz=48000, channels=1)
    mic_capture = _FakeCapture(frames=[high_rate_frame], sample_rate_hz=48000)

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: recording_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=3))

    clock = {"value": 0.0}

    def _fake_monotonic() -> float:
        clock["value"] += 1.0
        return clock["value"]

    monkeypatch.setattr("narada.cli.time.monotonic", _fake_monotonic)

    _run_start_for_tests()

    assert recording_engine.sample_rates
    assert recording_engine.sample_rates[0] == 48000
    assert recording_engine.asr_presets
    assert recording_engine.asr_presets[0] == "balanced"
    assert "sample rate transcript" in out_path.read_text(encoding="utf-8")


def test_start_live_loop_uses_idle_sleep_not_one_second(monkeypatch: Any, tmp_path: Path) -> None:
    out_path = tmp_path / "idle-sleep.txt"
    cfg = _runtime_config("mic", out_path)
    fake_engine = _FakeEngine("idle sleep transcript")
    mic_capture = _FakeCapture(frames=[_frame()])

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: mic_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())

    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr("narada.cli.time.sleep", _fake_sleep)

    _run_start_for_tests()

    assert sleep_calls
    assert all(seconds == pytest.approx(0.01, rel=1e-6) for seconds in sleep_calls)


def test_start_worker_exception_exits_with_code_2(
    monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "worker-error.txt"
    cfg = _runtime_config("mic", out_path)
    fake_engine = _FakeEngine("unused")
    failing_capture = _FailingCapture(error=RuntimeError("worker boom"))

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (AudioDevice(1, "Mic", "input"), None, []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_mic_capture", lambda *_args, **_kwargs: failing_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=10))
    with pytest.raises(typer.Exit) as exc_info:
        _run_start_for_tests()

    assert exc_info.value.exit_code == 2
    captured = capsys.readouterr()
    assert "Audio capture error" in captured.err


def test_start_worker_device_disconnection_exits_with_code_2(
    monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "worker-disconnect.txt"
    cfg = _runtime_config("system", out_path)
    fake_engine = _FakeEngine("unused")
    failing_capture = _FailingCapture(error=DeviceDisconnectedError("device unavailable"))

    monkeypatch.setattr("narada.cli.build_runtime_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        "narada.cli._resolve_selected_devices",
        lambda *_args, **_kwargs: (None, AudioDevice(2, "Loopback", "loopback"), []),
    )
    monkeypatch.setattr("narada.cli.discover_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "narada.cli.build_start_model_preflight",
        lambda *_args, **_kwargs: StartModelPreflight(
            selected_engine="faster-whisper",
            selected_available=True,
            recommended_engine=None,
            messages=(),
        ),
    )
    monkeypatch.setattr("narada.cli.build_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr("narada.cli.open_system_capture", lambda *_args, **_kwargs: failing_capture)
    monkeypatch.setattr("narada.cli.sys.stdin", _TTYStdin())
    monkeypatch.setattr("narada.cli.time.sleep", _interrupt_after_sleep_calls(limit=10))
    with pytest.raises(typer.Exit) as exc_info:
        _run_start_for_tests()

    assert exc_info.value.exit_code == 2
    captured = capsys.readouterr()
    assert "Device disconnected" in captured.err


def test_start_rejects_serve_options_without_serve() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["start", "--bind", "127.0.0.1"])
    assert result.exit_code != 0
    normalized = result.output.lower()
    assert "require" in normalized
    assert "--serve" in normalized
    token_result = runner.invoke(app, ["start", "--serve-token", "abc"])
    assert token_result.exit_code != 0
    token_normalized = token_result.output.lower()
    assert "require" in token_normalized
    assert "--serve" in token_normalized
