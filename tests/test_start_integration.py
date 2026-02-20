from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from narada.asr.base import AsrEngine, TranscriptionRequest, TranscriptSegment
from narada.asr.model_discovery import StartModelPreflight
from narada.audio.capture import CapturedFrame, CaptureError, DeviceDisconnectedError
from narada.cli import app, start_command
from narada.config import RuntimeConfig
from narada.devices import AudioDevice


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


def _runtime_config(
    mode: str,
    out_path: Path,
    *,
    wall_flush_seconds: float = 60.0,
    capture_queue_warn_seconds: float = 120.0,
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
        bind="127.0.0.1",
        port=8787,
        model_dir_faster_whisper=None,
        model_dir_whisper_cpp=None,
    )


def _frame() -> CapturedFrame:
    return CapturedFrame(pcm_bytes=b"\x00\x00\x10\x00", sample_rate_hz=16000, channels=1)


def _interrupt_after_sleep_calls(limit: int = 4):
    state = {"calls": 0}

    def _sleep(_seconds: float) -> None:
        state["calls"] += 1
        if state["calls"] >= limit:
            raise KeyboardInterrupt

    return _sleep


def _run_start_for_tests(**kwargs: Any) -> None:
    defaults: dict[str, Any] = {
        "serve": False,
        "bind": None,
        "port": None,
        "qr": False,
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

    def _start_server(transcript_path: Path, bind: str, port: int) -> _FakeRunningServer:
        calls["transcript_path"] = transcript_path
        calls["bind"] = bind
        calls["port"] = port
        return fake_server

    monkeypatch.setattr("narada.cli.start_transcript_server", _start_server)
    monkeypatch.setattr("narada.cli.render_ascii_qr", lambda _url: "QR")

    _run_start_for_tests(serve=True, qr=True)
    assert calls["transcript_path"] == out_path
    assert calls["bind"] == cfg.bind
    assert calls["port"] == cfg.port
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


def test_start_live_loop_uses_idle_sleep_not_one_second(
    monkeypatch: Any, tmp_path: Path
) -> None:
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
    assert "require `--serve`" in result.output.lower()
