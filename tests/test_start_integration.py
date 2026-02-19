from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from narada.asr.base import AsrEngine, TranscriptionRequest, TranscriptSegment
from narada.asr.model_discovery import StartModelPreflight
from narada.audio.capture import CapturedFrame
from narada.cli import start_command
from narada.config import RuntimeConfig
from narada.devices import AudioDevice


class _TTYStdin:
    def isatty(self) -> bool:
        return True


@dataclass
class _FakeCapture:
    frames: list[CapturedFrame]
    closed: bool = False

    def read_frame(self) -> CapturedFrame | None:
        if not self.frames:
            return None
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


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


def _runtime_config(mode: str, out_path: Path) -> RuntimeConfig:
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
        bind="127.0.0.1",
        port=8787,
        model_dir_faster_whisper=None,
        model_dir_whisper_cpp=None,
    )


def _frame() -> CapturedFrame:
    return CapturedFrame(pcm_bytes=b"\x00\x00\x10\x00", sample_rate_hz=16000, channels=1)


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
    monkeypatch.setattr(
        "narada.cli.time.sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    start_command()
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
    monkeypatch.setattr(
        "narada.cli.time.sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    start_command()
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
    monkeypatch.setattr(
        "narada.cli.time.sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    start_command()
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
    monkeypatch.setattr(
        "narada.cli.time.sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    start_command()

    assert selected_engine_name["value"] == "whisper-cpp"
    assert "fallback transcript" in out_path.read_text(encoding="utf-8")
