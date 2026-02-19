import subprocess
from pathlib import Path

import pytest

from narada.asr.base import EngineUnavailableError, TranscriptionRequest
from narada.asr.whisper_cpp_engine import WhisperCppEngine


def test_whisper_cpp_transcribe_reads_cli_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    captured_cmd: list[str] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured_cmd[:] = cmd
        base = Path(cmd[cmd.index("-of") + 1])
        base.with_suffix(".txt").write_text("transcribed output", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("en",),
        model="small",
        compute="cpu",
    )
    result = engine.transcribe(request)

    assert result[0].text == "transcribed output"
    assert "-ng" in captured_cmd
    assert captured_cmd[captured_cmd.index("-ng") + 1] == "0"


def test_whisper_cpp_cuda_compute_sets_gpu_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    captured_cmd: list[str] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured_cmd[:] = cmd
        base = Path(cmd[cmd.index("-of") + 1])
        base.with_suffix(".txt").write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("auto",),
        model="small",
        compute="cuda",
    )
    _ = engine.transcribe(request)
    assert captured_cmd[captured_cmd.index("-ng") + 1] == "99"


def test_whisper_cpp_missing_model_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(tmp_path / "missing"))
    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli")
    request = TranscriptionRequest(
        pcm_bytes=b"\x00\x00",
        sample_rate_hz=16000,
        languages=("en",),
        model="small",
        compute="cpu",
    )
    with pytest.raises(EngineUnavailableError):
        engine.transcribe(request)
