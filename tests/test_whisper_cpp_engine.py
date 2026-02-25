import subprocess
from pathlib import Path

import pytest

from narada.asr.base import EngineUnavailableError, TranscriptionRequest
from narada.asr.whisper_cpp_engine import WhisperCppEngine


def _build_fake_run(*, help_text: str, command_sink: list[list[str]]):
    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        command_sink.append(cmd)
        if cmd[1] in {"-h", "--help"}:
            return subprocess.CompletedProcess(cmd, 0, help_text, "")
        base = Path(cmd[cmd.index("-of") + 1])
        base.with_suffix(".txt").write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


def _request(compute: str) -> TranscriptionRequest:
    return TranscriptionRequest(
        pcm_bytes=b"\x00\x00\x10\x00",
        sample_rate_hz=16000,
        languages=("en",),
        model="small",
        compute=compute,
    )


def test_whisper_cpp_is_available_requires_cli() -> None:
    engine = WhisperCppEngine(which_fn=lambda _: None)
    assert not engine.is_available()


def test_whisper_cpp_is_available_with_cli() -> None:
    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli")
    assert engine.is_available()


def test_whisper_cpp_transcribe_cpu_uses_no_gpu_flag_without_numeric_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    captured_cmds: list[list[str]] = []
    fake_run = _build_fake_run(
        help_text="usage: whisper-cli ... --no-gpu ... cuda metal",
        command_sink=captured_cmds,
    )

    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    result = engine.transcribe(_request("cpu"))

    assert result[0].text == "ok"
    transcribe_cmd = captured_cmds[-1]
    assert "--no-gpu" in transcribe_cmd
    assert "-ngl" not in transcribe_cmd
    assert "--gpu-layers" not in transcribe_cmd


def test_whisper_cpp_transcribe_cpu_uses_short_ng_when_no_long_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    captured_cmds: list[list[str]] = []
    fake_run = _build_fake_run(
        help_text="usage: whisper-cli ... -ng ...",
        command_sink=captured_cmds,
    )
    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    _ = engine.transcribe(_request("cpu"))

    transcribe_cmd = captured_cmds[-1]
    assert "-ng" in transcribe_cmd
    assert "--no-gpu" not in transcribe_cmd
    ng_index = transcribe_cmd.index("-ng")
    if ng_index + 1 < len(transcribe_cmd):
        next_token = transcribe_cmd[ng_index + 1]
        assert not next_token.isdigit()


def test_whisper_cpp_cuda_compute_does_not_disable_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    captured_cmds: list[list[str]] = []
    fake_run = _build_fake_run(
        help_text="usage: whisper-cli ... --no-gpu ... --gpu-layers N ... cuda",
        command_sink=captured_cmds,
    )

    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    _ = engine.transcribe(_request("cuda"))

    transcribe_cmd = captured_cmds[-1]
    assert "--no-gpu" not in transcribe_cmd
    assert "-ng" not in transcribe_cmd
    assert "--gpu-layers" not in transcribe_cmd
    assert "-ngl" not in transcribe_cmd


def test_whisper_cpp_probe_capabilities_detects_flags_and_backend_hints() -> None:
    WhisperCppEngine.clear_cache_for_tests()
    captured_cmds: list[list[str]] = []
    fake_run = _build_fake_run(
        help_text="usage: whisper-cli --no-gpu --gpu-layers N supports cuda metal",
        command_sink=captured_cmds,
    )
    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    capabilities = engine.probe_cli_capabilities()

    assert capabilities.no_gpu_flag == "--no-gpu"
    assert capabilities.gpu_layers_flag == "--gpu-layers"
    assert capabilities.backend_hints == ("cuda", "metal")


def test_whisper_cpp_missing_model_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(tmp_path / "missing"))
    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli")
    with pytest.raises(EngineUnavailableError):
        engine.transcribe(_request("cpu"))


def test_whisper_cpp_timeout_retries_once_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    transcribe_attempts = {"count": 0}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1] in {"-h", "--help"}:
            return subprocess.CompletedProcess(cmd, 0, "usage: whisper-cli --no-gpu", "")
        transcribe_attempts["count"] += 1
        if transcribe_attempts["count"] == 1:
            timeout = float(kwargs.get("timeout", 0.0))
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        base = Path(cmd[cmd.index("-of") + 1])
        base.with_suffix(".txt").write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    result = engine.transcribe(_request("cpu"))

    assert result[0].text == "ok"
    assert transcribe_attempts["count"] == 2


def test_whisper_cpp_timeout_retry_exhaustion_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1] in {"-h", "--help"}:
            return subprocess.CompletedProcess(cmd, 0, "usage: whisper-cli --no-gpu", "")
        timeout = float(kwargs.get("timeout", 0.0))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    with pytest.raises(EngineUnavailableError):
        engine.transcribe(_request("cpu"))


def test_whisper_cpp_timeout_on_cuda_retries_with_cpu_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WhisperCppEngine.clear_cache_for_tests()
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"model")
    monkeypatch.setenv("NARADA_WHISPER_CPP_MODEL_DIR", str(model_dir))

    transcribe_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1] in {"-h", "--help"}:
            return subprocess.CompletedProcess(cmd, 0, "usage: whisper-cli --no-gpu cuda", "")
        transcribe_cmds.append(cmd)
        if len(transcribe_cmds) == 1:
            timeout = float(kwargs.get("timeout", 0.0))
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        base = Path(cmd[cmd.index("-of") + 1])
        base.with_suffix(".txt").write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    engine = WhisperCppEngine(which_fn=lambda _: "whisper-cli", run_fn=fake_run)
    result = engine.transcribe(_request("cuda"))

    assert result[0].text == "ok"
    assert len(transcribe_cmds) == 2
    assert "--no-gpu" not in transcribe_cmds[0]
    assert "--no-gpu" in transcribe_cmds[1]
