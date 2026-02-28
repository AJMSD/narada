from __future__ import annotations

from pathlib import Path

import pytest

from narada.asr.model_download import ModelPreparationError, ensure_engine_model_available


def test_ensure_faster_whisper_cached_model_skips_download(tmp_path: Path) -> None:
    model_dir = tmp_path / "fw"
    cached = model_dir / "faster-whisper-small"
    cached.mkdir(parents=True)
    (cached / "model.bin").write_bytes(b"weights")
    messages: list[str] = []

    resolved = ensure_engine_model_available(
        engine_name="faster-whisper",
        model_name="small",
        faster_whisper_model_dir=model_dir,
        emit=messages.append,
    )

    assert resolved == cached
    assert any("Using cached faster-whisper model" in line for line in messages)


def test_ensure_faster_whisper_downloads_into_override_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "fw"
    calls: dict[str, object] = {}
    messages: list[str] = []

    def _snapshot_download(**kwargs: object) -> str:
        calls.update(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model.bin").write_bytes(b"weights")
        return str(local_dir)

    monkeypatch.setattr(
        "narada.asr.model_download._load_hf_download_api",
        lambda: (lambda **_kwargs: "", _snapshot_download),
    )

    resolved = ensure_engine_model_available(
        engine_name="faster-whisper",
        model_name="small",
        faster_whisper_model_dir=model_dir,
        emit=messages.append,
    )

    expected = model_dir / "faster-whisper-small"
    assert calls["repo_id"] == "Systran/faster-whisper-small"
    assert calls["local_dir"] == str(expected)
    assert (resolved / "model.bin").exists()
    assert any("Downloading faster-whisper model 'small'" in line for line in messages)


def test_ensure_whisper_cpp_downloads_into_override_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "wc"
    calls: dict[str, object] = {}
    messages: list[str] = []

    def _hf_hub_download(**kwargs: object) -> str:
        calls.update(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = local_dir / str(kwargs["filename"])
        downloaded.write_bytes(b"weights")
        return str(downloaded)

    monkeypatch.setattr(
        "narada.asr.model_download._load_hf_download_api",
        lambda: (_hf_hub_download, lambda **_kwargs: ""),
    )

    resolved = ensure_engine_model_available(
        engine_name="whisper-cpp",
        model_name="small",
        whisper_cpp_model_dir=model_dir,
        emit=messages.append,
    )

    assert calls["repo_id"] == "ggerganov/whisper.cpp"
    assert calls["filename"] == "ggml-small.bin"
    assert calls["local_dir"] == str(model_dir)
    assert resolved == model_dir / "ggml-small.bin"
    assert resolved.exists()
    assert any("Downloading whisper-cpp model 'small'" in line for line in messages)


def test_ensure_model_download_network_error_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "wc"

    def _hf_hub_download(**_kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr(
        "narada.asr.model_download._load_hf_download_api",
        lambda: (_hf_hub_download, lambda **_kwargs: ""),
    )

    with pytest.raises(ModelPreparationError) as exc_info:
        ensure_engine_model_available(
            engine_name="whisper-cpp",
            model_name="small",
            whisper_cpp_model_dir=model_dir,
        )

    message = str(exc_info.value)
    assert "Failed to download whisper-cpp model 'small'" in message
    assert str(model_dir / "ggml-small.bin") in message
    assert "Check internet access and write permissions" in message


def test_ensure_model_download_permission_error_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "fw"

    def _snapshot_download(**_kwargs: object) -> str:
        raise PermissionError("access denied")

    monkeypatch.setattr(
        "narada.asr.model_download._load_hf_download_api",
        lambda: (lambda **_kwargs: "", _snapshot_download),
    )

    with pytest.raises(ModelPreparationError) as exc_info:
        ensure_engine_model_available(
            engine_name="faster-whisper",
            model_name="small",
            faster_whisper_model_dir=model_dir,
        )

    message = str(exc_info.value)
    assert "Failed to download faster-whisper model 'small'" in message
    assert str(model_dir / "faster-whisper-small") in message
    assert "write permissions" in message
