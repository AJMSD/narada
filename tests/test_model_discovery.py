from pathlib import Path

from narada.asr.model_discovery import (
    build_start_model_preflight,
    discover_models,
)


def test_model_discovery_no_models_present(tmp_path: Path) -> None:
    discovery = discover_models(
        "small",
        faster_whisper_model_dir=tmp_path / "fw",
        whisper_cpp_model_dir=tmp_path / "wc",
    )
    assert not discovery.any_present
    preflight = build_start_model_preflight(discovery, "faster-whisper")
    assert not preflight.selected_available
    assert preflight.recommended_engine is None
    assert any("Download faster-whisper model" in line for line in preflight.messages)
    assert any("Download whisper.cpp model" in line for line in preflight.messages)


def test_model_discovery_faster_whisper_only(tmp_path: Path) -> None:
    fw_dir = tmp_path / "fw" / "faster-whisper-small"
    fw_dir.mkdir(parents=True)
    (fw_dir / "model.bin").write_bytes(b"weights")

    discovery = discover_models(
        "small",
        faster_whisper_model_dir=tmp_path / "fw",
        whisper_cpp_model_dir=tmp_path / "wc",
    )
    preflight = build_start_model_preflight(discovery, "whisper-cpp")
    assert discovery.faster_whisper.present
    assert not discovery.whisper_cpp.present
    assert preflight.recommended_engine == "faster-whisper"
    assert any("Narada will run with faster-whisper" in line for line in preflight.messages)


def test_model_discovery_whisper_cpp_only(tmp_path: Path) -> None:
    wc_dir = tmp_path / "wc"
    wc_dir.mkdir(parents=True)
    (wc_dir / "ggml-small.bin").write_bytes(b"weights")

    discovery = discover_models(
        "small",
        faster_whisper_model_dir=tmp_path / "fw",
        whisper_cpp_model_dir=wc_dir,
    )
    preflight = build_start_model_preflight(discovery, "faster-whisper")
    assert not discovery.faster_whisper.present
    assert discovery.whisper_cpp.present
    assert preflight.recommended_engine == "whisper-cpp"
    assert any("Narada will run with whisper-cpp" in line for line in preflight.messages)
