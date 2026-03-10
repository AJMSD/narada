from pathlib import Path

import pytest

from narada import doctor
from narada.asr.whisper_cpp_engine import WhisperCliCapabilities
from narada.doctor import DoctorCheck


def test_doctor_model_checks_reports_optional_missing(tmp_path: Path) -> None:
    fw_dir = tmp_path / "fw" / "faster-whisper-small"
    fw_dir.mkdir(parents=True)
    (fw_dir / "model.bin").write_bytes(b"x")

    checks = doctor._model_checks(  # noqa: SLF001
        "small",
        faster_whisper_model_dir=tmp_path / "fw",
        whisper_cpp_model_dir=tmp_path / "wc",
    )
    readiness = next(item for item in checks if item.name == "Model readiness")
    assert readiness.status == "PASS"
    whisper = next(item for item in checks if "whisper-cpp" in item.name)
    assert whisper.status == "INFO"
    assert "Download:" in whisper.message
    assert "Setup:" in whisper.message


def test_has_failures_ignores_info_status() -> None:
    checks = [
        DoctorCheck(name="x", status="PASS", message="ok"),
        DoctorCheck(name="y", status="INFO", message="optional"),
    ]
    assert not doctor.has_failures(checks)


def test_whisper_cli_compatibility_mentions_auto_downgrade_when_backend_hints_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWhisperEngine:
        def is_available(self) -> bool:
            return True

        def probe_cli_capabilities(self) -> WhisperCliCapabilities:
            return WhisperCliCapabilities(
                no_gpu_flag="--no-gpu",
                gpu_layers_flag=None,
                backend_hints=(),
            )

    monkeypatch.setattr(doctor, "WhisperCppEngine", lambda: _FakeWhisperEngine())

    check = doctor._whisper_cli_compatibility_check()  # noqa: SLF001

    assert check.status == "PASS"
    assert "compute=auto" in check.message
    assert "compute=cuda|metal" in check.message
