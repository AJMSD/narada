from __future__ import annotations

import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from narada.asr.base import build_engine
from narada.asr.model_discovery import discover_models
from narada.audio.backends import linux, macos, windows
from narada.devices import AudioDevice, enumerate_devices


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str


def _status(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def _python_check() -> DoctorCheck:
    ok = sys.version_info >= (3, 11)
    msg = f"Python {platform.python_version()} (requires >= 3.11)"
    return DoctorCheck(name="Python version", status=_status(ok), message=msg)


def _engine_checks() -> list[DoctorCheck]:
    availability: dict[str, bool] = {}
    for engine_name in ("faster-whisper", "whisper-cpp"):
        engine = build_engine(engine_name)
        availability[engine_name] = engine.is_available()

    checks: list[DoctorCheck] = []
    any_available = any(availability.values())
    for engine_name in ("faster-whisper", "whisper-cpp"):
        available = availability[engine_name]
        if available:
            status = "PASS"
            message = "Available"
        elif any_available:
            status = "INFO"
            message = (
                "Not installed in current environment (optional while another engine is available)"
            )
        else:
            status = "WARN"
            message = "Not installed in current environment"
        checks.append(
            DoctorCheck(name=f"ASR engine: {engine_name}", status=status, message=message)
        )
    return checks


def _model_checks(
    model_name: str,
    faster_whisper_model_dir: Path | None,
    whisper_cpp_model_dir: Path | None,
) -> list[DoctorCheck]:
    discovery = discover_models(
        model_name,
        faster_whisper_model_dir=faster_whisper_model_dir,
        whisper_cpp_model_dir=whisper_cpp_model_dir,
    )
    checks: list[DoctorCheck] = []
    any_present = discovery.any_present

    for item in (discovery.faster_whisper, discovery.whisper_cpp):
        if item.present:
            checks.append(
                DoctorCheck(
                    name=f"Model ({item.engine}:{item.model_name})",
                    status="PASS",
                    message=f"Found at {item.model_path}",
                )
            )
        else:
            status = "INFO" if any_present else "WARN"
            checks.append(
                DoctorCheck(
                    name=f"Model ({item.engine}:{item.model_name})",
                    status=status,
                    message=(
                        f"Missing at {item.model_path}. "
                        f"Download: {item.model_url} | Setup: {item.setup_url}"
                    ),
                )
            )

    checks.append(
        DoctorCheck(
            name="Model readiness",
            status="PASS" if any_present else "WARN",
            message=(
                "At least one model is available for local transcription."
                if any_present
                else "No local model files detected. Download at least one model."
            ),
        )
    )
    return checks


def _audio_probe(devices: Sequence[AudioDevice]) -> DoctorCheck:
    os_name = platform.system().lower()
    if os_name == "windows":
        probe = windows.probe(devices)
    elif os_name == "linux":
        probe = linux.probe(devices)
    elif os_name == "darwin":
        probe = macos.probe(devices)
    else:
        return DoctorCheck("Audio backend", "WARN", f"Unsupported platform: {platform.system()}")

    ok = probe.supports_mic_capture or probe.supports_system_capture
    return DoctorCheck(
        name=f"Audio backend ({probe.backend})",
        status=_status(ok),
        message=probe.summary,
    )


def _device_check(devices: Sequence[AudioDevice]) -> DoctorCheck:
    return DoctorCheck(
        name="Audio devices",
        status=_status(len(devices) > 0),
        message=f"Detected {len(devices)} device endpoints",
    )


def _path_check(output_path: Path | None) -> DoctorCheck:
    if output_path is None:
        return DoctorCheck(
            name="Transcript path",
            status="PASS",
            message="No transcript path provided. Skipping writeability check.",
        )
    target_dir = output_path.parent
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return DoctorCheck(
            name="Transcript path",
            status="WARN",
            message=f"Cannot prepare output directory '{target_dir}': {exc}",
        )
    writable = target_dir.exists() and os_access_write(target_dir)
    return DoctorCheck(
        name="Transcript path",
        status=_status(writable),
        message=f"Output directory {'is' if writable else 'is not'} writable: {target_dir}",
    )


def os_access_write(path: Path) -> bool:
    try:
        test_path = path / ".narada-write-test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def run_doctor(
    output_path: Path | None = None,
    model_name: str = "small",
    faster_whisper_model_dir: Path | None = None,
    whisper_cpp_model_dir: Path | None = None,
) -> list[DoctorCheck]:
    devices = enumerate_devices()
    checks = [_python_check(), _device_check(devices), _audio_probe(devices), *_engine_checks()]
    checks.extend(
        _model_checks(
            model_name,
            faster_whisper_model_dir=faster_whisper_model_dir,
            whisper_cpp_model_dir=whisper_cpp_model_dir,
        )
    )
    checks.append(_path_check(output_path))
    if platform.system().lower() == "darwin":
        has_blackhole = any("blackhole" in device.name.lower() for device in devices)
        checks.append(
            DoctorCheck(
                name="macOS virtual device",
                status=_status(has_blackhole),
                message=(
                    "BlackHole-style loopback device found."
                    if has_blackhole
                    else (
                        "No BlackHole device found. Install a virtual loopback "
                        "device for system audio capture."
                    )
                ),
            )
        )
    return checks


def format_doctor_report(checks: Sequence[DoctorCheck]) -> str:
    if not checks:
        return "No doctor checks available."

    name_width = max(len("Check"), max(len(item.name) for item in checks))
    status_width = max(len("Status"), max(len(item.status) for item in checks))
    lines = [
        f"{'Check':<{name_width}}  {'Status':<{status_width}}  Message",
        f"{'-' * name_width}  {'-' * status_width}  {'-' * 7}",
    ]
    for item in checks:
        lines.append(f"{item.name:<{name_width}}  {item.status:<{status_width}}  {item.message}")
    return "\n".join(lines)


def has_failures(checks: Sequence[DoctorCheck]) -> bool:
    return any(item.status.upper() == "WARN" for item in checks)
