from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from narada.audio.backends import linux, macos, windows
from narada.devices import AudioDevice, DeviceResolutionError, enumerate_devices, resolve_device

MessageEmitter = Callable[[str], None]
ConfirmPrompt = Callable[[str], bool]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SetupActionResult:
    performed: bool
    succeeded: bool
    message: str


@dataclass(frozen=True)
class TeardownStep:
    description: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class StartSetupResult:
    runtime_ready: bool
    loopback_ready: bool
    message: str | None = None
    teardown_steps: tuple[TeardownStep, ...] = ()
    system_selector_override: str | None = None

    @property
    def ok(self) -> bool:
        return self.runtime_ready and self.loopback_ready


@dataclass(frozen=True)
class _SystemCaptureReadiness:
    ready: bool
    message: str


def _run_command(command: Sequence[str]) -> CommandResult:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandResult(
        returncode=int(completed.returncode),
        stdout=(completed.stdout or "").strip(),
        stderr=(completed.stderr or "").strip(),
    )


def _normalize_os_name(raw_os: str | None = None) -> str:
    value = (raw_os or platform.system()).strip().lower()
    if value == "darwin":
        return "macos"
    return value


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _find_whisper_cli(which_fn: Callable[[str], str | None] = shutil.which) -> str | None:
    for binary_name in ("whisper-cli", "whisper-cpp"):
        candidate = which_fn(binary_name)
        if candidate:
            return candidate
    return None


def _resolve_selected_system_device(
    *,
    selector: str | None,
    devices: Sequence[AudioDevice],
) -> AudioDevice:
    if selector is None:
        raise DeviceResolutionError("Mode requires --system (or NARADA_SYSTEM).")
    return resolve_device(selector, devices, {"output", "loopback", "monitor"})


def _check_system_capture_readiness(
    *,
    os_name: str,
    selector: str | None,
    devices: Sequence[AudioDevice],
) -> _SystemCaptureReadiness:
    try:
        selected_device = _resolve_selected_system_device(selector=selector, devices=devices)
    except DeviceResolutionError as exc:
        return _SystemCaptureReadiness(ready=False, message=str(exc))

    try:
        if os_name == "windows":
            windows.resolve_system_capture_device(selected_device, devices)
            loopback_issue = windows.loopback_support_error()
            if loopback_issue is not None:
                return _SystemCaptureReadiness(ready=False, message=loopback_issue)
            return _SystemCaptureReadiness(ready=True, message="Windows system capture ready.")
        if os_name == "linux":
            linux.resolve_system_capture_device(selected_device, devices)
            return _SystemCaptureReadiness(ready=True, message="Linux system capture ready.")
        if os_name == "macos":
            macos.resolve_system_capture_device(selected_device, devices)
            return _SystemCaptureReadiness(ready=True, message="macOS system capture ready.")
    except DeviceResolutionError as exc:
        return _SystemCaptureReadiness(ready=False, message=str(exc))

    return _SystemCaptureReadiness(ready=True, message="System capture readiness check skipped.")


def _linux_parse_default_sink(pactl_info: str) -> str | None:
    for raw_line in pactl_info.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("default sink:"):
            continue
        value = line.split(":", 1)[1].strip()
        if value:
            return value
    return None


def _linux_extract_module_id(load_stdout: str) -> str | None:
    match = re.search(r"^\s*(\d+)\s*$", load_stdout)
    if match is None:
        return None
    return match.group(1)


def _linux_select_auto_monitor(
    *, devices: Sequence[AudioDevice], sink_name: str, description_hint: str
) -> AudioDevice | None:
    lowered_sink = sink_name.lower()
    lowered_hint = description_hint.lower()
    candidates = [
        item
        for item in devices
        if item.type in {"monitor", "loopback", "output"}
        and (
            lowered_sink in item.name.lower()
            or lowered_hint in item.name.lower()
            or "narada auto loopback" in item.name.lower()
        )
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.type != "monitor", item.id))
    return candidates[0]


def _manual_system_capture_guidance(os_name: str) -> str:
    if os_name == "windows":
        return (
            "Ensure PyAudioWPatch is installed and a WASAPI loopback source is available. "
            "Enable a playback device or Stereo Mix if your driver supports it."
        )
    if os_name == "linux":
        return (
            "Configure a PulseAudio/PipeWire monitor source and select it with --system. "
            "If using PipeWire, verify `pactl list short sources` shows a monitor device."
        )
    if os_name == "macos":
        return (
            "Install a virtual loopback device (for example BlackHole), route output through it, "
            "and select the virtual device with --system."
        )
    return "System capture prerequisites are missing for this platform."


def ensure_whisper_cpp_runtime(
    *,
    interactive: bool,
    emit: MessageEmitter,
    confirm: ConfirmPrompt,
    run_command: Callable[[Sequence[str]], CommandResult] = _run_command,
    which_fn: Callable[[str], str | None] = shutil.which,
    python_executable: str = sys.executable,
) -> SetupActionResult:
    cli_path = _find_whisper_cli(which_fn)
    if cli_path is not None:
        return SetupActionResult(
            performed=False,
            succeeded=True,
            message=f"whisper.cpp runtime is available at {cli_path}.",
        )

    if not interactive:
        return SetupActionResult(
            performed=False,
            succeeded=False,
            message=(
                "whisper.cpp runtime is missing and auto-setup is disabled "
                "in non-interactive mode. "
                'Install with: python -m pip install "whisper.cpp-cli>=0.0.3"'
            ),
        )

    if not confirm("whisper.cpp runtime is missing. Install it now via pip?"):
        return SetupActionResult(
            performed=False,
            succeeded=False,
            message=(
                "whisper.cpp runtime is required for --engine whisper-cpp. "
                'Install with: python -m pip install "whisper.cpp-cli>=0.0.3"'
            ),
        )

    emit("Installing whisper.cpp runtime via pip (whisper.cpp-cli>=0.0.3)...")
    install_result = run_command(
        [python_executable, "-m", "pip", "install", "whisper.cpp-cli>=0.0.3"]
    )
    if install_result.returncode != 0:
        details = _first_non_empty(install_result.stderr, install_result.stdout)
        return SetupActionResult(
            performed=True,
            succeeded=False,
            message=(
                "Failed to install whisper.cpp runtime automatically. "
                f"{details or 'pip exited with a non-zero status.'}"
            ),
        )

    cli_path = _find_whisper_cli(which_fn)
    if cli_path is None:
        return SetupActionResult(
            performed=True,
            succeeded=False,
            message=(
                "whisper.cpp runtime installation completed but CLI binary was not found on PATH. "
                "Restart your shell and verify `whisper-cpp --help`."
            ),
        )

    return SetupActionResult(
        performed=True,
        succeeded=True,
        message=f"Installed whisper.cpp runtime at {cli_path}.",
    )


def ensure_system_capture_prereqs(
    *,
    os_name: str,
    mode: str,
    system_selector: str | None,
    interactive: bool,
    emit: MessageEmitter,
    confirm: ConfirmPrompt,
    run_command: Callable[[Sequence[str]], CommandResult] = _run_command,
    enumerate_devices_fn: Callable[[], list[AudioDevice]] = enumerate_devices,
    python_executable: str = sys.executable,
) -> tuple[SetupActionResult, tuple[TeardownStep, ...], str | None]:
    if mode != "system":
        return (
            SetupActionResult(
                performed=False, succeeded=True, message="System capture setup skipped."
            ),
            (),
            None,
        )

    devices = enumerate_devices_fn()
    readiness = _check_system_capture_readiness(
        os_name=os_name, selector=system_selector, devices=devices
    )
    if readiness.ready:
        return (
            SetupActionResult(performed=False, succeeded=True, message=readiness.message),
            (),
            None,
        )

    if not interactive:
        return (
            SetupActionResult(
                performed=False,
                succeeded=False,
                message=(f"{readiness.message} {_manual_system_capture_guidance(os_name)}"),
            ),
            (),
            None,
        )

    if not confirm("System capture prerequisites are missing. Attempt automatic setup now?"):
        return (
            SetupActionResult(
                performed=False,
                succeeded=False,
                message=f"{readiness.message} {_manual_system_capture_guidance(os_name)}",
            ),
            (),
            None,
        )

    if os_name == "windows":
        issue = windows.loopback_support_error()
        if issue is not None and "PyAudioWPatch is required" in issue:
            emit("Installing PyAudioWPatch for Windows live capture...")
            install_result = run_command(
                [python_executable, "-m", "pip", "install", "PyAudioWPatch>=0.2.12.8"]
            )
            if install_result.returncode != 0:
                details = _first_non_empty(install_result.stderr, install_result.stdout)
                return (
                    SetupActionResult(
                        performed=True,
                        succeeded=False,
                        message=(
                            "Failed to install PyAudioWPatch automatically. "
                            f"{details or 'pip exited with a non-zero status.'}"
                        ),
                    ),
                    (),
                    None,
                )
            devices = enumerate_devices_fn()
            post = _check_system_capture_readiness(
                os_name=os_name, selector=system_selector, devices=devices
            )
            return (
                SetupActionResult(
                    performed=True,
                    succeeded=post.ready,
                    message=post.message
                    if post.ready
                    else f"{post.message} {_manual_system_capture_guidance(os_name)}",
                ),
                (),
                None,
            )
        return (
            SetupActionResult(
                performed=False,
                succeeded=False,
                message=f"{readiness.message} {_manual_system_capture_guidance(os_name)}",
            ),
            (),
            None,
        )

    if os_name == "linux":
        if shutil.which("pactl") is None:
            return (
                SetupActionResult(
                    performed=False,
                    succeeded=False,
                    message=(
                        "Automatic Linux setup requires `pactl`, but it was not found. "
                        f"{_manual_system_capture_guidance(os_name)}"
                    ),
                ),
                (),
                None,
            )

        sink_name = f"narada_auto_loopback_{os.getpid()}"
        sink_description = "Narada Auto Loopback"
        load_result = run_command(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={sink_name}",
                f"sink_properties=device.description={sink_description}",
            ]
        )
        if load_result.returncode != 0:
            details = _first_non_empty(load_result.stderr, load_result.stdout)
            return (
                SetupActionResult(
                    performed=True,
                    succeeded=False,
                    message=(
                        "Failed to create a temporary loopback sink. "
                        f"{details or 'pactl load-module failed.'}"
                    ),
                ),
                (),
                None,
            )

        teardown_steps: list[TeardownStep] = []
        module_id = _linux_extract_module_id(load_result.stdout)
        if module_id is not None:
            teardown_steps.append(
                TeardownStep(
                    description="Unload temporary Narada loopback sink",
                    command=("pactl", "unload-module", module_id),
                )
            )

        info_result = run_command(["pactl", "info"])
        default_sink = (
            _linux_parse_default_sink(info_result.stdout) if info_result.returncode == 0 else None
        )
        set_default_result = run_command(["pactl", "set-default-sink", sink_name])
        if set_default_result.returncode == 0 and default_sink and default_sink != sink_name:
            teardown_steps.append(
                TeardownStep(
                    description="Restore previous default sink",
                    command=("pactl", "set-default-sink", default_sink),
                )
            )

        sink_inputs_result = run_command(["pactl", "list", "short", "sink-inputs"])
        if sink_inputs_result.returncode == 0:
            for raw_line in sink_inputs_result.stdout.splitlines():
                parts = raw_line.split()
                if not parts:
                    continue
                _ = run_command(["pactl", "move-sink-input", parts[0], sink_name])

        devices = enumerate_devices_fn()
        post = _check_system_capture_readiness(
            os_name=os_name, selector=system_selector, devices=devices
        )
        if post.ready:
            return (
                SetupActionResult(
                    performed=True,
                    succeeded=True,
                    message="Linux system capture prerequisites were configured for this session.",
                ),
                tuple(teardown_steps),
                None,
            )

        auto_monitor = _linux_select_auto_monitor(
            devices=devices,
            sink_name=sink_name,
            description_hint=sink_description,
        )
        if auto_monitor is not None:
            override_selector = str(auto_monitor.id)
            override_readiness = _check_system_capture_readiness(
                os_name=os_name,
                selector=override_selector,
                devices=devices,
            )
            if override_readiness.ready:
                return (
                    SetupActionResult(
                        performed=True,
                        succeeded=True,
                        message=(
                            "Linux system capture prerequisites were configured for this session. "
                            f"Using temporary monitor device {auto_monitor.id}:{auto_monitor.name}."
                        ),
                    ),
                    tuple(teardown_steps),
                    override_selector,
                )

        return (
            SetupActionResult(
                performed=True,
                succeeded=False,
                message=f"{post.message} {_manual_system_capture_guidance(os_name)}",
            ),
            tuple(teardown_steps),
            None,
        )

    if os_name == "macos":
        if shutil.which("brew") is None:
            return (
                SetupActionResult(
                    performed=False,
                    succeeded=False,
                    message=(
                        "Automatic macOS setup requires Homebrew (`brew`) "
                        "for BlackHole installation. "
                        f"{_manual_system_capture_guidance(os_name)}"
                    ),
                ),
                (),
                None,
            )

        if not confirm(
            "Install BlackHole (virtual loopback) now via Homebrew? "
            "This may request admin privileges."
        ):
            return (
                SetupActionResult(
                    performed=False,
                    succeeded=False,
                    message=f"{readiness.message} {_manual_system_capture_guidance(os_name)}",
                ),
                (),
                None,
            )

        emit("Installing BlackHole via Homebrew (`brew install --cask blackhole-2ch`)...")
        install_result = run_command(["brew", "install", "--cask", "blackhole-2ch"])
        if install_result.returncode != 0:
            details = _first_non_empty(install_result.stderr, install_result.stdout)
            return (
                SetupActionResult(
                    performed=True,
                    succeeded=False,
                    message=(
                        "Failed to install BlackHole automatically. "
                        f"{details or 'Homebrew command failed.'}"
                    ),
                ),
                (),
                None,
            )

        devices = enumerate_devices_fn()
        post = _check_system_capture_readiness(
            os_name=os_name, selector=system_selector, devices=devices
        )
        return (
            SetupActionResult(
                performed=True,
                succeeded=post.ready,
                message=post.message
                if post.ready
                else (
                    f"{post.message} Install completed; if needed, route output through the "
                    "virtual device in Audio MIDI Setup and retry."
                ),
            ),
            (),
            None,
        )

    return (
        SetupActionResult(
            performed=False,
            succeeded=True,
            message="System capture setup skipped on unsupported platform.",
        ),
        (),
        None,
    )


def prepare_start_setup(
    *,
    mode: str,
    engine_name: str,
    system_selector: str | None,
    interactive: bool,
    emit: MessageEmitter,
    confirm: ConfirmPrompt,
    run_command: Callable[[Sequence[str]], CommandResult] = _run_command,
    enumerate_devices_fn: Callable[[], list[AudioDevice]] = enumerate_devices,
    python_executable: str = sys.executable,
    which_fn: Callable[[str], str | None] = shutil.which,
) -> StartSetupResult:
    normalized_os = _normalize_os_name()
    teardown_steps: list[TeardownStep] = []
    system_selector_override: str | None = None

    runtime_ready = True
    runtime_message: str | None = None
    if engine_name.strip().lower() == "whisper-cpp":
        runtime_result = ensure_whisper_cpp_runtime(
            interactive=interactive,
            emit=emit,
            confirm=confirm,
            run_command=run_command,
            which_fn=which_fn,
            python_executable=python_executable,
        )
        runtime_ready = runtime_result.succeeded
        runtime_message = runtime_result.message
        if runtime_result.performed or not runtime_result.succeeded:
            emit(runtime_result.message)

    loopback_result, loopback_teardown, selector_override = ensure_system_capture_prereqs(
        os_name=normalized_os,
        mode=mode,
        system_selector=system_selector,
        interactive=interactive,
        emit=emit,
        confirm=confirm,
        run_command=run_command,
        enumerate_devices_fn=enumerate_devices_fn,
        python_executable=python_executable,
    )
    loopback_ready = loopback_result.succeeded
    teardown_steps.extend(loopback_teardown)
    if selector_override is not None:
        system_selector_override = selector_override
    if loopback_result.performed or not loopback_result.succeeded:
        emit(loopback_result.message)

    message = None
    if not runtime_ready:
        message = runtime_message
    elif not loopback_ready:
        message = loopback_result.message

    return StartSetupResult(
        runtime_ready=runtime_ready,
        loopback_ready=loopback_ready,
        message=message,
        teardown_steps=tuple(teardown_steps),
        system_selector_override=system_selector_override,
    )


def run_setup_teardown(
    *,
    steps: Sequence[TeardownStep],
    emit: MessageEmitter,
    run_command: Callable[[Sequence[str]], CommandResult] = _run_command,
) -> None:
    for step in reversed(list(steps)):
        emit(f"Running setup cleanup step: {step.description}")
        result = run_command(step.command)
        if result.returncode == 0:
            continue
        details = _first_non_empty(result.stderr, result.stdout)
        emit(
            f"Warning: setup cleanup step failed ({' '.join(step.command)}): "
            f"{details or 'non-zero exit status'}"
        )
