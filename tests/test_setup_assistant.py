from __future__ import annotations

from collections.abc import Sequence

import pytest

from narada.devices import AudioDevice
from narada.setup.assistant import (
    CommandResult,
    ensure_system_capture_prereqs,
    ensure_whisper_cpp_runtime,
    prepare_start_setup,
)


def _never_confirm(_prompt: str) -> bool:
    raise AssertionError("confirm should not be called")


def test_ensure_whisper_cpp_runtime_skips_when_cli_present() -> None:
    result = ensure_whisper_cpp_runtime(
        interactive=True,
        emit=lambda _message: None,
        confirm=_never_confirm,
        which_fn=lambda name: "/usr/bin/whisper-cli" if name == "whisper-cli" else None,
    )

    assert result.succeeded
    assert not result.performed


def test_ensure_whisper_cpp_runtime_installs_when_missing() -> None:
    state = {"installed": False}
    commands: list[tuple[str, ...]] = []

    def _which(name: str) -> str | None:
        if name in {"whisper-cli", "whisper-cpp"} and state["installed"]:
            return "/usr/local/bin/whisper-cpp"
        return None

    def _run(command: Sequence[str]) -> CommandResult:
        commands.append(tuple(command))
        state["installed"] = True
        return CommandResult(returncode=0, stdout="ok", stderr="")

    result = ensure_whisper_cpp_runtime(
        interactive=True,
        emit=lambda _message: None,
        confirm=lambda _prompt: True,
        run_command=_run,
        which_fn=_which,
        python_executable="python",
    )

    assert result.succeeded
    assert result.performed
    assert ("python", "-m", "pip", "install", "whisper.cpp-cli>=0.0.3") in commands


def test_ensure_whisper_cpp_runtime_decline_returns_actionable_error() -> None:
    result = ensure_whisper_cpp_runtime(
        interactive=True,
        emit=lambda _message: None,
        confirm=lambda _prompt: False,
        which_fn=lambda _name: None,
    )

    assert not result.succeeded
    assert "whisper.cpp runtime is required" in result.message


def test_ensure_whisper_cpp_runtime_non_interactive_does_not_install() -> None:
    commands: list[tuple[str, ...]] = []

    def _run(command: Sequence[str]) -> CommandResult:
        commands.append(tuple(command))
        return CommandResult(returncode=0, stdout="", stderr="")

    result = ensure_whisper_cpp_runtime(
        interactive=False,
        emit=lambda _message: None,
        confirm=lambda _prompt: True,
        run_command=_run,
        which_fn=lambda _name: None,
    )

    assert not result.succeeded
    assert not commands


def test_linux_system_setup_creates_session_loopback_and_returns_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    enumerate_count = {"value": 0}

    def _enumerate_devices() -> list[AudioDevice]:
        enumerate_count["value"] += 1
        if enumerate_count["value"] <= 1:
            return [
                AudioDevice(
                    id=1,
                    name="Speakers",
                    type="output",
                    system_device_id=1,
                    system_device_type="output",
                )
            ]
        return [
            AudioDevice(
                id=1,
                name="Speakers",
                type="output",
                system_device_id=1,
                system_device_type="output",
            ),
            AudioDevice(
                id=2,
                name="Narada Auto Loopback monitor",
                type="monitor",
                system_device_id=2,
                system_device_type="monitor",
            ),
        ]

    def _run(command: Sequence[str]) -> CommandResult:
        tokenized = tuple(command)
        calls.append(tokenized)
        if tokenized[:3] == ("pactl", "load-module", "module-null-sink"):
            return CommandResult(returncode=0, stdout="55", stderr="")
        if tokenized[:2] == ("pactl", "info"):
            return CommandResult(
                returncode=0, stdout="Default Sink: alsa_output.pci-0000", stderr=""
            )
        if tokenized[:3] == ("pactl", "list", "short"):
            return CommandResult(returncode=0, stdout="7\tfoo\tbar", stderr="")
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("narada.setup.assistant.shutil.which", lambda binary: "/bin/pactl")

    result, teardown_steps, selector_override = ensure_system_capture_prereqs(
        os_name="linux",
        mode="system",
        system_selector="1",
        interactive=True,
        emit=lambda _message: None,
        confirm=lambda _prompt: True,
        run_command=_run,
        enumerate_devices_fn=_enumerate_devices,
    )

    assert result.succeeded
    assert result.performed
    assert selector_override == "2"
    assert any(step.command[0:2] == ("pactl", "unload-module") for step in teardown_steps)
    assert any(command[:2] == ("pactl", "set-default-sink") for command in calls)


def test_linux_system_setup_fails_when_pactl_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "narada.setup.assistant.shutil.which",
        lambda binary: None if binary == "pactl" else "/bin/ok",
    )
    result, teardown_steps, selector_override = ensure_system_capture_prereqs(
        os_name="linux",
        mode="system",
        system_selector="1",
        interactive=True,
        emit=lambda _message: None,
        confirm=lambda _prompt: True,
        run_command=lambda _command: CommandResult(returncode=0, stdout="", stderr=""),
        enumerate_devices_fn=lambda: [AudioDevice(id=1, name="Speakers", type="output")],
    )

    assert not result.succeeded
    assert not teardown_steps
    assert selector_override is None
    assert "requires `pactl`" in result.message


def test_macos_system_setup_installs_blackhole(monkeypatch: pytest.MonkeyPatch) -> None:
    enumerate_count = {"value": 0}

    def _enumerate_devices() -> list[AudioDevice]:
        enumerate_count["value"] += 1
        if enumerate_count["value"] <= 1:
            return [
                AudioDevice(
                    id=1,
                    name="Built-in Output",
                    type="output",
                    system_device_id=1,
                    system_device_type="output",
                )
            ]
        return [
            AudioDevice(
                id=1,
                name="Built-in Output",
                type="output",
                system_device_id=1,
                system_device_type="output",
            ),
            AudioDevice(
                id=2,
                name="BlackHole 2ch",
                type="output",
                system_device_id=2,
                system_device_type="output",
            ),
        ]

    commands: list[tuple[str, ...]] = []

    def _run(command: Sequence[str]) -> CommandResult:
        commands.append(tuple(command))
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "narada.setup.assistant.shutil.which",
        lambda binary: "/opt/homebrew/bin/brew" if binary == "brew" else "/bin/ok",
    )

    result, teardown_steps, selector_override = ensure_system_capture_prereqs(
        os_name="macos",
        mode="system",
        system_selector="1",
        interactive=True,
        emit=lambda _message: None,
        confirm=lambda _prompt: True,
        run_command=_run,
        enumerate_devices_fn=_enumerate_devices,
    )

    assert result.succeeded
    assert result.performed
    assert ("brew", "install", "--cask", "blackhole-2ch") in commands
    assert teardown_steps == ()
    assert selector_override is None


def test_windows_system_setup_installs_pyaudiowpatch(monkeypatch: pytest.MonkeyPatch) -> None:
    issue_state = {"missing": True}
    commands: list[tuple[str, ...]] = []

    def _loopback_issue() -> str | None:
        if issue_state["missing"]:
            return "PyAudioWPatch is required for Windows live capture."
        return None

    def _run(command: Sequence[str]) -> CommandResult:
        commands.append(tuple(command))
        issue_state["missing"] = False
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("narada.setup.assistant.windows.loopback_support_error", _loopback_issue)
    result, teardown_steps, selector_override = ensure_system_capture_prereqs(
        os_name="windows",
        mode="system",
        system_selector="1",
        interactive=True,
        emit=lambda _message: None,
        confirm=lambda _prompt: True,
        run_command=_run,
        enumerate_devices_fn=lambda: [
            AudioDevice(
                id=1,
                name="Speakers",
                type="output",
                hostapi="Windows WASAPI",
                system_device_id=1,
                system_device_type="output",
            )
        ],
        python_executable="python",
    )

    assert result.succeeded
    assert result.performed
    assert ("python", "-m", "pip", "install", "PyAudioWPatch>=0.2.12.8") in commands
    assert teardown_steps == ()
    assert selector_override is None


def test_prepare_start_setup_non_interactive_does_not_install() -> None:
    seen_commands: list[tuple[str, ...]] = []

    def _run(command: Sequence[str]) -> CommandResult:
        seen_commands.append(tuple(command))
        return CommandResult(returncode=0, stdout="", stderr="")

    result = prepare_start_setup(
        mode="mic",
        engine_name="whisper-cpp",
        system_selector=None,
        interactive=False,
        emit=lambda _message: None,
        confirm=lambda _prompt: True,
        run_command=_run,
        enumerate_devices_fn=lambda: [],
        which_fn=lambda _name: None,
    )

    assert not result.ok
    assert result.message is not None
    assert not seen_commands
