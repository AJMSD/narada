from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import typer

from narada.asr.base import EngineUnavailableError, TranscriptSegment, build_engine
from narada.config import ConfigError, ConfigOverrides, build_runtime_config
from narada.devices import (
    DEVICE_TYPES,
    AmbiguousDeviceError,
    DeviceResolutionError,
    DeviceType,
    devices_to_json,
    enumerate_devices,
    filter_devices,
    format_devices_table,
    resolve_device,
)
from narada.doctor import format_doctor_report, has_failures, run_doctor
from narada.pipeline import ConfidenceGate
from narada.redaction import redact_text
from narada.server import serve_transcript_file
from narada.writer import TranscriptWriter

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Narada local transcript CLI.")


def _resolve_selected_devices(mode: str, mic: str | None, system: str | None) -> None:
    devices = enumerate_devices()
    if mode in {"mic", "mixed"} and mic:
        resolve_device(mic, devices, {"input"})
    if mode in {"system", "mixed"} and system:
        resolve_device(system, devices, {"output", "loopback", "monitor"})


def _elapsed(started_at: float) -> str:
    total = int(time.time() - started_at)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


@app.command("devices")
def devices_command(
    device_type: str | None = typer.Option(None, "--type", help="input|output|loopback|monitor"),
    search: str | None = typer.Option(None, "--search", help="Case-insensitive name filter."),
    json_output: bool = typer.Option(False, "--json", help="Return JSON output."),
) -> None:
    if device_type is not None and device_type not in DEVICE_TYPES:
        raise typer.BadParameter(
            f"Unsupported device type '{device_type}'. Expected one of: {', '.join(DEVICE_TYPES)}.",
            param_hint="--type",
        )
    normalized_type: DeviceType | None = None
    if device_type is not None:
        normalized_type = cast(DeviceType, device_type)

    items = enumerate_devices()
    filtered = filter_devices(items, device_type=normalized_type, search=search)
    if json_output:
        typer.echo(devices_to_json(filtered))
        return
    typer.echo(format_devices_table(filtered))


@app.command("start")
def start_command(
    mode: str | None = typer.Option(None, "--mode", help="mic|system|mixed"),
    mic: str | None = typer.Option(None, "--mic", help="Device ID or name for microphone input."),
    system: str | None = typer.Option(
        None, "--system", help="Device ID or name for system output/loopback."
    ),
    out: Path | None = typer.Option(None, "--out", help="Transcript output path."),
    model: str | None = typer.Option(None, "--model", help="tiny|small|medium|large"),
    compute: str | None = typer.Option(None, "--compute", help="cpu|cuda|metal|auto"),
    engine: str | None = typer.Option(None, "--engine", help="faster-whisper|whisper-cpp"),
    language: str | None = typer.Option(
        None,
        "--language",
        help="auto or comma-separated language list (example: hindi,english).",
    ),
    allow_multilingual: bool | None = typer.Option(
        None,
        "--allow-multilingual",
        help="Required when --language contains more than one language.",
    ),
    redact: str | None = typer.Option(None, "--redact", help="on|off"),
    noise_suppress: str | None = typer.Option(None, "--noise-suppress", help="off|rnnoise|webrtc"),
    agc: str | None = typer.Option(None, "--agc", help="on|off"),
    gate: str | None = typer.Option(None, "--gate", help="on|off"),
    gate_threshold_db: float | None = typer.Option(None, "--gate-threshold-db"),
    confidence_threshold: float | None = typer.Option(None, "--confidence-threshold"),
) -> None:
    overrides = ConfigOverrides(
        mode=mode,
        mic=mic,
        system=system,
        out=out,
        model=model,
        compute=compute,
        engine=engine,
        language=language,
        allow_multilingual=allow_multilingual,
        redact=redact,
        noise_suppress=noise_suppress,
        agc=agc,
        gate=gate,
        gate_threshold_db=gate_threshold_db,
        confidence_threshold=confidence_threshold,
    )

    try:
        config = build_runtime_config(overrides)
        _resolve_selected_devices(config.mode, config.mic, config.system)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except AmbiguousDeviceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except DeviceResolutionError as exc:
        raise typer.BadParameter(str(exc)) from exc

    engine_instance = build_engine(config.engine)
    if not engine_instance.is_available() and sys.stdin.isatty():
        raise typer.BadParameter(
            f"Selected engine '{config.engine}' is unavailable. "
            "Install dependencies or pipe text input for dry run."
        )

    gate_state = ConfidenceGate(config.confidence_threshold)

    typer.echo(
        "REC "
        + datetime.now().isoformat(timespec="seconds")
        + f" | mode={config.mode} | model={config.model}"
    )
    typer.echo(f"Writing transcript to: {config.out}")
    if sys.stdin.isatty():
        typer.echo("No piped input detected. Waiting for Ctrl+C.")

    started_at = time.time()
    try:
        with TranscriptWriter(config.out) as writer:
            if sys.stdin.isatty():
                while True:
                    typer.echo(
                        f"\rREC {_elapsed(started_at)} | mode={config.mode} "
                        f"| model={config.model} | rtf=n/a",
                        nl=False,
                    )
                    time.sleep(1.0)
            else:
                for line in sys.stdin:
                    segments = [
                        TranscriptSegment(text=line.strip(), confidence=1.0, start_s=0.0, end_s=0.0)
                    ]
                    committed = gate_state.ingest(segments)
                    for item in committed:
                        text = redact_text(item.text) if config.redact else item.text
                        writer.append_line(text)
                for pending in gate_state.drain_pending():
                    text = redact_text(pending.text) if config.redact else pending.text
                    writer.append_line(text)
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


@app.command("serve")
def serve_command(
    file: Path | None = typer.Option(None, "--file", help="Transcript file path."),
    port: int | None = typer.Option(None, "--port", help="HTTP port."),
    qr: bool = typer.Option(False, "--qr", help="Print an ASCII QR code."),
    bind: str | None = typer.Option(None, "--bind", help="Bind address."),
) -> None:
    transcript_file = file
    if transcript_file is None:
        transcript_file = Path(os.environ.get("NARADA_OUT", "./transcripts/session.txt"))
    serve_transcript_file(
        transcript_path=transcript_file,
        bind=bind or os.environ.get("NARADA_BIND", "0.0.0.0"),
        port=port or int(os.environ.get("NARADA_PORT", "8787")),
        show_qr=qr,
    )


@app.command("doctor")
def doctor_command(
    file: Path | None = typer.Option(
        None, "--file", help="Optional transcript file path to validate."
    ),
) -> None:
    checks = run_doctor(file)
    typer.echo(format_doctor_report(checks))
    if has_failures(checks):
        raise typer.Exit(code=1)


def main() -> None:
    try:
        app()
    except EngineUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    main()
