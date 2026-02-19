from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import typer

from narada.asr.base import (
    AsrEngine,
    EngineUnavailableError,
    TranscriptionRequest,
    TranscriptSegment,
    build_engine,
)
from narada.audio.capture import (
    CaptureError,
    DeviceDisconnectedError,
    open_mic_capture,
    open_system_capture,
    pcm16le_to_float32,
)
from narada.audio.mixer import AudioChunk, DriftResyncState, mix_audio_chunks
from narada.config import ConfigError, ConfigOverrides, RuntimeConfig, build_runtime_config
from narada.devices import (
    DEVICE_TYPES,
    AmbiguousDeviceError,
    AudioDevice,
    DeviceResolutionError,
    DeviceType,
    devices_to_json,
    enumerate_devices,
    filter_devices,
    format_devices_table,
    resolve_device,
)
from narada.doctor import format_doctor_report, has_failures, run_doctor
from narada.logging_setup import setup_logging
from narada.pipeline import AudioChunkWindow, ConfidenceGate, OverlapChunker
from narada.redaction import redact_text
from narada.server import serve_transcript_file
from narada.start_runtime import MonoAudioFrame, mono_frame_to_pcm16le, parse_input_line
from narada.writer import TranscriptWriter

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Narada local transcript CLI.")
logger = logging.getLogger("narada.cli")


@app.callback()
def app_callback(
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Optional log file path."),
) -> None:
    setup_logging(debug=debug, log_file=log_file)


def _resolve_selected_devices(
    mode: str, mic: str | None, system: str | None
) -> tuple[AudioDevice | None, AudioDevice | None, list[AudioDevice]]:
    devices = enumerate_devices()
    resolved_mic: AudioDevice | None = None
    resolved_system: AudioDevice | None = None
    if mode in {"mic", "mixed"} and mic:
        resolved_mic = resolve_device(mic, devices, {"input"})
    if mode in {"system", "mixed"} and system:
        resolved_system = resolve_device(system, devices, {"output", "loopback", "monitor"})
    return resolved_mic, resolved_system, devices


def _elapsed(started_at: float) -> str:
    total = int(time.time() - started_at)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _transcribe_windows(
    windows: list[AudioChunkWindow],
    *,
    engine_instance: AsrEngine,
    config: RuntimeConfig,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for window in windows:
        request = TranscriptionRequest(
            pcm_bytes=window.pcm_bytes,
            sample_rate_hz=window.sample_rate_hz,
            languages=config.languages,
            model=config.model,
            compute=config.compute,
        )
        result = engine_instance.transcribe(request)
        segments.extend(result)
    return segments


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
        mic_device, system_device, all_devices = _resolve_selected_devices(
            config.mode, config.mic, config.system
        )
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
    engine_available = engine_instance.is_available()

    typer.echo(
        "REC "
        + datetime.now().isoformat(timespec="seconds")
        + f" | mode={config.mode} | model={config.model}"
    )
    typer.echo(f"Writing transcript to: {config.out}")
    if sys.stdin.isatty():
        typer.echo("No piped input detected. Waiting for Ctrl+C.")

    started_at = time.time()
    warned_missing_engine_for_audio = False
    mic_capture = None
    system_capture = None
    mixed_resync_state = DriftResyncState()
    stopped_by_user = False
    try:
        with TranscriptWriter(config.out) as writer:
            audio_chunker = OverlapChunker(chunk_duration_s=6.0, overlap_duration_s=1.5)
            if sys.stdin.isatty():
                if config.mode in {"mic", "mixed"} and mic_device is not None:
                    mic_capture = open_mic_capture(device=mic_device)
                if config.mode in {"system", "mixed"} and system_device is not None:
                    system_capture = open_system_capture(
                        device=system_device,
                        all_devices=all_devices,
                    )

                try:
                    while True:
                        segments: list[TranscriptSegment] = []
                        audio_windows: list[AudioChunkWindow] = []
                        if config.mode == "mic" and mic_capture is not None:
                            mic_frame = mic_capture.read_frame()
                            if mic_frame:
                                audio_windows.extend(
                                    audio_chunker.ingest(
                                        mic_frame.pcm_bytes,
                                        mic_frame.sample_rate_hz,
                                        mic_frame.channels,
                                    )
                                )
                        elif config.mode == "system" and system_capture is not None:
                            system_frame = system_capture.read_frame()
                            if system_frame:
                                audio_windows.extend(
                                    audio_chunker.ingest(
                                        system_frame.pcm_bytes,
                                        system_frame.sample_rate_hz,
                                        system_frame.channels,
                                    )
                                )
                        elif (
                            config.mode == "mixed"
                            and mic_capture is not None
                            and system_capture is not None
                        ):
                            mic_frame = mic_capture.read_frame()
                            system_frame = system_capture.read_frame()
                            if mic_frame and system_frame:
                                mixed_samples, mixed_rate = mix_audio_chunks(
                                    AudioChunk(
                                        samples=pcm16le_to_float32(mic_frame.pcm_bytes),
                                        sample_rate_hz=mic_frame.sample_rate_hz,
                                        channels=mic_frame.channels,
                                    ),
                                    AudioChunk(
                                        samples=pcm16le_to_float32(system_frame.pcm_bytes),
                                        sample_rate_hz=system_frame.sample_rate_hz,
                                        channels=system_frame.channels,
                                    ),
                                    resync_state=mixed_resync_state,
                                )
                                audio_windows.extend(
                                    audio_chunker.ingest(
                                        mono_frame_to_pcm16le(
                                            MonoAudioFrame(
                                                samples=tuple(mixed_samples),
                                                sample_rate_hz=mixed_rate,
                                            )
                                        ),
                                        mixed_rate,
                                        1,
                                    )
                                )

                        if audio_windows and engine_available:
                            segments.extend(
                                _transcribe_windows(
                                    audio_windows,
                                    engine_instance=engine_instance,
                                    config=config,
                                )
                            )

                        committed = gate_state.ingest(segments)
                        for item in committed:
                            text = redact_text(item.text) if config.redact else item.text
                            writer.append_line(text)
                        typer.echo(
                            f"\rREC {_elapsed(started_at)} | mode={config.mode} "
                            f"| model={config.model} | rtf=n/a",
                            nl=False,
                        )
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    stopped_by_user = True
            else:
                for line in sys.stdin:
                    try:
                        parsed = parse_input_line(line, config.mode)
                    except ValueError as exc:
                        logger.warning("Skipping invalid stdin input: %s", exc)
                        continue
                    if parsed is None:
                        continue
                    stdin_segments: list[TranscriptSegment] = []
                    if parsed.text:
                        stdin_segments.append(
                            TranscriptSegment(
                                text=parsed.text,
                                confidence=parsed.confidence,
                                start_s=0.0,
                                end_s=0.0,
                                is_final=True,
                            )
                        )
                    if parsed.audio:
                        if not engine_available:
                            if not warned_missing_engine_for_audio:
                                typer.echo(
                                    "Selected ASR engine is unavailable; "
                                    "audio payloads are skipped.",
                                    err=True,
                                )
                                warned_missing_engine_for_audio = True
                        else:
                            audio_windows = audio_chunker.ingest(
                                mono_frame_to_pcm16le(parsed.audio),
                                parsed.audio.sample_rate_hz,
                                1,
                            )
                            stdin_segments.extend(
                                _transcribe_windows(
                                    audio_windows,
                                    engine_instance=engine_instance,
                                    config=config,
                                )
                            )
                    committed = gate_state.ingest(stdin_segments)
                    for item in committed:
                        text = redact_text(item.text) if config.redact else item.text
                        writer.append_line(text)

            remaining_windows = audio_chunker.flush(force=True)
            if remaining_windows and engine_available:
                tail_segments = _transcribe_windows(
                    remaining_windows,
                    engine_instance=engine_instance,
                    config=config,
                )
                committed_tail = gate_state.ingest(tail_segments)
                for item in committed_tail:
                    text = redact_text(item.text) if config.redact else item.text
                    writer.append_line(text)

            for pending in gate_state.drain_pending():
                text = redact_text(pending.text) if config.redact else pending.text
                writer.append_line(text)
    except DeviceDisconnectedError as exc:
        typer.echo(f"\nDevice disconnected: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except CaptureError as exc:
        typer.echo(f"\nAudio capture error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        if mic_capture is not None:
            mic_capture.close()
        if system_capture is not None:
            system_capture.close()
    if stopped_by_user:
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
