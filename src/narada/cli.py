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
from narada.asr.model_discovery import build_start_model_preflight, discover_models
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
    EndpointType,
    devices_to_json,
    enumerate_devices,
    filter_devices,
    format_devices_table,
    resolve_device,
)
from narada.doctor import format_doctor_report, has_failures, run_doctor
from narada.logging_setup import setup_logging
from narada.performance import RuntimePerformance
from narada.pipeline import AudioChunkWindow, ConfidenceGate, OverlapChunker
from narada.redaction import redact_text
from narada.server import (
    RunningTranscriptServer,
    render_ascii_qr,
    serve_transcript_file,
    start_transcript_server,
)
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
    selectable_devices = enumerate_devices()
    all_devices = enumerate_devices(include_all=True)
    resolved_mic: AudioDevice | None = None
    resolved_system: AudioDevice | None = None
    if mode in {"mic", "mixed"} and mic:
        resolved_mic = resolve_device(mic, selectable_devices, {"input"})
    if mode in {"system", "mixed"} and system:
        resolved_system = resolve_device(
            system,
            selectable_devices,
            {"output", "loopback", "monitor"},
        )
    return resolved_mic, resolved_system, all_devices


def _elapsed(started_at: float) -> str:
    total = int(time.time() - started_at)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _safe_echo(message: str, *, err: bool = False, nl: bool = True) -> None:
    try:
        typer.echo(message, err=err, nl=nl)
    except OSError:
        fallback_stream = sys.__stderr__ if err else sys.__stdout__
        if fallback_stream is None:
            return
        text = f"{message}\n" if nl else message
        try:
            fallback_stream.write(text)
            fallback_stream.flush()
        except Exception:
            return


def _transcribe_windows(
    windows: list[AudioChunkWindow],
    *,
    engine_instance: AsrEngine,
    config: RuntimeConfig,
) -> tuple[list[TranscriptSegment], float, float]:
    segments: list[TranscriptSegment] = []
    audio_seconds = 0.0
    processing_seconds = 0.0
    for window in windows:
        bytes_per_second = window.sample_rate_hz * window.channels * 2
        if bytes_per_second > 0:
            audio_seconds += len(window.pcm_bytes) / bytes_per_second
        request = TranscriptionRequest(
            pcm_bytes=window.pcm_bytes,
            sample_rate_hz=window.sample_rate_hz,
            languages=config.languages,
            model=config.model,
            compute=config.compute,
        )
        started = time.perf_counter()
        result = engine_instance.transcribe(request)
        processing_seconds += time.perf_counter() - started
        segments.extend(result)
    return segments, audio_seconds, processing_seconds


@app.command("devices")
def devices_command(
    device_type: str | None = typer.Option(None, "--type", help="input|output|loopback|monitor"),
    search: str | None = typer.Option(None, "--search", help="Case-insensitive name filter."),
    all_devices: bool = typer.Option(
        False,
        "--all",
        help="Show raw backend endpoints without automatic deduplication.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Return JSON output."),
) -> None:
    if device_type is not None and device_type not in DEVICE_TYPES:
        raise typer.BadParameter(
            f"Unsupported device type '{device_type}'. Expected one of: {', '.join(DEVICE_TYPES)}.",
            param_hint="--type",
        )
    normalized_type: EndpointType | None = None
    if device_type is not None:
        normalized_type = cast(EndpointType, device_type)

    items = enumerate_devices(include_all=all_devices)
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
    serve: bool = typer.Option(
        False,
        "--serve",
        help="Start LAN transcript server alongside recording.",
    ),
    bind: str | None = typer.Option(
        None,
        "--bind",
        help="Bind address used when --serve is enabled.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="HTTP port used when --serve is enabled.",
    ),
    qr: bool = typer.Option(
        False,
        "--qr",
        help="Print an ASCII QR code when --serve is enabled.",
    ),
    model_dir_faster_whisper: Path | None = typer.Option(
        None,
        "--model-dir-faster-whisper",
        help="Optional local model directory override for faster-whisper.",
    ),
    model_dir_whisper_cpp: Path | None = typer.Option(
        None,
        "--model-dir-whisper-cpp",
        help="Optional local model directory override for whisper.cpp.",
    ),
) -> None:
    if not serve and (bind is not None or port is not None or qr):
        raise typer.BadParameter(
            "`--bind`, `--port`, and `--qr` require `--serve` with `narada start`."
        )

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
        bind=bind,
        port=port,
        model_dir_faster_whisper=model_dir_faster_whisper,
        model_dir_whisper_cpp=model_dir_whisper_cpp,
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

    model_discovery = discover_models(
        config.model,
        faster_whisper_model_dir=config.model_dir_faster_whisper,
        whisper_cpp_model_dir=config.model_dir_whisper_cpp,
    )
    preflight = build_start_model_preflight(model_discovery, config.engine)
    for message in preflight.messages:
        typer.echo(message)

    selected_engine: str = config.engine
    if preflight.recommended_engine is not None and preflight.recommended_engine != config.engine:
        selected_engine = preflight.recommended_engine
        typer.echo(f"Switching to {selected_engine} based on available local models.")

    engine_instance = build_engine(
        selected_engine,
        faster_whisper_model_dir=config.model_dir_faster_whisper,
        whisper_cpp_model_dir=config.model_dir_whisper_cpp,
    )
    if not engine_instance.is_available() and sys.stdin.isatty():
        raise typer.BadParameter(
            f"Selected engine '{selected_engine}' is unavailable. "
            "Install dependencies or pipe text input for dry run."
        )

    gate_state = ConfidenceGate(config.confidence_threshold)
    engine_available = engine_instance.is_available()

    typer.echo(
        "REC "
        + datetime.now().isoformat(timespec="seconds")
        + f" | mode={config.mode} | engine={selected_engine} | model={config.model}"
    )
    typer.echo(f"Writing transcript to: {config.out}")
    if sys.stdin.isatty():
        typer.echo("No piped input detected. Waiting for Ctrl+C.")

    started_at = time.time()
    warned_missing_engine_for_audio = False
    running_server: RunningTranscriptServer | None = None
    mic_capture = None
    system_capture = None
    mixed_resync_state = DriftResyncState()
    performance = RuntimePerformance()
    stopped_by_user = False
    try:
        if serve:
            try:
                running_server = start_transcript_server(config.out, config.bind, config.port)
            except OSError as exc:
                raise typer.BadParameter(
                    f"Unable to start server on {config.bind}:{config.port}: {exc}"
                ) from exc
            typer.echo(f"Serving transcript from {config.out}")
            typer.echo(f"URL: {running_server.access_url}")
            if config.bind == "0.0.0.0":
                typer.echo("Warning: server bound to all interfaces on local network.")
            if qr:
                typer.echo(render_ascii_qr(running_server.access_url))

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
                        cycle_started_at = time.perf_counter()
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
                            window_segments, audio_s, processing_s = _transcribe_windows(
                                audio_windows,
                                engine_instance=engine_instance,
                                config=config,
                            )
                            performance.record_transcription(
                                audio_seconds=audio_s,
                                processing_seconds=processing_s,
                            )
                            segments.extend(window_segments)

                        committed = gate_state.ingest(segments)
                        for item in committed:
                            text = redact_text(item.text) if config.redact else item.text
                            writer.append_line(text)
                            performance.record_commit_latency(
                                elapsed_seconds=time.perf_counter() - cycle_started_at
                            )
                        typer.echo(
                            f"\rREC {_elapsed(started_at)} | mode={config.mode} "
                            f"| model={config.model} | {performance.status_fragment()}",
                            nl=False,
                        )
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    stopped_by_user = True
            else:
                for line in sys.stdin:
                    cycle_started_at = time.perf_counter()
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
                            window_segments, audio_s, processing_s = _transcribe_windows(
                                audio_windows,
                                engine_instance=engine_instance,
                                config=config,
                            )
                            performance.record_transcription(
                                audio_seconds=audio_s,
                                processing_seconds=processing_s,
                            )
                            stdin_segments.extend(window_segments)
                    committed = gate_state.ingest(stdin_segments)
                    for item in committed:
                        text = redact_text(item.text) if config.redact else item.text
                        writer.append_line(text)
                        performance.record_commit_latency(
                            elapsed_seconds=time.perf_counter() - cycle_started_at
                        )

            remaining_windows = audio_chunker.flush(force=True)
            if remaining_windows and engine_available:
                tail_segments, audio_s, processing_s = _transcribe_windows(
                    remaining_windows,
                    engine_instance=engine_instance,
                    config=config,
                )
                performance.record_transcription(
                    audio_seconds=audio_s,
                    processing_seconds=processing_s,
                )
                committed_tail = gate_state.ingest(tail_segments)
                for item in committed_tail:
                    tail_started_at = time.perf_counter()
                    text = redact_text(item.text) if config.redact else item.text
                    writer.append_line(text)
                    performance.record_commit_latency(
                        elapsed_seconds=time.perf_counter() - tail_started_at
                    )

            for pending in gate_state.drain_pending():
                pending_started_at = time.perf_counter()
                text = redact_text(pending.text) if config.redact else pending.text
                writer.append_line(text)
                performance.record_commit_latency(
                    elapsed_seconds=time.perf_counter() - pending_started_at
                )
    except DeviceDisconnectedError as exc:
        _safe_echo(f"\nDevice disconnected: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except CaptureError as exc:
        _safe_echo(f"\nAudio capture error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        if mic_capture is not None:
            mic_capture.close()
        if system_capture is not None:
            system_capture.close()
        if running_server is not None:
            running_server.stop()
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
    model: str | None = typer.Option(None, "--model", help="tiny|small|medium|large"),
    model_dir_faster_whisper: Path | None = typer.Option(
        None,
        "--model-dir-faster-whisper",
        help="Optional local model directory override for faster-whisper.",
    ),
    model_dir_whisper_cpp: Path | None = typer.Option(
        None,
        "--model-dir-whisper-cpp",
        help="Optional local model directory override for whisper.cpp.",
    ),
) -> None:
    model_name = (model or os.environ.get("NARADA_MODEL", "small")).strip().lower()
    checks = run_doctor(
        output_path=file,
        model_name=model_name,
        faster_whisper_model_dir=(
            model_dir_faster_whisper
            or (
                Path(os.environ["NARADA_MODEL_DIR_FASTER_WHISPER"])
                if "NARADA_MODEL_DIR_FASTER_WHISPER" in os.environ
                else None
            )
        ),
        whisper_cpp_model_dir=(
            model_dir_whisper_cpp
            or (
                Path(os.environ["NARADA_MODEL_DIR_WHISPER_CPP"])
                if "NARADA_MODEL_DIR_WHISPER_CPP" in os.environ
                else None
            )
        ),
    )
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
