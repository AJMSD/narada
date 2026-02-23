from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from collections import deque
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
    CapturedFrame,
    CaptureError,
    CaptureHandle,
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
from narada.live_notes import AsrResult, AsrTask, IntervalPlanner, SessionSpool
from narada.logging_setup import setup_logging
from narada.performance import RuntimePerformance
from narada.pipeline import (
    AudioChunkWindow,
    ConfidenceGate,
    OverlapChunker,
    StabilizedConfidenceGate,
)
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
_BACKLOG_WARNING_INTERVAL_S = 30.0
_ASR_TASK_QUEUE_MAX = 2


class _LiveStatusRenderer:
    def __init__(self) -> None:
        stream = sys.stdout
        self._stream = stream if hasattr(stream, "write") else None
        self._line_count = 0
        self._supports_ansi = bool(
            self._stream is not None and hasattr(self._stream, "isatty") and self._stream.isatty()
        )

    def render(self, lines: list[str]) -> None:
        if not lines:
            return
        normalized = [line.rstrip() for line in lines]
        if not self._supports_ansi or self._stream is None:
            for line in normalized:
                _safe_echo(line)
            return
        try:
            if self._line_count > 0:
                self._stream.write(f"\x1b[{self._line_count}F")
            max_lines = max(self._line_count, len(normalized))
            for index in range(max_lines):
                self._stream.write("\x1b[2K")
                if index < len(normalized):
                    self._stream.write(normalized[index])
                self._stream.write("\n")
            self._stream.flush()
            self._line_count = len(normalized)
        except Exception:
            self._supports_ansi = False
            self._line_count = 0
            for line in normalized:
                _safe_echo(line)


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


def _estimate_shutdown_eta_seconds(
    *,
    asr_backlog_s: float,
    performance: RuntimePerformance,
) -> float:
    backlog_s = max(0.0, asr_backlog_s)
    realtime_factor = performance.realtime_factor
    if realtime_factor is None or realtime_factor <= 0.0:
        realtime_factor = 1.0
    return backlog_s * realtime_factor


def _build_live_status_lines(
    *,
    config: RuntimeConfig,
    started_at: float,
    performance: RuntimePerformance,
    asr_backlog_warning_s: float | None = None,
    shutdown_eta_s: float | None = None,
    shutdown_reason: str | None = None,
) -> list[str]:
    lines = [
        f"REC {_elapsed(started_at)} | mode={config.mode} | model={config.model}",
        performance.status_fragment(),
    ]
    if asr_backlog_warning_s is not None:
        lines.append(
            f"Warning: ASR backlog is {asr_backlog_warning_s:.1f}s. "
            "Notes may lag behind live audio."
        )
    if shutdown_eta_s is not None:
        reason_text = f" ({shutdown_reason})" if shutdown_reason else ""
        lines.append(
            "Application stopping. ASR completing first. "
            f"Will stop in about ~{shutdown_eta_s:.1f}s{reason_text}"
        )
    return lines


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
            asr_preset=config.asr_preset,
        )
        started = time.perf_counter()
        result = engine_instance.transcribe(request)
        processing_seconds += time.perf_counter() - started
        segments.extend(result)
    return segments, audio_seconds, processing_seconds


def _write_committed_lines(
    *,
    committed: list[object],
    writer: TranscriptWriter,
    config: RuntimeConfig,
    performance: RuntimePerformance,
    started_at: float,
) -> None:
    for item in committed:
        text = cast(str, getattr(item, "text", "")).strip()
        if not text:
            continue
        if config.redact:
            text = redact_text(text)
        writer.append_line(text)
        performance.record_commit_latency(elapsed_seconds=time.perf_counter() - started_at)


def _transcribe_audio_windows(
    *,
    audio_windows: list[AudioChunkWindow],
    engine_available: bool,
    engine_instance: AsrEngine,
    config: RuntimeConfig,
    performance: RuntimePerformance,
    gate_state: ConfidenceGate,
    writer: TranscriptWriter,
    started_at: float,
) -> None:
    if not audio_windows or not engine_available:
        return
    segments, audio_s, processing_s = _transcribe_windows(
        audio_windows,
        engine_instance=engine_instance,
        config=config,
    )
    performance.record_transcription(
        audio_seconds=audio_s,
        processing_seconds=processing_s,
    )
    committed = gate_state.ingest(segments)
    _write_committed_lines(
        committed=cast(list[object], committed),
        writer=writer,
        config=config,
        performance=performance,
        started_at=started_at,
    )


def _capture_worker_loop(
    *,
    source_name: str,
    capture_handle: CaptureHandle,
    frame_queue: queue.Queue[CapturedFrame],
    stop_event: threading.Event,
    error_queue: queue.Queue[tuple[str, Exception]],
) -> None:
    while not stop_event.is_set():
        try:
            frame = capture_handle.read_frame()
        except Exception as exc:  # pragma: no cover - OS/backend-specific at runtime
            error_queue.put((source_name, exc))
            stop_event.set()
            return
        if frame is not None:
            frame_queue.put(frame)


def _estimate_capture_backlog_seconds(
    *,
    queued_frames: int,
    blocksize: int,
    sample_rate_hz: int,
) -> float:
    if queued_frames <= 0:
        return 0.0
    if blocksize <= 0 or sample_rate_hz <= 0:
        return 0.0
    return queued_frames * (blocksize / sample_rate_hz)


def _maybe_warn_capture_backlog(
    *,
    source_name: str,
    queued_frames: int,
    blocksize: int,
    sample_rate_hz: int,
    warn_threshold_s: float,
    now_monotonic: float,
    last_warned_at: float | None,
) -> float | None:
    backlog_s = _estimate_capture_backlog_seconds(
        queued_frames=queued_frames,
        blocksize=blocksize,
        sample_rate_hz=sample_rate_hz,
    )
    if backlog_s < warn_threshold_s:
        return last_warned_at
    if last_warned_at is not None and (
        now_monotonic - last_warned_at < _BACKLOG_WARNING_INTERVAL_S
    ):
        return last_warned_at
    message = (
        f"Warning: {source_name} capture backlog is {backlog_s:.1f}s "
        f"({queued_frames} queued frames). ASR may be falling behind."
    )
    logger.warning(message)
    _safe_echo(message, err=True)
    return now_monotonic


def _estimate_asr_backlog_seconds(
    *,
    planner_backlog_s: float,
    queued_tasks: int,
    interval_s: float,
) -> float:
    if planner_backlog_s < 0:
        planner_backlog_s = 0.0
    if queued_tasks < 0:
        queued_tasks = 0
    if interval_s < 0:
        interval_s = 0.0
    return planner_backlog_s + (queued_tasks * interval_s)


def _maybe_warn_asr_backlog(
    *,
    backlog_s: float,
    warn_threshold_s: float,
    now_monotonic: float,
    last_warned_at: float | None,
) -> float | None:
    if backlog_s < warn_threshold_s:
        return last_warned_at
    if last_warned_at is not None and (
        now_monotonic - last_warned_at < _BACKLOG_WARNING_INTERVAL_S
    ):
        return last_warned_at
    message = f"Warning: ASR backlog is {backlog_s:.1f}s. Notes may lag behind live audio."
    logger.warning(message)
    return now_monotonic


def _asr_worker_loop(
    *,
    task_queue: queue.Queue[AsrTask | None],
    result_queue: queue.Queue[AsrResult],
    spool: SessionSpool,
    engine_instance: AsrEngine,
    config: RuntimeConfig,
) -> None:
    while True:
        try:
            task = task_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if task is None:
            return
        started = time.perf_counter()
        try:
            pcm_bytes = spool.read_range(start_byte=task.start_byte, end_byte=task.end_byte)
            request = TranscriptionRequest(
                pcm_bytes=pcm_bytes,
                sample_rate_hz=task.sample_rate_hz,
                languages=config.languages,
                model=config.model,
                compute=config.compute,
                asr_preset=config.asr_preset,
            )
            segments = tuple(engine_instance.transcribe(request))
            result_queue.put(
                AsrResult(
                    task=task,
                    segments=segments,
                    audio_seconds=task.audio_seconds,
                    processing_seconds=time.perf_counter() - started,
                    error=None,
                )
            )
        except Exception as exc:  # pragma: no cover - backend/runtime-specific errors
            result_queue.put(
                AsrResult(
                    task=task,
                    segments=tuple(),
                    audio_seconds=task.audio_seconds,
                    processing_seconds=time.perf_counter() - started,
                    error=str(exc),
                )
            )


def _enqueue_interval_task(
    *,
    task_queue: queue.Queue[AsrTask | None],
    task: AsrTask,
) -> bool:
    try:
        task_queue.put_nowait(task)
    except queue.Full:
        return False
    return True


def _drain_asr_results(
    *,
    result_queue: queue.Queue[AsrResult],
    gate_state: StabilizedConfidenceGate,
    writer: TranscriptWriter,
    config: RuntimeConfig,
    performance: RuntimePerformance,
    started_at: float,
) -> int:
    drained = 0
    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break
        drained += 1
        if result.error is not None:
            message = (
                f"Warning: ASR task '{result.task.label}' failed "
                f"for bytes [{result.task.start_byte}, {result.task.end_byte}): {result.error}"
            )
            logger.warning(message)
            _safe_echo(message, err=True)
            continue
        performance.record_transcription(
            audio_seconds=result.audio_seconds,
            processing_seconds=result.processing_seconds,
        )
        committed = gate_state.ingest(result.segments, is_final_window=result.task.is_final)
        _write_committed_lines(
            committed=cast(list[object], committed),
            writer=writer,
            config=config,
            performance=performance,
            started_at=started_at,
        )
    return drained


def _run_tty_notes_first(
    *,
    config: RuntimeConfig,
    mic_device: AudioDevice | None,
    system_device: AudioDevice | None,
    all_devices: list[AudioDevice],
    engine_instance: AsrEngine,
    writer: TranscriptWriter,
    performance: RuntimePerformance,
    started_at: float,
    mixed_resync_state: DriftResyncState,
) -> bool:
    stopped_by_user = False
    shutdown_reason = "stop request"
    mic_capture: CaptureHandle | None = None
    system_capture: CaptureHandle | None = None
    mic_queue: queue.Queue[CapturedFrame] | None = None
    system_queue: queue.Queue[CapturedFrame] | None = None
    capture_threads: list[threading.Thread] = []
    capture_errors: queue.Queue[tuple[str, Exception]] = queue.Queue()
    capture_stop = threading.Event()
    asr_task_queue: queue.Queue[AsrTask | None] = queue.Queue(maxsize=_ASR_TASK_QUEUE_MAX)
    asr_result_queue: queue.Queue[AsrResult] = queue.Queue()
    asr_thread: threading.Thread | None = None
    spool: SessionSpool | None = None
    spool_keep_files = config.keep_spool
    status_renderer = _LiveStatusRenderer()
    live_gate_state = StabilizedConfidenceGate(
        threshold=config.confidence_threshold,
        holdback_windows=config.notes_commit_holdback_windows,
    )
    planner = IntervalPlanner(
        interval_seconds=config.notes_interval_seconds,
        overlap_seconds=config.notes_overlap_seconds,
    )

    def _drain_capture_queue_to_pending(
        source_queue: queue.Queue[CapturedFrame] | None,
        target_pending: deque[CapturedFrame],
    ) -> None:
        if source_queue is None:
            return
        while True:
            try:
                target_pending.append(source_queue.get_nowait())
            except queue.Empty:
                break

    def _ingest_pcm_for_notes(
        pcm_bytes: bytes,
        sample_rate_hz: int,
        channels: int,
        *,
        now_monotonic: float,
    ) -> None:
        if spool is None:
            raise CaptureError("Live spool is unavailable.")
        try:
            record = spool.append_frame(
                pcm_bytes=pcm_bytes,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise CaptureError(f"Failed to write live spool data: {exc}") from exc
        planner.ingest_record(record, now_monotonic=now_monotonic)

    def _render_live_status(*, asr_backlog_s: float, include_shutdown_notice: bool = False) -> None:
        warning_backlog_s = (
            asr_backlog_s if asr_backlog_s >= config.asr_backlog_warn_seconds else None
        )
        shutdown_eta_s = (
            _estimate_shutdown_eta_seconds(asr_backlog_s=asr_backlog_s, performance=performance)
            if include_shutdown_notice
            else None
        )
        status_renderer.render(
            _build_live_status_lines(
                config=config,
                started_at=started_at,
                performance=performance,
                asr_backlog_warning_s=warning_backlog_s,
                shutdown_eta_s=shutdown_eta_s,
                shutdown_reason=shutdown_reason if include_shutdown_notice else None,
            )
        )

    try:
        if config.mode in {"mic", "mixed"} and mic_device is not None:
            mic_capture = open_mic_capture(device=mic_device)
            mic_queue = queue.Queue()
        if config.mode in {"system", "mixed"} and system_device is not None:
            system_capture = open_system_capture(
                device=system_device,
                all_devices=all_devices,
            )
            system_queue = queue.Queue()

        if mic_capture is not None and mic_queue is not None:
            mic_thread = threading.Thread(
                target=_capture_worker_loop,
                kwargs={
                    "source_name": "mic",
                    "capture_handle": mic_capture,
                    "frame_queue": mic_queue,
                    "stop_event": capture_stop,
                    "error_queue": capture_errors,
                },
                daemon=True,
                name="narada-mic-capture",
            )
            mic_thread.start()
            capture_threads.append(mic_thread)
        if system_capture is not None and system_queue is not None:
            system_thread = threading.Thread(
                target=_capture_worker_loop,
                kwargs={
                    "source_name": "system",
                    "capture_handle": system_capture,
                    "frame_queue": system_queue,
                    "stop_event": capture_stop,
                    "error_queue": capture_errors,
                },
                daemon=True,
                name="narada-system-capture",
            )
            system_thread.start()
            capture_threads.append(system_thread)

        spool = SessionSpool(
            base_dir=config.out.parent,
            prefix=f"narada-{config.out.stem}-spool",
            flush_interval_seconds=config.spool_flush_interval_seconds,
            flush_bytes=config.spool_flush_bytes,
        )
        asr_thread = threading.Thread(
            target=_asr_worker_loop,
            kwargs={
                "task_queue": asr_task_queue,
                "result_queue": asr_result_queue,
                "spool": spool,
                "engine_instance": engine_instance,
                "config": config,
            },
            daemon=True,
            name="narada-asr-worker",
        )
        asr_thread.start()

        mic_pending: deque[CapturedFrame] = deque()
        system_pending: deque[CapturedFrame] = deque()
        next_status_at = time.monotonic()
        last_capture_backlog_warning_at: dict[str, float | None] = {
            "mic": None,
            "system": None,
        }
        last_asr_backlog_warning_at: float | None = None

        while True:
            cycle_started_at = time.perf_counter()
            processed_any = False
            try:
                source_name, worker_exc = capture_errors.get_nowait()
            except queue.Empty:
                source_name = ""
                worker_exc = None
            if worker_exc is not None:
                if isinstance(worker_exc, (DeviceDisconnectedError, CaptureError)):
                    raise worker_exc
                raise CaptureError(f"{source_name} capture failed: {worker_exc}") from worker_exc

            _drain_capture_queue_to_pending(mic_queue, mic_pending)
            _drain_capture_queue_to_pending(system_queue, system_pending)
            if mic_pending or system_pending:
                processed_any = True

            now_monotonic = time.monotonic()
            if config.mode == "mic":
                while mic_pending:
                    mic_frame = mic_pending.popleft()
                    _ingest_pcm_for_notes(
                        mic_frame.pcm_bytes,
                        mic_frame.sample_rate_hz,
                        mic_frame.channels,
                        now_monotonic=now_monotonic,
                    )
            elif config.mode == "system":
                while system_pending:
                    system_frame = system_pending.popleft()
                    _ingest_pcm_for_notes(
                        system_frame.pcm_bytes,
                        system_frame.sample_rate_hz,
                        system_frame.channels,
                        now_monotonic=now_monotonic,
                    )
            elif config.mode == "mixed" and mic_capture is not None and system_capture is not None:
                while mic_pending and system_pending:
                    mic_frame = mic_pending.popleft()
                    system_frame = system_pending.popleft()
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
                    _ingest_pcm_for_notes(
                        mono_frame_to_pcm16le(
                            MonoAudioFrame(samples=tuple(mixed_samples), sample_rate_hz=mixed_rate)
                        ),
                        mixed_rate,
                        1,
                        now_monotonic=now_monotonic,
                    )

            while not asr_task_queue.full():
                task = planner.pop_next_ready_task(now_monotonic=now_monotonic)
                if task is None:
                    break
                if not _enqueue_interval_task(task_queue=asr_task_queue, task=task):
                    break
                processed_any = True

            drained = _drain_asr_results(
                result_queue=asr_result_queue,
                gate_state=live_gate_state,
                writer=writer,
                config=config,
                performance=performance,
                started_at=cycle_started_at,
            )
            if drained > 0:
                processed_any = True

            capture_backlog_values: list[float] = []
            if mic_capture is not None and mic_queue is not None:
                mic_queued_frames = mic_queue.qsize() + len(mic_pending)
                capture_backlog_values.append(
                    _estimate_capture_backlog_seconds(
                        queued_frames=mic_queued_frames,
                        blocksize=mic_capture.blocksize,
                        sample_rate_hz=mic_capture.sample_rate_hz,
                    )
                )
                last_capture_backlog_warning_at["mic"] = _maybe_warn_capture_backlog(
                    source_name="mic",
                    queued_frames=mic_queued_frames,
                    blocksize=mic_capture.blocksize,
                    sample_rate_hz=mic_capture.sample_rate_hz,
                    warn_threshold_s=config.capture_queue_warn_seconds,
                    now_monotonic=now_monotonic,
                    last_warned_at=last_capture_backlog_warning_at["mic"],
                )
            if system_capture is not None and system_queue is not None:
                system_queued_frames = system_queue.qsize() + len(system_pending)
                capture_backlog_values.append(
                    _estimate_capture_backlog_seconds(
                        queued_frames=system_queued_frames,
                        blocksize=system_capture.blocksize,
                        sample_rate_hz=system_capture.sample_rate_hz,
                    )
                )
                last_capture_backlog_warning_at["system"] = _maybe_warn_capture_backlog(
                    source_name="system",
                    queued_frames=system_queued_frames,
                    blocksize=system_capture.blocksize,
                    sample_rate_hz=system_capture.sample_rate_hz,
                    warn_threshold_s=config.capture_queue_warn_seconds,
                    now_monotonic=now_monotonic,
                    last_warned_at=last_capture_backlog_warning_at["system"],
                )
            capture_backlog_s = max(capture_backlog_values) if capture_backlog_values else 0.0
            asr_backlog_s = _estimate_asr_backlog_seconds(
                planner_backlog_s=planner.pending_backlog_seconds(),
                queued_tasks=asr_task_queue.qsize(),
                interval_s=config.notes_interval_seconds,
            )
            last_asr_backlog_warning_at = _maybe_warn_asr_backlog(
                backlog_s=asr_backlog_s,
                warn_threshold_s=config.asr_backlog_warn_seconds,
                now_monotonic=now_monotonic,
                last_warned_at=last_asr_backlog_warning_at,
            )
            performance.set_backlogs(
                capture_backlog_s=capture_backlog_s,
                asr_backlog_s=asr_backlog_s,
            )
            dropped_frames = 0
            if mic_capture is not None:
                dropped_frames += mic_capture.stats_snapshot().dropped_frames
            if system_capture is not None:
                dropped_frames += system_capture.stats_snapshot().dropped_frames
            performance.set_dropped_frames(dropped_frames=dropped_frames)

            if now_monotonic >= next_status_at:
                _render_live_status(asr_backlog_s=asr_backlog_s)
                next_status_at = now_monotonic + 1.0

            if not processed_any:
                time.sleep(0.01)
    except KeyboardInterrupt:
        stopped_by_user = True
        shutdown_reason = "Ctrl+C"
    except Exception:
        shutdown_reason = "runtime error"
        spool_keep_files = True
        raise
    finally:
        finalization_started_at = time.perf_counter()
        capture_stop.set()
        for thread in capture_threads:
            thread.join(timeout=1.0)

        mic_pending_final: deque[CapturedFrame] = deque()
        system_pending_final: deque[CapturedFrame] = deque()
        _drain_capture_queue_to_pending(mic_queue, mic_pending_final)
        _drain_capture_queue_to_pending(system_queue, system_pending_final)

        now_monotonic = time.monotonic()
        if config.mode == "mic":
            while mic_pending_final:
                mic_frame = mic_pending_final.popleft()
                _ingest_pcm_for_notes(
                    mic_frame.pcm_bytes,
                    mic_frame.sample_rate_hz,
                    mic_frame.channels,
                    now_monotonic=now_monotonic,
                )
        elif config.mode == "system":
            while system_pending_final:
                system_frame = system_pending_final.popleft()
                _ingest_pcm_for_notes(
                    system_frame.pcm_bytes,
                    system_frame.sample_rate_hz,
                    system_frame.channels,
                    now_monotonic=now_monotonic,
                )
        elif config.mode == "mixed" and mic_capture is not None and system_capture is not None:
            while mic_pending_final and system_pending_final:
                mic_frame = mic_pending_final.popleft()
                system_frame = system_pending_final.popleft()
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
                _ingest_pcm_for_notes(
                    mono_frame_to_pcm16le(
                        MonoAudioFrame(samples=tuple(mixed_samples), sample_rate_hz=mixed_rate)
                    ),
                    mixed_rate,
                    1,
                    now_monotonic=now_monotonic,
                )

        while not asr_task_queue.full():
            task = planner.pop_next_ready_task(now_monotonic=now_monotonic)
            if task is None:
                break
            _enqueue_interval_task(task_queue=asr_task_queue, task=task)

        final_tasks = planner.build_final_tasks(now_monotonic=now_monotonic)
        for task in final_tasks:
            while not _enqueue_interval_task(task_queue=asr_task_queue, task=task):
                _drain_asr_results(
                    result_queue=asr_result_queue,
                    gate_state=live_gate_state,
                    writer=writer,
                    config=config,
                    performance=performance,
                    started_at=time.perf_counter(),
                )
                if asr_thread is None or not asr_thread.is_alive():
                    break
                asr_backlog_s = _estimate_asr_backlog_seconds(
                    planner_backlog_s=planner.pending_backlog_seconds(),
                    queued_tasks=asr_task_queue.qsize(),
                    interval_s=config.notes_interval_seconds,
                )
                performance.set_backlogs(capture_backlog_s=0.0, asr_backlog_s=asr_backlog_s)
                _render_live_status(asr_backlog_s=asr_backlog_s, include_shutdown_notice=True)
                time.sleep(0.01)

        asr_backlog_s = _estimate_asr_backlog_seconds(
            planner_backlog_s=planner.pending_backlog_seconds(),
            queued_tasks=asr_task_queue.qsize(),
            interval_s=config.notes_interval_seconds,
        )
        performance.set_backlogs(capture_backlog_s=0.0, asr_backlog_s=asr_backlog_s)
        _render_live_status(asr_backlog_s=asr_backlog_s, include_shutdown_notice=True)

        while True:
            try:
                asr_task_queue.put_nowait(None)
                break
            except queue.Full:
                _drain_asr_results(
                    result_queue=asr_result_queue,
                    gate_state=live_gate_state,
                    writer=writer,
                    config=config,
                    performance=performance,
                    started_at=time.perf_counter(),
                )
                if asr_thread is None or not asr_thread.is_alive():
                    break
                asr_backlog_s = _estimate_asr_backlog_seconds(
                    planner_backlog_s=planner.pending_backlog_seconds(),
                    queued_tasks=asr_task_queue.qsize(),
                    interval_s=config.notes_interval_seconds,
                )
                performance.set_backlogs(capture_backlog_s=0.0, asr_backlog_s=asr_backlog_s)
                _render_live_status(asr_backlog_s=asr_backlog_s, include_shutdown_notice=True)
                time.sleep(0.01)

        if asr_thread is not None:
            while asr_thread.is_alive():
                _drain_asr_results(
                    result_queue=asr_result_queue,
                    gate_state=live_gate_state,
                    writer=writer,
                    config=config,
                    performance=performance,
                    started_at=time.perf_counter(),
                )
                asr_backlog_s = _estimate_asr_backlog_seconds(
                    planner_backlog_s=planner.pending_backlog_seconds(),
                    queued_tasks=asr_task_queue.qsize(),
                    interval_s=config.notes_interval_seconds,
                )
                performance.set_backlogs(capture_backlog_s=0.0, asr_backlog_s=asr_backlog_s)
                _render_live_status(asr_backlog_s=asr_backlog_s, include_shutdown_notice=True)
                asr_thread.join(timeout=0.1)

        _drain_asr_results(
            result_queue=asr_result_queue,
            gate_state=live_gate_state,
            writer=writer,
            config=config,
            performance=performance,
            started_at=time.perf_counter(),
        )
        performance.set_backlogs(capture_backlog_s=0.0, asr_backlog_s=0.0)
        _render_live_status(asr_backlog_s=0.0, include_shutdown_notice=True)
        pending_live = live_gate_state.drain_pending(force_low_conf=True)
        _write_committed_lines(
            committed=cast(list[object], pending_live),
            writer=writer,
            config=config,
            performance=performance,
            started_at=time.perf_counter(),
        )
        performance.record_end_to_notes(
            elapsed_seconds=time.perf_counter() - finalization_started_at
        )

        if mic_capture is not None:
            mic_capture.close()
        if system_capture is not None:
            system_capture.close()
        if spool is not None:
            spool.cleanup(keep_files=spool_keep_files)

    return stopped_by_user


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
    wall_flush_seconds: float | None = typer.Option(
        None,
        "--wall-flush-seconds",
        help="Force transcript flush/commit at this wall-clock interval in live mode (0 disables).",
    ),
    capture_queue_warn_seconds: float | None = typer.Option(
        None,
        "--capture-queue-warn-seconds",
        help="Warn when estimated live capture backlog exceeds this many seconds.",
    ),
    notes_interval_seconds: float | None = typer.Option(
        None,
        "--notes-interval-seconds",
        help="Interval size (seconds) for notes-first background ASR windows.",
    ),
    notes_overlap_seconds: float | None = typer.Option(
        None,
        "--notes-overlap-seconds",
        help="Overlap size (seconds) between consecutive notes-first ASR windows.",
    ),
    notes_commit_holdback_windows: int | None = typer.Option(
        None,
        "--notes-commit-holdback-windows",
        help="How many ASR windows to hold back before committing notes.",
    ),
    asr_backlog_warn_seconds: float | None = typer.Option(
        None,
        "--asr-backlog-warn-seconds",
        help="Warn when estimated ASR backlog exceeds this many seconds.",
    ),
    keep_spool: bool | None = typer.Option(
        None,
        "--keep-spool/--no-keep-spool",
        help="Keep raw spool files for post-run debugging/recovery.",
    ),
    spool_flush_interval_seconds: float | None = typer.Option(
        None,
        "--spool-flush-interval-seconds",
        help=(
            "Flush live spool files at this interval (seconds). Use 0 to disable interval trigger."
        ),
    ),
    spool_flush_bytes: int | None = typer.Option(
        None,
        "--spool-flush-bytes",
        help="Flush live spool files when pending bytes reach this threshold. Use 0 to disable.",
    ),
    writer_fsync_mode: str | None = typer.Option(
        None,
        "--writer-fsync-mode",
        help="line|periodic transcript fsync policy.",
    ),
    writer_fsync_lines: int | None = typer.Option(
        None,
        "--writer-fsync-lines",
        help="In periodic fsync mode, fsync after this many committed lines (0 disables).",
    ),
    writer_fsync_seconds: float | None = typer.Option(
        None,
        "--writer-fsync-seconds",
        help="In periodic fsync mode, fsync after this many seconds (0 disables).",
    ),
    asr_preset: str | None = typer.Option(
        None,
        "--asr-preset",
        help="fast|balanced|accurate decode preset.",
    ),
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
    serve_token: str | None = typer.Option(
        None,
        "--serve-token",
        help="Optional auth token required by transcript HTTP endpoints.",
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
    if not serve and (bind is not None or port is not None or qr or serve_token is not None):
        raise typer.BadParameter(
            "`--bind`, `--port`, `--qr`, and `--serve-token` require `--serve` with `narada start`."
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
        wall_flush_seconds=wall_flush_seconds,
        capture_queue_warn_seconds=capture_queue_warn_seconds,
        notes_interval_seconds=notes_interval_seconds,
        notes_overlap_seconds=notes_overlap_seconds,
        notes_commit_holdback_windows=notes_commit_holdback_windows,
        asr_backlog_warn_seconds=asr_backlog_warn_seconds,
        keep_spool=keep_spool,
        spool_flush_interval_seconds=spool_flush_interval_seconds,
        spool_flush_bytes=spool_flush_bytes,
        writer_fsync_mode=writer_fsync_mode,
        writer_fsync_lines=writer_fsync_lines,
        writer_fsync_seconds=writer_fsync_seconds,
        asr_preset=asr_preset,
        serve_token=serve_token,
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
                running_server = start_transcript_server(
                    config.out,
                    config.bind,
                    config.port,
                    serve_token=config.serve_token,
                )
            except OSError as exc:
                raise typer.BadParameter(
                    f"Unable to start server on {config.bind}:{config.port}: {exc}"
                ) from exc
            typer.echo(f"Serving transcript from {config.out}")
            typer.echo(f"URL: {running_server.access_url}")
            if config.bind == "0.0.0.0":
                typer.echo("Warning: server bound to all interfaces on local network.")
                if config.serve_token is None:
                    typer.echo(
                        "Warning: LAN server is unauthenticated. "
                        "Set --serve-token to require access."
                    )
            if qr:
                typer.echo(render_ascii_qr(running_server.access_url))

        with TranscriptWriter(
            config.out,
            fsync_mode=config.writer_fsync_mode,
            fsync_lines=config.writer_fsync_lines,
            fsync_seconds=config.writer_fsync_seconds,
        ) as writer:
            if sys.stdin.isatty():
                stopped_by_user = _run_tty_notes_first(
                    config=config,
                    mic_device=mic_device,
                    system_device=system_device,
                    all_devices=all_devices,
                    engine_instance=engine_instance,
                    writer=writer,
                    performance=performance,
                    started_at=started_at,
                    mixed_resync_state=mixed_resync_state,
                )
                if stopped_by_user:
                    typer.echo("\nStopped.")
                return
            audio_chunker = OverlapChunker(chunk_duration_s=2.0, overlap_duration_s=0.5)
            if sys.stdin.isatty():
                if config.mode in {"mic", "mixed"} and mic_device is not None:
                    mic_capture = open_mic_capture(device=mic_device)
                if config.mode in {"system", "mixed"} and system_device is not None:
                    system_capture = open_system_capture(
                        device=system_device,
                        all_devices=all_devices,
                    )

                mic_queue: queue.Queue[CapturedFrame] | None = (
                    queue.Queue() if mic_capture is not None else None
                )
                system_queue: queue.Queue[CapturedFrame] | None = (
                    queue.Queue() if system_capture is not None else None
                )
                capture_stop = threading.Event()
                capture_errors: queue.Queue[tuple[str, Exception]] = queue.Queue()
                capture_threads: list[threading.Thread] = []
                if mic_capture is not None and mic_queue is not None:
                    mic_thread = threading.Thread(
                        target=_capture_worker_loop,
                        kwargs={
                            "source_name": "mic",
                            "capture_handle": mic_capture,
                            "frame_queue": mic_queue,
                            "stop_event": capture_stop,
                            "error_queue": capture_errors,
                        },
                        daemon=True,
                        name="narada-mic-capture",
                    )
                    mic_thread.start()
                    capture_threads.append(mic_thread)
                if system_capture is not None and system_queue is not None:
                    system_thread = threading.Thread(
                        target=_capture_worker_loop,
                        kwargs={
                            "source_name": "system",
                            "capture_handle": system_capture,
                            "frame_queue": system_queue,
                            "stop_event": capture_stop,
                            "error_queue": capture_errors,
                        },
                        daemon=True,
                        name="narada-system-capture",
                    )
                    system_thread.start()
                    capture_threads.append(system_thread)

                mic_pending: deque[CapturedFrame] = deque()
                system_pending: deque[CapturedFrame] = deque()
                next_status_at = time.monotonic()
                next_flush_deadline: float | None = None
                if config.wall_flush_seconds > 0.0:
                    next_flush_deadline = time.monotonic() + config.wall_flush_seconds
                last_backlog_warning_at: dict[str, float | None] = {
                    "mic": None,
                    "system": None,
                }

                try:
                    while True:
                        cycle_started_at = time.perf_counter()
                        processed_any = False
                        audio_windows: list[AudioChunkWindow] = []
                        try:
                            source_name, worker_exc = capture_errors.get_nowait()
                        except queue.Empty:
                            source_name = ""
                            worker_exc = None
                        if worker_exc is not None:
                            if isinstance(worker_exc, (DeviceDisconnectedError, CaptureError)):
                                raise worker_exc
                            raise CaptureError(
                                f"{source_name} capture failed: {worker_exc}"
                            ) from worker_exc

                        if mic_queue is not None:
                            while True:
                                try:
                                    mic_pending.append(mic_queue.get_nowait())
                                except queue.Empty:
                                    break
                                processed_any = True
                        if system_queue is not None:
                            while True:
                                try:
                                    system_pending.append(system_queue.get_nowait())
                                except queue.Empty:
                                    break
                                processed_any = True

                        if config.mode == "mic":
                            while mic_pending:
                                mic_frame = mic_pending.popleft()
                                audio_windows.extend(
                                    audio_chunker.ingest(
                                        mic_frame.pcm_bytes,
                                        mic_frame.sample_rate_hz,
                                        mic_frame.channels,
                                    )
                                )
                        elif config.mode == "system":
                            while system_pending:
                                system_frame = system_pending.popleft()
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
                            while mic_pending and system_pending:
                                mic_frame = mic_pending.popleft()
                                system_frame = system_pending.popleft()
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

                        _transcribe_audio_windows(
                            audio_windows=audio_windows,
                            engine_available=engine_available,
                            engine_instance=engine_instance,
                            config=config,
                            performance=performance,
                            gate_state=gate_state,
                            writer=writer,
                            started_at=cycle_started_at,
                        )

                        now_monotonic = time.monotonic()
                        if next_flush_deadline is not None and now_monotonic >= next_flush_deadline:
                            forced_windows: list[AudioChunkWindow] = []
                            while (
                                next_flush_deadline is not None
                                and now_monotonic >= next_flush_deadline
                            ):
                                forced_windows.extend(audio_chunker.flush(force=True))
                                next_flush_deadline += config.wall_flush_seconds
                            _transcribe_audio_windows(
                                audio_windows=forced_windows,
                                engine_available=engine_available,
                                engine_instance=engine_instance,
                                config=config,
                                performance=performance,
                                gate_state=gate_state,
                                writer=writer,
                                started_at=cycle_started_at,
                            )
                            if forced_windows:
                                processed_any = True

                        if mic_capture is not None and mic_queue is not None:
                            mic_queued_frames = mic_queue.qsize() + len(mic_pending)
                            last_backlog_warning_at["mic"] = _maybe_warn_capture_backlog(
                                source_name="mic",
                                queued_frames=mic_queued_frames,
                                blocksize=mic_capture.blocksize,
                                sample_rate_hz=mic_capture.sample_rate_hz,
                                warn_threshold_s=config.capture_queue_warn_seconds,
                                now_monotonic=now_monotonic,
                                last_warned_at=last_backlog_warning_at["mic"],
                            )
                        if system_capture is not None and system_queue is not None:
                            system_queued_frames = system_queue.qsize() + len(system_pending)
                            last_backlog_warning_at["system"] = _maybe_warn_capture_backlog(
                                source_name="system",
                                queued_frames=system_queued_frames,
                                blocksize=system_capture.blocksize,
                                sample_rate_hz=system_capture.sample_rate_hz,
                                warn_threshold_s=config.capture_queue_warn_seconds,
                                now_monotonic=now_monotonic,
                                last_warned_at=last_backlog_warning_at["system"],
                            )

                        if now_monotonic >= next_status_at:
                            typer.echo(
                                f"\rREC {_elapsed(started_at)} | mode={config.mode} "
                                f"| model={config.model} | {performance.status_fragment()}",
                                nl=False,
                            )
                            next_status_at = now_monotonic + 1.0

                        if not processed_any:
                            time.sleep(0.01)
                except KeyboardInterrupt:
                    stopped_by_user = True
                finally:
                    capture_stop.set()
                    for thread in capture_threads:
                        thread.join(timeout=1.0)
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
                    _write_committed_lines(
                        committed=cast(list[object], committed),
                        writer=writer,
                        config=config,
                        performance=performance,
                        started_at=cycle_started_at,
                    )

            remaining_windows = audio_chunker.flush(force=True)
            _transcribe_audio_windows(
                audio_windows=remaining_windows,
                engine_available=engine_available,
                engine_instance=engine_instance,
                config=config,
                performance=performance,
                gate_state=gate_state,
                writer=writer,
                started_at=time.perf_counter(),
            )

            pending = gate_state.drain_pending()
            _write_committed_lines(
                committed=cast(list[object], pending),
                writer=writer,
                config=config,
                performance=performance,
                started_at=time.perf_counter(),
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
    serve_token: str | None = typer.Option(
        None,
        "--serve-token",
        help="Optional auth token required by transcript HTTP endpoints.",
    ),
) -> None:
    transcript_file = file
    if transcript_file is None:
        transcript_file = Path(os.environ.get("NARADA_OUT", "./transcripts/session.txt"))
    serve_transcript_file(
        transcript_path=transcript_file,
        bind=bind or os.environ.get("NARADA_BIND", "0.0.0.0"),
        port=port or int(os.environ.get("NARADA_PORT", "8787")),
        show_qr=qr,
        serve_token=serve_token or os.environ.get("NARADA_SERVE_TOKEN"),
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
