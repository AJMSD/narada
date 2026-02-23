from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

Mode = Literal["mic", "system", "mixed"]
Model = Literal["tiny", "small", "medium", "large"]
Compute = Literal["cpu", "cuda", "metal", "auto"]
Engine = Literal["faster-whisper", "whisper-cpp"]
NoiseSuppress = Literal["off", "rnnoise", "webrtc"]
WriterFsyncMode = Literal["line", "periodic"]
AsrPreset = Literal["fast", "balanced", "accurate"]

MODE_VALUES: tuple[Mode, ...] = ("mic", "system", "mixed")
MODEL_VALUES: tuple[Model, ...] = ("tiny", "small", "medium", "large")
COMPUTE_VALUES: tuple[Compute, ...] = ("cpu", "cuda", "metal", "auto")
ENGINE_VALUES: tuple[Engine, ...] = ("faster-whisper", "whisper-cpp")
NOISE_VALUES: tuple[NoiseSuppress, ...] = ("off", "rnnoise", "webrtc")
WRITER_FSYNC_MODE_VALUES: tuple[WriterFsyncMode, ...] = ("line", "periodic")
ASR_PRESET_VALUES: tuple[AsrPreset, ...] = ("fast", "balanced", "accurate")

ENV_KEYS: dict[str, str] = {
    "mode": "NARADA_MODE",
    "mic": "NARADA_MIC",
    "system": "NARADA_SYSTEM",
    "out": "NARADA_OUT",
    "model": "NARADA_MODEL",
    "compute": "NARADA_COMPUTE",
    "engine": "NARADA_ENGINE",
    "language": "NARADA_LANGUAGE",
    "allow_multilingual": "NARADA_ALLOW_MULTILINGUAL",
    "redact": "NARADA_REDACT",
    "noise_suppress": "NARADA_NOISE_SUPPRESS",
    "agc": "NARADA_AGC",
    "gate": "NARADA_GATE",
    "gate_threshold_db": "NARADA_GATE_THRESHOLD_DB",
    "confidence_threshold": "NARADA_CONFIDENCE_THRESHOLD",
    "wall_flush_seconds": "NARADA_WALL_FLUSH_SECONDS",
    "capture_queue_warn_seconds": "NARADA_CAPTURE_QUEUE_WARN_SECONDS",
    "notes_interval_seconds": "NARADA_NOTES_INTERVAL_SECONDS",
    "notes_overlap_seconds": "NARADA_NOTES_OVERLAP_SECONDS",
    "notes_commit_holdback_windows": "NARADA_NOTES_COMMIT_HOLDBACK_WINDOWS",
    "asr_backlog_warn_seconds": "NARADA_ASR_BACKLOG_WARN_SECONDS",
    "keep_spool": "NARADA_KEEP_SPOOL",
    "spool_flush_interval_seconds": "NARADA_SPOOL_FLUSH_INTERVAL_SECONDS",
    "spool_flush_bytes": "NARADA_SPOOL_FLUSH_BYTES",
    "writer_fsync_mode": "NARADA_WRITER_FSYNC_MODE",
    "writer_fsync_lines": "NARADA_WRITER_FSYNC_LINES",
    "writer_fsync_seconds": "NARADA_WRITER_FSYNC_SECONDS",
    "asr_preset": "NARADA_ASR_PRESET",
    "serve_token": "NARADA_SERVE_TOKEN",
    "bind": "NARADA_BIND",
    "port": "NARADA_PORT",
    "model_dir_faster_whisper": "NARADA_MODEL_DIR_FASTER_WHISPER",
    "model_dir_whisper_cpp": "NARADA_MODEL_DIR_WHISPER_CPP",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "english": "en",
    "en": "en",
    "hindi": "hi",
    "hi": "hi",
    "spanish": "es",
    "es": "es",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
    "chinese": "zh",
    "zh": "zh",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    mode: Mode
    mic: str | None
    system: str | None
    out: Path
    model: Model
    compute: Compute
    engine: Engine
    languages: tuple[str, ...]
    allow_multilingual: bool
    redact: bool
    noise_suppress: NoiseSuppress
    agc: bool
    gate: bool
    gate_threshold_db: float
    confidence_threshold: float
    wall_flush_seconds: float
    capture_queue_warn_seconds: float
    notes_interval_seconds: float
    notes_overlap_seconds: float
    notes_commit_holdback_windows: int
    asr_backlog_warn_seconds: float
    keep_spool: bool
    spool_flush_interval_seconds: float
    spool_flush_bytes: int
    writer_fsync_mode: WriterFsyncMode
    writer_fsync_lines: int
    writer_fsync_seconds: float
    asr_preset: AsrPreset
    serve_token: str | None
    bind: str
    port: int
    model_dir_faster_whisper: Path | None
    model_dir_whisper_cpp: Path | None


@dataclass(frozen=True)
class ConfigOverrides:
    mode: str | None = None
    mic: str | None = None
    system: str | None = None
    out: Path | None = None
    model: str | None = None
    compute: str | None = None
    engine: str | None = None
    language: str | None = None
    allow_multilingual: bool | None = None
    redact: str | None = None
    noise_suppress: str | None = None
    agc: str | None = None
    gate: str | None = None
    gate_threshold_db: float | None = None
    confidence_threshold: float | None = None
    wall_flush_seconds: float | None = None
    capture_queue_warn_seconds: float | None = None
    notes_interval_seconds: float | None = None
    notes_overlap_seconds: float | None = None
    notes_commit_holdback_windows: int | None = None
    asr_backlog_warn_seconds: float | None = None
    keep_spool: bool | None = None
    spool_flush_interval_seconds: float | None = None
    spool_flush_bytes: int | None = None
    writer_fsync_mode: str | None = None
    writer_fsync_lines: int | None = None
    writer_fsync_seconds: float | None = None
    asr_preset: str | None = None
    serve_token: str | None = None
    bind: str | None = None
    port: int | None = None
    model_dir_faster_whisper: Path | None = None
    model_dir_whisper_cpp: Path | None = None


def default_output_path(now: datetime | None = None) -> Path:
    current = now or datetime.now(UTC)
    stamp = current.strftime("%Y%m%d-%H%M%S")
    return Path("transcripts") / f"narada-{stamp}.txt"


def load_env_values(env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = env or os.environ
    resolved: dict[str, str] = {}
    for name, key in ENV_KEYS.items():
        value = source.get(key)
        if value is not None and value != "":
            resolved[name] = value
    return resolved


def parse_languages(raw_value: str, allow_multilingual: bool) -> tuple[str, ...]:
    tokens = [token.strip().lower() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise ConfigError("Language list is empty.")

    normalized: list[str] = []
    for token in tokens:
        mapped = LANGUAGE_ALIASES.get(token)
        if mapped:
            normalized.append(mapped)
            continue
        if len(token) == 2 and token.isalpha():
            normalized.append(token)
            continue
        raise ConfigError(
            f"Unsupported language token '{token}'. "
            "Use language names, two-letter codes, or 'auto'."
        )

    if len(normalized) > 1 and not allow_multilingual:
        raise ConfigError("Multiple languages require --allow-multilingual.")
    if "auto" in normalized and len(normalized) > 1:
        raise ConfigError("'auto' cannot be combined with explicit languages.")
    return tuple(normalized)


def _choose_string(cli_value: str | None, env_value: str | None, default: str) -> str:
    if cli_value is not None:
        return cli_value
    if env_value is not None:
        return env_value
    return default


def _choose_path(cli_value: Path | None, env_value: str | None) -> Path:
    if cli_value is not None:
        return cli_value
    if env_value:
        return Path(env_value)
    return default_output_path()


def _choose_optional_path(cli_value: Path | None, env_value: str | None) -> Path | None:
    if cli_value is not None:
        return cli_value
    if env_value:
        return Path(env_value)
    return None


def _parse_on_off(raw_value: str, field_name: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"on", "1", "true", "yes"}:
        return True
    if normalized in {"off", "0", "false", "no"}:
        return False
    raise ConfigError(f"Invalid value for {field_name}: '{raw_value}'. Expected on/off.")


def _parse_bool(raw_value: str, field_name: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Invalid value for {field_name}: '{raw_value}'.")


def _parse_mode(raw_value: str) -> Mode:
    normalized = raw_value.strip().lower()
    if normalized not in MODE_VALUES:
        raise ConfigError(f"Invalid mode '{raw_value}'. Expected one of: {', '.join(MODE_VALUES)}.")
    return cast(Mode, normalized)


def _parse_model(raw_value: str) -> Model:
    normalized = raw_value.strip().lower()
    if normalized not in MODEL_VALUES:
        raise ConfigError(
            f"Invalid model '{raw_value}'. Expected one of: {', '.join(MODE_VALUES)}."
        )
    return cast(Model, normalized)


def _parse_compute(raw_value: str) -> Compute:
    normalized = raw_value.strip().lower()
    if normalized not in COMPUTE_VALUES:
        raise ConfigError(
            f"Invalid compute '{raw_value}'. Expected one of: {', '.join(COMPUTE_VALUES)}."
        )
    return cast(Compute, normalized)


def _parse_engine(raw_value: str) -> Engine:
    normalized = raw_value.strip().lower()
    if normalized not in ENGINE_VALUES:
        raise ConfigError(
            f"Invalid engine '{raw_value}'. Expected one of: {', '.join(ENGINE_VALUES)}."
        )
    return cast(Engine, normalized)


def _parse_noise(raw_value: str) -> NoiseSuppress:
    normalized = raw_value.strip().lower()
    if normalized not in NOISE_VALUES:
        raise ConfigError(
            f"Invalid noise suppress value '{raw_value}'. "
            f"Expected one of: {', '.join(NOISE_VALUES)}."
        )
    return cast(NoiseSuppress, normalized)


def _parse_writer_fsync_mode(raw_value: str) -> WriterFsyncMode:
    normalized = raw_value.strip().lower()
    if normalized not in WRITER_FSYNC_MODE_VALUES:
        raise ConfigError(
            f"Invalid writer fsync mode '{raw_value}'. "
            f"Expected one of: {', '.join(WRITER_FSYNC_MODE_VALUES)}."
        )
    return cast(WriterFsyncMode, normalized)


def _parse_asr_preset(raw_value: str) -> AsrPreset:
    normalized = raw_value.strip().lower()
    if normalized not in ASR_PRESET_VALUES:
        raise ConfigError(
            f"Invalid ASR preset '{raw_value}'. Expected one of: {', '.join(ASR_PRESET_VALUES)}."
        )
    return cast(AsrPreset, normalized)


def build_runtime_config(
    overrides: ConfigOverrides,
    env: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    env_values = load_env_values(env)

    mode = _parse_mode(_choose_string(overrides.mode, env_values.get("mode"), "mic"))
    model = _parse_model(_choose_string(overrides.model, env_values.get("model"), "small"))
    compute = _parse_compute(_choose_string(overrides.compute, env_values.get("compute"), "auto"))
    engine = _parse_engine(
        _choose_string(overrides.engine, env_values.get("engine"), "faster-whisper")
    )

    mic = overrides.mic if overrides.mic is not None else env_values.get("mic")
    system = overrides.system if overrides.system is not None else env_values.get("system")
    out = _choose_path(overrides.out, env_values.get("out"))

    allow_multilingual: bool
    if overrides.allow_multilingual is not None:
        allow_multilingual = overrides.allow_multilingual
    elif "allow_multilingual" in env_values:
        allow_multilingual = _parse_bool(env_values["allow_multilingual"], "allow_multilingual")
    else:
        allow_multilingual = False

    raw_language = _choose_string(overrides.language, env_values.get("language"), "auto")
    languages = parse_languages(raw_language, allow_multilingual)

    redact = _parse_on_off(
        _choose_string(overrides.redact, env_values.get("redact"), "off"), "redact"
    )
    noise_suppress = _parse_noise(
        _choose_string(overrides.noise_suppress, env_values.get("noise_suppress"), "off")
    )
    agc = _parse_on_off(_choose_string(overrides.agc, env_values.get("agc"), "off"), "agc")
    gate = _parse_on_off(_choose_string(overrides.gate, env_values.get("gate"), "off"), "gate")

    gate_threshold_raw = _choose_string(
        str(overrides.gate_threshold_db) if overrides.gate_threshold_db is not None else None,
        env_values.get("gate_threshold_db"),
        "-35.0",
    )
    confidence_raw = _choose_string(
        str(overrides.confidence_threshold) if overrides.confidence_threshold is not None else None,
        env_values.get("confidence_threshold"),
        "0.65",
    )
    wall_flush_seconds_raw = _choose_string(
        str(overrides.wall_flush_seconds) if overrides.wall_flush_seconds is not None else None,
        env_values.get("wall_flush_seconds"),
        "60.0",
    )
    capture_queue_warn_seconds_raw = _choose_string(
        (
            str(overrides.capture_queue_warn_seconds)
            if overrides.capture_queue_warn_seconds is not None
            else None
        ),
        env_values.get("capture_queue_warn_seconds"),
        "120.0",
    )
    notes_interval_seconds_raw = _choose_string(
        str(overrides.notes_interval_seconds)
        if overrides.notes_interval_seconds is not None
        else None,
        env_values.get("notes_interval_seconds"),
        "12.0",
    )
    notes_overlap_seconds_raw = _choose_string(
        (
            str(overrides.notes_overlap_seconds)
            if overrides.notes_overlap_seconds is not None
            else None
        ),
        env_values.get("notes_overlap_seconds"),
        "1.5",
    )
    notes_commit_holdback_raw = _choose_string(
        str(overrides.notes_commit_holdback_windows)
        if overrides.notes_commit_holdback_windows is not None
        else None,
        env_values.get("notes_commit_holdback_windows"),
        "1",
    )
    asr_backlog_warn_seconds_raw = _choose_string(
        str(overrides.asr_backlog_warn_seconds)
        if overrides.asr_backlog_warn_seconds is not None
        else None,
        env_values.get("asr_backlog_warn_seconds"),
        "45.0",
    )
    keep_spool_raw = _choose_string(
        str(overrides.keep_spool).lower() if overrides.keep_spool is not None else None,
        env_values.get("keep_spool"),
        "false",
    )
    spool_flush_interval_seconds_raw = _choose_string(
        (
            str(overrides.spool_flush_interval_seconds)
            if overrides.spool_flush_interval_seconds is not None
            else None
        ),
        env_values.get("spool_flush_interval_seconds"),
        "0.25",
    )
    spool_flush_bytes_raw = _choose_string(
        str(overrides.spool_flush_bytes) if overrides.spool_flush_bytes is not None else None,
        env_values.get("spool_flush_bytes"),
        "65536",
    )
    writer_fsync_mode_raw = _choose_string(
        overrides.writer_fsync_mode,
        env_values.get("writer_fsync_mode"),
        "line",
    )
    writer_fsync_lines_raw = _choose_string(
        str(overrides.writer_fsync_lines) if overrides.writer_fsync_lines is not None else None,
        env_values.get("writer_fsync_lines"),
        "20",
    )
    writer_fsync_seconds_raw = _choose_string(
        (
            str(overrides.writer_fsync_seconds)
            if overrides.writer_fsync_seconds is not None
            else None
        ),
        env_values.get("writer_fsync_seconds"),
        "1.0",
    )
    asr_preset_raw = _choose_string(
        overrides.asr_preset,
        env_values.get("asr_preset"),
        "balanced",
    )
    serve_token = _choose_string(overrides.serve_token, env_values.get("serve_token"), "").strip()
    if not serve_token:
        serve_token = None
    bind = _choose_string(overrides.bind, env_values.get("bind"), "0.0.0.0")
    port_raw = _choose_string(
        str(overrides.port) if overrides.port is not None else None,
        env_values.get("port"),
        "8787",
    )

    try:
        gate_threshold_db = float(gate_threshold_raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid gate threshold: '{gate_threshold_raw}'.") from exc

    try:
        confidence_threshold = float(confidence_raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid confidence threshold: '{confidence_raw}'.") from exc
    try:
        wall_flush_seconds = float(wall_flush_seconds_raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid wall flush interval: '{wall_flush_seconds_raw}'.") from exc
    try:
        capture_queue_warn_seconds = float(capture_queue_warn_seconds_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid capture queue warning threshold: '{capture_queue_warn_seconds_raw}'."
        ) from exc
    try:
        notes_interval_seconds = float(notes_interval_seconds_raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid notes interval: '{notes_interval_seconds_raw}'.") from exc
    try:
        notes_overlap_seconds = float(notes_overlap_seconds_raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid notes overlap: '{notes_overlap_seconds_raw}'.") from exc
    try:
        notes_commit_holdback_windows = int(notes_commit_holdback_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid notes commit holdback windows: '{notes_commit_holdback_raw}'."
        ) from exc
    try:
        asr_backlog_warn_seconds = float(asr_backlog_warn_seconds_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid ASR backlog warning threshold: '{asr_backlog_warn_seconds_raw}'."
        ) from exc
    keep_spool = _parse_bool(keep_spool_raw, "keep_spool")
    try:
        spool_flush_interval_seconds = float(spool_flush_interval_seconds_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid spool flush interval: '{spool_flush_interval_seconds_raw}'."
        ) from exc
    try:
        spool_flush_bytes = int(spool_flush_bytes_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid spool flush byte threshold: '{spool_flush_bytes_raw}'."
        ) from exc
    writer_fsync_mode = _parse_writer_fsync_mode(writer_fsync_mode_raw)
    try:
        writer_fsync_lines = int(writer_fsync_lines_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid writer fsync line threshold: '{writer_fsync_lines_raw}'."
        ) from exc
    try:
        writer_fsync_seconds = float(writer_fsync_seconds_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid writer fsync interval: '{writer_fsync_seconds_raw}'."
        ) from exc
    asr_preset = _parse_asr_preset(asr_preset_raw)

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ConfigError("Confidence threshold must be between 0.0 and 1.0.")
    if wall_flush_seconds < 0.0:
        raise ConfigError("Wall flush seconds must be >= 0.0.")
    if capture_queue_warn_seconds <= 0.0:
        raise ConfigError("Capture queue warning threshold must be > 0.0.")
    if notes_interval_seconds <= 0.0:
        raise ConfigError("Notes interval must be > 0.0.")
    if notes_overlap_seconds < 0.0:
        raise ConfigError("Notes overlap must be >= 0.0.")
    if notes_overlap_seconds >= notes_interval_seconds:
        raise ConfigError("Notes overlap must be smaller than notes interval.")
    if notes_commit_holdback_windows < 0:
        raise ConfigError("Notes commit holdback windows must be >= 0.")
    if asr_backlog_warn_seconds <= 0.0:
        raise ConfigError("ASR backlog warning threshold must be > 0.0.")
    if spool_flush_interval_seconds < 0.0:
        raise ConfigError("Spool flush interval seconds must be >= 0.0.")
    if spool_flush_bytes < 0:
        raise ConfigError("Spool flush bytes must be >= 0.")
    if writer_fsync_lines < 0:
        raise ConfigError("Writer fsync line threshold must be >= 0.")
    if writer_fsync_seconds < 0.0:
        raise ConfigError("Writer fsync interval seconds must be >= 0.0.")
    if (
        writer_fsync_mode == "periodic"
        and writer_fsync_lines == 0
        and writer_fsync_seconds == 0.0
    ):
        raise ConfigError(
            "Writer fsync periodic mode requires writer_fsync_lines > 0 or "
            "writer_fsync_seconds > 0."
        )

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid port: '{port_raw}'.") from exc

    if not 1 <= port <= 65535:
        raise ConfigError("Port must be between 1 and 65535.")

    if mode in {"mic", "mixed"} and not mic:
        raise ConfigError("Mode requires --mic (or NARADA_MIC).")
    if mode in {"system", "mixed"} and not system:
        raise ConfigError("Mode requires --system (or NARADA_SYSTEM).")

    model_dir_faster_whisper = _choose_optional_path(
        overrides.model_dir_faster_whisper,
        env_values.get("model_dir_faster_whisper"),
    )
    model_dir_whisper_cpp = _choose_optional_path(
        overrides.model_dir_whisper_cpp,
        env_values.get("model_dir_whisper_cpp"),
    )

    return RuntimeConfig(
        mode=mode,
        mic=mic,
        system=system,
        out=out,
        model=model,
        compute=compute,
        engine=engine,
        languages=languages,
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
