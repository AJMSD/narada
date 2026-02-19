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

MODE_VALUES: tuple[Mode, ...] = ("mic", "system", "mixed")
MODEL_VALUES: tuple[Model, ...] = ("tiny", "small", "medium", "large")
COMPUTE_VALUES: tuple[Compute, ...] = ("cpu", "cuda", "metal", "auto")
ENGINE_VALUES: tuple[Engine, ...] = ("faster-whisper", "whisper-cpp")
NOISE_VALUES: tuple[NoiseSuppress, ...] = ("off", "rnnoise", "webrtc")

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
    "bind": "NARADA_BIND",
    "port": "NARADA_PORT",
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
    bind: str
    port: int


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
    bind: str | None = None
    port: int | None = None


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

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ConfigError("Confidence threshold must be between 0.0 and 1.0.")

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
        bind=bind,
        port=port,
    )
