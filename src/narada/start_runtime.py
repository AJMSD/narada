from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Literal

from narada.audio.mixer import AudioChunk, mix_audio_chunks

StartMode = Literal["mic", "system", "mixed"]


@dataclass(frozen=True)
class MonoAudioFrame:
    samples: tuple[float, ...]
    sample_rate_hz: int


@dataclass(frozen=True)
class ParsedInput:
    text: str | None = None
    confidence: float = 1.0
    audio: MonoAudioFrame | None = None


def _parse_numeric_sequence(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of numbers.")
    parsed: list[float] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (int, float)):
            raise ValueError(f"{field_name}[{idx}] must be numeric.")
        parsed.append(float(item))
    return tuple(parsed)


def _parse_int(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _parse_audio_chunk(payload: dict[str, Any], prefix: str = "") -> AudioChunk:
    samples = _parse_numeric_sequence(payload.get(f"{prefix}samples"), f"{prefix}samples")
    sample_rate_hz = _parse_int(payload.get(f"{prefix}sample_rate_hz"), f"{prefix}sample_rate_hz")
    channels_raw = payload.get(f"{prefix}channels", 1)
    channels = _parse_int(channels_raw, f"{prefix}channels")
    if sample_rate_hz <= 0:
        raise ValueError(f"{prefix}sample_rate_hz must be positive.")
    if channels <= 0:
        raise ValueError(f"{prefix}channels must be positive.")
    return AudioChunk(samples=samples, sample_rate_hz=sample_rate_hz, channels=channels)


def parse_input_line(line: str, mode: StartMode) -> ParsedInput | None:
    cleaned = line.strip()
    if not cleaned:
        return None

    if not cleaned.startswith("{"):
        return ParsedInput(text=cleaned, confidence=1.0)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON input: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object.")

    text_field = payload.get("text")
    if isinstance(text_field, str):
        confidence_raw = payload.get("confidence", 1.0)
        if not isinstance(confidence_raw, (int, float)):
            raise ValueError("confidence must be numeric.")
        confidence = float(confidence_raw)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        return ParsedInput(text=text_field.strip(), confidence=confidence)

    if mode == "mixed":
        mic_payload = payload.get("mic")
        system_payload = payload.get("system")
        if not isinstance(mic_payload, dict) or not isinstance(system_payload, dict):
            raise ValueError("Mixed mode JSON requires 'mic' and 'system' objects.")
        mic_chunk = _parse_audio_chunk(mic_payload)
        system_chunk = _parse_audio_chunk(system_payload)
        mixed_samples, mixed_rate = mix_audio_chunks(mic_chunk, system_chunk)
        return ParsedInput(
            audio=MonoAudioFrame(samples=tuple(mixed_samples), sample_rate_hz=mixed_rate)
        )

    if "audio" in payload:
        nested = payload["audio"]
        if not isinstance(nested, dict):
            raise ValueError("'audio' must be an object when provided.")
        chunk = _parse_audio_chunk(nested)
    else:
        chunk = _parse_audio_chunk(payload)
    return ParsedInput(
        audio=MonoAudioFrame(
            samples=chunk.samples,
            sample_rate_hz=chunk.sample_rate_hz,
        )
    )


def mono_frame_to_pcm16le(frame: MonoAudioFrame) -> bytes:
    raw = bytearray()
    for sample in frame.samples:
        clipped = max(-1.0, min(1.0, sample))
        as_int = int(clipped * 32767)
        raw.extend(struct.pack("<h", as_int))
    return bytes(raw)
