from __future__ import annotations

from collections.abc import Sequence


def resample_linear(samples: Sequence[float], src_rate: int, dst_rate: int) -> list[float]:
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if not samples:
        return []
    if src_rate == dst_rate:
        return list(samples)

    ratio = dst_rate / src_rate
    target_length = max(1, int(len(samples) * ratio))
    output: list[float] = []
    for idx in range(target_length):
        source_position = idx / ratio
        lower = int(source_position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = source_position - lower
        interpolated = (1.0 - fraction) * samples[lower] + fraction * samples[upper]
        output.append(interpolated)
    return output


def normalize_peak(samples: Sequence[float], target_peak: float = 0.9) -> list[float]:
    if not samples:
        return []
    if target_peak <= 0:
        raise ValueError("target_peak must be positive.")

    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        return list(samples)
    scale = min(1.0, target_peak / peak)
    return [sample * scale for sample in samples]


def mix_streams(
    mic: Sequence[float], system: Sequence[float], headroom: float = 0.8
) -> list[float]:
    if not 0 < headroom <= 1.0:
        raise ValueError("headroom must be between 0 and 1.")

    total = max(len(mic), len(system))
    mixed: list[float] = []
    for idx in range(total):
        mic_sample = mic[idx] if idx < len(mic) else 0.0
        system_sample = system[idx] if idx < len(system) else 0.0
        value = (mic_sample + system_sample) * 0.5 * headroom
        mixed.append(max(-1.0, min(1.0, value)))
    return mixed
