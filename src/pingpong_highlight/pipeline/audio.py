from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from pingpong_highlight.pipeline.media import finish_decoder, open_audio_decoder, read_exact
from pingpong_highlight.pipeline.models import AudioFeatures, ImpactEvent, MediaInfo
from pingpong_highlight.pipeline.normalize import contextual_robust_z

ProgressCallback = Callable[[float], None]


def pick_impact_events(
    times: np.ndarray,
    scores: np.ndarray,
    *,
    minimum_score: float = 3.0,
    minimum_spacing: float = 0.16,
) -> list[ImpactEvent]:
    if scores.size < 3:
        return []
    local_maximum = (scores[1:-1] >= scores[:-2]) & (scores[1:-1] > scores[2:])
    candidates = np.flatnonzero(local_maximum) + 1
    candidates = candidates[scores[candidates] >= minimum_score]

    accepted: list[int] = []
    for index in candidates[np.argsort(scores[candidates])[::-1]]:
        if all(abs(float(times[index] - times[other])) >= minimum_spacing for other in accepted):
            accepted.append(int(index))
    accepted.sort()
    return [
        ImpactEvent(time=float(times[index]), strength=float(min(scores[index] / 6.0, 2.0)))
        for index in accepted
    ]


def analyze_audio(
    path: Path,
    media: MediaInfo,
    *,
    sample_rate: int = 16_000,
    progress: ProgressCallback | None = None,
) -> AudioFeatures:
    if not media.has_audio:
        if progress:
            progress(1.0)
        return AudioFeatures.empty()

    process = open_audio_decoder(path, sample_rate)
    if process.stdout is None:
        raise RuntimeError("FFmpeg audio pipe was not created")

    frame_size = 512
    hop_size = 256
    window = np.hanning(frame_size).astype(np.float32)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    high_frequency = frequencies >= 1_600

    pending = np.empty(0, dtype=np.float32)
    pending_start = 0
    previous_spectrum: np.ndarray | None = None
    feature_times: list[np.ndarray] = []
    flux_values: list[np.ndarray] = []
    high_values: list[np.ndarray] = []
    rms_values: list[np.ndarray] = []
    last_progress = -1.0

    while True:
        raw = read_exact(process.stdout, 1024 * 1024)
        if not raw:
            break
        usable_size = len(raw) - (len(raw) % np.dtype("<f4").itemsize)
        if usable_size == 0:
            continue
        samples = np.frombuffer(raw[:usable_size], dtype="<f4")
        pending = np.concatenate((pending, samples))
        frame_count = 1 + (pending.size - frame_size) // hop_size
        if frame_count <= 0:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(pending, frame_size)[::hop_size]
        windows = windows[:frame_count]
        spectrum = np.abs(np.fft.rfft(windows * window, axis=1))
        power = np.square(spectrum)
        total_power = np.sum(power, axis=1) + 1e-12
        high_power = np.sum(power[:, high_frequency], axis=1)

        previous = np.vstack(
            (
                previous_spectrum if previous_spectrum is not None else spectrum[0],
                spectrum[:-1],
            )
        )
        positive_flux = np.sum(np.maximum(spectrum - previous, 0.0), axis=1)
        flux_ratio = positive_flux / (np.sum(previous, axis=1) + 1e-12)
        high_ratio = high_power / total_power
        rms = np.sqrt(np.mean(np.square(windows), axis=1) + 1e-12)

        starts = pending_start + np.arange(frame_count) * hop_size
        feature_times.append((starts + frame_size / 2) / sample_rate)
        flux_values.append(np.log1p(flux_ratio * (0.35 + high_ratio) * 100.0))
        high_values.append(np.log1p(high_power))
        rms_values.append(np.log1p(rms * 100.0))
        previous_spectrum = spectrum[-1]

        consumed = frame_count * hop_size
        pending = pending[consumed:]
        pending_start += consumed

        current_progress = min(1.0, pending_start / (media.duration * sample_rate))
        if progress and current_progress - last_progress >= 0.01:
            progress(current_progress)
            last_progress = current_progress

    finish_decoder(process, "audio")
    if not feature_times:
        if progress:
            progress(1.0)
        return AudioFeatures.empty()

    times = np.concatenate(feature_times)
    flux = contextual_robust_z(np.concatenate(flux_values), times)
    high = contextual_robust_z(np.concatenate(high_values), times)
    rms = contextual_robust_z(np.concatenate(rms_values), times)
    scores = (
        0.62 * np.maximum(flux, 0.0) + 0.28 * np.maximum(high, 0.0) + 0.10 * np.maximum(rms, 0.0)
    )
    events = pick_impact_events(times, scores)
    if progress:
        progress(1.0)
    return AudioFeatures(times=times, scores=scores, events=events)
