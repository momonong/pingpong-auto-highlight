from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from pingpong_highlight.pipeline.media import finish_decoder, open_video_decoder, read_exact
from pingpong_highlight.pipeline.models import MediaInfo, MotionFeatures
from pingpong_highlight.pipeline.normalize import contextual_robust_z, moving_average

ProgressCallback = Callable[[float], None]


def _localized_motion(previous: np.ndarray, current: np.ndarray, grid: int = 8) -> float:
    difference = np.abs(current.astype(np.int16) - previous.astype(np.int16)).astype(np.float32)
    difference[difference < 7.0] = 0.0
    height, width = difference.shape
    block_height = height // grid
    block_width = width // grid
    cropped = difference[: block_height * grid, : block_width * grid]
    blocks = cropped.reshape(grid, block_height, grid, block_width).mean(axis=(1, 3))
    flattened = blocks.ravel()
    top_count = max(1, flattened.size // 8)
    local = float(np.mean(np.partition(flattened, -top_count)[-top_count:]))
    background = float(np.median(flattened))
    global_change = float(np.mean(difference))

    # A cut or sudden exposure change affects almost every block and is not player motion.
    if global_change > 85.0 and background > 55.0:
        return 0.0
    return max(0.0, (local - 0.65 * background) / 255.0)


def analyze_motion(
    path: Path,
    media: MediaInfo,
    *,
    fps: float = 8.0,
    frame_size: int = 320,
    progress: ProgressCallback | None = None,
) -> MotionFeatures:
    process = open_video_decoder(path, fps, frame_size)
    if process.stdout is None:
        raise RuntimeError("FFmpeg video pipe was not created")

    bytes_per_frame = frame_size * frame_size
    previous: np.ndarray | None = None
    times: list[float] = []
    raw_scores: list[float] = []
    frame_index = 0
    last_progress = -1.0

    while True:
        raw = read_exact(process.stdout, bytes_per_frame)
        if len(raw) != bytes_per_frame:
            break
        current = np.frombuffer(raw, dtype=np.uint8).reshape(frame_size, frame_size)
        if previous is not None:
            times.append(frame_index / fps)
            raw_scores.append(_localized_motion(previous, current))
        previous = current.copy()
        frame_index += 1

        current_progress = min(1.0, (frame_index / fps) / media.duration)
        if progress and current_progress - last_progress >= 0.01:
            progress(current_progress)
            last_progress = current_progress

    finish_decoder(process, "video")
    if not times:
        if progress:
            progress(1.0)
        return MotionFeatures.empty()

    time_array = np.asarray(times, dtype=np.float64)
    normalized = contextual_robust_z(np.asarray(raw_scores), time_array)
    activity = moving_average(np.maximum(normalized, 0.0), max(1, round(fps * 0.4)))
    if progress:
        progress(1.0)
    return MotionFeatures(times=time_array, scores=activity)
