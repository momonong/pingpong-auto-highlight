from __future__ import annotations

import numpy as np


def contextual_robust_z(
    values: np.ndarray,
    times: np.ndarray,
    *,
    block_seconds: float = 60.0,
) -> np.ndarray:
    """Robustly normalize changing recording conditions without a large rolling matrix."""
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    if values.size != times.size:
        raise ValueError("values and times must have equal lengths")

    result = np.zeros(values.shape, dtype=np.float64)
    total_blocks = max(1, int(np.ceil((float(times[-1]) + 1e-9) / block_seconds)))
    for block in range(total_blocks):
        start = block * block_seconds
        end = start + block_seconds
        target = (times >= start) & (times < end)
        if not np.any(target):
            continue
        context = (times >= max(0.0, start - block_seconds)) & (times < end + block_seconds)
        sample = values[context]
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        scale = max(1.4826 * mad, float(np.std(sample)) * 0.15, 1e-9)
        result[target] = (values[target] - median) / scale
    return np.clip(result, -4.0, 12.0)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == 0 or width <= 1:
        return values.astype(np.float64, copy=True)
    width = min(width, values.size)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")
