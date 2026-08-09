from __future__ import annotations

import numpy as np

from pingpong_highlight.pipeline.detect import DetectionConfig, detect_highlights
from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    ImpactEvent,
    MotionFeatures,
)


def test_audio_events_form_ranked_padded_rally() -> None:
    events = [ImpactEvent(time=time, strength=1.0) for time in (5.0, 5.4, 5.9, 6.3, 6.8)]
    audio = AudioFeatures(np.empty(0), np.empty(0), events)
    motion_times = np.arange(0, 12, 0.125)
    motion_scores = np.where((motion_times >= 4.8) & (motion_times <= 7.0), 1.8, 0.1)
    motion = MotionFeatures(motion_times, motion_scores)

    highlights = detect_highlights(12.0, audio, motion)

    assert len(highlights) == 1
    assert highlights[0].start == 2.5
    assert highlights[0].end == 8.8
    assert highlights[0].hit_count == 5
    assert highlights[0].rank == 1


def test_separate_rallies_are_ranked_but_returned_chronologically() -> None:
    events = [
        *[ImpactEvent(time=time, strength=0.8) for time in (3.0, 3.5, 4.0)],
        *[ImpactEvent(time=time, strength=1.2) for time in (14.0, 14.4, 14.8, 15.2, 15.6)],
    ]
    highlights = detect_highlights(
        20.0,
        AudioFeatures(np.empty(0), np.empty(0), events),
        MotionFeatures.empty(),
        DetectionConfig(merge_gap=1.0),
    )

    assert len(highlights) == 2
    assert highlights[0].start < highlights[1].start
    assert highlights[0].rank == 2
    assert highlights[1].rank == 1


def test_motion_only_fallback_works_without_audio() -> None:
    times = np.arange(0, 10, 0.125)
    scores = np.zeros_like(times)
    scores[(times >= 3.0) & (times <= 6.0)] = 1.5

    highlights = detect_highlights(
        10.0,
        AudioFeatures.empty(),
        MotionFeatures(times, scores),
    )

    assert len(highlights) == 1
    assert highlights[0].hit_count == 0
    assert "audio fallback" in highlights[0].reason
