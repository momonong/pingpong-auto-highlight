from __future__ import annotations

import numpy as np

from pingpong_highlight.pipeline.detect import detect_points
from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    ImpactEvent,
    MotionFeatures,
)


def test_audio_events_form_ranked_padded_point() -> None:
    events = [ImpactEvent(time=time, strength=1.0) for time in (5.0, 5.4, 5.9, 6.3, 6.8)]
    audio = AudioFeatures(np.empty(0), np.empty(0), events)
    motion_times = np.arange(0, 12, 0.125)
    motion_scores = np.where((motion_times >= 4.8) & (motion_times <= 7.0), 1.8, 0.1)
    motion = MotionFeatures(motion_times, motion_scores)

    detection = detect_points(12.0, audio, motion)

    assert len(detection.candidates) == 1
    assert len(detection.points) == 1
    assert detection.points[0].start == 3.8
    assert detection.points[0].end == 7.8
    assert detection.points[0].impact_count == 5
    assert detection.points[0].rank == 1


def test_separate_points_are_ranked_but_returned_chronologically() -> None:
    events = [
        *[ImpactEvent(time=time, strength=0.8) for time in (3.0, 3.5, 4.0)],
        *[ImpactEvent(time=time, strength=1.2) for time in (14.0, 14.4, 14.8, 15.2, 15.6)],
    ]
    detection = detect_points(
        20.0,
        AudioFeatures(np.empty(0), np.empty(0), events),
        MotionFeatures.empty(),
    )

    assert len(detection.points) == 2
    assert detection.points[0].start < detection.points[1].start
    assert detection.points[0].rank == 2
    assert detection.points[1].rank == 1


def test_nearby_points_share_the_quiet_gap_instead_of_merging() -> None:
    events = [
        *[ImpactEvent(time=time, strength=1.0) for time in (3.0, 3.4, 3.8)],
        *[ImpactEvent(time=time, strength=1.0) for time in (5.4, 5.8, 6.2)],
    ]
    detection = detect_points(
        10.0,
        AudioFeatures(np.empty(0), np.empty(0), events),
        MotionFeatures.empty(),
    )

    assert len(detection.candidates) == 2
    assert detection.candidates[0].end == detection.candidates[1].start == 4.6


def test_motion_only_fallback_works_without_audio() -> None:
    times = np.arange(0, 10, 0.125)
    scores = np.zeros_like(times)
    scores[(times >= 3.0) & (times <= 6.0)] = 1.5

    detection = detect_points(
        10.0,
        AudioFeatures.empty(),
        MotionFeatures(times, scores),
    )

    assert len(detection.points) == 1
    assert detection.points[0].impact_count == 0
    assert "audio fallback" in detection.points[0].reason
