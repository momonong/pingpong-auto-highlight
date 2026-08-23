from __future__ import annotations

import numpy as np

from pingpong_highlight.pipeline.detect import (
    DetectionConfig,
    _select_candidates,
    detect_points,
)
from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    ImpactEvent,
    MotionFeatures,
    PointCandidate,
)


def _candidate(start: float, score: float, duration: float = 1.0) -> PointCandidate:
    return PointCandidate(
        start=start,
        end=start + duration,
        score=score,
        impact_count=3,
        motion_score=1.0,
        reason="test candidate",
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
    assert detection.points[0].start == 3.5
    assert detection.points[0].end == 8.3
    assert detection.points[0].rally_start == 5.0
    assert detection.points[0].rally_end == 6.8
    assert detection.points[0].to_dict()["pre_context_seconds"] == 1.5
    assert detection.points[0].to_dict()["post_context_seconds"] == 1.5
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
        DetectionConfig(minimum_point_score_ratio=0.0),
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
    assert detection.points[0].end == detection.points[1].start == 4.6


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


def test_reel_budget_counts_full_padded_duration_for_direct_cuts() -> None:
    events = [
        *[ImpactEvent(time=time, strength=1.0) for time in (3.0, 3.4, 3.8)],
        *[ImpactEvent(time=time, strength=0.9) for time in (12.0, 12.4, 12.8)],
    ]
    detection = detect_points(
        20.0,
        AudioFeatures(np.empty(0), np.empty(0), events),
        MotionFeatures.empty(),
        DetectionConfig(
            max_points=2,
            target_reel_duration=7.5,
            minimum_point_score_ratio=0.0,
            pre_roll=1.5,
            post_roll=1.5,
        ),
    )

    assert len(detection.candidates) == 2
    assert len(detection.points) == 1
    assert detection.points[0].duration == 3.8


def test_relative_threshold_accepts_equality_and_rejects_lower_scores() -> None:
    candidates = [
        _candidate(1.0, 10.0),
        _candidate(4.0, 8.7),
        _candidate(7.0, 8.699),
    ]

    decisions, selected, threshold = _select_candidates(
        20.0,
        candidates,
        DetectionConfig(minimum_point_score_ratio=0.87, target_reel_duration=20.0),
    )

    assert threshold == 8.7
    assert [candidate.score for candidate in selected] == [10.0, 8.7]
    assert [candidate.selection for candidate in decisions] == [
        "selected",
        "selected",
        "below-score-threshold",
    ]


def test_default_selection_has_no_six_point_quota() -> None:
    candidates = [_candidate(index * 3.0, 10.0) for index in range(7)]

    decisions, selected, _threshold = _select_candidates(
        25.0,
        candidates,
        DetectionConfig(target_reel_duration=100.0),
    )

    assert len(selected) == 7
    assert all(candidate.selection == "selected" for candidate in decisions)


def test_explicit_point_cap_remains_an_optional_safety_limit() -> None:
    candidates = [_candidate(index * 3.0, 10.0) for index in range(4)]

    decisions, selected, _threshold = _select_candidates(
        20.0,
        candidates,
        DetectionConfig(max_points=2, target_reel_duration=20.0),
    )

    assert len(selected) == 2
    assert [candidate.selection for candidate in decisions] == [
        "selected",
        "selected",
        "point-cap",
        "point-cap",
    ]


def test_budget_does_not_force_a_minimum_number_of_points() -> None:
    decisions, selected, _threshold = _select_candidates(
        10.0,
        [_candidate(1.0, 10.0, duration=4.0)],
        DetectionConfig(target_reel_duration=3.0),
    )

    assert selected == []
    assert decisions[0].selection == "duration-budget"


def test_rejected_neighbour_does_not_trim_selected_point_context() -> None:
    events = [
        *[ImpactEvent(time=time, strength=1.0) for time in (3.0, 3.4, 3.8)],
        *[ImpactEvent(time=time, strength=0.1) for time in (5.4, 5.8, 6.2)],
    ]

    detection = detect_points(
        10.0,
        AudioFeatures(np.empty(0), np.empty(0), events),
        MotionFeatures.empty(),
        DetectionConfig(minimum_point_score_ratio=0.99),
    )

    assert len(detection.candidates) == 2
    assert len(detection.points) == 1
    assert detection.points[0].rally_end == 3.8
    assert detection.points[0].end == 5.3
