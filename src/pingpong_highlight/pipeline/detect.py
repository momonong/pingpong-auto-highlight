from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    ImpactEvent,
    MotionFeatures,
    Point,
    PointDetection,
)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Rules for turning impact evidence into individual scored points."""

    minimum_impacts: int = 3
    maximum_impact_gap: float = 1.45
    minimum_point_span: float = 0.5
    maximum_point_span: float = 18.0
    pre_roll: float = 1.2
    post_roll: float = 1.0
    max_points: int = 6
    target_reel_duration: float = 55.0
    transition_duration: float = 0.35
    minimum_points_before_budget: int = 3


@dataclass(slots=True)
class _Candidate:
    start: float
    end: float
    score: float
    impact_count: int
    motion_score: float
    reason: str


def _motion_level(motion: MotionFeatures, start: float, end: float) -> float:
    if motion.scores.size == 0:
        return 0.0
    selected = (motion.times >= start) & (motion.times <= end)
    if not np.any(selected):
        return 0.0
    values = motion.scores[selected]
    return float(0.55 * np.mean(values) + 0.45 * np.percentile(values, 85))


def _group_impacts(events: list[ImpactEvent], config: DetectionConfig) -> list[list[ImpactEvent]]:
    if not events:
        return []
    groups: list[list[ImpactEvent]] = []
    current = [events[0]]
    for event in events[1:]:
        gap = event.time - current[-1].time
        span = event.time - current[0].time
        if gap > config.maximum_impact_gap or span > config.maximum_point_span:
            groups.append(current)
            current = [event]
        else:
            current.append(event)
    groups.append(current)
    return groups


def _audio_candidates(
    audio: AudioFeatures,
    motion: MotionFeatures,
    config: DetectionConfig,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for group in _group_impacts(audio.events, config):
        if len(group) < config.minimum_impacts:
            continue
        span = group[-1].time - group[0].time
        if span < config.minimum_point_span:
            continue

        gaps = np.diff([event.time for event in group])
        rhythmic = float(np.mean((gaps >= 0.16) & (gaps <= 1.05))) if gaps.size else 0.0
        motion_score = _motion_level(motion, group[0].time - 0.4, group[-1].time + 0.4)
        tempo = (len(group) - 1) / max(span, 0.3)
        impact_strength = float(np.mean([event.strength for event in group]))
        score = (
            3.4 * np.log1p(len(group))
            + 1.25 * min(tempo, 3.5)
            + 1.7 * min(motion_score, 4.0)
            + 1.2 * rhythmic
            + 0.7 * impact_strength
            + 0.08 * min(span, 12.0)
        )
        candidates.append(
            _Candidate(
                start=group[0].time,
                end=group[-1].time,
                score=float(score),
                impact_count=len(group),
                motion_score=motion_score,
                reason=f"{len(group)} rhythmic impact transients within one point",
            )
        )
    return candidates


def _motion_candidates(motion: MotionFeatures, config: DetectionConfig) -> list[_Candidate]:
    if motion.scores.size == 0:
        return []
    active_indices = np.flatnonzero(motion.scores >= 0.75)
    if active_indices.size == 0:
        return []

    groups: list[list[int]] = [[int(active_indices[0])]]
    for index in active_indices[1:]:
        group = groups[-1]
        gap = motion.times[index] - motion.times[group[-1]]
        span = motion.times[index] - motion.times[group[0]]
        if gap > 0.85 or span > config.maximum_point_span:
            groups.append([int(index)])
        else:
            group.append(int(index))

    candidates: list[_Candidate] = []
    for group in groups:
        start = float(motion.times[group[0]])
        end = float(motion.times[group[-1]])
        if len(group) < 5 or end - start < 1.25:
            continue
        values = motion.scores[group]
        motion_score = float(0.5 * np.mean(values) + 0.5 * np.percentile(values, 90))
        score = 2.3 * np.log1p(end - start) + 2.0 * min(motion_score, 6.0)
        candidates.append(
            _Candidate(
                start=start,
                end=end,
                score=float(score),
                impact_count=0,
                motion_score=motion_score,
                reason="sustained localized play motion (audio fallback)",
            )
        )
    return candidates


def _pad_candidates(
    duration: float,
    candidates: list[_Candidate],
    config: DetectionConfig,
) -> list[Point]:
    ordered = sorted(candidates, key=lambda candidate: candidate.start)
    points: list[Point] = []
    for index, candidate in enumerate(ordered):
        start = max(0.0, candidate.start - config.pre_roll)
        end = min(duration, candidate.end + config.post_roll)

        # Neighbouring points may be close together. Divide the quiet gap instead of
        # duplicating the next serve or previous reaction in both exported clips.
        if index:
            divider = (ordered[index - 1].end + candidate.start) / 2
            start = max(start, divider)
        if index + 1 < len(ordered):
            divider = (candidate.end + ordered[index + 1].start) / 2
            end = min(end, divider)
        if end <= start:
            continue
        points.append(
            Point(
                start=round(start, 3),
                end=round(end, 3),
                score=round(candidate.score, 3),
                impact_count=candidate.impact_count,
                motion_score=round(candidate.motion_score, 3),
                reason=candidate.reason,
            )
        )
    return points


def _select_points(candidates: list[Point], config: DetectionConfig) -> list[Point]:
    ranked = sorted(candidates, key=lambda point: point.score, reverse=True)
    selected: list[Point] = []
    reel_duration = 0.0
    for point in ranked:
        if len(selected) >= config.max_points:
            break
        added_duration = point.duration
        if selected:
            added_duration -= min(
                config.transition_duration,
                selected[-1].duration / 4,
                point.duration / 4,
            )
        exceeds_budget = reel_duration + added_duration > config.target_reel_duration
        if exceeds_budget and len(selected) >= config.minimum_points_before_budget:
            continue
        selected.append(point)
        reel_duration += added_duration

    ranked_selected = [replace(point, rank=rank) for rank, point in enumerate(selected, start=1)]
    return sorted(ranked_selected, key=lambda point: point.start)


def detect_points(
    duration: float,
    audio: AudioFeatures,
    motion: MotionFeatures,
    config: DetectionConfig | None = None,
) -> PointDetection:
    config = config or DetectionConfig()
    raw_candidates = _audio_candidates(audio, motion, config)
    if not raw_candidates:
        raw_candidates = _motion_candidates(motion, config)
    candidates = _pad_candidates(duration, raw_candidates, config)
    return PointDetection(candidates=candidates, points=_select_points(candidates, config))
