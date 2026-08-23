from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    ImpactEvent,
    MotionFeatures,
    Point,
    PointCandidate,
    PointDetection,
)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Rules for turning impact evidence into individual scored points."""

    minimum_impacts: int = 3
    maximum_impact_gap: float = 1.45
    minimum_point_span: float = 0.5
    maximum_point_span: float = 18.0
    pre_roll: float = 1.5
    post_roll: float = 1.5
    minimum_point_score_ratio: float = 0.87
    max_points: int | None = None
    target_reel_duration: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_point_score_ratio <= 1.0:
            raise ValueError("minimum_point_score_ratio must be between 0 and 1")
        if self.max_points is not None and self.max_points <= 0:
            raise ValueError("max_points must be positive or None")
        if self.target_reel_duration is not None and self.target_reel_duration <= 0:
            raise ValueError("target_reel_duration must be positive or None")


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
) -> list[PointCandidate]:
    candidates: list[PointCandidate] = []
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
            PointCandidate(
                start=group[0].time,
                end=group[-1].time,
                score=float(score),
                impact_count=len(group),
                motion_score=motion_score,
                reason=f"{len(group)} rhythmic impact transients within one point",
            )
        )
    return candidates


def _motion_candidates(
    motion: MotionFeatures,
    config: DetectionConfig,
) -> list[PointCandidate]:
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

    candidates: list[PointCandidate] = []
    for group in groups:
        start = float(motion.times[group[0]])
        end = float(motion.times[group[-1]])
        if len(group) < 5 or end - start < 1.25:
            continue
        values = motion.scores[group]
        motion_score = float(0.5 * np.mean(values) + 0.5 * np.percentile(values, 90))
        score = 2.3 * np.log1p(end - start) + 2.0 * min(motion_score, 6.0)
        candidates.append(
            PointCandidate(
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
    candidates: list[PointCandidate],
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
                rally_start=round(candidate.start, 3),
                rally_end=round(candidate.end, 3),
                rank=candidate.rank or 0,
            )
        )
    return points


def _candidate_clip_duration(
    duration: float,
    candidate: PointCandidate,
    config: DetectionConfig,
) -> float:
    start = max(0.0, candidate.start - config.pre_roll)
    end = min(duration, candidate.end + config.post_roll)
    return max(0.0, end - start)


def _select_candidates(
    duration: float,
    candidates: list[PointCandidate],
    config: DetectionConfig,
) -> tuple[list[PointCandidate], list[PointCandidate], float | None]:
    if not candidates:
        return [], [], None

    best_score = max(candidate.score for candidate in candidates)
    score_threshold = best_score * config.minimum_point_score_ratio
    decisions = ["below-score-threshold"] * len(candidates)
    ranked_indices = sorted(
        (
            index
            for index, candidate in enumerate(candidates)
            if candidate.score >= score_threshold
        ),
        key=lambda index: (
            -candidates[index].score,
            candidates[index].start,
            candidates[index].end,
        ),
    )

    selected_indices: list[int] = []
    reel_duration = 0.0
    for index in ranked_indices:
        if config.max_points is not None and len(selected_indices) >= config.max_points:
            decisions[index] = "point-cap"
            continue

        added_duration = _candidate_clip_duration(duration, candidates[index], config)
        if (
            config.target_reel_duration is not None
            and reel_duration + added_duration > config.target_reel_duration
        ):
            decisions[index] = "duration-budget"
            continue

        decisions[index] = "selected"
        selected_indices.append(index)
        reel_duration += added_duration

    rank_by_index = {index: rank for rank, index in enumerate(selected_indices, start=1)}
    decided_candidates = [
        replace(
            candidate,
            selection=decisions[index],
            rank=rank_by_index.get(index),
        )
        for index, candidate in enumerate(candidates)
    ]
    selected_candidates = [decided_candidates[index] for index in selected_indices]
    return (
        sorted(decided_candidates, key=lambda candidate: candidate.start),
        selected_candidates,
        score_threshold,
    )


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
    candidates, selected_candidates, score_threshold = _select_candidates(
        duration,
        raw_candidates,
        config,
    )
    points = _pad_candidates(duration, selected_candidates, config)
    return PointDetection(
        candidates=candidates,
        points=points,
        effective_score_threshold=(
            round(score_threshold, 6) if score_threshold is not None else None
        ),
    )
