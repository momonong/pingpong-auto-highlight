from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    Highlight,
    ImpactEvent,
    MotionFeatures,
)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    minimum_hits: int = 3
    maximum_hit_gap: float = 2.2
    minimum_rally_span: float = 0.35
    pre_roll: float = 2.5
    post_roll: float = 2.0
    merge_gap: float = 2.0
    maximum_clip_duration: float = 45.0
    max_highlights: int = 12


@dataclass(slots=True)
class _Candidate:
    start: float
    end: float
    score: float
    hit_count: int
    motion_score: float
    reason: str


def _motion_level(motion: MotionFeatures, start: float, end: float) -> float:
    if motion.scores.size == 0:
        return 0.0
    selected = (motion.times >= start) & (motion.times <= end)
    if not np.any(selected):
        return 0.0
    values = motion.scores[selected]
    return float(0.6 * np.mean(values) + 0.4 * np.percentile(values, 85))


def _group_impacts(events: list[ImpactEvent], config: DetectionConfig) -> list[list[ImpactEvent]]:
    if not events:
        return []
    groups: list[list[ImpactEvent]] = []
    current = [events[0]]
    maximum_content_span = config.maximum_clip_duration - config.pre_roll - config.post_roll
    for event in events[1:]:
        gap = event.time - current[-1].time
        span = event.time - current[0].time
        if gap > config.maximum_hit_gap or span > maximum_content_span:
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
        if len(group) < config.minimum_hits:
            continue
        span = group[-1].time - group[0].time
        if span < config.minimum_rally_span:
            continue
        motion_score = _motion_level(motion, group[0].time - 0.5, group[-1].time + 0.5)
        tempo = (len(group) - 1) / max(span, 0.25)
        impact_strength = float(np.mean([event.strength for event in group]))
        score = (
            3.0 * np.log1p(len(group))
            + 1.1 * min(tempo, 5.0)
            + 1.4 * min(motion_score, 5.0)
            + 0.8 * impact_strength
            + 0.06 * span
        )
        candidates.append(
            _Candidate(
                start=group[0].time,
                end=group[-1].time,
                score=float(score),
                hit_count=len(group),
                motion_score=motion_score,
                reason=f"{len(group)} impact transients with sustained play motion",
            )
        )
    return candidates


def _motion_candidates(motion: MotionFeatures, config: DetectionConfig) -> list[_Candidate]:
    if motion.scores.size == 0:
        return []
    active_indices = np.flatnonzero(motion.scores >= 0.65)
    if active_indices.size == 0:
        return []

    groups: list[list[int]] = [[int(active_indices[0])]]
    for index in active_indices[1:]:
        if motion.times[index] - motion.times[groups[-1][-1]] > 1.1:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))

    candidates: list[_Candidate] = []
    for group in groups:
        start = float(motion.times[group[0]])
        end = float(motion.times[group[-1]])
        if len(group) < 5 or end - start < 1.25:
            continue
        values = motion.scores[group]
        motion_score = float(0.5 * np.mean(values) + 0.5 * np.percentile(values, 90))
        score = 2.2 * np.log1p(end - start) + 2.0 * min(motion_score, 6.0)
        candidates.append(
            _Candidate(
                start=start,
                end=end,
                score=float(score),
                hit_count=0,
                motion_score=motion_score,
                reason="sustained localized player motion (audio fallback)",
            )
        )
    return candidates


def _merge_candidates(candidates: list[_Candidate], config: DetectionConfig) -> list[_Candidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda candidate: candidate.start)
    merged = [ordered[0]]
    maximum_content_span = config.maximum_clip_duration - config.pre_roll - config.post_roll
    for candidate in ordered[1:]:
        current = merged[-1]
        combined_end = max(current.end, candidate.end)
        can_merge = (
            candidate.start - current.end <= config.merge_gap
            and combined_end - current.start <= maximum_content_span
        )
        if not can_merge:
            merged.append(candidate)
            continue
        total_hits = current.hit_count + candidate.hit_count
        current.end = combined_end
        current.score = max(current.score, candidate.score) + 0.25 * min(
            current.score, candidate.score
        )
        current.hit_count = total_hits
        current.motion_score = max(current.motion_score, candidate.motion_score)
        current.reason = (
            f"{total_hits} impact transients across adjacent activity"
            if total_hits
            else "adjacent sustained player-motion segments"
        )
    return merged


def detect_highlights(
    duration: float,
    audio: AudioFeatures,
    motion: MotionFeatures,
    config: DetectionConfig | None = None,
) -> list[Highlight]:
    config = config or DetectionConfig()
    candidates = _audio_candidates(audio, motion, config)
    if not candidates:
        candidates = _motion_candidates(motion, config)
    candidates = _merge_candidates(candidates, config)

    padded: list[Highlight] = []
    for candidate in candidates:
        start = max(0.0, candidate.start - config.pre_roll)
        end = min(duration, candidate.end + config.post_roll)
        if end <= start:
            continue
        padded.append(
            Highlight(
                start=round(start, 3),
                end=round(end, 3),
                score=round(candidate.score, 3),
                hit_count=candidate.hit_count,
                motion_score=round(candidate.motion_score, 3),
                reason=candidate.reason,
            )
        )

    ranked = sorted(padded, key=lambda item: item.score, reverse=True)[: config.max_highlights]
    ranks = {id(item): rank for rank, item in enumerate(ranked, start=1)}
    return sorted(
        (replace(item, rank=ranks[id(item)]) for item in ranked),
        key=lambda item: item.start,
    )
