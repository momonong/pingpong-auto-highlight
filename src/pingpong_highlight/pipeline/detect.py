from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pingpong_highlight.pipeline.audio import pick_impact_events
from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    ImpactEvent,
    ImpactGroupDiagnostic,
    MotionFeatures,
    Point,
    PointCandidate,
    PointDetection,
)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Rules for turning impact evidence into individual scored points."""

    minimum_impacts: int = 3
    minimum_strong_impacts: int = 2
    weak_impact_score: float = 2.0
    strong_impact_score: float = 3.0
    minimum_impact_spacing: float = 0.16
    maximum_unverified_impact_gap: float = 1.45
    maximum_impact_gap: float = 2.75
    minimum_point_span: float = 0.5
    maximum_point_span: float = 18.0
    core_boundary_search_seconds: float = 1.0
    quiet_motion_threshold: float = 0.25
    quiet_motion_duration: float = 0.75
    quiet_group_split_duration: float = 1.5
    motion_activity_threshold: float = 0.75
    maximum_motion_gap: float = 0.85
    minimum_motion_samples: int = 5
    minimum_motion_span: float = 1.25
    minimum_isolated_motion_score: float = 2.0
    motion_attachment_tolerance: float = 0.25
    pre_roll: float = 1.5
    post_roll: float = 1.5
    minimum_point_score_ratio: float = 0.87
    max_points: int | None = None
    target_reel_duration: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_impacts <= 0 or self.minimum_strong_impacts <= 0:
            raise ValueError("impact count minimums must be positive")
        if self.minimum_strong_impacts > self.minimum_impacts:
            raise ValueError("minimum_strong_impacts cannot exceed minimum_impacts")
        if self.weak_impact_score < 0 or self.strong_impact_score < 0:
            raise ValueError("impact score thresholds must be non-negative")
        if self.weak_impact_score > self.strong_impact_score:
            raise ValueError("weak_impact_score cannot exceed strong_impact_score")
        if (
            self.minimum_impact_spacing <= 0
            or self.maximum_unverified_impact_gap <= 0
            or self.maximum_impact_gap <= 0
        ):
            raise ValueError("impact spacing and gap must be positive")
        if self.maximum_unverified_impact_gap > self.maximum_impact_gap:
            raise ValueError(
                "maximum_unverified_impact_gap cannot exceed maximum_impact_gap"
            )
        if self.maximum_point_span <= 0 or self.minimum_point_span < 0:
            raise ValueError("point spans must be non-negative and ordered")
        if self.minimum_point_span > self.maximum_point_span:
            raise ValueError("minimum_point_span cannot exceed maximum_point_span")
        if (
            self.core_boundary_search_seconds < 0
            or self.quiet_motion_duration <= 0
            or self.quiet_group_split_duration <= 0
        ):
            raise ValueError("motion boundary durations are invalid")
        if self.minimum_motion_samples <= 0 or self.minimum_motion_span < 0:
            raise ValueError("motion candidate minimums are invalid")
        if (
            self.quiet_motion_threshold < 0
            or self.motion_activity_threshold < 0
            or self.maximum_motion_gap <= 0
            or self.minimum_isolated_motion_score < 0
            or self.motion_attachment_tolerance < 0
        ):
            raise ValueError("motion score thresholds and gaps are invalid")
        if self.pre_roll < 0 or self.post_roll < 0:
            raise ValueError("clip padding must be non-negative")
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


def _has_sustained_quiet_motion(
    motion: MotionFeatures,
    start: float,
    end: float,
    config: DetectionConfig,
) -> bool:
    """Use sustained quiet as negative evidence that two impact bursts are separate points."""

    if motion.times.size == 0:
        return True
    indices = np.flatnonzero((motion.times >= start) & (motion.times <= end))
    if indices.size == 0:
        return True
    times = motion.times[indices]
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    sample_period = float(np.median(positive_steps)) if positive_steps.size else 0.125
    required = max(1, int(np.ceil(config.quiet_group_split_duration / sample_period)))
    longest = 0
    current = 0
    for value in motion.scores[indices]:
        current = current + 1 if value < config.quiet_motion_threshold else 0
        longest = max(longest, current)
    return longest >= required


def _group_impacts(
    events: list[ImpactEvent],
    config: DetectionConfig,
    motion: MotionFeatures | None = None,
) -> list[list[ImpactEvent]]:
    if not events:
        return []
    groups: list[list[ImpactEvent]] = []
    current = [events[0]]
    for event in events[1:]:
        gap = event.time - current[-1].time
        span = event.time - current[0].time
        ambiguous_gap_is_quiet = (
            gap > config.maximum_unverified_impact_gap
            and _has_sustained_quiet_motion(
                motion or MotionFeatures.empty(),
                current[-1].time,
                event.time,
                config,
            )
        )
        if (
            gap > config.maximum_impact_gap
            or span > config.maximum_point_span
            or ambiguous_gap_is_quiet
        ):
            groups.append(current)
            current = [event]
        else:
            current.append(event)
    groups.append(current)
    return groups


def _tiered_impacts(
    audio: AudioFeatures,
    config: DetectionConfig,
) -> tuple[list[ImpactEvent], set[float]]:
    """Return weak impact peaks plus the exact timestamps that also pass the strong tier."""

    if audio.times.size and audio.scores.size:
        weak = pick_impact_events(
            audio.times,
            audio.scores,
            minimum_score=config.weak_impact_score,
            minimum_spacing=config.minimum_impact_spacing,
        )
        strong = pick_impact_events(
            audio.times,
            audio.scores,
            minimum_score=config.strong_impact_score,
            minimum_spacing=config.minimum_impact_spacing,
        )
        return weak, {event.time for event in strong}

    # Hand-authored/synthetic AudioFeatures predate raw signal persistence. Treat
    # their events as accepted strong peaks so this postprocessor remains usable.
    return list(audio.events), {event.time for event in audio.events}


def _quiet_run_boundary(
    motion: MotionFeatures,
    anchor: float,
    *,
    direction: int,
    duration: float,
    config: DetectionConfig,
) -> tuple[float, str]:
    search = config.core_boundary_search_seconds
    cap = min(duration, anchor + direction * search)
    lower, upper = sorted((anchor, cap))
    if motion.times.size == 0 or search == 0:
        return max(0.0, cap), "search-cap-no-motion"

    indices = np.flatnonzero((motion.times >= lower) & (motion.times <= upper))
    if indices.size == 0:
        return max(0.0, cap), "search-cap-no-samples"
    times = motion.times[indices]
    scores = motion.scores[indices]
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    sample_period = float(np.median(positive_steps)) if positive_steps.size else 0.125
    required = max(1, int(np.ceil(config.quiet_motion_duration / sample_period)))
    quiet = scores < config.quiet_motion_threshold

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_quiet in enumerate(quiet):
        if is_quiet and run_start is None:
            run_start = index
        if run_start is not None and (not is_quiet or index == len(quiet) - 1):
            run_end = index if is_quiet and index == len(quiet) - 1 else index - 1
            if run_end - run_start + 1 >= required:
                runs.append((run_start, run_end))
            run_start = None

    if not runs:
        return max(0.0, cap), "search-cap-active-motion"
    if direction < 0:
        _run_start, run_end = runs[-1]
        return max(0.0, float(times[run_end])), "quiet-run"
    run_start, _run_end = runs[0]
    return min(duration, float(times[run_start])), "quiet-run"


def _audio_candidates(
    duration: float,
    audio: AudioFeatures,
    motion: MotionFeatures,
    config: DetectionConfig,
) -> tuple[list[PointCandidate], list[ImpactGroupDiagnostic]]:
    candidates: list[PointCandidate] = []
    diagnostics: list[ImpactGroupDiagnostic] = []
    weak_events, strong_times = _tiered_impacts(audio, config)
    for group in _group_impacts(weak_events, config, motion):
        impact_times = tuple(event.time for event in group)
        impact_strengths = tuple(event.strength for event in group)
        strong_count = sum(event.time in strong_times for event in group)
        if len(group) < config.minimum_impacts:
            diagnostics.append(
                ImpactGroupDiagnostic(
                    start=group[0].time,
                    end=group[-1].time,
                    impact_times=impact_times,
                    impact_strengths=impact_strengths,
                    accepted=False,
                    decision="too-few-impacts",
                    strong_impact_count=strong_count,
                )
            )
            continue
        if strong_count < config.minimum_strong_impacts:
            diagnostics.append(
                ImpactGroupDiagnostic(
                    start=group[0].time,
                    end=group[-1].time,
                    impact_times=impact_times,
                    impact_strengths=impact_strengths,
                    accepted=False,
                    decision="too-few-strong-impacts",
                    strong_impact_count=strong_count,
                )
            )
            continue
        span = group[-1].time - group[0].time
        if span < config.minimum_point_span:
            diagnostics.append(
                ImpactGroupDiagnostic(
                    start=group[0].time,
                    end=group[-1].time,
                    impact_times=impact_times,
                    impact_strengths=impact_strengths,
                    accepted=False,
                    decision="span-too-short",
                    strong_impact_count=strong_count,
                )
            )
            continue

        core_start, start_reason = _quiet_run_boundary(
            motion,
            group[0].time,
            direction=-1,
            duration=duration,
            config=config,
        )
        core_end, end_reason = _quiet_run_boundary(
            motion,
            group[-1].time,
            direction=1,
            duration=duration,
            config=config,
        )
        boundary_reason = f"start:{start_reason};end:{end_reason}"

        gaps = np.diff([event.time for event in group])
        rhythmic = float(np.mean((gaps >= 0.16) & (gaps <= 1.05))) if gaps.size else 0.0
        motion_score = _motion_level(motion, group[0].time - 0.4, group[-1].time + 0.4)
        tempo = (len(group) - 1) / max(span, 0.3)
        impact_strength = float(np.mean([event.strength for event in group]))
        score_components = (
            ("impact_count", float(3.4 * np.log1p(len(group)))),
            ("tempo", float(1.25 * min(tempo, 3.5))),
            ("motion", float(1.7 * min(motion_score, 4.0))),
            ("rhythmicity", float(1.2 * rhythmic)),
            ("impact_strength", float(0.7 * impact_strength)),
            ("span", float(0.08 * min(span, 12.0))),
        )
        score = sum(value for _name, value in score_components)
        candidates.append(
            PointCandidate(
                start=core_start,
                end=core_end,
                score=float(score),
                impact_count=len(group),
                motion_score=motion_score,
                reason=(
                    f"{len(group)} weak and {strong_count} strong impact transients "
                    "within one point"
                ),
                origin="audio",
                impact_times=impact_times,
                impact_strengths=impact_strengths,
                tempo=tempo,
                rhythmic_fraction=rhythmic,
                mean_impact_strength=impact_strength,
                score_components=score_components,
                strong_impact_count=strong_count,
                core_boundary_reason=boundary_reason,
            )
        )
        diagnostics.append(
            ImpactGroupDiagnostic(
                start=group[0].time,
                end=group[-1].time,
                impact_times=impact_times,
                impact_strengths=impact_strengths,
                accepted=True,
                decision="candidate",
                strong_impact_count=strong_count,
                core_start=core_start,
                core_end=core_end,
                core_boundary_reason=boundary_reason,
            )
        )
    return candidates, diagnostics


def _motion_candidates(
    motion: MotionFeatures,
    config: DetectionConfig,
) -> list[PointCandidate]:
    if motion.scores.size == 0:
        return []
    active_indices = np.flatnonzero(motion.scores >= config.motion_activity_threshold)
    if active_indices.size == 0:
        return []

    groups: list[list[int]] = [[int(active_indices[0])]]
    for index in active_indices[1:]:
        group = groups[-1]
        gap = motion.times[index] - motion.times[group[-1]]
        span = motion.times[index] - motion.times[group[0]]
        if gap > config.maximum_motion_gap or span > config.maximum_point_span:
            groups.append([int(index)])
        else:
            group.append(int(index))

    candidates: list[PointCandidate] = []
    for group in groups:
        start = float(motion.times[group[0]])
        end = float(motion.times[group[-1]])
        if (
            len(group) < config.minimum_motion_samples
            or end - start < config.minimum_motion_span
        ):
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
                reason="sustained localized play motion",
                origin="motion",
                score_components=(
                    ("span", float(2.3 * np.log1p(end - start))),
                    ("motion", float(2.0 * min(motion_score, 6.0))),
                ),
            )
        )
    return candidates


def _interval_gap(left: PointCandidate, right: PointCandidate) -> float:
    if left.end < right.start:
        return right.start - left.end
    if right.end < left.start:
        return left.start - right.end
    return 0.0


def _deconflict_audio_boundaries(
    candidates: list[PointCandidate],
    motion: MotionFeatures,
) -> list[PointCandidate]:
    """Give neighbouring audio cores one shared boundary after motion expansion."""

    ordered = sorted(candidates, key=lambda candidate: (candidate.start, candidate.end))
    deconflicted: list[PointCandidate] = []
    for candidate in ordered:
        if not deconflicted or deconflicted[-1].end <= candidate.start:
            deconflicted.append(candidate)
            continue

        left = deconflicted[-1]
        raw_left_end = left.impact_times[-1] if left.impact_times else left.end
        raw_right_start = candidate.impact_times[0] if candidate.impact_times else candidate.start
        lower, upper = sorted((raw_left_end, raw_right_start))
        indices = np.flatnonzero((motion.times >= lower) & (motion.times <= upper))
        if indices.size:
            local_scores = motion.scores[indices]
            divider = float(motion.times[indices[int(np.argmin(local_scores))]])
        else:
            divider = (raw_left_end + raw_right_start) / 2
        divider = min(max(divider, left.start), candidate.end)
        deconflicted[-1] = replace(
            left,
            end=divider,
            core_boundary_reason=f"{left.core_boundary_reason};end:shared-divider",
        )
        deconflicted.append(
            replace(
                candidate,
                start=divider,
                core_boundary_reason=(
                    f"{candidate.core_boundary_reason};start:shared-divider"
                ),
            )
        )
    return deconflicted


def _sync_audio_group_boundaries(
    diagnostics: list[ImpactGroupDiagnostic],
    candidates: list[PointCandidate],
) -> list[ImpactGroupDiagnostic]:
    resolved = {candidate.impact_times: candidate for candidate in candidates}
    return [
        replace(
            diagnostic,
            core_start=resolved[diagnostic.impact_times].start,
            core_end=resolved[diagnostic.impact_times].end,
            core_boundary_reason=resolved[diagnostic.impact_times].core_boundary_reason,
        )
        if diagnostic.accepted and diagnostic.impact_times in resolved
        else diagnostic
        for diagnostic in diagnostics
    ]


def _fuse_candidates(
    audio_candidates: list[PointCandidate],
    motion_candidates: list[PointCandidate],
    config: DetectionConfig,
) -> list[PointCandidate]:
    attachments: dict[int, list[PointCandidate]] = {
        index: [] for index in range(len(audio_candidates))
    }
    isolated: list[PointCandidate] = []
    for motion_candidate in motion_candidates:
        compatible = [
            index
            for index, audio_candidate in enumerate(audio_candidates)
            if _interval_gap(audio_candidate, motion_candidate)
            <= config.motion_attachment_tolerance
        ]
        if compatible:
            best = max(
                compatible,
                key=lambda index: (
                    max(
                        0.0,
                        min(audio_candidates[index].end, motion_candidate.end)
                        - max(audio_candidates[index].start, motion_candidate.start),
                    ),
                    -_interval_gap(audio_candidates[index], motion_candidate),
                    -abs(
                        (audio_candidates[index].start + audio_candidates[index].end)
                        - (motion_candidate.start + motion_candidate.end)
                    ),
                    -index,
                ),
            )
            attachments[best].append(motion_candidate)
        elif motion_candidate.motion_score >= config.minimum_isolated_motion_score:
            isolated.append(motion_candidate)

    fused: list[PointCandidate] = []
    for index, candidate in enumerate(audio_candidates):
        attached = sorted(attachments[index], key=lambda item: (item.start, item.end))
        if not attached:
            fused.append(candidate)
            continue
        fused.append(
            replace(
                candidate,
                origin="audio-motion",
                attached_motion_intervals=tuple((item.start, item.end) for item in attached),
                attached_motion_score=max(item.motion_score for item in attached),
            )
        )
    return sorted(
        [*fused, *isolated],
        key=lambda candidate: (candidate.start, candidate.end, candidate.origin),
    )


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
        (index for index, candidate in enumerate(candidates) if candidate.score >= score_threshold),
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
    audio_candidates, audio_groups = _audio_candidates(duration, audio, motion, config)
    audio_candidates = _deconflict_audio_boundaries(audio_candidates, motion)
    audio_groups = _sync_audio_group_boundaries(audio_groups, audio_candidates)
    motion_candidates = _motion_candidates(motion, config)
    raw_candidates = _fuse_candidates(audio_candidates, motion_candidates, config)
    origins = {candidate.origin for candidate in raw_candidates}
    has_audio = any(origin.startswith("audio") for origin in origins)
    has_motion = "motion" in origins or "audio-motion" in origins
    if has_audio and has_motion:
        candidate_mode = "audio-motion"
    elif has_audio:
        candidate_mode = "audio"
    elif "motion" in origins:
        candidate_mode = "motion"
    else:
        candidate_mode = "none"
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
        audio_groups=audio_groups,
        motion_candidates=motion_candidates,
        candidate_mode=candidate_mode,
    )
