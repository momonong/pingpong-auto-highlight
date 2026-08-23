from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    has_audio: bool
    audio_codec: str | None = None
    rotation: int = 0
    video_profile: str | None = None
    pixel_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True, slots=True)
class ImpactEvent:
    time: float
    strength: float


@dataclass(slots=True)
class AudioFeatures:
    times: np.ndarray
    scores: np.ndarray
    events: list[ImpactEvent]

    @classmethod
    def empty(cls) -> AudioFeatures:
        return cls(np.empty(0), np.empty(0), [])


@dataclass(slots=True)
class MotionFeatures:
    times: np.ndarray
    scores: np.ndarray

    @classmethod
    def empty(cls) -> MotionFeatures:
        return cls(np.empty(0), np.empty(0))


@dataclass(frozen=True, slots=True)
class Point:
    start: float
    end: float
    score: float
    impact_count: int
    motion_score: float
    rank: int = 0
    reason: str = ""
    rally_start: float | None = None
    rally_end: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        rally_start = self.start if self.rally_start is None else self.rally_start
        rally_end = self.end if self.rally_end is None else self.rally_end
        return asdict(self) | {
            "clip_start": self.start,
            "clip_end": self.end,
            "rally_start": rally_start,
            "rally_end": rally_end,
            "duration": round(self.duration, 3),
            "rally_duration": round(rally_end - rally_start, 3),
            "pre_context_seconds": round(rally_start - self.start, 3),
            "post_context_seconds": round(self.end - rally_end, 3),
        }


@dataclass(frozen=True, slots=True)
class PointCandidate:
    start: float
    end: float
    score: float
    impact_count: int
    motion_score: float
    reason: str
    selection: str = "candidate"
    rank: int | None = None
    origin: str = "audio"
    impact_times: tuple[float, ...] = ()
    impact_strengths: tuple[float, ...] = ()
    tempo: float = 0.0
    rhythmic_fraction: float = 0.0
    mean_impact_strength: float = 0.0
    score_components: tuple[tuple[str, float], ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "rally_start": round(self.start, 3),
            "rally_end": round(self.end, 3),
            "rally_duration": round(self.duration, 3),
            "score": round(self.score, 6),
            "impact_count": self.impact_count,
            "motion_score": round(self.motion_score, 6),
            "reason": self.reason,
            "selection": self.selection,
            "rank": self.rank,
            "origin": self.origin,
            "impact_times": [round(value, 6) for value in self.impact_times],
            "impact_strengths": [round(value, 6) for value in self.impact_strengths],
            "tempo": round(self.tempo, 6),
            "rhythmic_fraction": round(self.rhythmic_fraction, 6),
            "mean_impact_strength": round(self.mean_impact_strength, 6),
            "score_components": {name: round(value, 6) for name, value in self.score_components},
        }


@dataclass(frozen=True, slots=True)
class ImpactGroupDiagnostic:
    start: float
    end: float
    impact_times: tuple[float, ...]
    impact_strengths: tuple[float, ...]
    accepted: bool
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "span": round(self.end - self.start, 6),
            "impact_count": len(self.impact_times),
            "impact_times": [round(value, 6) for value in self.impact_times],
            "impact_strengths": [round(value, 6) for value in self.impact_strengths],
            "accepted": self.accepted,
            "decision": self.decision,
        }


@dataclass(slots=True)
class PointDetection:
    candidates: list[PointCandidate]
    points: list[Point]
    effective_score_threshold: float | None = None
    audio_groups: list[ImpactGroupDiagnostic] = field(default_factory=list)
    motion_candidates: list[PointCandidate] = field(default_factory=list)
    candidate_mode: str = "none"
