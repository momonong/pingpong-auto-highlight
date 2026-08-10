from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(slots=True)
class PointDetection:
    candidates: list[Point]
    points: list[Point]
