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

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"duration": round(self.duration, 3)}


@dataclass(slots=True)
class PointDetection:
    candidates: list[Point]
    points: list[Point]
