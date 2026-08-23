from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pingpong_highlight.config import Settings
from pingpong_highlight.pipeline.audio import analyze_audio
from pingpong_highlight.pipeline.detect import DetectionConfig, detect_points
from pingpong_highlight.pipeline.media import (
    export_clip,
    probe_media,
)
from pingpong_highlight.pipeline.motion import analyze_motion

ProgressCallback = Callable[[float, str], None]


class HighlightProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(
        self,
        source: Path,
        output_dir: Path,
        progress: ProgressCallback | None = None,
        *,
        source_name: str | None = None,
    ) -> dict[str, Any]:
        report = progress or (lambda _value, _stage: None)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_name = source_name or source.name

        report(0.01, "probing")
        media = probe_media(source)
        report(0.04, "audio-analysis")
        audio = analyze_audio(
            source,
            media,
            sample_rate=self.settings.audio_sample_rate,
            progress=lambda value: report(0.04 + value * 0.30, "audio-analysis"),
        )
        report(0.35, "motion-analysis")
        motion = analyze_motion(
            source,
            media,
            fps=self.settings.video_sample_fps,
            frame_size=self.settings.analysis_frame_size,
            progress=lambda value: report(0.35 + value * 0.38, "motion-analysis"),
        )

        report(0.75, "detecting-points")
        detection = detect_points(
            media.duration,
            audio,
            motion,
            DetectionConfig(
                minimum_point_score_ratio=(
                    self.settings.library_minimum_point_score_ratio
                ),
                max_points=self.settings.max_points,
                target_reel_duration=None,
                pre_roll=self.settings.clip_pre_roll_seconds,
                post_roll=self.settings.clip_post_roll_seconds,
            ),
        )
        points = detection.points
        best_candidate_score = max(
            (candidate.score for candidate in detection.candidates),
            default=0.0,
        )
        recommendation_threshold = (
            best_candidate_score * self.settings.minimum_point_score_ratio
        )
        recommended_count = sum(
            candidate.score >= recommendation_threshold
            for candidate in detection.candidates
        )
        counted_decisions = Counter(
            candidate.selection for candidate in detection.candidates
        )
        selection_counts = {
            decision: counted_decisions[decision]
            for decision in (
                "selected",
                "below-score-threshold",
                "duration-budget",
                "point-cap",
            )
        }

        files: list[dict[str, str]] = []
        warnings: list[str] = []
        for index, point in enumerate(points, start=1):
            report(
                0.78 + 0.21 * ((index - 1) / max(1, len(points))),
                f"saving-highlight-{index}",
            )
            filename = f"highlight_{index:03d}_rank_{point.rank:03d}.mp4"
            clip_path = output_dir / filename
            export_clip(source, clip_path, point.start, point.end)
            files.append({"name": filename, "kind": "highlight"})

        result: dict[str, Any] = {
            "algorithm_version": "highlight-library-v3",
            "source_name": source_name,
            "media": media.to_dict() | {"path": source_name},
            "summary": {
                "point_count": len(points),
                "candidate_point_count": len(detection.candidates),
                "eligible_candidate_count": (
                    len(detection.candidates)
                    - selection_counts["below-score-threshold"]
                ),
                "recommended_candidate_count": recommended_count,
                "impact_count": len(audio.events),
                "motion_sample_count": int(motion.scores.size),
                "library_duration": round(sum(point.duration for point in points), 3),
                "reel_duration": None,
                "used_motion_only_rescue": any(
                    candidate.impact_count == 0 for candidate in detection.candidates
                ),
            },
            "selection": {
                "policy": "relative-score-threshold",
                "library_minimum_point_score_ratio": (
                    self.settings.library_minimum_point_score_ratio
                ),
                "recommendation_score_ratio": (
                    self.settings.minimum_point_score_ratio
                ),
                "effective_score_threshold": detection.effective_score_threshold,
                "recommendation_score_threshold": round(
                    recommendation_threshold,
                    6,
                ) if best_candidate_score > 0 else None,
                "maximum_reel_seconds": None,
                "maximum_points": self.settings.max_points,
                "decision_counts": selection_counts,
            },
            "editing": {
                "unit": "scored-point",
                "layout": "reusable-source-aspect-clips",
                "width": media.width,
                "height": media.height,
                "fps": round(media.fps, 3),
                "assembly": "deferred-to-library",
                "clip_pre_roll_seconds": self.settings.clip_pre_roll_seconds,
                "clip_post_roll_seconds": self.settings.clip_post_roll_seconds,
                "target_reel_seconds": None,
            },
            "candidates": [candidate.to_dict() for candidate in detection.candidates],
            "points": [point.to_dict() for point in points],
            "warnings": warnings,
            "files": [*files, {"name": "analysis.json", "kind": "analysis"}],
        }
        analysis_path = output_dir / "analysis.json"
        analysis_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report(1.0, "completed")
        return result
