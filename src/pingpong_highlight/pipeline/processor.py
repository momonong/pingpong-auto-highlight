from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pingpong_highlight.config import Settings
from pingpong_highlight.pipeline.audio import analyze_audio
from pingpong_highlight.pipeline.detect import DetectionConfig, detect_points
from pingpong_highlight.pipeline.media import (
    MediaError,
    build_point_reel,
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
                max_points=self.settings.max_points,
                target_reel_duration=self.settings.reel_target_seconds,
                transition_duration=self.settings.reel_transition_seconds,
                pre_roll=self.settings.clip_pre_roll_seconds,
                post_roll=self.settings.clip_post_roll_seconds,
            ),
        )
        points = detection.points

        files: list[dict[str, str]] = []
        clip_paths: list[Path] = []
        warnings: list[str] = []
        for index, point in enumerate(points, start=1):
            report(
                0.78 + 0.16 * ((index - 1) / max(1, len(points))),
                f"exporting-point-{index}",
            )
            filename = f"point_{index:03d}_rank_{point.rank:02d}.mp4"
            clip_path = output_dir / filename
            export_clip(source, clip_path, point.start, point.end)
            clip_paths.append(clip_path)
            files.append({"name": filename, "kind": "point"})

        reel_duration: float | None = None
        reel_media = None
        if clip_paths:
            report(0.95, "editing-point-reel")
            reel_path = output_dir / "best_points_reel.mp4"
            try:
                build_point_reel(
                    clip_paths,
                    reel_path,
                    transition_duration=self.settings.reel_transition_seconds,
                )
            except MediaError as exc:
                warnings.append(str(exc))
            else:
                files.insert(0, {"name": reel_path.name, "kind": "reel"})
                reel_media = probe_media(reel_path)
                reel_duration = reel_media.duration

        result: dict[str, Any] = {
            "algorithm_version": "point-reel-v3",
            "source_name": source_name,
            "media": media.to_dict() | {"path": source_name},
            "summary": {
                "point_count": len(points),
                "candidate_point_count": len(detection.candidates),
                "impact_count": len(audio.events),
                "motion_sample_count": int(motion.scores.size),
                "reel_duration": round(reel_duration, 3) if reel_duration is not None else None,
                "used_motion_only_fallback": any(point.impact_count == 0 for point in points),
            },
            "editing": {
                "unit": "scored-point",
                "layout": "source-aspect",
                "width": reel_media.width if reel_media is not None else media.width,
                "height": reel_media.height if reel_media is not None else media.height,
                "fps": round(reel_media.fps, 3) if reel_media is not None else round(media.fps, 3),
                "transition": "cross-dissolve",
                "transition_seconds": self.settings.reel_transition_seconds,
                "clip_pre_roll_seconds": self.settings.clip_pre_roll_seconds,
                "clip_post_roll_seconds": self.settings.clip_post_roll_seconds,
                "target_reel_seconds": self.settings.reel_target_seconds,
                "final_point_fades_out": False,
            },
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
