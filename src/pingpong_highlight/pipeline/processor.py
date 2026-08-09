from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pingpong_highlight.config import Settings
from pingpong_highlight.pipeline.audio import analyze_audio
from pingpong_highlight.pipeline.detect import DetectionConfig, detect_highlights
from pingpong_highlight.pipeline.media import (
    MediaError,
    concatenate_clips,
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

        report(0.75, "detecting-rallies")
        highlights = detect_highlights(
            media.duration,
            audio,
            motion,
            DetectionConfig(max_highlights=self.settings.max_highlights),
        )

        files: list[dict[str, str]] = []
        clip_paths: list[Path] = []
        warnings: list[str] = []
        for index, highlight in enumerate(highlights, start=1):
            report(
                0.78 + 0.18 * ((index - 1) / max(1, len(highlights))),
                f"exporting-highlight-{index}",
            )
            filename = f"highlight_{index:03d}_rank_{highlight.rank:02d}.mp4"
            clip_path = output_dir / filename
            export_clip(source, clip_path, highlight.start, highlight.end)
            clip_paths.append(clip_path)
            files.append({"name": filename, "kind": "clip"})

        if clip_paths:
            report(0.97, "building-reel")
            reel_path = output_dir / "highlight_reel.mp4"
            manifest = output_dir / ".concat.txt"
            try:
                concatenate_clips(clip_paths, reel_path, manifest)
            except MediaError as exc:
                warnings.append(str(exc))
            else:
                files.insert(0, {"name": reel_path.name, "kind": "reel"})
            finally:
                manifest.unlink(missing_ok=True)

        result: dict[str, Any] = {
            "algorithm_version": "signal-fusion-v1",
            "source_name": source_name,
            "media": media.to_dict() | {"path": source_name},
            "summary": {
                "highlight_count": len(highlights),
                "impact_count": len(audio.events),
                "motion_sample_count": int(motion.scores.size),
                "used_motion_only_fallback": any(
                    highlight.hit_count == 0 for highlight in highlights
                ),
            },
            "highlights": [highlight.to_dict() for highlight in highlights],
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
