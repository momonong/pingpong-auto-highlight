from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pingpong_highlight.pipeline.media import (
    _social_reel_command,
    build_social_reel,
    export_clip,
    probe_media,
)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_probe_and_frame_accurate_export(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=16000:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )

    info = probe_media(source)
    assert info.width == 320
    assert info.height == 240
    assert info.has_audio
    assert info.duration == pytest.approx(2.0, abs=0.15)

    clip = tmp_path / "clip.mp4"
    export_clip(source, clip, 0.4, 1.3)
    clipped = probe_media(clip)
    assert clipped.duration == pytest.approx(0.9, abs=0.18)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_social_reel_is_vertical_and_dissolves_only_between_points(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=2.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:sample_rate=48000:duration=2.4",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    first = tmp_path / "point-one.mp4"
    second = tmp_path / "point-two.mp4"
    export_clip(source, first, 0.0, 1.0)
    export_clip(source, second, 1.2, 2.2)

    reel = tmp_path / "reel.mp4"
    build_social_reel(
        [first, second],
        reel,
        transition_duration=0.2,
        width=360,
        height=640,
        fps=24,
    )

    info = probe_media(reel)
    assert (info.width, info.height) == (360, 640)
    assert info.duration == pytest.approx(1.8, abs=0.25)

    command = _social_reel_command(
        [first, second],
        [1.0, 1.0],
        reel,
        transition_duration=0.2,
        width=360,
        height=640,
        fps=24,
        with_audio=True,
        encoder="libx264",
    )
    filter_graph = command[command.index("-filter_complex") + 1]
    assert filter_graph.count("xfade=transition=fade") == 1
    assert "fade=t=out" not in filter_graph
