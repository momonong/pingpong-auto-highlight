from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pingpong_highlight.pipeline.media import export_clip, probe_media


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
