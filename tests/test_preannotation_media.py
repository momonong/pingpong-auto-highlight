from __future__ import annotations

import shutil
import subprocess

import pytest

from pingpong_highlight.pipeline.media import probe_media
from pingpong_highlight.preannotate import extract


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
def test_vfr_rotation_and_source_offset(tmp_path):
    original = tmp_path / "vfr.mp4"
    rotated = tmp_path / "rotated.mp4"
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=12:duration=5",
            "-vf",
            "select='not(eq(mod(n,3),1))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(original),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(original),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
        capture_output=True,
    )
    assert probe_media(rotated).rotation == 90
    command = extract(rotated, clip, 1000, 3000, fps=2, width=96)
    info = probe_media(clip)
    assert (info.width, info.height, info.rotation) == (96, 160, 0)
    assert abs(info.duration - 2) < 0.05
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(clip),
        ],
        text=True,
    )
    # Frame CSV avoids depending on optional SEI side-data JSON formatting.
    pts = [float(line.split(",")[0]) for line in output.splitlines() if line.strip()]
    assert pts == [0, 0.5, 1, 1.5]
    assert [round(t * 1000) + 1000 for t in pts] == [1000, 1500, 2000, 2500]
    assert command[command.index("-ss") + 1] == "1.0"
    with pytest.raises(FileExistsError):
        extract(rotated, clip, 1000, 3000)
