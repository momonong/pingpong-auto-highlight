from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from pingpong_highlight.config import Settings
from pingpong_highlight.pipeline.processor import HighlightProcessor


def _write_impacts(path: Path, duration: float = 8.0, sample_rate: int = 16_000) -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.0004, int(duration * sample_rate))
    for event_time in (1.2, 1.65, 2.1, 2.55, 3.0, 3.45, 3.9):
        length = int(0.018 * sample_rate)
        index = int(event_time * sample_rate)
        phase = np.arange(length) / sample_rate
        impact = 0.8 * np.sin(2 * np.pi * 3_200 * phase) * np.exp(-phase * 180)
        samples[index : index + length] += impact
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_end_to_end_signal_fusion_pipeline(tmp_path: Path) -> None:
    audio = tmp_path / "impacts.wav"
    source = tmp_path / "phone.mp4"
    _write_impacts(audio)
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
            "testsrc2=size=480x270:rate=30:duration=8",
            "-i",
            str(audio),
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

    settings = Settings(
        data_dir=tmp_path / "data",
        upload_token="test",
        reel_target_seconds=10.0,
    )
    settings.ensure_directories()
    output = tmp_path / "output"
    result = HighlightProcessor(settings).run(source, output)

    assert result["summary"]["impact_count"] >= 5
    assert result["summary"]["point_count"] >= 1
    assert result["editing"]["unit"] == "scored-point"
    assert result["editing"]["layout"] == "source-aspect"
    assert result["editing"]["width"] == 480
    assert result["editing"]["height"] == 270
    assert result["editing"]["clip_pre_roll_seconds"] == 1.5
    assert result["editing"]["clip_post_roll_seconds"] == 1.5
    assert result["editing"]["transition"] == "hard-cut"
    assert result["editing"]["transition_seconds"] == 0.0
    assert result["editing"]["target_reel_seconds"] == 10.0
    assert result["editing"]["final_point_fades_out"] is False
    assert (output / "best_points_reel.mp4").is_file()
    assert (output / "analysis.json").is_file()
