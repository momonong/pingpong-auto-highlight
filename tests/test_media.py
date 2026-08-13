from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import pingpong_highlight.pipeline.media as media_module
from pingpong_highlight.pipeline.media import (
    _clip_command,
    _point_reel_command,
    _social_reel_command,
    _streaming_video_options,
    _video_decoder_command,
    build_point_reel,
    build_social_reel,
    export_clip,
    probe_media,
)


def test_gpu_commands_request_cuda_before_video_input(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "clip.mp4"

    decoder = _video_decoder_command(source, 8.0, 320, use_nvdec=True)
    clip = _clip_command(
        source,
        destination,
        1.0,
        2.0,
        encoder="h264_nvenc",
        use_nvdec=True,
    )

    assert decoder[decoder.index("-hwaccel") + 1] == "cuda"
    assert decoder.index("-hwaccel") < decoder.index("-i")
    assert clip[clip.index("-hwaccel") + 1] == "cuda"
    assert clip.index("-hwaccel") < clip.index("-i")
    assert "h264_nvenc" in clip
    assert clip[clip.index("-vf") + 1] == "fps=30,format=yuv420p"


def test_browser_output_caps_fps_and_bitrate() -> None:
    command = _point_reel_command(
        [Path("point.mp4")],
        [2.0],
        Path("reel.mp4"),
        transition_duration=0.2,
        width=1920,
        height=1080,
        fps=120.0,
        with_audio=False,
        encoder="h264_nvenc",
    )

    assert command[command.index("-r") + 1] == "30.000000"
    assert command[command.index("-b:v") + 1] == "8M"
    assert command[command.index("-rc") + 1] == "vbr"
    assert command[command.index("-maxrate") + 1] == "12M"
    assert command[command.index("-bufsize") + 1] == "24M"
    assert command[command.index("-g") + 1] == "60"

    software = _streaming_video_options(
        "libx264",
        width=1920,
        height=1080,
        fps=30.0,
    )
    assert software[software.index("-maxrate") + 1] == "12M"
    assert "-crf" in software

    social = _social_reel_command(
        [Path("point.mp4")],
        [2.0],
        Path("social.mp4"),
        transition_duration=0.2,
        width=1080,
        height=1920,
        fps=120,
        with_audio=False,
        encoder="h264_nvenc",
    )
    social_filter = social[social.index("-filter_complex") + 1]
    assert "fps=30" in social_filter
    assert social[social.index("-r") + 1] == "30"

    low_fps_clip = _clip_command(
        Path("source.mp4"),
        Path("clip.mp4"),
        0.0,
        1.0,
        encoder="libx264",
        fps=24.0,
    )
    assert low_fps_clip[low_fps_clip.index("-vf") + 1] == "fps=24,format=yuv420p"


def test_nvenc_detection_requires_a_successful_runtime_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def failed_runtime_probe(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="h264_nvenc", stderr="missing driver")

    media_module.has_nvenc.cache_clear()
    monkeypatch.setattr(media_module.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(media_module, "_run", failed_runtime_probe)
    try:
        assert not media_module.has_nvenc()
    finally:
        media_module.has_nvenc.cache_clear()

    assert calls
    assert "-frames:v" in calls[0]
    assert "h264_nvenc" in calls[0]


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
    assert info.video_codec == "h264"
    assert info.pixel_format == "yuv420p"

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
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_point_reel_preserves_source_geometry(tmp_path: Path) -> None:
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
    build_point_reel([first, second], reel, transition_duration=0.2)

    info = probe_media(reel)
    assert (info.width, info.height) == (320, 180)
    assert info.duration == pytest.approx(1.8, abs=0.25)
    assert info.video_codec == "h264"
    assert info.pixel_format == "yuv420p"

    command = _point_reel_command(
        [first, second],
        [1.0, 1.0],
        reel,
        transition_duration=0.2,
        width=320,
        height=180,
        fps=30.0,
        with_audio=True,
        encoder="libx264",
    )
    filter_graph = command[command.index("-filter_complex") + 1]
    assert filter_graph.count("xfade=transition=fade") == 1
    assert "fade=t=out" not in filter_graph
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
