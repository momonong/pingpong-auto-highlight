from __future__ import annotations

import json
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


def _video_color_metadata(path: Path) -> tuple[str | None, str | None]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,color_range",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return stream.get("pix_fmt"), stream.get("color_range")


def _filtered_frame_md5(path: Path, video_filter: str) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            video_filter,
            "-frames:v",
            "1",
            "-an",
            "-f",
            "md5",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
    assert clip[clip.index("-vf") + 1] == (
        "settb=AVTB,setpts=PTS-STARTPTS,fps=30,"
        "scale=in_range=auto:out_range=tv,format=yuv420p"
    )
    assert clip[clip.index("-af") + 1] == "asetpts=PTS-STARTPTS"
    assert "-avoid_negative_ts" not in clip
    assert clip[clip.index("-movflags") + 1] == "+faststart+negative_cts_offsets"


def test_browser_output_caps_fps_and_bitrate() -> None:
    command = _point_reel_command(
        [Path("point.mp4")],
        Path("reel.mp4"),
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
        Path("social.mp4"),
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
    assert low_fps_clip[low_fps_clip.index("-vf") + 1] == (
        "settb=AVTB,setpts=PTS-STARTPTS,fps=24,"
        "scale=in_range=auto:out_range=tv,format=yuv420p"
    )


def test_hard_cut_command_maps_silent_and_single_clip_reels() -> None:
    def command(clips: list[Path], *, with_audio: bool) -> list[str]:
        return _point_reel_command(
            clips,
            Path("reel.mp4"),
            width=320,
            height=180,
            fps=30.0,
            with_audio=with_audio,
            encoder="libx264",
        )

    silent_reel = command([Path("one.mp4"), Path("two.mp4")], with_audio=False)
    silent_filter = silent_reel[silent_reel.index("-filter_complex") + 1]
    assert "[v0][v1]concat=n=2:v=1:a=0[vout]" in silent_filter
    assert silent_reel[silent_reel.index("-map") + 1] == "[vout]"
    assert "-an" in silent_reel

    single_silent = command([Path("one.mp4")], with_audio=False)
    assert "concat=" not in single_silent[single_silent.index("-filter_complex") + 1]
    assert single_silent[single_silent.index("-map") + 1] == "[v0]"
    assert "-an" in single_silent

    single_audio = command([Path("one.mp4")], with_audio=True)
    mapped_streams = [
        single_audio[index + 1] for index, value in enumerate(single_audio) if value == "-map"
    ]
    assert "concat=" not in single_audio[single_audio.index("-filter_complex") + 1]
    assert mapped_streams == ["[v0]", "[a0]"]
    assert "-an" not in single_audio


def test_point_reel_adds_silence_for_a_cross_source_clip_without_audio() -> None:
    command = _point_reel_command(
        [Path("with-audio.mp4"), Path("silent.mp4")],
        Path("reel.mp4"),
        width=1920,
        height=1080,
        fps=30.0,
        with_audio=True,
        encoder="libx264",
        audio_presence=[True, False],
        durations=[4.0, 5.5],
    )

    filter_graph = command[command.index("-filter_complex") + 1]
    assert "[0:a:0]aresample=48000" in filter_graph
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in filter_graph
    assert "atrim=duration=5.500000" in filter_graph
    assert "concat=n=2:v=1:a=1[vout][aout]" in filter_graph


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
def test_social_reel_is_vertical_and_uses_direct_cuts(tmp_path: Path) -> None:
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
        width=360,
        height=640,
        fps=24,
    )

    info = probe_media(reel)
    assert (info.width, info.height) == (360, 640)
    assert info.duration == pytest.approx(2.0, abs=0.25)
    assert info.video_codec == "h264"
    assert info.pixel_format == "yuv420p"

    command = _social_reel_command(
        [first, second],
        reel,
        width=360,
        height=640,
        fps=24,
        with_audio=True,
        encoder="libx264",
    )
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1[vout][aout]" in filter_graph
    assert "xfade" not in filter_graph
    assert "acrossfade" not in filter_graph
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
    build_point_reel([first, second], reel)

    info = probe_media(reel)
    assert (info.width, info.height) == (320, 180)
    assert info.duration == pytest.approx(2.0, abs=0.25)
    assert info.video_codec == "h264"
    assert info.pixel_format == "yuv420p"

    command = _point_reel_command(
        [first, second],
        reel,
        width=320,
        height=180,
        fps=30.0,
        with_audio=True,
        encoder="libx264",
    )
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1[vout][aout]" in filter_graph
    assert "xfade" not in filter_graph
    assert "acrossfade" not in filter_graph
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_point_reel_normalizes_large_input_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "high-offset.mp4"
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
            "testsrc2=size=320x180:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:sample_rate=48000:duration=1",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-output_ts_offset",
            "1000",
            str(source),
        ],
        check=True,
    )

    reel = tmp_path / "reel.mp4"
    build_point_reel([source, source], reel)

    info = probe_media(reel)
    assert info.duration == pytest.approx(2.0, abs=0.25)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_full_range_phone_video_is_normalized_for_browser_playback(tmp_path: Path) -> None:
    source = tmp_path / "full-range-source.mp4"
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
            "testsrc2=size=320x180:rate=30:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=900:sample_rate=48000:duration=1.2",
            "-vf",
            "setparams=range=pc",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "pc",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    assert _video_color_metadata(source)[1] == "pc"

    point = tmp_path / "point.mp4"
    reel = tmp_path / "reel.mp4"
    direct_reel = tmp_path / "direct-reel.mp4"
    social_reel = tmp_path / "social-reel.mp4"
    export_clip(source, point, 0.0, 1.0)
    build_point_reel([point], reel)
    build_point_reel([source], direct_reel)
    build_social_reel([source], social_reel, width=360, height=640, fps=24)

    for output in (point, reel, direct_reel, social_reel):
        pixel_format, color_range = _video_color_metadata(output)
        assert pixel_format == "yuv420p"
        assert color_range != "pc"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_limited_range_10_bit_video_is_not_squeezed_again(tmp_path: Path) -> None:
    source = tmp_path / "limited-10-bit.mkv"
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
            "testsrc2=size=320x180:rate=24:duration=1",
            "-vf",
            "format=yuv420p10le,setparams=range=tv",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p10le",
            "-color_range",
            "tv",
            str(source),
        ],
        check=True,
    )
    assert _video_color_metadata(source) == ("yuv420p10le", "tv")
    assert _filtered_frame_md5(source, "format=yuv420p") == _filtered_frame_md5(
        source,
        "scale=in_range=auto:out_range=tv,format=yuv420p",
    )

    reel = tmp_path / "limited-reel.mp4"
    build_point_reel([source], reel)

    pixel_format, color_range = _video_color_metadata(reel)
    assert pixel_format == "yuv420p"
    assert color_range != "pc"
