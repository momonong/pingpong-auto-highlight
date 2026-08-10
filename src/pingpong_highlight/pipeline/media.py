from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from pingpong_highlight.pipeline.models import MediaInfo


class MediaError(RuntimeError):
    pass


def _startup_info() -> tuple[int, subprocess.STARTUPINFO | None]:
    if os.name != "nt":
        return 0, None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return subprocess.CREATE_NO_WINDOW, info


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    flags, startupinfo = _startup_info()
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=flags,
        startupinfo=startupinfo,
    )


def require_media_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise MediaError(f"Missing required media tools: {', '.join(missing)}")


def _parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_media(path: Path) -> MediaInfo:
    require_media_tools()
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe could not read {path.name}: {result.stderr.strip()}")

    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
    except (json.JSONDecodeError, StopIteration) as exc:
        raise MediaError(f"No readable video stream in {path.name}") from exc

    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    format_duration = payload.get("format", {}).get("duration")
    stream_duration = video.get("duration")
    try:
        duration = float(format_duration or stream_duration)
    except (TypeError, ValueError) as exc:
        raise MediaError(f"Video duration is unavailable for {path.name}") from exc
    if duration <= 0:
        raise MediaError(f"Video duration must be positive for {path.name}")

    rotation = 0
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = round(float(side_data["rotation"])) % 360
            break
    if not rotation:
        try:
            rotation = round(float(video.get("tags", {}).get("rotate", 0))) % 360
        except (TypeError, ValueError):
            rotation = 0

    return MediaInfo(
        path=path.resolve(),
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        video_codec=str(video.get("codec_name") or "unknown"),
        has_audio=audio is not None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        rotation=rotation,
        video_profile=str(video.get("profile")) if video.get("profile") else None,
        pixel_format=str(video.get("pix_fmt")) if video.get("pix_fmt") else None,
    )


def _popen_stdout(command: list[str]) -> subprocess.Popen[bytes]:
    flags, startupinfo = _startup_info()
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
        startupinfo=startupinfo,
    )


def open_audio_decoder(path: Path, sample_rate: int) -> subprocess.Popen[bytes]:
    return _popen_stdout(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )


def _video_decoder_command(
    path: Path,
    fps: float,
    frame_size: int,
    *,
    use_nvdec: bool,
) -> list[str]:
    filter_graph = (
        f"fps={fps},"
        f"scale={frame_size}:{frame_size}:force_original_aspect_ratio=decrease,"
        f"pad={frame_size}:{frame_size}:(ow-iw)/2:(oh-ih)/2:color=black,format=gray"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if use_nvdec:
        command.extend(["-hwaccel", "cuda"])
    command.extend(
        [
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            filter_graph,
            "-fps_mode",
            "cfr",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    return command


def open_video_decoder(
    path: Path,
    fps: float,
    frame_size: int,
    *,
    use_nvdec: bool = False,
) -> subprocess.Popen[bytes]:
    return _popen_stdout(
        _video_decoder_command(path, fps, frame_size, use_nvdec=use_nvdec)
    )


def finish_decoder(process: subprocess.Popen[bytes], label: str) -> None:
    if process.stdout:
        process.stdout.close()
    stderr = ""
    if process.stderr:
        try:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
        finally:
            process.stderr.close()
    return_code = process.wait()
    if return_code != 0:
        raise MediaError(f"FFmpeg {label} decoding failed: {stderr.strip()}")


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@lru_cache(maxsize=1)
def has_nvenc() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x320:d=0.04",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
    )
    return result.returncode == 0


def _nvidia_library_available(names: tuple[str, ...]) -> bool:
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        return True
    return False


@lru_cache(maxsize=1)
def has_nvdec() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    library_names = ("nvcuvid.dll",) if os.name == "nt" else ("libnvcuvid.so.1",)
    if not _nvidia_library_available(library_names):
        return False
    result = _run(["ffmpeg", "-hide_banner", "-hwaccels"])
    accelerators = {line.strip().lower() for line in result.stdout.splitlines()}
    return result.returncode == 0 and "cuda" in accelerators


def _clip_command(
    source: Path,
    destination: Path,
    start: float,
    end: float,
    *,
    encoder: str,
    use_nvdec: bool = False,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if use_nvdec:
        command.extend(["-hwaccel", "cuda"])
    command.extend(
        [
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-sn",
            "-dn",
            "-vf",
            "format=yuv420p",
        ]
    )
    if encoder == "h264_nvenc":
        command.extend(["-c:v", encoder, "-preset", "p5", "-cq", "21", "-b:v", "0"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "20"])
    command.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(destination),
        ]
    )
    return command


def export_clip(source: Path, destination: Path, start: float, end: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoders = ["h264_nvenc", "libx264"] if has_nvenc() else ["libx264"]
    nvdec_available = has_nvdec()
    errors: list[str] = []
    for encoder in encoders:
        result = _run(
            _clip_command(
                source,
                destination,
                start,
                end,
                encoder=encoder,
                use_nvdec=encoder == "h264_nvenc" and nvdec_available,
            )
        )
        if result.returncode == 0:
            return
        destination.unlink(missing_ok=True)
        errors.append(f"{encoder}: {result.stderr.strip()}")
    raise MediaError("Could not export highlight clip. " + " | ".join(errors))


def concatenate_clips(clips: list[Path], destination: Path, manifest: Path) -> None:
    if not clips:
        return
    lines = []
    for clip in clips:
        escaped = clip.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise MediaError(f"Could not build highlight reel: {result.stderr.strip()}")


def _point_reel_command(
    clips: list[Path],
    durations: list[float],
    destination: Path,
    *,
    transition_duration: float,
    width: int,
    height: int,
    fps: float,
    with_audio: bool,
    encoder: str,
    use_nvdec: bool = False,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        if use_nvdec:
            command.extend(["-hwaccel", "cuda"])
        command.extend(["-i", str(clip)])

    filters: list[str] = []
    fps_value = f"{fps:.6f}"
    for index in range(len(clips)):
        filters.append(
            f"[{index}:v:0]fps={fps_value},settb=AVTB,setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,format=yuv420p[v{index}]"
        )
        if with_audio:
            filters.append(
                f"[{index}:a:0]aresample=48000,"
                "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )

    video_label = "v0"
    audio_label = "a0" if with_audio else None
    cumulative_duration = durations[0]
    for index in range(1, len(clips)):
        dissolve = min(
            transition_duration,
            durations[index - 1] / 4,
            durations[index] / 4,
        )
        offset = cumulative_duration - dissolve
        next_video = f"vx{index}"
        filters.append(
            f"[{video_label}][v{index}]xfade=transition=fade:"
            f"duration={dissolve:.6f}:offset={offset:.6f}[{next_video}]"
        )
        video_label = next_video
        cumulative_duration += durations[index] - dissolve
        if with_audio and audio_label is not None:
            next_audio = f"ax{index}"
            filters.append(
                f"[{audio_label}][a{index}]acrossfade=d={dissolve:.6f}:"
                f"c1=tri:c2=tri[{next_audio}]"
            )
            audio_label = next_audio

    command.extend(["-filter_complex", ";".join(filters), "-map", f"[{video_label}]"])
    if with_audio and audio_label is not None:
        command.extend(["-map", f"[{audio_label}]"])
    else:
        command.append("-an")

    if encoder == "h264_nvenc":
        command.extend(["-c:v", encoder, "-preset", "p5", "-cq", "21", "-b:v", "0"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "20"])
    command.extend(["-pix_fmt", "yuv420p"])
    if with_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(
        [
            "-r",
            fps_value,
            "-movflags",
            "+faststart",
            "-shortest",
            str(destination),
        ]
    )
    return command


def _validate_browser_compatible_reel(path: Path) -> None:
    media = probe_media(path)
    if media.video_codec != "h264" or media.pixel_format != "yuv420p":
        raise MediaError(
            "Reel is not browser-compatible H.264 yuv420p "
            f"(got {media.video_codec} {media.pixel_format or 'unknown pixel format'})"
        )


def build_point_reel(
    clips: list[Path],
    destination: Path,
    *,
    transition_duration: float = 0.35,
) -> None:
    """Build a source-aspect point montage with cross-dissolves between points."""
    if not clips:
        return
    if transition_duration < 0:
        raise ValueError("Transition duration cannot be negative")

    media = [probe_media(clip) for clip in clips]
    durations = [item.duration for item in media]
    first = media[0]
    width = first.width - first.width % 2
    height = first.height - first.height % 2
    fps = first.fps if first.fps > 0 else 30.0
    if width <= 0 or height <= 0:
        raise MediaError("Point clips do not have valid output dimensions")
    with_audio = all(item.has_audio for item in media)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoders = ["h264_nvenc", "libx264"] if has_nvenc() else ["libx264"]
    nvdec_available = has_nvdec()
    errors: list[str] = []
    for encoder in encoders:
        command = _point_reel_command(
            clips,
            durations,
            destination,
            transition_duration=transition_duration,
            width=width,
            height=height,
            fps=fps,
            with_audio=with_audio,
            encoder=encoder,
            use_nvdec=encoder == "h264_nvenc" and nvdec_available,
        )
        result = _run(command)
        if result.returncode == 0:
            try:
                _validate_browser_compatible_reel(destination)
            except MediaError as exc:
                errors.append(f"{encoder}: {exc}")
                destination.unlink(missing_ok=True)
                continue
            return
        destination.unlink(missing_ok=True)
        errors.append(f"{encoder}: {result.stderr.strip()}")
    raise MediaError("Could not build point reel. " + " | ".join(errors))


def _social_reel_command(
    clips: list[Path],
    durations: list[float],
    destination: Path,
    *,
    transition_duration: float,
    width: int,
    height: int,
    fps: int,
    with_audio: bool,
    encoder: str,
    use_nvdec: bool = False,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        if use_nvdec:
            command.extend(["-hwaccel", "cuda"])
        command.extend(["-i", str(clip)])

    filters: list[str] = []
    background_width = max(2, width // 2 - (width // 2) % 2)
    background_height = max(2, height // 2 - (height // 2) % 2)
    for index in range(len(clips)):
        filters.extend(
            [
                (
                    f"[{index}:v:0]fps={fps},settb=AVTB,setpts=PTS-STARTPTS,"
                    f"split=2[bgsrc{index}][fgsrc{index}]"
                ),
                (
                    f"[bgsrc{index}]scale={background_width}:{background_height}:"
                    "force_original_aspect_ratio=increase,"
                    f"crop={background_width}:{background_height},boxblur=18:2,"
                    f"scale={width}:{height},eq=brightness=-0.18:saturation=0.75[bg{index}]"
                ),
                (
                    f"[fgsrc{index}]scale={width}:{height}:"
                    f"force_original_aspect_ratio=decrease,setsar=1[fg{index}]"
                ),
                (
                    f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2:shortest=1,"
                    f"format=yuv420p[v{index}]"
                ),
            ]
        )
        if with_audio:
            filters.append(
                f"[{index}:a:0]aresample=48000,"
                "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )

    video_label = "v0"
    audio_label = "a0" if with_audio else None
    cumulative_duration = durations[0]
    for index in range(1, len(clips)):
        dissolve = min(
            transition_duration,
            durations[index - 1] / 4,
            durations[index] / 4,
        )
        offset = cumulative_duration - dissolve
        next_video = f"vx{index}"
        filters.append(
            f"[{video_label}][v{index}]xfade=transition=fade:"
            f"duration={dissolve:.6f}:offset={offset:.6f}[{next_video}]"
        )
        video_label = next_video
        cumulative_duration += durations[index] - dissolve
        if with_audio and audio_label is not None:
            next_audio = f"ax{index}"
            filters.append(
                f"[{audio_label}][a{index}]acrossfade=d={dissolve:.6f}:"
                f"c1=tri:c2=tri[{next_audio}]"
            )
            audio_label = next_audio

    command.extend(["-filter_complex", ";".join(filters), "-map", f"[{video_label}]"])
    if with_audio and audio_label is not None:
        command.extend(["-map", f"[{audio_label}]"])
    else:
        command.append("-an")

    if encoder == "h264_nvenc":
        command.extend(["-c:v", encoder, "-preset", "p5", "-cq", "21", "-b:v", "0"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "20"])
    command.extend(["-pix_fmt", "yuv420p"])
    if with_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(
        [
            "-r",
            str(fps),
            "-movflags",
            "+faststart",
            "-shortest",
            str(destination),
        ]
    )
    return command


def build_social_reel(
    clips: list[Path],
    destination: Path,
    *,
    transition_duration: float = 0.35,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> None:
    """Build a vertical point montage with cross-dissolves between points.

    The final point has no fade-out: each dissolve is placed only at a boundary
    where a following point exists.
    """
    if not clips:
        return
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Reel dimensions must be positive even numbers")
    if fps <= 0:
        raise ValueError("Reel fps must be positive")
    if transition_duration < 0:
        raise ValueError("Transition duration cannot be negative")

    media = [probe_media(clip) for clip in clips]
    durations = [item.duration for item in media]
    with_audio = all(item.has_audio for item in media)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoders = ["h264_nvenc", "libx264"] if has_nvenc() else ["libx264"]
    nvdec_available = has_nvdec()
    errors: list[str] = []
    for encoder in encoders:
        command = _social_reel_command(
            clips,
            durations,
            destination,
            transition_duration=transition_duration,
            width=width,
            height=height,
            fps=fps,
            with_audio=with_audio,
            encoder=encoder,
            use_nvdec=encoder == "h264_nvenc" and nvdec_available,
        )
        result = _run(command)
        if result.returncode == 0:
            try:
                _validate_browser_compatible_reel(destination)
            except MediaError as exc:
                errors.append(f"{encoder}: {exc}")
                destination.unlink(missing_ok=True)
                continue
            return
        destination.unlink(missing_ok=True)
        errors.append(f"{encoder}: {result.stderr.strip()}")
    raise MediaError("Could not build social reel. " + " | ".join(errors))
