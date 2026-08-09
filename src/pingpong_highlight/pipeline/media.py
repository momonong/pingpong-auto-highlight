from __future__ import annotations

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


def open_video_decoder(path: Path, fps: float, frame_size: int) -> subprocess.Popen[bytes]:
    filter_graph = (
        f"fps={fps},"
        f"scale={frame_size}:{frame_size}:force_original_aspect_ratio=decrease,"
        f"pad={frame_size}:{frame_size}:(ow-iw)/2:(oh-ih)/2:color=black,format=gray"
    )
    return _popen_stdout(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
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
    result = _run(["ffmpeg", "-hide_banner", "-encoders"])
    return result.returncode == 0 and "h264_nvenc" in result.stdout


def _clip_command(
    source: Path,
    destination: Path,
    start: float,
    end: float,
    *,
    encoder: str,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
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
    errors: list[str] = []
    for encoder in encoders:
        result = _run(_clip_command(source, destination, start, end, encoder=encoder))
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
