from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    upload_token: str
    public_url: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_bytes: int = 100 * 1024**3
    max_chunk_bytes: int = 32 * 1024**2
    download_min_free_bytes: int = 2 * 1024**3
    video_sample_fps: float = 8.0
    analysis_frame_size: int = 320
    audio_sample_rate: int = 16_000
    max_points: int = 6
    reel_target_seconds: float = 55.0
    reel_transition_seconds: float = 0.35
    worker_count: int = 1

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def drive_imports_dir(self) -> Path:
        return self.data_dir / "drive-imports"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "state.sqlite3"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.uploads_dir,
            self.outputs_dir,
            self.work_dir,
            self.drive_imports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        *,
        data_dir: Path | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> Settings:
        root = (
            Path(data_dir or os.getenv("PINGPONG_DATA_DIR", Path.cwd() / "data"))
            .expanduser()
            .resolve()
        )
        root.mkdir(parents=True, exist_ok=True)

        token = os.getenv("PINGPONG_UPLOAD_TOKEN") or _read_or_create_token(root)
        settings = cls(
            data_dir=root,
            upload_token=token,
            public_url=os.getenv("PINGPONG_PUBLIC_URL") or None,
            host=host or os.getenv("PINGPONG_HOST", "0.0.0.0"),
            port=port or _env_int("PINGPONG_PORT", 8000),
            max_upload_bytes=_env_int("PINGPONG_MAX_UPLOAD_BYTES", 100 * 1024**3),
            max_chunk_bytes=_env_int("PINGPONG_MAX_CHUNK_BYTES", 32 * 1024**2),
            download_min_free_bytes=_env_int(
                "PINGPONG_DOWNLOAD_MIN_FREE_BYTES", 2 * 1024**3
            ),
            video_sample_fps=_env_float("PINGPONG_VIDEO_SAMPLE_FPS", 8.0),
            analysis_frame_size=_env_int("PINGPONG_ANALYSIS_FRAME_SIZE", 320),
            audio_sample_rate=_env_int("PINGPONG_AUDIO_SAMPLE_RATE", 16_000),
            max_points=_env_int(
                "PINGPONG_MAX_POINTS",
                _env_int("PINGPONG_MAX_HIGHLIGHTS", 6),
            ),
            reel_target_seconds=_env_float("PINGPONG_REEL_TARGET_SECONDS", 55.0),
            reel_transition_seconds=_env_float("PINGPONG_REEL_TRANSITION_SECONDS", 0.35),
            worker_count=_env_int("PINGPONG_WORKERS", 1),
        )
        settings.ensure_directories()
        return settings


def _read_or_create_token(root: Path) -> str:
    token_path = root / ".upload-token"
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    if token:
        return token

    token = secrets.token_urlsafe(24)
    token_path.write_text(token, encoding="utf-8")
    return token
