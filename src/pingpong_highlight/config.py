from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_optional_positive_int(name: str, fallback_name: str | None = None) -> int | None:
    value = os.getenv(name)
    if (value is None or not value.strip()) and fallback_name is not None:
        value = os.getenv(fallback_name)
    if value is None or not value.strip():
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be zero or a positive integer")
    return parsed or None


def _env_optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


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
    library_minimum_point_score_ratio: float = 0.70
    minimum_point_score_ratio: float = 0.87
    max_points: int | None = None
    reel_target_seconds: float = 55.0
    clip_pre_roll_seconds: float = 1.5
    clip_post_roll_seconds: float = 1.5
    worker_count: int = 1
    pcloud_remote: str = "highlightcraft-pcloud"
    pcloud_root: str = "HighlightCraft"
    rclone_binary: str = "rclone"
    rclone_config: Path | None = None
    pcloud_bwlimit: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.library_minimum_point_score_ratio <= 1.0:
            raise ValueError(
                "library_minimum_point_score_ratio must be between 0 and 1"
            )
        if not 0.0 <= self.minimum_point_score_ratio <= 1.0:
            raise ValueError("minimum_point_score_ratio must be between 0 and 1")
        if self.library_minimum_point_score_ratio > self.minimum_point_score_ratio:
            raise ValueError(
                "library_minimum_point_score_ratio cannot exceed "
                "minimum_point_score_ratio"
            )
        if self.max_points is not None and self.max_points < 0:
            raise ValueError("max_points must be zero, positive, or None")
        if self.max_points == 0:
            object.__setattr__(self, "max_points", None)
        if self.reel_target_seconds <= 0:
            raise ValueError("reel_target_seconds must be positive")
        normalized_remote = self.pcloud_remote.strip()
        if not normalized_remote or any(
            character in normalized_remote for character in ":/\\"
        ) or any(
            character.isspace() or ord(character) < 32
            for character in normalized_remote
        ):
            raise ValueError("pcloud_remote must be a plain rclone remote name")
        object.__setattr__(self, "pcloud_remote", normalized_remote)
        normalized_root = self.pcloud_root.strip().strip("/\\").replace("\\", "/")
        root_parts = PurePosixPath(normalized_root).parts
        if (
            not root_parts
            or ":" in normalized_root
            or ".." in root_parts
            or any(ord(character) < 32 for character in normalized_root)
        ):
            raise ValueError("pcloud_root must be a safe relative remote path")
        object.__setattr__(self, "pcloud_root", PurePosixPath(*root_parts).as_posix())

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
    def compilations_dir(self) -> Path:
        return self.data_dir / "compilations"

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
            self.compilations_dir,
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
            library_minimum_point_score_ratio=_env_float(
                "PINGPONG_LIBRARY_MIN_POINT_SCORE_RATIO",
                0.70,
            ),
            minimum_point_score_ratio=_env_float(
                "PINGPONG_MIN_POINT_SCORE_RATIO",
                0.87,
            ),
            max_points=_env_optional_positive_int(
                "PINGPONG_MAX_POINTS",
                "PINGPONG_MAX_HIGHLIGHTS",
            ),
            reel_target_seconds=_env_float("PINGPONG_REEL_TARGET_SECONDS", 55.0),
            clip_pre_roll_seconds=_env_float("PINGPONG_CLIP_PRE_ROLL_SECONDS", 1.5),
            clip_post_roll_seconds=_env_float("PINGPONG_CLIP_POST_ROLL_SECONDS", 1.5),
            worker_count=_env_int("PINGPONG_WORKERS", 1),
            pcloud_remote=os.getenv(
                "PINGPONG_PCLOUD_REMOTE",
                "highlightcraft-pcloud",
            ),
            pcloud_root=os.getenv("PINGPONG_PCLOUD_ROOT", "HighlightCraft"),
            rclone_binary=os.getenv("PINGPONG_RCLONE_BINARY", "rclone"),
            rclone_config=_env_optional_path("PINGPONG_RCLONE_CONFIG"),
            pcloud_bwlimit=os.getenv("PINGPONG_PCLOUD_BWLIMIT") or None,
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
