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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


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
    minimum_point_score_ratio: float = 0.87
    max_points: int | None = None
    reel_target_seconds: float = 55.0
    clip_pre_roll_seconds: float = 1.5
    clip_post_roll_seconds: float = 1.5
    worker_count: int = 1
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str | None = None
    session_ttl_seconds: int = 7 * 24 * 60 * 60
    session_cookie_secure: bool = False
    legacy_token_auth_enabled: bool = False
    trusted_proxy_provider: str = "none"
    maintenance_token: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_point_score_ratio <= 1.0:
            raise ValueError("minimum_point_score_ratio must be between 0 and 1")
        if self.max_points is not None and self.max_points < 0:
            raise ValueError("max_points must be zero, positive, or None")
        if self.max_points == 0:
            object.__setattr__(self, "max_points", None)
        if self.reel_target_seconds <= 0:
            raise ValueError("reel_target_seconds must be positive")
        username = self.bootstrap_admin_username.strip().casefold()
        if not username:
            raise ValueError("bootstrap_admin_username must not be blank")
        object.__setattr__(self, "bootstrap_admin_username", username)
        if self.bootstrap_admin_password == "":
            object.__setattr__(self, "bootstrap_admin_password", None)
        if self.session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        provider = self.trusted_proxy_provider.strip().casefold()
        if provider not in {"none", "ngrok", "cloudflare"}:
            raise ValueError("trusted_proxy_provider must be 'none', 'ngrok', or 'cloudflare'")
        object.__setattr__(self, "trusted_proxy_provider", provider)
        if self.maintenance_token is None:
            object.__setattr__(self, "maintenance_token", self.upload_token)

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
            download_min_free_bytes=_env_int("PINGPONG_DOWNLOAD_MIN_FREE_BYTES", 2 * 1024**3),
            video_sample_fps=_env_float("PINGPONG_VIDEO_SAMPLE_FPS", 8.0),
            analysis_frame_size=_env_int("PINGPONG_ANALYSIS_FRAME_SIZE", 320),
            audio_sample_rate=_env_int("PINGPONG_AUDIO_SAMPLE_RATE", 16_000),
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
            bootstrap_admin_username=(os.getenv("PINGPONG_BOOTSTRAP_ADMIN_USERNAME", "admin")),
            bootstrap_admin_password=(os.getenv("PINGPONG_BOOTSTRAP_ADMIN_PASSWORD") or None),
            session_ttl_seconds=_env_int(
                "PINGPONG_SESSION_TTL_SECONDS",
                7 * 24 * 60 * 60,
            ),
            session_cookie_secure=_env_bool(
                "PINGPONG_SESSION_COOKIE_SECURE",
                False,
            ),
            legacy_token_auth_enabled=_env_bool(
                "PINGPONG_ENABLE_LEGACY_TOKEN_AUTH",
                False,
            ),
            trusted_proxy_provider=os.getenv(
                "PINGPONG_TRUSTED_PROXY_PROVIDER",
                "none",
            ),
            maintenance_token=_read_or_create_secret(root / ".maintenance-token"),
        )
        settings.ensure_directories()
        return settings


def _read_or_create_token(root: Path) -> str:
    return _read_or_create_secret(root / ".upload-token")


def _read_or_create_secret(token_path: Path) -> str:
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    if token:
        token_path.chmod(0o600)
        return token

    token = secrets.token_urlsafe(24)
    try:
        descriptor = os.open(
            token_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError(f"Secret file is empty: {token_path}") from None
        token_path.chmod(0o600)
        return token
    try:
        os.write(descriptor, f"{token}\n".encode())
    finally:
        os.close(descriptor)
    return token
