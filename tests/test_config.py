from pathlib import Path

import pytest

from pingpong_highlight.cli import _service_url
from pingpong_highlight.config import Settings


def test_public_url_overrides_container_address(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        upload_token="phone token",
        public_url="http://192.168.1.19:9000/",
    )

    assert _service_url(settings, "172.18.0.2") == "http://192.168.1.19:9000"


def test_public_url_reads_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_PUBLIC_URL", "https://clips.example.test")
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.public_url == "https://clips.example.test"


def test_clip_context_reads_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_CLIP_PRE_ROLL_SECONDS", "1.75")
    monkeypatch.setenv("PINGPONG_CLIP_POST_ROLL_SECONDS", "2.0")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.clip_pre_roll_seconds == 1.75
    assert settings.clip_post_roll_seconds == 2.0


def test_threshold_selection_defaults_to_relative_score_without_point_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.delenv("PINGPONG_MIN_POINT_SCORE_RATIO", raising=False)
    monkeypatch.delenv("PINGPONG_MAX_POINTS", raising=False)
    monkeypatch.delenv("PINGPONG_MAX_HIGHLIGHTS", raising=False)

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.minimum_point_score_ratio == 0.87
    assert settings.max_points is None


@pytest.mark.parametrize(("value", "expected"), [("0", None), ("8", 8)])
def test_point_cap_reads_zero_as_unlimited(
    tmp_path: Path,
    monkeypatch,
    value: str,
    expected: int | None,
) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_MAX_POINTS", value)

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.max_points == expected


def test_programmatic_zero_point_cap_is_also_unlimited(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, upload_token="test", max_points=0)

    assert settings.max_points is None


def test_negative_programmatic_point_cap_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_points"):
        Settings(data_dir=tmp_path, upload_token="test", max_points=-1)


def test_legacy_highlight_cap_remains_supported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.delenv("PINGPONG_MAX_POINTS", raising=False)
    monkeypatch.setenv("PINGPONG_MAX_HIGHLIGHTS", "5")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.max_points == 5


def test_blank_primary_point_cap_uses_legacy_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_MAX_POINTS", "")
    monkeypatch.setenv("PINGPONG_MAX_HIGHLIGHTS", "5")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.max_points == 5


def test_primary_point_cap_takes_precedence_over_legacy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_MAX_POINTS", "7")
    monkeypatch.setenv("PINGPONG_MAX_HIGHLIGHTS", "5")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.max_points == 7


def test_custom_score_ratio_reads_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_MIN_POINT_SCORE_RATIO", "0.9")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.minimum_point_score_ratio == 0.9


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_invalid_score_ratio_fails_during_configuration(
    tmp_path: Path,
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match="minimum_point_score_ratio"):
        Settings(
            data_dir=tmp_path,
            upload_token="test",
            minimum_point_score_ratio=ratio,
        )


def test_identity_settings_read_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_BOOTSTRAP_ADMIN_USERNAME", "Owner")
    monkeypatch.setenv("PINGPONG_BOOTSTRAP_ADMIN_PASSWORD", "private-password")
    monkeypatch.setenv("PINGPONG_SESSION_TTL_SECONDS", "3600")
    monkeypatch.setenv("PINGPONG_SESSION_COOKIE_SECURE", "yes")
    monkeypatch.setenv("PINGPONG_ENABLE_LEGACY_TOKEN_AUTH", "true")
    monkeypatch.setenv("PINGPONG_TRUSTED_PROXY_PROVIDER", "ngrok")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.bootstrap_admin_username == "owner"
    assert settings.bootstrap_admin_password == "private-password"
    assert settings.session_ttl_seconds == 3600
    assert settings.session_cookie_secure is True
    assert settings.legacy_token_auth_enabled is True
    assert settings.trusted_proxy_provider == "ngrok"
    assert settings.maintenance_token != settings.upload_token
    assert (tmp_path / ".maintenance-token").read_text(encoding="utf-8").strip() == (
        settings.maintenance_token
    )
    assert (tmp_path / ".maintenance-token").stat().st_mode & 0o777 == 0o600


def test_invalid_identity_settings_are_rejected(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="session_ttl_seconds"):
        Settings(data_dir=tmp_path, upload_token="test", session_ttl_seconds=0)

    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_SESSION_COOKIE_SECURE", "sometimes")
    with pytest.raises(ValueError, match="PINGPONG_SESSION_COOKIE_SECURE"):
        Settings.from_env(data_dir=tmp_path)

    with pytest.raises(ValueError, match="trusted_proxy_provider"):
        Settings(
            data_dir=tmp_path,
            upload_token="test",
            trusted_proxy_provider="untrusted",
        )
