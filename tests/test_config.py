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

    assert _service_url(settings, "172.18.0.2") == (
        "http://192.168.1.19:9000/#token=phone%20token"
    )


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
    monkeypatch.delenv("PINGPONG_LIBRARY_MIN_POINT_SCORE_RATIO", raising=False)
    monkeypatch.delenv("PINGPONG_MIN_POINT_SCORE_RATIO", raising=False)
    monkeypatch.delenv("PINGPONG_MAX_POINTS", raising=False)
    monkeypatch.delenv("PINGPONG_MAX_HIGHLIGHTS", raising=False)

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.library_minimum_point_score_ratio == 0.70
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


def test_custom_library_ratio_reads_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")
    monkeypatch.setenv("PINGPONG_LIBRARY_MIN_POINT_SCORE_RATIO", "0.65")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.library_minimum_point_score_ratio == 0.65


def test_library_ratio_cannot_exceed_recommendation_ratio(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        Settings(
            data_dir=tmp_path,
            upload_token="test",
            library_minimum_point_score_ratio=0.9,
            minimum_point_score_ratio=0.87,
        )


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
