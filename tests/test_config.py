from pathlib import Path

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
