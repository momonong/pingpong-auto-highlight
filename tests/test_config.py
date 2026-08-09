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
        "http://192.168.1.19:9000/?token=phone%20token"
    )


def test_public_url_reads_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PINGPONG_PUBLIC_URL", "https://clips.example.test")
    monkeypatch.setenv("PINGPONG_UPLOAD_TOKEN", "fixed-token")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.public_url == "https://clips.example.test"
