from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pingpong_highlight

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_references_stay_in_sync() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert pingpong_highlight.__version__ == version

    lockfile = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    project_package = next(
        package for package in lockfile["package"] if package["name"] == pyproject["project"]["name"]
    )
    assert project_package["version"] == version

    expected_references = {
        "Dockerfile": f"ARG APP_VERSION={version}",
        "README.md": f"Git/source 研究材料版本為 **{version}**",
        "src/pingpong_highlight/web.py": f'version="{version}"',
    }
    for relative_path, expected in expected_references.items():
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in content, f"{relative_path} is missing release version {version}"

    index = (ROOT / "src/pingpong_highlight/static/index.html").read_text(encoding="utf-8")
    assert f"/static/i18n.js?v={version}" in index
    assert f"/static/app.js?v={version}" in index
    assert f"/static/styles.css?v={version}" in index
    asset_versions = re.findall(r"/static/[^\" ]+\?v=([^\" ]+)", index)
    assert asset_versions and set(asset_versions) == {version}


def test_pull_only_image_reference_is_separate_from_source_version() -> None:
    compose = (ROOT / "compose.release.yaml").read_text(encoding="utf-8")
    historical_image = "docker.io/momonong/pingpong-auto-highlight:1.4.0"
    assert "build: !reset null" in compose
    assert f'image: "${{PINGPONG_IMAGE:-{historical_image}}}"' in compose

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"歷史 image 參照 `{historical_image}`" in readme
    assert "source 研究材料版本與 image 版本分開" in readme
    assert "本輪未 build/push 新 Docker image" in readme
    assert "未查核 registry 現況" in readme
    assert f"不能假定 `{pingpong_highlight.__version__}` image 可 pull" in readme
