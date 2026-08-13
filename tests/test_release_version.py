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

    expected_references = {
        "Dockerfile": f"ARG APP_VERSION={version}",
        "compose.release.yaml": f"pingpong-auto-highlight:{version}",
        "README.md": f"pingpong-auto-highlight:{version}",
        "src/pingpong_highlight/web.py": f'version="{version}"',
    }
    for relative_path, expected in expected_references.items():
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in content, f"{relative_path} is missing release version {version}"

    index = (ROOT / "src/pingpong_highlight/static/index.html").read_text(encoding="utf-8")
    assert f"/static/app.js?v={version}" in index
    assert f"/static/styles.css?v={version}" in index
