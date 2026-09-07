"""Isolated browser fixture: real auth/upload APIs, synthetic CPU-only processing and Drive.

Invoked by workspace.cjs with a fresh data directory and generated browser video.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import uvicorn

from pingpong_highlight.config import Settings
from pingpong_highlight.web import create_app


class PreviewProcessor:
    def run(self, source, output_dir, progress=None, *, source_name=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0.5, "editing-point-reel")
        time.sleep(2)
        shutil.copyfile(source, output_dir / "best_points_reel.mp4")
        result = {
            "source_name": source_name,
            "media": {"duration": 2},
            "summary": {"point_count": 1, "reel_duration": 2},
            "files": [{"name": "best_points_reel.mp4", "kind": "reel"}],
        }
        (output_dir / "analysis.json").write_text(json.dumps(result), encoding="utf-8")
        return result


class FixtureDrive:
    def resolve(self, link):
        return "Drive fixture.mp4"

    def download(self, link, output, progress):
        shutil.copyfile(sys.argv[3], output)
        size = output.stat().st_size
        progress(size, size)
        return output


if __name__ == "__main__":
    settings = Settings(
        data_dir=Path(sys.argv[1]),
        upload_token="browser-test-only",
        bootstrap_admin_password="browser-test-password",
        host="127.0.0.1",
        port=int(sys.argv[2]),
        max_chunk_bytes=1024,
        download_min_free_bytes=0,
    )
    app = create_app(settings, processor=PreviewProcessor(), drive_downloader=FixtureDrive())
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
