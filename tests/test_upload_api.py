from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from pingpong_highlight.config import Settings
from pingpong_highlight.web import create_app


class FakeProcessor:
    def __init__(self, expected: bytes):
        self.expected = expected

    def run(self, source: Path, output_dir: Path, progress=None, *, source_name=None):
        assert source.read_bytes() == self.expected
        assert source_name == "phone.mov"
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0.5, "fake-analysis")
        result = {
            "source_name": source_name,
            "media": {"duration": 1.0},
            "summary": {"point_count": 0, "impact_count": 0, "reel_duration": None},
            "files": [
                {"name": "best_points_reel.mp4", "kind": "reel"},
                {"name": "analysis.json", "kind": "analysis"},
            ],
        }
        (output_dir / "best_points_reel.mp4").write_bytes(b"fake-video")
        (output_dir / "analysis.json").write_text(json.dumps(result), encoding="utf-8")
        return result


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_dir=tmp_path,
        upload_token="test-secret",
        max_upload_bytes=1024,
        max_chunk_bytes=8,
    )
    settings.ensure_directories()
    return settings


def _headers(**extra: str) -> dict[str, str]:
    return {"X-Upload-Token": "test-secret", "Tus-Resumable": "1.0.0", **extra}


def test_resumable_upload_checksum_and_job_completion(tmp_path: Path) -> None:
    payload = b"a-real-video-payload"
    app = create_app(_settings(tmp_path), processor=FakeProcessor(payload))
    with TestClient(app) as client:
        filename = base64.b64encode(b"phone.mov").decode()
        created = client.post(
            "/api/uploads",
            headers=_headers(
                **{
                    "Upload-Length": str(len(payload)),
                    "Upload-Metadata": f"filename {filename}",
                }
            ),
        )
        assert created.status_code == 201
        location = created.headers["location"]

        first = payload[:8]
        digest = base64.b64encode(hashlib.sha256(first).digest()).decode()
        response = client.patch(
            location,
            headers=_headers(
                **{
                    "Upload-Offset": "0",
                    "Upload-Checksum": f"sha256 {digest}",
                    "Content-Type": "application/offset+octet-stream",
                }
            ),
            content=first,
        )
        assert response.status_code == 204
        assert response.headers["upload-offset"] == "8"

        head = client.head(location, headers=_headers())
        assert head.headers["upload-offset"] == "8"

        bad = client.patch(
            location,
            headers=_headers(
                **{
                    "Upload-Offset": "8",
                    "Upload-Checksum": "sha256 AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                    "Content-Type": "application/offset+octet-stream",
                }
            ),
            content=payload[8:16],
        )
        assert bad.status_code == 460
        assert client.head(location, headers=_headers()).headers["upload-offset"] == "8"

        offset = 8
        job_id = None
        while offset < len(payload):
            chunk = payload[offset : offset + 8]
            digest = base64.b64encode(hashlib.sha256(chunk).digest()).decode()
            response = client.patch(
                location,
                headers=_headers(
                    **{
                        "Upload-Offset": str(offset),
                        "Upload-Checksum": f"sha256 {digest}",
                        "Content-Type": "application/offset+octet-stream",
                    }
                ),
                content=chunk,
            )
            assert response.status_code == 204
            offset = int(response.headers["upload-offset"])
            job_id = response.headers.get("upload-job-id") or job_id

        assert job_id
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}", headers=_headers()).json()
            if job["status"] == "completed":
                break
            time.sleep(0.01)
        assert job["status"] == "completed"

        download = client.get(
            f"/api/jobs/{job_id}/files/analysis.json",
            headers={"X-Upload-Token": "test-secret"},
        )
        assert download.status_code == 200

        preview = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={"X-Upload-Token": "test-secret"},
        )
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "video/mp4"
        assert preview.headers["cache-control"] == "private, no-store"
        assert "content-disposition" not in preview.headers

        attachment = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4?download=true",
            headers={"X-Upload-Token": "test-secret"},
        )
        assert attachment.status_code == 200
        assert attachment.headers["content-disposition"].startswith("attachment;")


def test_api_rejects_missing_token(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), processor=FakeProcessor(b"x"))
    with TestClient(app) as client:
        assert client.get("/api/jobs").status_code == 401


def test_public_responses_have_security_and_cache_headers(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), processor=FakeProcessor(b"x"))
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["referrer-policy"] == "no-referrer"
        assert index.headers["x-content-type-options"] == "nosniff"
        assert index.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in index.headers["content-security-policy"]

        jobs = client.get("/api/jobs", headers={"X-Upload-Token": "test-secret"})
        assert jobs.status_code == 200
        assert jobs.headers["cache-control"] == "private, no-store"
