from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from gdown.exceptions import FileURLRetrievalError

from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.drive import DriveLink, DriveLinkError, parse_drive_link
from pingpong_highlight.web import create_app

FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
AUTH = {"X-Upload-Token": "test-secret"}


class FakeDriveDownloader:
    def __init__(
        self,
        payload: bytes,
        *,
        filename: str = "drive-video.mp4",
        failures: int = 0,
    ):
        self.payload = payload
        self.filename = filename
        self.failures = failures
        self.links: list[DriveLink] = []

    def resolve(self, link: DriveLink) -> str:
        self.links.append(link)
        if self.failures:
            self.failures -= 1
            raise FileURLRetrievalError("not publicly downloadable")
        return self.filename

    def download(self, link: DriveLink, output: Path, progress) -> Path:
        self.links.append(link)
        output.write_bytes(self.payload)
        progress(len(self.payload), len(self.payload))
        return output


class DriveProcessor:
    def __init__(self, expected: bytes, filename: str = "drive-video.mp4"):
        self.expected = expected
        self.filename = filename

    def run(self, source: Path, output_dir: Path, progress=None, *, source_name=None):
        assert source.read_bytes() == self.expected
        assert source_name == self.filename
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0.5, "fake-analysis")
        result = {
            "source_name": source_name,
            "media": {"duration": 1.0},
            "summary": {"point_count": 1, "reel_duration": 1.0},
            "files": [{"name": "best_points_reel.mp4", "kind": "reel"}],
        }
        (output_dir / "best_points_reel.mp4").write_bytes(b"fake-reel")
        (output_dir / "analysis.json").write_text(json.dumps(result), encoding="utf-8")
        return result


class ResumingDriveDownloader(FakeDriveDownloader):
    def download(self, link: DriveLink, output: Path, progress) -> Path:
        self.links.append(link)
        parts = list(output.parent.glob(f"{output.name}*.part"))
        assert len(parts) == 1
        prefix = parts[0].read_bytes()
        assert self.payload.startswith(prefix)
        output.write_bytes(self.payload)
        parts[0].unlink()
        progress(len(self.payload), len(self.payload))
        return output


def _settings(tmp_path: Path, *, max_upload_bytes: int = 1024) -> Settings:
    settings = Settings(
        data_dir=tmp_path,
        upload_token="test-secret",
        legacy_token_auth_enabled=True,
        max_upload_bytes=max_upload_bytes,
        max_chunk_bytes=8,
        download_min_free_bytes=0,
    )
    settings.ensure_directories()
    return settings


def _wait_for_import(client: TestClient, status: str) -> dict:
    for _ in range(200):
        imports = client.get("/api/drive-imports", headers=AUTH).json()["imports"]
        if imports and imports[0]["status"] == status:
            return imports[0]
        time.sleep(0.01)
    raise AssertionError(f"Drive import did not reach {status}")


def _wait_for_completed_job(client: TestClient) -> dict:
    for _ in range(200):
        jobs = client.get("/api/jobs", headers=AUTH).json()["jobs"]
        if jobs and jobs[0]["status"] == "completed":
            return jobs[0]
        time.sleep(0.01)
    raise AssertionError("Imported video job did not complete")


def test_parse_drive_file_links_and_reject_other_hosts() -> None:
    parsed = parse_drive_link(
        f"https://drive.google.com/file/u/0/d/{FILE_ID}/view?usp=sharing&resourcekey=0-exampleKey"
    )
    assert parsed == DriveLink(FILE_ID, "0-exampleKey")
    assert parse_drive_link(f"https://drive.google.com/open?id={FILE_ID}").file_id == FILE_ID

    invalid = [
        f"https://example.com/file/d/{FILE_ID}/view",
        f"https://drive.google.com.evil.example/file/d/{FILE_ID}/view",
        f"https://drive.google.com/drive/folders/{FILE_ID}",
        f"http://drive.google.com/file/d/{FILE_ID}/view",
    ]
    for url in invalid:
        try:
            parse_drive_link(url)
        except DriveLinkError:
            pass
        else:
            raise AssertionError(f"Unsafe Drive URL was accepted: {url}")


def test_public_drive_link_downloads_then_queues_existing_pipeline(tmp_path: Path) -> None:
    payload = b"drive-video-bytes"
    downloader = FakeDriveDownloader(payload)
    app = create_app(
        _settings(tmp_path),
        processor=DriveProcessor(payload),
        drive_downloader=downloader,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/drive-imports",
            headers=AUTH,
            json={"url": f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"},
        )
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert "file_id" not in response.json()

        job = _wait_for_completed_job(client)
        assert job["result"]["source_name"] == "drive-video.mp4"
        assert client.get("/api/drive-imports", headers=AUTH).json() == {"imports": []}

        records = app.state.database.list_drive_imports(include_completed=True)
        assert len(records) == 1
        assert records[0].status == "completed"
        assert records[0].offset == len(payload)
        upload = app.state.database.get_upload(records[0].upload_id)
        assert upload is not None
        assert upload.path.read_bytes() == payload


def test_drive_import_failure_can_retry_without_reposting_link(tmp_path: Path) -> None:
    payload = b"retry-video"
    downloader = FakeDriveDownloader(payload, failures=1)
    app = create_app(
        _settings(tmp_path),
        processor=DriveProcessor(payload),
        drive_downloader=downloader,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/drive-imports",
            headers=AUTH,
            json={"url": f"https://drive.google.com/file/d/{FILE_ID}/view"},
        ).json()
        failed = _wait_for_import(client, "failed")
        assert "知道連結的任何人可檢視" in failed["error"]

        retried = client.post(
            f"/api/drive-imports/{created['id']}/retry",
            headers=AUTH,
        )
        assert retried.status_code == 202
        _wait_for_completed_job(client)


def test_interrupted_drive_download_resumes_after_service_restart(tmp_path: Path) -> None:
    payload = b"partial-and-finished"
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    record = database.create_or_requeue_drive_import(FILE_ID, None)
    assert database.claim_drive_import(record.id)
    database.start_drive_import_download(record.id, "drive-video.mp4")
    prefix = payload[:7]
    database.update_drive_import_progress(record.id, len(prefix), len(payload))
    part = settings.drive_imports_dir / f"{record.id}.mp4.resume.part"
    part.write_bytes(prefix)

    app = create_app(
        settings,
        processor=DriveProcessor(payload),
        drive_downloader=ResumingDriveDownloader(payload),
    )
    with TestClient(app) as client:
        _wait_for_completed_job(client)

    updated = app.state.database.get_drive_import(record.id)
    assert updated is not None
    assert updated.status == "completed"
    assert updated.offset == len(payload)
    assert not part.exists()


def test_drive_import_enforces_auth_url_policy_and_size_limit(tmp_path: Path) -> None:
    payload = b"too-large"
    app = create_app(
        _settings(tmp_path, max_upload_bytes=4),
        processor=DriveProcessor(payload),
        drive_downloader=FakeDriveDownloader(payload),
    )
    valid_url = f"https://drive.google.com/file/d/{FILE_ID}/view"

    with TestClient(app) as client:
        assert client.post("/api/drive-imports", json={"url": valid_url}).status_code == 401
        rejected = client.post(
            "/api/drive-imports",
            headers=AUTH,
            json={"url": f"https://example.com/file/d/{FILE_ID}/view"},
        )
        assert rejected.status_code == 400

        accepted = client.post(
            "/api/drive-imports",
            headers=AUTH,
            json={"url": valid_url},
        )
        assert accepted.status_code == 202
        failed = _wait_for_import(client, "failed")
        assert "大小上限" in failed["error"]
        assert client.get("/api/jobs", headers=AUTH).json() == {"jobs": []}
