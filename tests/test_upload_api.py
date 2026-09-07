from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import anyio
from fastapi import Request
from fastapi.testclient import TestClient

from pingpong_highlight.config import Settings
from pingpong_highlight.web import (
    MEDIA_CHUNK_SIZE,
    MediaFileResponse,
    _stream_file_range,
    create_app,
)


def test_media_response_uses_large_chunks_for_docker_bind_mounts() -> None:
    assert MEDIA_CHUNK_SIZE >= 1024 * 1024
    assert MediaFileResponse.chunk_size == MEDIA_CHUNK_SIZE


def test_media_stream_uses_large_chunks(tmp_path: Path) -> None:
    media_path = tmp_path / "large.mp4"
    media_path.write_bytes(b"x" * (MEDIA_CHUNK_SIZE * 2 + 17))

    async def collect_sizes() -> list[int]:
        return [
            len(chunk)
            async for chunk in _stream_file_range(media_path, 0, media_path.stat().st_size)
        ]

    assert anyio.run(collect_sizes) == [MEDIA_CHUNK_SIZE, MEDIA_CHUNK_SIZE, 17]


def test_media_response_stops_streaming_after_disconnect(tmp_path: Path) -> None:
    media_path = tmp_path / "large.mp4"
    media_path.write_bytes(b"x" * (MEDIA_CHUNK_SIZE * 8))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/media",
        "raw_path": b"/media",
        "query_string": b"",
        "headers": [(b"range", b"bytes=0-")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
    }
    response = MediaFileResponse(media_path, Request(scope), media_type="video/mp4")
    messages: list[dict] = []

    async def exercise_disconnect() -> None:
        disconnected = anyio.Event()

        async def receive() -> dict[str, str]:
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            messages.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                disconnected.set()

        await response(scope, receive, send)

    anyio.run(exercise_disconnect)
    streamed_bytes = sum(
        len(message.get("body", b""))
        for message in messages
        if message["type"] == "http.response.body"
    )
    assert 0 < streamed_bytes < media_path.stat().st_size


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
        legacy_token_auth_enabled=True,
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

        active = client.get("/api/uploads", headers={"X-Upload-Token": "test-secret"})
        assert active.status_code == 200
        active_upload = active.json()["uploads"][0]
        assert active_upload["id"] == location.rsplit("/", 1)[-1]
        assert active_upload["filename"] == "phone.mov"
        assert active_upload["size"] == len(payload)
        assert active_upload["offset"] == 0
        assert active_upload["status"] == "uploading"
        assert active_upload["job_id"] is None
        assert active_upload["created_at"]
        assert active_upload["updated_at"]

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

        active_upload = client.get(
            "/api/uploads", headers={"X-Upload-Token": "test-secret"}
        ).json()["uploads"][0]
        assert active_upload["offset"] == 8
        assert active_upload["size"] == len(payload)

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
        assert client.get("/api/uploads", headers={"X-Upload-Token": "test-secret"}).json() == {
            "uploads": []
        }
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
        assert download.headers["cache-control"] == "private, no-store"

        preview = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={"X-Upload-Token": "test-secret"},
        )
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "video/mp4"
        assert preview.headers["cache-control"] == "private, no-store"
        assert preview.headers["accept-ranges"] == "bytes"
        assert "content-disposition" not in preview.headers

        preview_range = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={"X-Upload-Token": "test-secret", "Range": "bytes=2-5"},
        )
        assert preview_range.status_code == 206
        assert preview_range.content == b"ke-v"
        assert preview_range.headers["content-range"] == "bytes 2-5/10"
        assert preview_range.headers["content-length"] == "4"
        assert preview_range.headers["etag"]

        matching_if_range = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={
                "X-Upload-Token": "test-secret",
                "Range": "bytes=2-",
                "If-Range": preview_range.headers["etag"],
            },
        )
        assert matching_if_range.status_code == 206
        assert matching_if_range.content == b"ke-video"

        clamped_range = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={"X-Upload-Token": "test-secret", "Range": "bytes=2-999"},
        )
        assert clamped_range.status_code == 206
        assert clamped_range.content == b"ke-video"

        suffix_range = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={"X-Upload-Token": "test-secret", "Range": "bytes=-4"},
        )
        assert suffix_range.status_code == 206
        assert suffix_range.content == b"ideo"

        invalid_range = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={"X-Upload-Token": "test-secret", "Range": "bytes=20-30"},
        )
        assert invalid_range.status_code == 416
        assert invalid_range.headers["content-range"] == "bytes */10"

        stale_if_range = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4",
            headers={
                "X-Upload-Token": "test-secret",
                "Range": "bytes=2-5",
                "If-Range": '"stale"',
            },
        )
        assert stale_if_range.status_code == 200
        assert stale_if_range.content == b"fake-video"

        attachment = client.get(
            f"/api/jobs/{job_id}/files/best_points_reel.mp4?download=true",
            headers={"X-Upload-Token": "test-secret"},
        )
        assert attachment.status_code == 200
        assert attachment.headers["content-disposition"].startswith("attachment;")

        source = client.get(
            f"/api/jobs/{job_id}/source",
            headers={"X-Upload-Token": "test-secret", "Range": "bytes=0-3"},
        )
        assert source.status_code == 206
        assert source.content == payload[:4]
        assert source.headers["content-range"].startswith("bytes 0-3/")
        assert source.headers["cache-control"] == "private, no-store"

        annotations_url = f"/api/jobs/{job_id}/annotations"
        assert client.get(annotations_url, headers=_headers()).json()["annotations"] == []
        annotated = client.post(
            annotations_url,
            headers=_headers(**{"Content-Type": "application/json"}),
            json={
                "label": "highlight",
                "start": 0.1,
                "end": 0.8,
                "note": "  backhand counter  ",
            },
        )
        assert annotated.status_code == 201
        annotation = annotated.json()
        assert annotation["label"] == "highlight"
        assert annotation["duration"] == 0.7
        assert annotation["note"] == "backhand counter"
        assert client.get(annotations_url, headers=_headers()).json()["annotations"] == [annotation]

        invalid = client.post(
            annotations_url,
            headers=_headers(**{"Content-Type": "application/json"}),
            json={"label": "highlight", "start": 0.5, "end": 1.1},
        )
        assert invalid.status_code == 422

        deleted = client.delete(f"{annotations_url}/{annotation['id']}", headers=_headers())
        assert deleted.status_code == 204
        assert client.get(annotations_url, headers=_headers()).json()["annotations"] == []


def test_api_rejects_missing_token(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), processor=FakeProcessor(b"x"))
    with TestClient(app) as client:
        assert client.get("/api/jobs").status_code == 401
        assert client.get("/api/uploads").status_code == 401
        assert client.get("/api/jobs/missing/source").status_code == 401
        assert client.get("/api/jobs/missing/annotations").status_code == 401


def test_incomplete_upload_can_be_deleted_with_its_partial_file(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), processor=FakeProcessor(b"x"))
    with TestClient(app) as client:
        filename = base64.b64encode(b"duplicate.mp4").decode()
        created = client.post(
            "/api/uploads",
            headers=_headers(
                **{
                    "Upload-Length": "12",
                    "Upload-Metadata": f"filename {filename}",
                }
            ),
        )
        location = created.headers["location"]
        appended = client.patch(
            location,
            headers=_headers(
                **{
                    "Upload-Offset": "0",
                    "Content-Type": "application/offset+octet-stream",
                }
            ),
            content=b"partial",
        )
        assert appended.status_code == 204
        upload_id = location.rsplit("/", 1)[-1]
        record = app.state.database.get_upload(upload_id)
        assert record is not None
        part_path = app.state.uploads.part_path(record)
        assert part_path.read_bytes() == b"partial"

        deleted = client.delete(location, headers=_headers())

        assert deleted.status_code == 204
        assert deleted.headers["tus-extension"] == "creation,checksum,termination"
        assert app.state.database.get_upload(upload_id) is None
        assert not part_path.exists()
        assert client.get(location, headers=_headers()).status_code == 404
        assert client.get("/api/uploads", headers=_headers()).json() == {"uploads": []}


def test_completed_upload_session_cannot_be_deleted(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), processor=FakeProcessor(b"done"))
    with TestClient(app) as client:
        filename = base64.b64encode(b"done.mp4").decode()
        created = client.post(
            "/api/uploads",
            headers=_headers(
                **{
                    "Upload-Length": "4",
                    "Upload-Metadata": f"filename {filename}",
                }
            ),
        )
        location = created.headers["location"]
        completed = client.patch(
            location,
            headers=_headers(
                **{
                    "Upload-Offset": "0",
                    "Content-Type": "application/offset+octet-stream",
                }
            ),
            content=b"done",
        )
        assert completed.status_code == 204

        rejected = client.delete(location, headers=_headers())

        assert rejected.status_code == 409
        assert client.get(location, headers=_headers()).status_code == 200


def test_public_responses_have_security_and_cache_headers(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), processor=FakeProcessor(b"x"))
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["referrer-policy"] == "no-referrer"
        assert index.headers["x-content-type-options"] == "nosniff"
        assert index.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in index.headers["content-security-policy"]
        assert index.headers["cache-control"] == "no-cache"
        assert 'id="loginForm"' in index.text
        assert 'id="loginUsername"' in index.text
        assert 'id="loginPassword"' in index.text
        assert 'id="quickGuide"' in index.text
        assert 'id="adminPanel"' in index.text
        assert 'id="adminPasswordDialog"' in index.text
        assert 'id="adminResetPassword" type="password"' in index.text
        assert 'id="annotationWorkspace"' in index.text
        assert 'id="annotationWorkspaceVideo"' in index.text
        assert 'id="annotationDevBlock"' in index.text
        assert 'id="annotationDevList"' in index.text
        assert index.text.index('id="jobList"') < index.text.index('id="annotationDevBlock"')
        assert "登入同一帳號即可查看自己的所有影片" in index.text

        app_js = client.get("/static/app.js")
        assert app_js.status_code == 200
        assert app_js.headers["cache-control"] == "no-cache"
        assert "openAnnotationWorkspace" in app_js.text
        assert "renderAnnotationDevelopment" in app_js.text
        assert "lastAnnotationDevSignature" in app_js.text
        assert 't("annotation.openLabel", { filename })' in app_js.text
        assert "expandedResultJobIds" in app_js.text
        assert "jobRenderSignatures" in app_js.text
        assert "hydrateResultPanel" in app_js.text
        assert "function dehydrateResultPanel(panel)" in app_js.text
        assert 'source.removeAttribute("src");' in app_js.text
        assert "renderJobs(jobs)" in app_js.text
        assert 'data-result-job-id="${escapeHtml(jobId)}"' in app_js.text
        assert 't("result.srLabel", { source: sourceName })' in app_js.text
        assert 'aria-label="展開或收合' not in app_js.text
        assert 'data-src="${escapeHtml(previewUrl)}"' in app_js.text
        assert '<source src="${escapeHtml(previewUrl)}"' not in app_js.text
        assert '"toggle",' in app_js.text
        assert "renderAnnotationPanel" not in app_js.text
        assert ".annotation-video" not in app_js.text
        assert 'event.code === "KeyI"' in app_js.text
        assert 'event.code === "KeyO"' in app_js.text
        assert "function annotationWorkspaceNoteValue()" in app_js.text
        assert 'return selectedTags.join("、");' in app_js.text
        assert 'elements.annotationWorkspaceForm.addEventListener("submit"' in app_js.text
        assert "elements.annotationWorkspaceForm.requestSubmit();" in app_js.text
        assert 'input[type="checkbox"][name^="annotation-note-tag"]' in app_js.text
        assert (
            "annotationWorkspaceComposing || event.isComposing || event.keyCode === 229"
            in app_js.text
        )
        assert "note.length > annotationNoteMaxLength" in app_js.text
        assert 'input, select, textarea, button, a, [contenteditable="true"]' in app_js.text
        assert 'removeLocalStorage("pingpong-upload-token");' in app_js.text
        assert 't("admin.selfPasswordTitle")' in app_js.text
        assert "function finishAdminPassword(value)" in app_js.text
        assert "elements.adminPasswordForm.reset();" in app_js.text

        i18n_js = client.get("/static/i18n.js")
        assert i18n_js.status_code == 200
        assert i18n_js.headers["cache-control"] == "no-cache"
        assert '"language.currentEnglish"' in i18n_js.text

        index_html = client.get("/")
        assert index_html.status_code == 200
        assert "HighlightCraft — 桌球精彩集錦" in index_html.text
        assert "RallyCut" not in index_html.text
        assert 'id="annotationWorkspaceForm"' in index_html.text
        assert 'name="annotation-note-tag" value="相持"' in index_html.text
        assert 'name="annotation-note-tag" value="搶攻"' in index_html.text
        assert 'id="annotationWorkspaceNoteOtherToggle"' in index_html.text
        assert 'id="annotationWorkspaceNoteOtherField"' in index_html.text
        assert 'id="annotationWorkspaceNoteOther" type="text" maxlength="274"' in index_html.text

        styles = client.get("/static/styles.css")
        assert styles.status_code == 200
        assert styles.headers["cache-control"] == "no-cache"
        assert ".annotation-dev-block" in styles.text
        assert ".result-panel[open]" in styles.text
        assert ".reel-toggle-label" in styles.text

        jobs = client.get("/api/jobs", headers={"X-Upload-Token": "test-secret"})
        assert jobs.status_code == 200
        assert jobs.headers["cache-control"] == "private, no-store"
