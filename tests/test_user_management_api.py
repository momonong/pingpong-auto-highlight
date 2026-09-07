from __future__ import annotations

import base64
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from pingpong_highlight.config import Settings
from pingpong_highlight.web import create_app

ADMIN_PASSWORD = "admin-password-123"


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def run(
        self,
        source: Path,
        output_dir: Path,
        progress=None,
        *,
        source_name: str | None = None,
    ) -> dict[str, Any]:
        assert source.is_file()
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0.5, "test-processing")
        with self._lock:
            self.calls.append(source_name or source.name)
        result = {
            "source_name": source_name,
            "media": {"duration": 1.0},
            "summary": {"point_count": 1, "reel_duration": 1.0},
            "files": [{"name": "best_points_reel.mp4", "kind": "reel"}],
        }
        (output_dir / "best_points_reel.mp4").write_bytes(b"test-reel")
        return result


@dataclass(slots=True)
class HttpResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


class LiveHttpClient:
    """Small cookie-aware client that avoids Starlette's TestClient transport."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.cookies: dict[str, str] = {}

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        request_headers = dict(headers or {})
        if json_body is not None:
            assert body is None
            body = json.dumps(json_body).encode()
            request_headers["Content-Type"] = "application/json"
        if self.cookies:
            request_headers["Cookie"] = "; ".join(
                f"{name}={value}" for name, value in self.cookies.items()
            )

        connection = HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            response_body = response.read()
            for raw_cookie in response.headers.get_all("Set-Cookie", []):
                parsed = SimpleCookie()
                parsed.load(raw_cookie)
                for name, morsel in parsed.items():
                    if not morsel.value or morsel["max-age"] == "0":
                        self.cookies.pop(name, None)
                    else:
                        self.cookies[name] = morsel.value
            return HttpResponse(
                response.status,
                {name.casefold(): value for name, value in response.getheaders()},
                response_body,
            )
        finally:
            connection.close()

    def clone_without_cookies(self) -> LiveHttpClient:
        return LiveHttpClient(self.host, self.port)


@contextmanager
def _serve(app) -> Iterator[tuple[str, int]]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="critical",
            access_log=False,
            lifespan="on",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="test-uvicorn",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("Test Uvicorn server did not start")
    try:
        yield host, port
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=2)
        listener.close()
        assert not thread.is_alive(), "Test Uvicorn server did not stop"


@pytest.fixture
def live_app(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path,
        upload_token="legacy-header-secret",
        bootstrap_admin_username="admin",
        bootstrap_admin_password=ADMIN_PASSWORD,
        max_upload_bytes=1024,
        max_chunk_bytes=1024,
        worker_count=1,
    )
    processor = RecordingProcessor()
    app = create_app(settings, processor=processor)
    with _serve(app) as (host, port):
        yield app, LiveHttpClient(host, port), processor


def _login(client: LiveHttpClient, username: str, password: str) -> dict[str, Any]:
    response = client.request(
        "POST",
        "/api/auth/login",
        json_body={"username": username, "password": password},
    )
    assert response.status_code == 200, response.body
    assert "pingpong_session" in client.cookies
    return response.json()


def _create_user(
    admin: LiveHttpClient,
    username: str,
    password: str,
    *,
    display_name: str | None = None,
) -> dict[str, Any]:
    response = admin.request(
        "POST",
        "/api/admin/users",
        json_body={
            "username": username,
            "display_name": display_name or username.title(),
            "password": password,
            "role": "user",
        },
    )
    assert response.status_code == 201, response.body
    return response.json()


def _create_incomplete_upload(
    client: LiveHttpClient,
    filename: str,
    *,
    size: int = 10,
) -> str:
    encoded_name = base64.b64encode(filename.encode()).decode()
    response = client.request(
        "POST",
        "/api/uploads",
        headers={
            "Tus-Resumable": "1.0.0",
            "Upload-Length": str(size),
            "Upload-Metadata": f"filename {encoded_name},filetype dmlkZW8vbXA0",
        },
    )
    assert response.status_code == 201, response.body
    return response.headers["location"]


def _upload_video(client: LiveHttpClient, filename: str, payload: bytes) -> str:
    location = _create_incomplete_upload(client, filename, size=len(payload))
    response = client.request(
        "PATCH",
        location,
        body=payload,
        headers={
            "Tus-Resumable": "1.0.0",
            "Upload-Offset": "0",
            "Content-Type": "application/offset+octet-stream",
        },
    )
    assert response.status_code == 204, response.body
    return response.headers["upload-job-id"]


def _wait_for_job(client: LiveHttpClient, job_id: str, status: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.request("GET", f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.body
        job = response.json()
        if job["status"] == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not reach {status}")


def test_sessions_admin_user_lifecycle_and_query_token_rejection(live_app) -> None:
    _app, admin, _processor = live_app
    anonymous = admin.clone_without_cookies()

    assert anonymous.request("GET", "/api/auth/me").status_code == 401
    assert (
        anonymous.request("GET", "/api/jobs?token=legacy-header-secret&scope=all").status_code
        == 401
    )
    assert (
        anonymous.request(
            "GET",
            "/api/admin/users",
            headers={"X-Upload-Token": "legacy-header-secret"},
        ).status_code
        == 401
    )
    maintenance = anonymous.request(
        "GET",
        "/api/maintenance/active-work",
        headers={"X-Upload-Token": "legacy-header-secret"},
    )
    assert maintenance.status_code == 200
    assert maintenance.json()["active"] is False
    bad_login = anonymous.request(
        "POST",
        "/api/auth/login",
        json_body={"username": "admin", "password": "wrong-password"},
    )
    assert bad_login.status_code == 401

    admin_payload = _login(admin, " ADMIN ", ADMIN_PASSWORD)
    assert admin_payload["username"] == "admin"
    assert admin_payload["role"] == "admin"
    assert "password_hash" not in admin_payload

    alice_record = _create_user(admin, "Alice", "alice-password", display_name="Alice A")
    bob_record = _create_user(admin, "bob", "bob-password", display_name="Bob B")
    assert alice_record["username"] == "alice"
    assert bob_record["active"] is True

    duplicate = admin.request(
        "POST",
        "/api/admin/users",
        json_body={
            "username": "ALICE",
            "password": "another-password",
            "role": "user",
        },
    )
    assert duplicate.status_code == 409

    listed = admin.request("GET", "/api/admin/users")
    assert listed.status_code == 200
    assert listed.json()["total"] == 3
    assert {item["username"] for item in listed.json()["users"]} == {
        "admin",
        "alice",
        "bob",
    }

    bob = admin.clone_without_cookies()
    _login(bob, "bob", "bob-password")
    assert bob.request("GET", "/api/admin/users").status_code == 403

    changed = admin.request(
        "PATCH",
        f"/api/admin/users/{bob_record['id']}",
        json_body={"display_name": "Bob Updated", "password": "bob-new-password"},
    )
    assert changed.status_code == 200
    assert changed.json()["display_name"] == "Bob Updated"
    assert bob.request("GET", "/api/auth/me").status_code == 401
    assert (
        bob.clone_without_cookies()
        .request(
            "POST",
            "/api/auth/login",
            json_body={"username": "bob", "password": "bob-password"},
        )
        .status_code
        == 401
    )
    _login(bob, "bob", "bob-new-password")

    bob_other_session = admin.clone_without_cookies()
    _login(bob_other_session, "bob", "bob-new-password")
    own_password_change = bob.request(
        "POST",
        "/api/auth/change-password",
        json_body={
            "current_password": "bob-new-password",
            "new_password": "bob-self-service-password",
        },
    )
    assert own_password_change.status_code == 200
    assert own_password_change.json()["id"] == bob_record["id"]
    assert bob.request("GET", "/api/auth/me").status_code == 200
    assert bob_other_session.request("GET", "/api/auth/me").status_code == 401
    assert (
        bob.clone_without_cookies()
        .request(
            "POST",
            "/api/auth/login",
            json_body={"username": "bob", "password": "bob-new-password"},
        )
        .status_code
        == 401
    )
    bob_relogin = admin.clone_without_cookies()
    _login(bob_relogin, "bob", "bob-self-service-password")

    deactivated = admin.request(
        "PATCH",
        f"/api/admin/users/{bob_record['id']}",
        json_body={"active": False},
    )
    assert deactivated.status_code == 200
    assert deactivated.json()["active"] is False
    assert bob.request("GET", "/api/auth/me").status_code == 401
    assert bob_relogin.request("GET", "/api/auth/me").status_code == 401
    reactivated = admin.request(
        "PATCH",
        f"/api/admin/users/{bob_record['id']}",
        json_body={"active": True},
    )
    assert reactivated.status_code == 200

    cannot_disable_self = admin.request(
        "PATCH",
        f"/api/admin/users/{admin_payload['id']}",
        json_body={"active": False},
    )
    assert cannot_disable_self.status_code == 409

    logout = admin.request("POST", "/api/auth/logout")
    assert logout.status_code == 204
    assert logout.headers["clear-site-data"] == '"cache"'
    assert "pingpong_session" not in admin.cookies
    assert admin.request("GET", "/api/auth/me").status_code == 401


def test_two_users_are_isolated_while_admin_can_manage_all_jobs(live_app) -> None:
    app, admin, processor = live_app
    _login(admin, "admin", ADMIN_PASSWORD)
    alice_record = _create_user(admin, "alice", "alice-password")
    bob_record = _create_user(admin, "bob", "bob-password")

    alice = admin.clone_without_cookies()
    bob = admin.clone_without_cookies()
    _login(alice, "alice", "alice-password")
    _login(bob, "bob", "bob-password")

    alice_pending = _create_incomplete_upload(alice, "alice-pending.mp4")
    bob_pending = _create_incomplete_upload(bob, "bob-pending.mp4")
    alice_job_id = _upload_video(alice, "alice-match.mp4", b"alice-video")
    bob_job_id = _upload_video(bob, "bob-match.mp4", b"bob-video")
    _wait_for_job(alice, alice_job_id, "completed")
    _wait_for_job(bob, bob_job_id, "completed")

    alice_uploads = alice.request("GET", "/api/uploads?scope=mine").json()["uploads"]
    bob_uploads = bob.request("GET", "/api/uploads?scope=mine").json()["uploads"]
    assert [item["filename"] for item in alice_uploads] == ["alice-pending.mp4"]
    assert [item["filename"] for item in bob_uploads] == ["bob-pending.mp4"]
    assert alice.request("HEAD", bob_pending, headers={"Tus-Resumable": "1.0.0"}).status_code == 404
    assert bob.request("GET", alice_pending).status_code == 404

    alice_jobs = alice.request("GET", "/api/jobs?scope=mine&limit=50&offset=0").json()
    bob_jobs = bob.request("GET", "/api/jobs?scope=mine&limit=50&offset=0").json()
    assert alice_jobs["total"] == 1
    assert [item["id"] for item in alice_jobs["jobs"]] == [alice_job_id]
    assert bob_jobs["total"] == 1
    assert [item["id"] for item in bob_jobs["jobs"]] == [bob_job_id]
    assert alice.request("GET", "/api/jobs?scope=all").status_code == 403

    assert alice.request("GET", f"/api/jobs/{bob_job_id}").status_code == 404
    assert alice.request("GET", f"/api/jobs/{bob_job_id}/source").status_code == 404
    assert alice.request("GET", f"/api/jobs/{bob_job_id}/annotations").status_code == 404
    assert (
        alice.request("GET", f"/api/jobs/{bob_job_id}/files/best_points_reel.mp4").status_code
        == 404
    )

    all_uploads = admin.request("GET", "/api/uploads?scope=all").json()["uploads"]
    assert {item["owner"]["username"] for item in all_uploads} == {"alice", "bob"}
    all_jobs = admin.request("GET", "/api/jobs?scope=all&limit=50&offset=0").json()
    assert all_jobs["total"] == 2
    assert {item["id"] for item in all_jobs["jobs"]} == {alice_job_id, bob_job_id}
    assert {item["owner"]["username"] for item in all_jobs["jobs"]} == {
        "alice",
        "bob",
    }

    database = app.state.database
    source = app.state.settings.uploads_dir / "retry-source.mp4"
    source.write_bytes(b"retry-video")
    upload = database.create_upload(
        "retry-upload",
        "retry-source.mp4",
        source.stat().st_size,
        "video/mp4",
        source,
        user_id=alice_record["id"],
    )
    database.force_upload_offset(upload.id, upload.size)
    retry_job = database.complete_upload(upload.id)
    database.fail_job(retry_job.id, "intentional failure")

    assert bob.request("POST", f"/api/jobs/{retry_job.id}/retry").status_code == 404
    retried = alice.request("POST", f"/api/jobs/{retry_job.id}/retry")
    assert retried.status_code == 202, retried.body
    _wait_for_job(alice, retry_job.id, "completed")
    assert processor.calls.count("retry-source.mp4") == 1

    invalid_retry = alice.request("POST", f"/api/jobs/{retry_job.id}/retry")
    assert invalid_retry.status_code == 409
    reprocessed = alice.request("POST", f"/api/jobs/{retry_job.id}/reprocess")
    assert reprocessed.status_code == 202, reprocessed.body
    _wait_for_job(alice, retry_job.id, "completed")
    assert processor.calls.count("retry-source.mp4") == 2

    output_file = app.state.settings.outputs_dir / retry_job.id / "best_points_reel.mp4"
    assert source.is_file()
    assert output_file.is_file()
    source_download = alice.request(
        "GET",
        f"/api/jobs/{retry_job.id}/source?download=true",
    )
    assert source_download.status_code == 200
    assert "retry-source.mp4" in source_download.headers["content-disposition"]
    assert source_download.headers["cache-control"] == "private, no-store"
    assert bob.request("DELETE", f"/api/jobs/{retry_job.id}").status_code == 404
    deleted = alice.request("DELETE", f"/api/jobs/{retry_job.id}")
    assert deleted.status_code == 204, deleted.body
    assert database.get_job(retry_job.id) is None
    assert database.get_upload(upload.id) is None
    assert not source.exists()
    assert not output_file.exists()

    storage = admin.request("GET", "/api/storage")
    assert storage.status_code == 200
    assert storage.json()["summary"]["upload_count"] >= 4
    assert bob_record["id"] != alice_record["id"]
