from __future__ import annotations

import base64
import hashlib
import ipaddress
import math
import mimetypes
import os
import secrets
import shutil
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import formatdate
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote

import anyio
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pingpong_highlight.auth import (
    generate_session_token,
    hash_password,
    hash_session_token,
    normalize_username,
    verify_password,
)
from pingpong_highlight.cleanup import FilesystemCleanup
from pingpong_highlight.config import Settings
from pingpong_highlight.db import (
    AnnotationRecord,
    Database,
    DriveImportRecord,
    JobRecord,
    StateConflict,
    UploadRecord,
    UserRecord,
)
from pingpong_highlight.drive import (
    DriveDownloader,
    DriveImportError,
    DriveImportManager,
)
from pingpong_highlight.jobs import JobManager
from pingpong_highlight.pipeline.processor import HighlightProcessor
from pingpong_highlight.uploads import UploadError, UploadStore

TUS_VERSION = "1.0.0"
MEDIA_CACHE_CONTROL = "private, no-store"
MEDIA_CHUNK_SIZE = 1024 * 1024
SESSION_COOKIE = "pingpong_session"
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 1024
LOGIN_WINDOW_SECONDS = 10.0
LOGIN_ATTEMPT_LIMIT = 10


class LoginRateLimiter:
    """Small in-process limiter that bounds expensive password verification work."""

    def __init__(
        self,
        *,
        window_seconds: float = LOGIN_WINDOW_SECONDS,
        attempt_limit: int = LOGIN_ATTEMPT_LIMIT,
    ) -> None:
        self.window_seconds = window_seconds
        self.attempt_limit = attempt_limit
        self._client_attempts: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def consume(self, client: str) -> int | None:
        """Reserve one attempt, or return the number of seconds before retry."""

        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            for key, attempts in list(self._client_attempts.items()):
                while attempts and attempts[0] <= cutoff:
                    attempts.popleft()
                if not attempts:
                    del self._client_attempts[key]

            attempts = self._client_attempts.setdefault(client, deque())
            if len(attempts) >= self.attempt_limit:
                return max(1, math.ceil(self.window_seconds - (now - attempts[0])))

            attempts.append(now)
        return None


def _login_client_address(request: Request, *, proxy_provider: str) -> str:
    direct = request.client.host if request.client is not None else "unknown"
    if proxy_provider == "none":
        return direct

    try:
        direct_address = ipaddress.ip_address(direct)
    except ValueError:
        return direct
    if not (direct_address.is_private or direct_address.is_loopback):
        return direct

    if proxy_provider == "cloudflare":
        forwarded = request.headers.get("CF-Connecting-IP", "").strip()
    elif proxy_provider == "ngrok":
        values = request.headers.get("X-Forwarded-For", "").split(",")
        forwarded = values[-1].strip() if values else ""
    else:
        return direct
    try:
        return str(ipaddress.ip_address(forwarded))
    except ValueError:
        return direct


async def _stream_file_range(path: Path, start: int, end: int) -> AsyncIterator[bytes]:
    async with await anyio.open_file(path, "rb") as media_file:
        await media_file.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = await media_file.read(min(MEDIA_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _parse_media_range(value: str, file_size: int) -> tuple[int, int]:
    try:
        units, requested = value.split("=", 1)
        if units.strip().lower() != "bytes" or "," in requested:
            raise ValueError
        start_value, separator, end_value = requested.strip().partition("-")
        if not separator:
            raise ValueError
        if start_value:
            start = int(start_value)
            end = min(int(end_value) + 1, file_size) if end_value else file_size
        else:
            suffix_length = int(end_value)
            if suffix_length <= 0:
                raise ValueError
            start = max(file_size - suffix_length, 0)
            end = file_size
        if start < 0 or start >= file_size or end <= start:
            raise ValueError
        return start, end
    except ValueError as exc:
        raise HTTPException(
            status_code=416,
            detail="Requested media range is not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        ) from exc


class MediaFileResponse(StreamingResponse):
    """Cancellation-aware byte-range response for large local media."""

    chunk_size = MEDIA_CHUNK_SIZE

    def __init__(
        self,
        path: Path,
        request: Request,
        *,
        media_type: str,
        filename: str | None = None,
    ) -> None:
        stat_result = path.stat()
        file_size = stat_result.st_size
        last_modified = formatdate(stat_result.st_mtime, usegmt=True)
        etag_value = f"{stat_result.st_mtime}-{file_size}"
        etag = f'"{hashlib.md5(etag_value.encode(), usedforsecurity=False).hexdigest()}"'
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": MEDIA_CACHE_CONTROL,
            "ETag": etag,
            "Last-Modified": last_modified,
        }
        if filename is not None:
            encoded_filename = quote(filename)
            if encoded_filename != filename:
                disposition = f"attachment; filename*=utf-8''{encoded_filename}"
            else:
                disposition = f'attachment; filename="{filename}"'
            headers["Content-Disposition"] = disposition

        start = 0
        end = file_size
        status_code = 200
        range_header = request.headers.get("range")
        if_range = request.headers.get("if-range")
        if range_header and (if_range is None or if_range in {etag, last_modified}):
            start, end = _parse_media_range(range_header, file_size)
            status_code = 206
            headers["Content-Range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["Content-Length"] = str(end - start)

        super().__init__(
            _stream_file_range(path, start, end),
            status_code=status_code,
            headers=headers,
            media_type=media_type,
        )


class DriveImportRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    display_name: str | None = Field(default=None, max_length=80)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    role: Literal["admin", "user"] = "user"


class UpdateUserRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)
    password: str | None = Field(
        default=None,
        min_length=MIN_PASSWORD_LENGTH,
        max_length=MAX_PASSWORD_LENGTH,
    )
    role: Literal["admin", "user"] | None = None
    active: bool | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=MAX_PASSWORD_LENGTH,
    )


class AnnotationRequest(BaseModel):
    label: Literal["highlight", "exclude"] = "highlight"
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    note: str = Field(default="", max_length=300)


def _tus_headers(settings: Settings) -> dict[str, str]:
    return {
        "Tus-Resumable": TUS_VERSION,
        "Tus-Version": TUS_VERSION,
        "Tus-Extension": "creation,checksum,termination",
        "Tus-Checksum-Algorithm": "sha1,sha256",
        "Tus-Max-Size": str(settings.max_upload_bytes),
        "Cache-Control": "no-store",
    }


def _metadata(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not value:
        return result
    for item in value.split(","):
        key, separator, encoded = item.strip().partition(" ")
        if not key:
            continue
        if not separator:
            result[key] = ""
            continue
        try:
            result[key] = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise UploadError(400, f"Invalid Upload-Metadata value for {key}") from exc
    return result


def _user_payload(record: UserRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "username": record.username,
        "display_name": record.display_name,
        "role": record.role,
        "active": record.active,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _upload_payload(
    record: UploadRecord,
    *,
    owner: UserRecord | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": record.id,
        "filename": record.filename,
        "size": record.size,
        "offset": record.offset,
        "content_type": record.content_type,
        "status": record.status,
        "job_id": record.job_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    if owner is not None:
        payload["owner"] = _user_payload(owner)
    return payload


def _job_payload(
    record: JobRecord,
    *,
    upload: UploadRecord | None = None,
    owner: UserRecord | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    result = record.result
    if result:
        result = dict(result)
        result["files"] = [
            item
            | {
                "url": f"/api/jobs/{record.id}/files/{quote(item['name'])}",
            }
            for item in result.get("files", [])
        ]
    payload: dict[str, Any] = {
        "id": record.id,
        "upload_id": record.upload_id,
        "status": record.status,
        "progress": record.progress,
        "stage": record.stage,
        "error": record.error,
        "result": result,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    if upload is not None:
        payload["filename"] = upload.filename
        payload["source_name"] = upload.filename
        payload["user_id"] = upload.user_id
    if owner is not None:
        payload["owner"] = _user_payload(owner)
    if source_type is not None:
        payload["source_type"] = source_type
    return payload


def _drive_import_payload(
    record: DriveImportRecord,
    *,
    owner: UserRecord | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": record.id,
        "filename": record.filename,
        "size": record.size,
        "offset": record.offset,
        "status": record.status,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    if owner is not None:
        payload["owner"] = _user_payload(owner)
    return payload


def _annotation_payload(record: AnnotationRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "label": record.label,
        "start": record.start,
        "end": record.end,
        "duration": round(record.end - record.start, 3),
        "note": record.note,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _read_or_create_admin_password(settings: Settings) -> str:
    if settings.bootstrap_admin_password:
        if not MIN_PASSWORD_LENGTH <= len(settings.bootstrap_admin_password) <= MAX_PASSWORD_LENGTH:
            raise RuntimeError(
                f"Bootstrap admin password must be {MIN_PASSWORD_LENGTH}-"
                f"{MAX_PASSWORD_LENGTH} characters"
            )
        return settings.bootstrap_admin_password

    password_path = settings.data_dir / ".admin-password"
    try:
        password = password_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        password = ""
    if password:
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            raise RuntimeError(
                f"Bootstrap password file must contain {MIN_PASSWORD_LENGTH}-"
                f"{MAX_PASSWORD_LENGTH} characters"
            )
        password_path.chmod(0o600)
        return password
    if password_path.exists():
        raise RuntimeError(f"Bootstrap password file is empty: {password_path}")

    password = secrets.token_urlsafe(18)
    try:
        descriptor = os.open(
            password_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        password = password_path.read_text(encoding="utf-8").strip()
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            raise RuntimeError(f"Bootstrap password file is invalid: {password_path}") from None
        password_path.chmod(0o600)
        return password
    try:
        os.write(descriptor, f"{password}\n".encode())
    finally:
        os.close(descriptor)
    return password


def _ensure_bootstrap_admin(settings: Settings, database: Database) -> UserRecord:
    configured = database.get_user_by_username(settings.bootstrap_admin_username)
    active_admins = [
        user
        for user in database.list_users(limit=500, include_inactive=False)
        if user.role == "admin"
    ]
    if configured is not None and configured.active and configured.role == "admin":
        admin = configured
    elif active_admins:
        admin = active_admins[0]
    elif configured is not None:
        updated = database.update_user(configured.id, role="admin", active=True)
        assert updated is not None
        admin = updated
    else:
        password = _read_or_create_admin_password(settings)
        admin = database.create_user(
            settings.bootstrap_admin_username,
            hash_password(password),
            display_name="系統管理員",
            role="admin",
        )
    database.claim_unowned_data(admin.id)
    return admin


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _tree_usage(root: Path) -> tuple[int, int]:
    size = 0
    count = 0
    if not root.exists():
        return size, count
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            size += path.stat().st_size
            count += 1
        except FileNotFoundError:
            continue
    return size, count


async def _authorize_request(request: Request) -> UserRecord:
    database: Database = request.app.state.database
    settings: Settings = request.app.state.settings
    session_token = request.cookies.get(SESSION_COOKIE, "")
    if session_token:
        resolved = database.resolve_session(hash_session_token(session_token))
        if resolved is not None:
            session, user = resolved
            database.touch_session(session.id)
            request.state.auth_session_id = session.id
            return user

    supplied = request.headers.get("X-Upload-Token", "")
    if (
        settings.legacy_token_auth_enabled
        and supplied
        and secrets.compare_digest(supplied, settings.upload_token)
    ):
        legacy_owner = database.get_user(request.app.state.bootstrap_admin.id)
        if legacy_owner is not None and legacy_owner.active:
            request.state.auth_session_id = None
            # Explicit legacy mode keeps old upload clients working against only
            # the bootstrap owner's data. It never confers administrator powers.
            return replace(legacy_owner, role="user")
    raise HTTPException(status_code=401, detail="Authentication required")


AuthenticatedUser = Annotated[UserRecord, Depends(_authorize_request)]


async def _require_admin(user: AuthenticatedUser) -> UserRecord:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user


Administrator = Annotated[UserRecord, Depends(_require_admin)]


def create_app(
    settings: Settings | None = None,
    *,
    processor: HighlightProcessor | None = None,
    drive_downloader: DriveDownloader | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    database = Database(settings.database_path)
    bootstrap_admin = _ensure_bootstrap_admin(settings, database)
    cleanup = FilesystemCleanup(settings, database)
    uploads = UploadStore(settings, database, cleanup)
    jobs = JobManager(settings, database, processor)
    drive_imports = DriveImportManager(
        settings,
        database,
        uploads,
        jobs.enqueue,
        drive_downloader,
        cleanup,
    )
    static_dir = Path(__file__).parent / "static"
    login_rate_limiter = LoginRateLimiter()
    login_hash_limiter = anyio.CapacityLimiter(2)
    login_pending_slots = threading.BoundedSemaphore(4)
    invalid_login_password_hash = hash_password(secrets.token_urlsafe(32))

    async def run_password_work(function: Callable[..., Any], *args: Any) -> Any:
        if not login_pending_slots.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail="Credential verification is busy; try again shortly",
                headers={"Retry-After": "1"},
            )
        try:
            return await anyio.to_thread.run_sync(
                function,
                *args,
                limiter=login_hash_limiter,
            )
        finally:
            login_pending_slots.release()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        database.delete_expired_sessions()
        cleanup.discard_obsolete_previous_artifacts()
        cleanup.drain()
        uploads.reconcile()
        jobs.start()
        drive_imports.start()
        yield
        drive_imports.shutdown()
        jobs.shutdown()

    app = FastAPI(
        title="Ping-Pong Auto Highlight",
        version="1.4.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.cleanup = cleanup
    app.state.uploads = uploads
    app.state.jobs = jobs
    app.state.drive_imports = drive_imports
    app.state.bootstrap_admin = bootstrap_admin

    @app.middleware("http")
    async def secure_responses(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; connect-src 'self'; "
            "img-src 'self' data:; media-src 'self' blob:; object-src 'none'; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
        )
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "private, no-store")
        elif request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    def owner_scope(user: UserRecord) -> str | None:
        return None if user.role == "admin" else user.id

    def owned_upload(upload_id: str, user: UserRecord) -> UploadRecord:
        record = database.get_upload(upload_id, user_id=owner_scope(user))
        if record is None:
            raise HTTPException(status_code=404, detail="Upload not found")
        return record

    def owned_drive_import(import_id: str, user: UserRecord) -> DriveImportRecord:
        record = database.get_drive_import(import_id, user_id=owner_scope(user))
        if record is None:
            raise HTTPException(status_code=404, detail="Drive import not found")
        return record

    def owned_job(job_id: str, user: UserRecord) -> JobRecord:
        record = database.get_job(job_id, user_id=owner_scope(user))
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return record

    def require_tus(request: Request) -> None:
        if request.headers.get("Tus-Resumable") != TUS_VERSION:
            raise UploadError(412, f"Tus-Resumable must be {TUS_VERSION}")

    @app.exception_handler(UploadError)
    async def handle_upload_error(_request: Request, exc: UploadError) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=_tus_headers(settings) | exc.headers,
        )

    @app.exception_handler(DriveImportError)
    async def handle_drive_import_error(_request: Request, exc: DriveImportError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/maintenance/active-work")
    async def maintenance_active_work(request: Request) -> dict[str, Any]:
        supplied = request.headers.get("X-Upload-Token", "")
        maintenance_token = settings.maintenance_token or settings.upload_token
        if not supplied or not secrets.compare_digest(supplied, maintenance_token):
            raise HTTPException(status_code=401, detail="Maintenance credential required")
        active_imports = [
            record
            for record in database.list_drive_imports()
            if record.status in {"queued", "resolving", "downloading"}
        ]
        # Read jobs/uploads after imports so a Drive import that transitions into
        # a queued job between snapshots cannot disappear from both sides.
        summary = database.get_storage_summary()
        return {
            "active": bool(
                summary.uploading_count
                or summary.queued_count
                or summary.processing_count
                or active_imports
            ),
            "jobs": {
                "queued": summary.queued_count,
                "processing": summary.processing_count,
                "completed": summary.completed_count,
            },
            "drive_imports": {"active": len(active_imports)},
            "uploads": {"incomplete": summary.uploading_count},
        }

    @app.post("/api/auth/login")
    async def login(payload: LoginRequest, request: Request) -> JSONResponse:
        try:
            username = normalize_username(payload.username)
        except ValueError:
            username = ""
        client_address = _login_client_address(
            request,
            proxy_provider=settings.trusted_proxy_provider,
        )
        retry_after = login_rate_limiter.consume(client_address)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts; try again later",
                headers={"Retry-After": str(retry_after)},
            )
        user = database.get_user_by_username(username) if username else None
        password_hash = user.password_hash if user is not None else invalid_login_password_hash
        password_matches = await run_password_work(
            verify_password,
            payload.password,
            password_hash,
        )
        if user is None or not user.active or not password_matches:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        token = generate_session_token()
        expires_at = datetime.now(UTC) + timedelta(seconds=settings.session_ttl_seconds)
        try:
            database.create_session(
                user.id,
                hash_session_token(token),
                expires_at,
                expected_password_hash=user.password_hash,
                expected_role=user.role,
            )
        except StateConflict as exc:
            raise HTTPException(status_code=401, detail="Invalid username or password") from exc
        response = JSONResponse(_user_payload(user))
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=settings.session_ttl_seconds,
            expires=expires_at,
            path="/",
            secure=settings.session_cookie_secure,
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/api/auth/me")
    async def auth_me(user: AuthenticatedUser) -> dict[str, Any]:
        return _user_payload(user)

    @app.post("/api/auth/logout")
    async def logout(request: Request) -> Response:
        token = request.cookies.get(SESSION_COOKIE, "")
        if token:
            database.revoke_session_by_token_hash(hash_session_token(token))
        response = Response(status_code=204)
        response.headers["Clear-Site-Data"] = '"cache"'
        response.delete_cookie(
            SESSION_COOKIE,
            path="/",
            secure=settings.session_cookie_secure,
            httponly=True,
            samesite="strict",
        )
        return response

    @app.post("/api/auth/change-password")
    async def change_password(
        payload: ChangePasswordRequest,
        request: Request,
        user: AuthenticatedUser,
    ) -> JSONResponse:
        client_address = _login_client_address(
            request,
            proxy_provider=settings.trusted_proxy_provider,
        )
        retry_after = login_rate_limiter.consume(client_address)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many credential attempts; try again later",
                headers={"Retry-After": str(retry_after)},
            )
        current_password_matches = await run_password_work(
            verify_password,
            payload.current_password,
            user.password_hash,
        )
        if not current_password_matches:
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        password_hash = await run_password_work(
            hash_password,
            payload.new_password,
        )
        token = generate_session_token()
        expires_at = datetime.now(UTC) + timedelta(seconds=settings.session_ttl_seconds)
        changed = database.change_password_and_create_session(
            user.id,
            expected_password_hash=user.password_hash,
            new_password_hash=password_hash,
            token_hash=hash_session_token(token),
            expires_at=expires_at,
        )
        if changed is None:
            raise HTTPException(
                status_code=401,
                detail="Credentials changed while this request was in progress; sign in again",
            )
        updated, _session = changed
        response = JSONResponse(_user_payload(updated))
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=settings.session_ttl_seconds,
            expires=expires_at,
            path="/",
            secure=settings.session_cookie_secure,
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/api/admin/users")
    async def list_users(
        _admin: Administrator,
    ) -> dict[str, Any]:
        users = database.list_users(limit=500)
        return {
            "users": [_user_payload(record) for record in users],
            "total": database.count_users(),
        }

    @app.post("/api/admin/users", status_code=201)
    async def create_user(
        payload: CreateUserRequest,
        _admin: Administrator,
    ) -> dict[str, Any]:
        password_hash = await run_password_work(
            hash_password,
            payload.password,
        )
        try:
            record = database.create_user(
                payload.username,
                password_hash,
                display_name=payload.display_name,
                role=payload.role,
            )
        except StateConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _user_payload(record)

    @app.patch("/api/admin/users/{user_id}")
    async def update_user(
        user_id: str,
        payload: UpdateUserRequest,
        admin: Administrator,
    ) -> dict[str, Any]:
        target = database.get_user(user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="User not found")
        fields = payload.model_fields_set
        if not fields:
            raise HTTPException(status_code=422, detail="No user changes were provided")
        for required_field in ("password", "role", "active"):
            if required_field in fields and getattr(payload, required_field) is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"{required_field} cannot be null",
                )
        if target.id == admin.id:
            if "active" in fields and payload.active is False:
                raise HTTPException(
                    status_code=409,
                    detail="You cannot deactivate your own account",
                )
            if "role" in fields and payload.role != "admin":
                raise HTTPException(status_code=409, detail="You cannot remove your own admin role")
        display_name: str | None = None
        if "display_name" in fields:
            display_name = (payload.display_name or target.username).strip()
        password_hash = None
        if "password" in fields and payload.password is not None:
            password_hash = await run_password_work(
                hash_password,
                payload.password,
            )
        try:
            updated = database.update_user(
                target.id,
                display_name=display_name,
                role=payload.role if "role" in fields else None,
                password_hash=password_hash,
                active=payload.active if "active" in fields else None,
            )
        except (StateConflict, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        assert updated is not None
        return _user_payload(updated)

    @app.get("/api/storage")
    async def storage_summary(
        _admin: Administrator,
    ) -> dict[str, Any]:
        summary = database.get_storage_summary()
        output_bytes, output_count = _tree_usage(settings.outputs_dir)
        used_bytes, _data_file_count = _tree_usage(settings.data_dir)
        disk = shutil.disk_usage(settings.data_dir)
        return {
            "summary": {
                "upload_count": summary.upload_count,
                "source_count": summary.upload_count,
                "source_bytes": summary.source_bytes,
                "output_count": output_count,
                "output_bytes": output_bytes,
                "used_bytes": used_bytes,
                "capacity_bytes": disk.total,
                "free_bytes": disk.free,
                "uploading_count": summary.uploading_count,
                "queued_count": summary.queued_count,
                "processing_count": summary.processing_count,
                "completed_count": summary.completed_count,
                "failed_count": summary.failed_count,
            }
        }

    @app.get("/api/config")
    async def public_config(_user: AuthenticatedUser) -> dict[str, Any]:
        return {
            "chunk_size": min(8 * 1024**2, settings.max_chunk_bytes),
            "max_upload_size": settings.max_upload_bytes,
            "video_sample_fps": settings.video_sample_fps,
            "minimum_point_score_ratio": settings.minimum_point_score_ratio,
            "max_points": settings.max_points,
            "reel_target_seconds": settings.reel_target_seconds,
            "clip_pre_roll_seconds": settings.clip_pre_roll_seconds,
            "clip_post_roll_seconds": settings.clip_post_roll_seconds,
        }

    @app.get("/api/uploads")
    async def list_uploads(
        user: AuthenticatedUser,
        scope: Literal["mine", "all"] = "mine",
    ) -> dict[str, list[dict[str, Any]]]:
        if scope == "all" and user.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator access required")
        user_id = None if scope == "all" else user.id
        return {
            "uploads": [
                _upload_payload(
                    upload,
                    owner=database.get_user(upload.user_id)
                    if scope == "all" and upload.user_id
                    else None,
                )
                for upload in database.list_uploads(
                    user_id=user_id,
                    status="uploading",
                    limit=500,
                )
            ]
        }

    @app.get("/api/drive-imports")
    async def list_drive_imports(
        user: AuthenticatedUser,
        scope: Literal["mine", "all"] = "mine",
    ) -> dict[str, list[dict[str, Any]]]:
        if scope == "all" and user.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator access required")
        user_id = None if scope == "all" else user.id
        return {
            "imports": [
                _drive_import_payload(
                    record,
                    owner=database.get_user(record.user_id)
                    if scope == "all" and record.user_id
                    else None,
                )
                for record in database.list_drive_imports(user_id=user_id)
            ]
        }

    @app.post(
        "/api/drive-imports",
        status_code=202,
    )
    async def create_drive_import(
        payload: DriveImportRequest,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        return _drive_import_payload(drive_imports.submit(payload.url, user_id=user.id))

    @app.post(
        "/api/drive-imports/{import_id}/retry",
        status_code=202,
    )
    async def retry_drive_import(
        import_id: str,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        owned_drive_import(import_id, user)
        return _drive_import_payload(drive_imports.retry(import_id, user_id=owner_scope(user)))

    @app.delete(
        "/api/drive-imports/{import_id}",
        status_code=204,
    )
    async def delete_drive_import(
        import_id: str,
        user: AuthenticatedUser,
    ) -> Response:
        owned_drive_import(import_id, user)
        drive_imports.delete(import_id, user_id=owner_scope(user))
        return Response(status_code=204)

    @app.options("/api/uploads")
    async def upload_options() -> Response:
        return Response(status_code=204, headers=_tus_headers(settings))

    @app.post("/api/uploads")
    async def create_upload(
        request: Request,
        user: AuthenticatedUser,
    ) -> Response:
        require_tus(request)
        try:
            upload_length = int(request.headers["Upload-Length"])
        except (KeyError, ValueError) as exc:
            raise UploadError(400, "A valid Upload-Length header is required") from exc
        metadata = _metadata(request.headers.get("Upload-Metadata"))
        record = uploads.create(
            metadata.get("filename", "video.mp4"),
            upload_length,
            metadata.get("filetype", "application/octet-stream"),
            user_id=user.id,
        )
        return Response(
            status_code=201,
            headers=_tus_headers(settings)
            | {
                "Location": f"/api/uploads/{record.id}",
                "Upload-Offset": "0",
                "Upload-Length": str(record.size),
            },
        )

    @app.head("/api/uploads/{upload_id}")
    async def head_upload(
        upload_id: str,
        request: Request,
        user: AuthenticatedUser,
    ) -> Response:
        require_tus(request)
        owned_upload(upload_id, user)
        record = uploads.get(upload_id)
        return Response(
            status_code=200,
            headers=_tus_headers(settings)
            | {
                "Upload-Offset": str(record.offset),
                "Upload-Length": str(record.size),
                "Upload-Metadata": "",
            },
        )

    @app.get("/api/uploads/{upload_id}")
    async def get_upload(
        upload_id: str,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        owned_upload(upload_id, user)
        return _upload_payload(uploads.get(upload_id))

    @app.delete("/api/uploads/{upload_id}")
    async def delete_upload(
        upload_id: str,
        request: Request,
        user: AuthenticatedUser,
    ) -> Response:
        require_tus(request)
        owned_upload(upload_id, user)
        await uploads.delete(upload_id)
        return Response(status_code=204, headers=_tus_headers(settings))

    @app.patch("/api/uploads/{upload_id}")
    async def append_upload(
        upload_id: str,
        request: Request,
        user: AuthenticatedUser,
    ) -> Response:
        require_tus(request)
        owned_upload(upload_id, user)
        if (
            request.headers.get("Content-Type", "").split(";", 1)[0]
            != "application/offset+octet-stream"
        ):
            raise UploadError(415, "Content-Type must be application/offset+octet-stream")
        try:
            expected_offset = int(request.headers["Upload-Offset"])
        except (KeyError, ValueError) as exc:
            raise UploadError(400, "A valid Upload-Offset header is required") from exc
        try:
            content_length = (
                int(request.headers["Content-Length"])
                if "Content-Length" in request.headers
                else None
            )
        except ValueError as exc:
            raise UploadError(400, "Content-Length is invalid") from exc

        record, job = await uploads.append(
            upload_id,
            expected_offset,
            request.stream(),
            content_length=content_length,
            checksum_header=request.headers.get("Upload-Checksum"),
            on_complete=jobs.enqueue,
        )
        response_headers = _tus_headers(settings) | {"Upload-Offset": str(record.offset)}
        if job:
            response_headers["Upload-Job-Id"] = job.id
        return Response(status_code=204, headers=response_headers)

    def enriched_job_payload(
        record: JobRecord,
        *,
        include_owner: bool = False,
        drive_upload_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        upload = database.get_upload(record.upload_id)
        owner = (
            database.get_user(upload.user_id)
            if include_owner and upload is not None and upload.user_id
            else None
        )
        source_type = (
            "google_drive"
            if drive_upload_ids is not None and record.upload_id in drive_upload_ids
            else "upload"
        )
        return _job_payload(
            record,
            upload=upload,
            owner=owner,
            source_type=source_type,
        )

    @app.get("/api/jobs")
    async def list_jobs(
        request: Request,
        user: AuthenticatedUser,
        scope: Literal["mine", "all"] | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if scope == "all" and user.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator access required")
        user_id = None if scope == "all" else user.id
        records = database.list_jobs(limit, offset=offset, user_id=user_id)
        drive_upload_ids = {
            record.upload_id
            for record in database.list_drive_imports(include_completed=True)
            if record.upload_id is not None
        }
        payload: dict[str, Any] = {
            "jobs": [
                enriched_job_payload(
                    record,
                    include_owner=scope == "all" and user.role == "admin",
                    drive_upload_ids=drive_upload_ids,
                )
                for record in records
            ]
        }
        if scope is not None or "limit" in request.query_params or "offset" in request.query_params:
            payload |= {
                "total": database.count_jobs(user_id=user_id),
                "limit": limit,
                "offset": offset,
            }
        return payload

    @app.get("/api/jobs/{job_id}")
    async def get_job(
        job_id: str,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        job = owned_job(job_id, user)
        drive_upload_ids = {
            record.upload_id
            for record in database.list_drive_imports(include_completed=True)
            if record.upload_id is not None
        }
        return enriched_job_payload(
            job,
            include_owner=user.role == "admin",
            drive_upload_ids=drive_upload_ids,
        )

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    async def retry_job(
        job_id: str,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        owned_job(job_id, user)
        try:
            if not jobs.retry(job_id, user_id=owner_scope(user)):
                raise HTTPException(status_code=404, detail="Job not found")
        except StateConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        updated = database.get_job(job_id)
        assert updated is not None
        return enriched_job_payload(updated, include_owner=user.role == "admin")

    @app.post("/api/jobs/{job_id}/reprocess", status_code=202)
    async def reprocess_job(
        job_id: str,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        owned_job(job_id, user)
        try:
            if not jobs.reprocess(job_id, user_id=owner_scope(user)):
                raise HTTPException(status_code=404, detail="Job not found")
        except StateConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        updated = database.get_job(job_id)
        assert updated is not None
        return enriched_job_payload(updated, include_owner=user.role == "admin")

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def delete_job(
        job_id: str,
        user: AuthenticatedUser,
    ) -> Response:
        job = owned_job(job_id, user)
        upload = database.get_upload(job.upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="Source video not found")
        if not _is_within(upload.path, settings.uploads_dir):
            raise HTTPException(status_code=500, detail="Stored source path is invalid")
        try:
            cleanup_targets = [
                (upload.path, "file"),
                (UploadStore.part_path(upload), "file"),
                *(
                    (temporary, "file")
                    for temporary in settings.uploads_dir.glob(f".{upload.id}.*.chunk")
                ),
                (settings.outputs_dir / job.id, "tree"),
                (settings.work_dir / job.id, "tree"),
                (settings.work_dir / f".{job.id}.previous", "tree"),
            ]
            deleted = database.delete_job(
                job.id,
                user_id=owner_scope(user),
                cleanup_targets=cleanup_targets,
            )
        except StateConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if deleted is None:
            raise HTTPException(status_code=404, detail="Job not found")
        cleanup.drain()
        return Response(status_code=204)

    def job_and_upload(
        job_id: str,
        user: UserRecord,
    ) -> tuple[JobRecord, UploadRecord]:
        job = owned_job(job_id, user)
        upload = database.get_upload(job.upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="Source video not found")
        return job, upload

    @app.get("/api/jobs/{job_id}/source")
    async def stream_source(
        job_id: str,
        request: Request,
        user: AuthenticatedUser,
        download: bool = False,
    ) -> MediaFileResponse:
        _job, upload = job_and_upload(job_id, user)
        source = upload.path.resolve()
        if not _is_within(source, settings.uploads_dir) or not source.is_file():
            raise HTTPException(status_code=404, detail="Source video not found")
        guessed_type = mimetypes.guess_type(upload.filename)[0]
        media_type = (
            upload.content_type
            if upload.content_type.startswith("video/")
            else guessed_type or "application/octet-stream"
        )
        return MediaFileResponse(
            source,
            request,
            media_type=media_type,
            filename=upload.filename if download else None,
        )

    @app.get("/api/jobs/{job_id}/annotations")
    async def list_annotations(
        job_id: str,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        job, _upload = job_and_upload(job_id, user)
        return {
            "job_id": job.id,
            "upload_id": job.upload_id,
            "annotations": [
                _annotation_payload(record) for record in database.list_annotations(job.upload_id)
            ],
        }

    @app.post(
        "/api/jobs/{job_id}/annotations",
        status_code=201,
    )
    async def create_annotation(
        job_id: str,
        payload: AnnotationRequest,
        user: AuthenticatedUser,
    ) -> dict[str, Any]:
        job, _upload = job_and_upload(job_id, user)
        if not math.isfinite(payload.start) or not math.isfinite(payload.end):
            raise HTTPException(status_code=422, detail="Annotation times must be finite")
        if payload.end <= payload.start:
            raise HTTPException(status_code=422, detail="Annotation end must follow start")
        duration = (job.result or {}).get("media", {}).get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise HTTPException(status_code=409, detail="Source duration is not available yet")
        if payload.start >= duration or payload.end > duration:
            raise HTTPException(status_code=422, detail="Annotation exceeds source duration")
        record = database.create_annotation(
            job.upload_id,
            label=payload.label,
            start=round(payload.start, 3),
            end=round(payload.end, 3),
            note=payload.note.strip(),
        )
        return _annotation_payload(record)

    @app.delete(
        "/api/jobs/{job_id}/annotations/{annotation_id}",
        status_code=204,
    )
    async def delete_annotation(
        job_id: str,
        annotation_id: str,
        user: AuthenticatedUser,
    ) -> Response:
        job, _upload = job_and_upload(job_id, user)
        if not database.delete_annotation(job.upload_id, annotation_id):
            raise HTTPException(status_code=404, detail="Annotation not found")
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/files/{filename}")
    async def download_file(
        job_id: str,
        filename: str,
        request: Request,
        user: AuthenticatedUser,
        download: bool = False,
    ) -> Response:
        job = owned_job(job_id, user)
        if not job.result:
            raise HTTPException(status_code=404, detail="Result not found")
        allowed = {item["name"] for item in job.result.get("files", [])}
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="Result file not found")
        job_dir = (settings.outputs_dir / job_id).resolve()
        path = (job_dir / filename).resolve()
        if path.parent != job_dir or not path.is_file():
            raise HTTPException(status_code=404, detail="Result file not found")
        media_type = "application/json" if path.suffix == ".json" else "video/mp4"
        if media_type != "video/mp4":
            return FileResponse(
                path,
                media_type=media_type,
                filename=filename if download else None,
            )
        return MediaFileResponse(
            path,
            request,
            media_type=media_type,
            filename=filename if download else None,
        )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
