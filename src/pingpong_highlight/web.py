from __future__ import annotations

import base64
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database, DriveImportRecord, JobRecord, UploadRecord
from pingpong_highlight.drive import (
    DriveDownloader,
    DriveImportError,
    DriveImportManager,
)
from pingpong_highlight.jobs import JobManager
from pingpong_highlight.pipeline.processor import HighlightProcessor
from pingpong_highlight.uploads import UploadError, UploadStore

TUS_VERSION = "1.0.0"


class DriveImportRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


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


def _upload_payload(record: UploadRecord) -> dict[str, Any]:
    return {
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


def _job_payload(record: JobRecord) -> dict[str, Any]:
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
    return {
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


def _drive_import_payload(record: DriveImportRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "filename": record.filename,
        "size": record.size,
        "offset": record.offset,
        "status": record.status,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def create_app(
    settings: Settings | None = None,
    *,
    processor: HighlightProcessor | None = None,
    drive_downloader: DriveDownloader | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    database = Database(settings.database_path)
    uploads = UploadStore(settings, database)
    jobs = JobManager(settings, database, processor)
    drive_imports = DriveImportManager(
        settings,
        database,
        uploads,
        jobs.enqueue,
        drive_downloader,
    )
    static_dir = Path(__file__).parent / "static"

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        uploads.reconcile()
        jobs.start()
        drive_imports.start()
        yield
        drive_imports.shutdown()
        jobs.shutdown()

    app = FastAPI(
        title="Ping-Pong Auto Highlight",
        version="0.11.1",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.uploads = uploads
    app.state.jobs = jobs
    app.state.drive_imports = drive_imports

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
            response.headers["Cache-Control"] = "private, no-store"
        return response

    async def authorize(request: Request) -> None:
        supplied = request.headers.get("X-Upload-Token") or request.query_params.get("token") or ""
        if not secrets.compare_digest(supplied, settings.upload_token):
            raise HTTPException(status_code=401, detail="Invalid upload token")

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
    async def handle_drive_import_error(
        _request: Request, exc: DriveImportError
    ) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config", dependencies=[Depends(authorize)])
    async def public_config() -> dict[str, Any]:
        return {
            "chunk_size": min(8 * 1024**2, settings.max_chunk_bytes),
            "max_upload_size": settings.max_upload_bytes,
            "video_sample_fps": settings.video_sample_fps,
            "max_points": settings.max_points,
            "reel_target_seconds": settings.reel_target_seconds,
        }

    @app.get("/api/uploads", dependencies=[Depends(authorize)])
    async def list_uploads() -> dict[str, list[dict[str, Any]]]:
        return {
            "uploads": [
                _upload_payload(upload) for upload in database.list_incomplete_uploads()
            ]
        }

    @app.get("/api/drive-imports", dependencies=[Depends(authorize)])
    async def list_drive_imports() -> dict[str, list[dict[str, Any]]]:
        return {
            "imports": [
                _drive_import_payload(record)
                for record in database.list_drive_imports()
            ]
        }

    @app.post(
        "/api/drive-imports",
        dependencies=[Depends(authorize)],
        status_code=202,
    )
    async def create_drive_import(payload: DriveImportRequest) -> dict[str, Any]:
        return _drive_import_payload(drive_imports.submit(payload.url))

    @app.post(
        "/api/drive-imports/{import_id}/retry",
        dependencies=[Depends(authorize)],
        status_code=202,
    )
    async def retry_drive_import(import_id: str) -> dict[str, Any]:
        return _drive_import_payload(drive_imports.retry(import_id))

    @app.delete(
        "/api/drive-imports/{import_id}",
        dependencies=[Depends(authorize)],
        status_code=204,
    )
    async def delete_drive_import(import_id: str) -> Response:
        drive_imports.delete(import_id)
        return Response(status_code=204)

    @app.options("/api/uploads")
    async def upload_options() -> Response:
        return Response(status_code=204, headers=_tus_headers(settings))

    @app.post("/api/uploads", dependencies=[Depends(authorize)])
    async def create_upload(request: Request) -> Response:
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

    @app.head("/api/uploads/{upload_id}", dependencies=[Depends(authorize)])
    async def head_upload(upload_id: str, request: Request) -> Response:
        require_tus(request)
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

    @app.get("/api/uploads/{upload_id}", dependencies=[Depends(authorize)])
    async def get_upload(upload_id: str) -> dict[str, Any]:
        return _upload_payload(uploads.get(upload_id))

    @app.delete("/api/uploads/{upload_id}", dependencies=[Depends(authorize)])
    async def delete_upload(upload_id: str, request: Request) -> Response:
        require_tus(request)
        await uploads.delete(upload_id)
        return Response(status_code=204, headers=_tus_headers(settings))

    @app.patch("/api/uploads/{upload_id}", dependencies=[Depends(authorize)])
    async def append_upload(upload_id: str, request: Request) -> Response:
        require_tus(request)
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

    @app.get("/api/jobs", dependencies=[Depends(authorize)])
    async def list_jobs() -> dict[str, list[dict[str, Any]]]:
        return {"jobs": [_job_payload(job) for job in database.list_jobs()]}

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(authorize)])
    async def get_job(job_id: str) -> dict[str, Any]:
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}/files/{filename}", dependencies=[Depends(authorize)])
    async def download_file(job_id: str, filename: str, download: bool = False) -> FileResponse:
        job = database.get_job(job_id)
        if job is None or not job.result:
            raise HTTPException(status_code=404, detail="Result not found")
        allowed = {item["name"] for item in job.result.get("files", [])}
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="Result file not found")
        job_dir = (settings.outputs_dir / job_id).resolve()
        path = (job_dir / filename).resolve()
        if path.parent != job_dir or not path.is_file():
            raise HTTPException(status_code=404, detail="Result file not found")
        media_type = "application/json" if path.suffix == ".json" else "video/mp4"
        return FileResponse(
            path,
            media_type=media_type,
            filename=filename if download else None,
        )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
