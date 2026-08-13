from __future__ import annotations

import base64
import hashlib
import math
import mimetypes
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from email.utils import formatdate
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pingpong_highlight.config import Settings
from pingpong_highlight.db import (
    AnnotationRecord,
    Database,
    DriveImportRecord,
    JobRecord,
    UploadRecord,
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
MEDIA_CACHE_CONTROL = "private, max-age=3600"
MEDIA_CHUNK_SIZE = 1024 * 1024


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
        version="1.2.3",
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
            response.headers.setdefault("Cache-Control", "private, no-store")
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
            "clip_pre_roll_seconds": settings.clip_pre_roll_seconds,
            "clip_post_roll_seconds": settings.clip_post_roll_seconds,
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

    def job_and_upload(job_id: str) -> tuple[JobRecord, UploadRecord]:
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        upload = database.get_upload(job.upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="Source video not found")
        return job, upload

    @app.get("/api/jobs/{job_id}/source", dependencies=[Depends(authorize)])
    async def stream_source(job_id: str, request: Request) -> MediaFileResponse:
        _job, upload = job_and_upload(job_id)
        source = upload.path.resolve()
        if not source.is_file():
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
        )

    @app.get("/api/jobs/{job_id}/annotations", dependencies=[Depends(authorize)])
    async def list_annotations(job_id: str) -> dict[str, Any]:
        job, _upload = job_and_upload(job_id)
        return {
            "job_id": job.id,
            "upload_id": job.upload_id,
            "annotations": [
                _annotation_payload(record)
                for record in database.list_annotations(job.upload_id)
            ],
        }

    @app.post(
        "/api/jobs/{job_id}/annotations",
        dependencies=[Depends(authorize)],
        status_code=201,
    )
    async def create_annotation(job_id: str, payload: AnnotationRequest) -> dict[str, Any]:
        job, _upload = job_and_upload(job_id)
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
        dependencies=[Depends(authorize)],
        status_code=204,
    )
    async def delete_annotation(job_id: str, annotation_id: str) -> Response:
        job, _upload = job_and_upload(job_id)
        if not database.delete_annotation(job.upload_id, annotation_id):
            raise HTTPException(status_code=404, detail="Annotation not found")
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/files/{filename}", dependencies=[Depends(authorize)])
    async def download_file(
        job_id: str, filename: str, request: Request, download: bool = False
    ) -> Response:
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
