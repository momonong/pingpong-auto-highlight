from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import shutil
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from pingpong_highlight.cleanup import FilesystemCleanup
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database, JobRecord, StateConflict, UploadRecord

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
SAFE_NAME = re.compile(r"[^\w.()\- ]+", re.UNICODE)
MAX_INCOMPLETE_UPLOADS_PER_USER = 3


class UploadError(RuntimeError):
    def __init__(self, status_code: int, detail: str, headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}


def clean_filename(filename: str) -> str:
    basename = Path(filename.replace("\\", "/")).name.strip()
    cleaned = SAFE_NAME.sub("_", basename).strip(" .")
    return cleaned[:180] or "video.mp4"


def _checksum(header: str | None) -> tuple[str, bytes] | None:
    if not header:
        return None
    algorithm, separator, encoded = header.partition(" ")
    if not separator or algorithm not in {"sha1", "sha256"}:
        raise UploadError(400, "Unsupported or malformed Upload-Checksum")
    try:
        digest = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise UploadError(400, "Malformed Upload-Checksum") from exc
    return algorithm, digest


class UploadStore:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        cleanup: FilesystemCleanup | None = None,
    ):
        self.settings = settings
        self.database = database
        self.cleanup = cleanup or FilesystemCleanup(settings, database)
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, upload_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(upload_id, asyncio.Lock())

    @staticmethod
    def part_path(record: UploadRecord) -> Path:
        return record.path.with_name(record.path.name + ".part")

    def create(
        self,
        filename: str,
        size: int,
        content_type: str,
        *,
        user_id: str | None = None,
    ) -> UploadRecord:
        if size <= 0:
            raise UploadError(400, "Upload-Length must be positive")
        if size > self.settings.max_upload_bytes:
            raise UploadError(413, "Video exceeds the configured upload size limit")
        if (
            user_id is not None
            and self.database.count_uploads(user_id=user_id, status="uploading")
            >= MAX_INCOMPLETE_UPLOADS_PER_USER
        ):
            raise UploadError(
                409,
                "Too many incomplete uploads; resume or delete one before starting another",
            )

        reserved = sum(
            max(0, record.size - record.offset)
            for record in self.database.list_incomplete_uploads()
        )
        free = shutil.disk_usage(self.settings.uploads_dir).free
        if free < self.settings.download_min_free_bytes + reserved + size:
            raise UploadError(507, "Not enough free disk space for this upload")

        upload_id = uuid.uuid4().hex
        filename = clean_filename(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            suffix = ".video"
        destination = self.settings.uploads_dir / f"{upload_id}{suffix}"
        record = self.database.create_upload(
            upload_id,
            filename,
            size,
            content_type or "application/octet-stream",
            destination,
            user_id=user_id,
        )
        self.part_path(record).touch(exist_ok=False)
        return record

    def import_completed_file(
        self,
        filename: str,
        content_type: str,
        source: Path,
        *,
        drive_import_id: str,
        user_id: str | None = None,
    ) -> tuple[UploadRecord, JobRecord]:
        filename = clean_filename(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UploadError(415, "Google Drive file must use a supported video extension")

        try:
            size = source.stat().st_size
        except FileNotFoundError as exc:
            raise UploadError(500, "Downloaded Google Drive file is missing") from exc
        if size <= 0:
            raise UploadError(400, "Downloaded Google Drive file is empty")
        if size > self.settings.max_upload_bytes:
            raise UploadError(413, "Video exceeds the configured upload size limit")

        upload_id = uuid.uuid4().hex
        destination = self.settings.uploads_dir / f"{upload_id}{suffix}"
        os.replace(source, destination)
        try:
            return self.database.register_completed_upload(
                upload_id,
                filename,
                size,
                content_type or "application/octet-stream",
                destination,
                drive_import_id=drive_import_id,
                user_id=user_id,
            )
        except Exception:
            os.replace(destination, source)
            raise

    def reconcile(self) -> None:
        for record in self.database.list_incomplete_uploads():
            if record.path.exists():
                actual = record.path.stat().st_size
                if actual != record.size:
                    raise StateConflict(
                        f"Completed upload {record.id} does not match its declared size"
                    )
                self.database.force_upload_offset(record.id, actual)
                self.database.complete_upload(record.id)
                continue
            part = self.part_path(record)
            actual = part.stat().st_size if part.exists() else 0
            if actual > record.size:
                raise StateConflict(f"Partial upload {record.id} is larger than declared")
            if actual == record.size and actual > 0:
                os.replace(part, record.path)
                self.database.force_upload_offset(record.id, actual)
                self.database.complete_upload(record.id)
                continue
            if actual != record.offset:
                self.database.force_upload_offset(record.id, actual)

    def get(self, upload_id: str) -> UploadRecord:
        record = self.database.get_upload(upload_id)
        if record is None:
            raise UploadError(404, "Upload not found")
        if record.status == "uploading":
            part = self.part_path(record)
            actual = part.stat().st_size if part.exists() else 0
            if actual != record.offset and actual <= record.size:
                self.database.force_upload_offset(upload_id, actual)
                record = self.database.get_upload(upload_id)
                assert record is not None
        return record

    async def delete(self, upload_id: str) -> UploadRecord:
        lock = await self._lock_for(upload_id)
        async with lock:
            record = self.get(upload_id)
            if record.status != "uploading":
                raise UploadError(409, "Only an incomplete upload can be deleted")
            targets = [
                (self.part_path(record), "file"),
                (record.path, "file"),
                *(
                    (temporary, "file")
                    for temporary in self.settings.uploads_dir.glob(f".{upload_id}.*.chunk")
                ),
            ]
            if not self.database.delete_incomplete_upload(
                upload_id,
                cleanup_targets=targets,
            ):
                raise UploadError(409, "Upload state changed before it could be deleted")
            self.cleanup.drain()
            return record

    async def append(
        self,
        upload_id: str,
        expected_offset: int,
        chunks: AsyncIterator[bytes],
        *,
        content_length: int | None,
        checksum_header: str | None,
        on_complete: Callable[[str], None] | None = None,
    ) -> tuple[UploadRecord, JobRecord | None]:
        if content_length is not None and content_length > self.settings.max_chunk_bytes:
            raise UploadError(413, "Upload chunk exceeds the configured chunk size limit")
        expected_checksum = _checksum(checksum_header)
        lock = await self._lock_for(upload_id)

        async with lock:
            record = self.get(upload_id)
            if record.status != "uploading":
                raise UploadError(409, "Upload is no longer writable")
            if expected_offset != record.offset:
                raise UploadError(
                    409, "Upload offset mismatch", {"Upload-Offset": str(record.offset)}
                )
            expected_write = (
                content_length
                if content_length is not None
                else min(self.settings.max_chunk_bytes, record.size - expected_offset)
            )
            if shutil.disk_usage(
                self.settings.uploads_dir
            ).free < self.settings.download_min_free_bytes + (2 * expected_write):
                raise UploadError(507, "Not enough free disk space to continue this upload")

            temporary = self.settings.uploads_dir / f".{upload_id}.{uuid.uuid4().hex}.chunk"
            algorithm = expected_checksum[0] if expected_checksum else "sha256"
            digest = hashlib.new(algorithm)
            received = 0
            try:
                with temporary.open("xb") as handle:
                    async for chunk in chunks:
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > self.settings.max_chunk_bytes:
                            raise UploadError(
                                413, "Upload chunk exceeds the configured chunk size limit"
                            )
                        if expected_offset + received > record.size:
                            raise UploadError(413, "Upload chunk exceeds the declared video length")
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())

                if content_length is not None and received != content_length:
                    raise UploadError(400, "Received chunk length does not match Content-Length")
                if received == 0:
                    raise UploadError(400, "Upload chunk must not be empty")
                if expected_checksum and digest.digest() != expected_checksum[1]:
                    raise UploadError(
                        460,
                        "Upload checksum mismatch",
                        {"Upload-Offset": str(record.offset)},
                    )

                part = self.part_path(record)
                actual_offset = part.stat().st_size if part.exists() else 0
                if actual_offset != expected_offset:
                    self.database.force_upload_offset(upload_id, actual_offset)
                    raise UploadError(
                        409,
                        "Upload offset changed concurrently",
                        {"Upload-Offset": str(actual_offset)},
                    )
                with part.open("ab") as destination, temporary.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                    destination.flush()
                    os.fsync(destination.fileno())

                new_offset = expected_offset + received
                self.database.advance_upload(upload_id, expected_offset, new_offset)
                job: JobRecord | None = None
                if new_offset == record.size:
                    os.replace(part, record.path)
                    job = self.database.complete_upload(upload_id)
                    if on_complete:
                        on_complete(job.id)
                updated = self.database.get_upload(upload_id)
                assert updated is not None
                return updated, job
            finally:
                temporary.unlink(missing_ok=True)
