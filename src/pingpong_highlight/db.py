from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class UploadRecord:
    id: str
    filename: str
    size: int
    offset: int
    content_type: str
    status: str
    path: Path
    job_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    upload_id: str
    status: str
    progress: float
    stage: str
    error: str | None
    result: dict[str, Any] | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class DriveImportRecord:
    id: str
    file_id: str
    resource_key: str | None
    filename: str | None
    size: int | None
    offset: int
    status: str
    error: str | None
    upload_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    id: str
    upload_id: str
    label: str
    start: float
    end: float
    note: str
    created_at: str
    updated_at: str


class StateConflict(RuntimeError):
    pass


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    size INTEGER NOT NULL CHECK (size >= 0),
                    offset INTEGER NOT NULL DEFAULT 0 CHECK (offset >= 0),
                    content_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    path TEXT NOT NULL,
                    job_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    upload_id TEXT NOT NULL UNIQUE REFERENCES uploads(id),
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL,
                    error TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS drive_imports (
                    id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    resource_key TEXT,
                    filename TEXT,
                    size INTEGER CHECK (size IS NULL OR size >= 0),
                    offset INTEGER NOT NULL DEFAULT 0 CHECK (offset >= 0),
                    status TEXT NOT NULL,
                    error TEXT,
                    upload_id TEXT REFERENCES uploads(id),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
                    label TEXT NOT NULL CHECK (label IN ('highlight', 'exclude')),
                    start REAL NOT NULL CHECK (start >= 0),
                    end REAL NOT NULL CHECK (end > start),
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS drive_imports_status_idx
                    ON drive_imports(status, created_at);
                CREATE INDEX IF NOT EXISTS annotations_upload_time_idx
                    ON annotations(upload_id, start, created_at);
                """
            )

    @staticmethod
    def _upload(row: sqlite3.Row | None) -> UploadRecord | None:
        if row is None:
            return None
        return UploadRecord(
            id=row["id"],
            filename=row["filename"],
            size=int(row["size"]),
            offset=int(row["offset"]),
            content_type=row["content_type"],
            status=row["status"],
            path=Path(row["path"]),
            job_id=row["job_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _job(row: sqlite3.Row | None) -> JobRecord | None:
        if row is None:
            return None
        return JobRecord(
            id=row["id"],
            upload_id=row["upload_id"],
            status=row["status"],
            progress=float(row["progress"]),
            stage=row["stage"],
            error=row["error"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _drive_import(row: sqlite3.Row | None) -> DriveImportRecord | None:
        if row is None:
            return None
        return DriveImportRecord(
            id=row["id"],
            file_id=row["file_id"],
            resource_key=row["resource_key"],
            filename=row["filename"],
            size=int(row["size"]) if row["size"] is not None else None,
            offset=int(row["offset"]),
            status=row["status"],
            error=row["error"],
            upload_id=row["upload_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _annotation(row: sqlite3.Row | None) -> AnnotationRecord | None:
        if row is None:
            return None
        return AnnotationRecord(
            id=row["id"],
            upload_id=row["upload_id"],
            label=row["label"],
            start=float(row["start"]),
            end=float(row["end"]),
            note=row["note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_upload(
        self,
        upload_id: str,
        filename: str,
        size: int,
        content_type: str,
        path: Path,
    ) -> UploadRecord:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO uploads
                    (id, filename, size, offset, content_type, status, path, created_at, updated_at)
                VALUES (?, ?, ?, 0, ?, 'uploading', ?, ?, ?)
                """,
                (upload_id, filename, size, content_type, str(path), timestamp, timestamp),
            )
        record = self.get_upload(upload_id)
        assert record is not None
        return record

    def register_completed_upload(
        self,
        upload_id: str,
        filename: str,
        size: int,
        content_type: str,
        path: Path,
        *,
        drive_import_id: str | None = None,
    ) -> tuple[UploadRecord, JobRecord]:
        timestamp = _now()
        job_id = uuid.uuid4().hex
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO uploads
                    (id, filename, size, offset, content_type, status, path, job_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    filename,
                    size,
                    size,
                    content_type,
                    str(path),
                    job_id,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO jobs
                    (id, upload_id, status, progress, stage, created_at, updated_at)
                VALUES (?, ?, 'queued', 0, 'queued', ?, ?)
                """,
                (job_id, upload_id, timestamp, timestamp),
            )
            if drive_import_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE drive_imports
                    SET status = 'completed', upload_id = ?, offset = ?, size = ?,
                        error = NULL, updated_at = ?
                    WHERE id = ? AND status = 'downloading'
                    """,
                    (upload_id, size, size, timestamp, drive_import_id),
                )
                if cursor.rowcount != 1:
                    raise StateConflict("Drive import state changed before completion")
            upload_row = connection.execute(
                "SELECT * FROM uploads WHERE id = ?", (upload_id,)
            ).fetchone()
            job_row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        upload = self._upload(upload_row)
        job = self._job(job_row)
        assert upload is not None and job is not None
        return upload, job

    def create_or_requeue_drive_import(
        self,
        file_id: str,
        resource_key: str | None,
    ) -> DriveImportRecord:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM drive_imports
                WHERE file_id = ? AND status != 'completed'
                ORDER BY created_at DESC LIMIT 1
                """,
                (file_id,),
            ).fetchone()
            record = self._drive_import(row)
            if record is None:
                import_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO drive_imports
                        (id, file_id, resource_key, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'queued', ?, ?)
                    """,
                    (import_id, file_id, resource_key, timestamp, timestamp),
                )
            else:
                import_id = record.id
                connection.execute(
                    """
                    UPDATE drive_imports
                    SET resource_key = COALESCE(?, resource_key),
                        status = CASE WHEN status = 'failed' THEN 'queued' ELSE status END,
                        error = CASE WHEN status = 'failed' THEN NULL ELSE error END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (resource_key, timestamp, import_id),
                )
            updated = connection.execute(
                "SELECT * FROM drive_imports WHERE id = ?", (import_id,)
            ).fetchone()
        result = self._drive_import(updated)
        assert result is not None
        return result

    def get_drive_import(self, import_id: str) -> DriveImportRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM drive_imports WHERE id = ?", (import_id,)
            ).fetchone()
        return self._drive_import(row)

    def list_drive_imports(
        self, *, include_completed: bool = False
    ) -> list[DriveImportRecord]:
        where = "" if include_completed else "WHERE status != 'completed'"
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM drive_imports {where} ORDER BY created_at DESC"
            ).fetchall()
        return [
            record
            for row in rows
            if (record := self._drive_import(row)) is not None
        ]

    def requeue_interrupted_drive_imports(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE drive_imports
                SET status = 'queued', error = NULL, updated_at = ?
                WHERE status IN ('resolving', 'downloading')
                """,
                (_now(),),
            )

    def claim_drive_import(self, import_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE drive_imports
                SET status = 'resolving', error = NULL, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (_now(), import_id),
            )
        return cursor.rowcount == 1

    def start_drive_import_download(self, import_id: str, filename: str) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE drive_imports
                SET filename = ?, status = 'downloading', updated_at = ?
                WHERE id = ? AND status = 'resolving'
                """,
                (filename, _now(), import_id),
            )
        if cursor.rowcount != 1:
            raise StateConflict("Drive import state changed before download")

    def update_drive_import_progress(
        self,
        import_id: str,
        offset: int,
        size: int | None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE drive_imports
                SET offset = ?, size = COALESCE(?, size), updated_at = ?
                WHERE id = ? AND status = 'downloading'
                """,
                (offset, size, _now(), import_id),
            )

    def fail_drive_import(self, import_id: str, error: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE drive_imports
                SET status = 'failed', error = ?, updated_at = ?
                WHERE id = ? AND status != 'completed'
                """,
                (error[:1000], _now(), import_id),
            )

    def retry_drive_import(self, import_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE drive_imports
                SET status = 'queued', error = NULL, updated_at = ?
                WHERE id = ? AND status = 'failed'
                """,
                (_now(), import_id),
            )
        return cursor.rowcount == 1

    def delete_drive_import(self, import_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM drive_imports WHERE id = ? AND status IN ('queued', 'failed')",
                (import_id,),
            )
        return cursor.rowcount == 1

    def get_upload(self, upload_id: str) -> UploadRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        return self._upload(row)

    def list_incomplete_uploads(self) -> list[UploadRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM uploads WHERE status = 'uploading' ORDER BY created_at"
            ).fetchall()
        return [record for row in rows if (record := self._upload(row)) is not None]

    def delete_incomplete_upload(self, upload_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM uploads WHERE id = ? AND status = 'uploading'",
                (upload_id,),
            )
        return cursor.rowcount == 1

    def force_upload_offset(self, upload_id: str, offset: int) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE uploads SET offset = ?, updated_at = ? WHERE id = ?",
                (offset, _now(), upload_id),
            )

    def advance_upload(self, upload_id: str, expected_offset: int, new_offset: int) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE uploads SET offset = ?, updated_at = ?
                WHERE id = ? AND offset = ? AND status = 'uploading'
                """,
                (new_offset, _now(), upload_id, expected_offset),
            )
            if cursor.rowcount != 1:
                raise StateConflict("Upload offset changed concurrently")

    def complete_upload(self, upload_id: str) -> JobRecord:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
            upload = self._upload(row)
            if upload is None:
                raise StateConflict("Upload does not exist")
            if upload.offset != upload.size:
                raise StateConflict("Upload is not complete")
            if upload.job_id:
                job_row = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (upload.job_id,)
                ).fetchone()
                job = self._job(job_row)
                if job is None:
                    raise StateConflict("Upload references a missing job")
                return job

            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO jobs
                    (id, upload_id, status, progress, stage, created_at, updated_at)
                VALUES (?, ?, 'queued', 0, 'queued', ?, ?)
                """,
                (job_id, upload_id, timestamp, timestamp),
            )
            connection.execute(
                """
                UPDATE uploads
                SET status = 'queued', job_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (job_id, timestamp, upload_id),
            )
            job_row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        job = self._job(job_row)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job(row)

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [record for row in rows if (record := self._job(row)) is not None]

    def create_annotation(
        self,
        upload_id: str,
        *,
        label: str,
        start: float,
        end: float,
        note: str = "",
    ) -> AnnotationRecord:
        annotation_id = uuid.uuid4().hex
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO annotations
                    (id, upload_id, label, start, end, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (annotation_id, upload_id, label, start, end, note, timestamp, timestamp),
            )
            row = connection.execute(
                "SELECT * FROM annotations WHERE id = ?", (annotation_id,)
            ).fetchone()
        annotation = self._annotation(row)
        assert annotation is not None
        return annotation

    def list_annotations(self, upload_id: str) -> list[AnnotationRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM annotations
                WHERE upload_id = ?
                ORDER BY start, created_at
                """,
                (upload_id,),
            ).fetchall()
        return [record for row in rows if (record := self._annotation(row)) is not None]

    def delete_annotation(self, upload_id: str, annotation_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM annotations WHERE id = ? AND upload_id = ?",
                (annotation_id, upload_id),
            )
        return cursor.rowcount == 1

    def list_queued_jobs(self) -> list[JobRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
        return [record for row in rows if (record := self._job(row)) is not None]

    def requeue_interrupted_jobs(self) -> None:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'queued', stage = 'queued-after-restart', updated_at = ?
                WHERE status = 'processing'
                """,
                (timestamp,),
            )
            connection.execute(
                """
                UPDATE uploads SET status = 'queued', updated_at = ?
                WHERE job_id IN (SELECT id FROM jobs WHERE status = 'queued')
                """,
                (timestamp,),
            )

    def claim_job(self, job_id: str) -> bool:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status = 'processing', stage = 'starting', updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (timestamp, job_id),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE uploads SET status = 'processing', updated_at = ? WHERE job_id = ?",
                    (timestamp, job_id),
                )
            return cursor.rowcount == 1

    def update_job(self, job_id: str, progress: float, stage: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET progress = ?, stage = ?, updated_at = ?
                WHERE id = ? AND status = 'processing'
                """,
                (max(0.0, min(1.0, progress)), stage, _now(), job_id),
            )

    def finish_job(self, job_id: str, result: dict[str, Any]) -> None:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed', progress = 1, stage = 'completed',
                    result_json = ?, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(result, ensure_ascii=False), timestamp, job_id),
            )
            connection.execute(
                "UPDATE uploads SET status = 'completed', updated_at = ? WHERE job_id = ?",
                (timestamp, job_id),
            )

    def fail_job(self, job_id: str, error: str) -> None:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', stage = 'failed', error = ?, updated_at = ?
                WHERE id = ?
                """,
                (error, timestamp, job_id),
            )
            connection.execute(
                "UPDATE uploads SET status = 'failed', updated_at = ? WHERE job_id = ?",
                (timestamp, job_id),
            )
