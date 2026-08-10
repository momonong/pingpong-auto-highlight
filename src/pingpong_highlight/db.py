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

                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
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
