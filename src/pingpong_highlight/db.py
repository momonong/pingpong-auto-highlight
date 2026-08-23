from __future__ import annotations

import json
import re
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


_FILENAME_TIMESTAMP = re.compile(r"(?:^|[_-])(\d{8})[_-](\d{6})")


def _recorded_at_from_filename(filename: str) -> str | None:
    match = _FILENAME_TIMESTAMP.search(filename)
    if match is None:
        return None
    try:
        recorded_at = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    # Phone filenames contain wall-clock time but no timezone. Keep that value
    # timezone-naive so the browser does not shift a late-night recording into
    # the following day when it formats or filters the library.
    return recorded_at.isoformat()


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
    recorded_at: str | None
    recorded_at_source: str | None
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


@dataclass(frozen=True, slots=True)
class HighlightClipRecord:
    id: str
    job_id: str
    upload_id: str
    clip_filename: str
    source_name: str
    source_created_at: str
    source_date: str
    source_date_source: str
    start: float
    end: float
    rally_start: float
    rally_end: float
    score: float
    relative_score: float
    source_rank: int
    reason: str
    library_version: str
    active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CompilationRecord:
    id: str
    name: str
    status: str
    file_name: str | None
    duration: float | None
    error: str | None
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
                    recorded_at TEXT,
                    recorded_at_source TEXT,
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

                CREATE TABLE IF NOT EXISTS highlight_clips (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
                    clip_filename TEXT NOT NULL,
                    start REAL NOT NULL CHECK (start >= 0),
                    end REAL NOT NULL CHECK (end > start),
                    rally_start REAL NOT NULL CHECK (rally_start >= 0),
                    rally_end REAL NOT NULL CHECK (rally_end > rally_start),
                    score REAL NOT NULL,
                    relative_score REAL NOT NULL CHECK (
                        relative_score >= 0 AND relative_score <= 1.000001
                    ),
                    source_rank INTEGER NOT NULL CHECK (source_rank > 0),
                    reason TEXT NOT NULL DEFAULT '',
                    library_version TEXT NOT NULL DEFAULT 'legacy-result',
                    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(job_id, clip_filename)
                );

                CREATE TABLE IF NOT EXISTS compilations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    file_name TEXT,
                    duration REAL CHECK (duration IS NULL OR duration >= 0),
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS compilation_items (
                    compilation_id TEXT NOT NULL REFERENCES compilations(id)
                        ON DELETE CASCADE,
                    highlight_id TEXT NOT NULL REFERENCES highlight_clips(id),
                    position INTEGER NOT NULL CHECK (position >= 0),
                    PRIMARY KEY (compilation_id, position),
                    UNIQUE (compilation_id, highlight_id)
                );

                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS drive_imports_status_idx
                    ON drive_imports(status, created_at);
                CREATE INDEX IF NOT EXISTS annotations_upload_time_idx
                    ON annotations(upload_id, start, created_at);
                CREATE INDEX IF NOT EXISTS highlight_clips_quality_idx
                    ON highlight_clips(relative_score DESC, score DESC);
                CREATE INDEX IF NOT EXISTS highlight_clips_source_idx
                    ON highlight_clips(job_id, source_rank);
                CREATE INDEX IF NOT EXISTS compilations_status_idx
                    ON compilations(status, created_at);
                """
            )
            upload_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(uploads)").fetchall()
            }
            if "recorded_at" not in upload_columns:
                connection.execute("ALTER TABLE uploads ADD COLUMN recorded_at TEXT")
            if "recorded_at_source" not in upload_columns:
                connection.execute(
                    "ALTER TABLE uploads ADD COLUMN recorded_at_source TEXT"
                )
            highlight_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(highlight_clips)"
                ).fetchall()
            }
            if "library_version" not in highlight_columns:
                connection.execute(
                    """
                    ALTER TABLE highlight_clips
                    ADD COLUMN library_version TEXT NOT NULL DEFAULT 'legacy-result'
                    """
                )
            if "active" not in highlight_columns:
                connection.execute(
                    """
                    ALTER TABLE highlight_clips
                    ADD COLUMN active INTEGER NOT NULL DEFAULT 1
                    CHECK (active IN (0, 1))
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS highlight_clips_active_idx
                ON highlight_clips(active, job_id, source_rank)
                """
            )
            for row in connection.execute(
                "SELECT id, filename, recorded_at_source FROM uploads"
            ).fetchall():
                recorded_at = _recorded_at_from_filename(row["filename"])
                if recorded_at is not None and row["recorded_at_source"] in {
                    None,
                    "filename",
                }:
                    connection.execute(
                        """
                        UPDATE uploads
                        SET recorded_at = ?, recorded_at_source = 'filename'
                        WHERE id = ?
                        """,
                        (recorded_at, row["id"]),
                    )
            self._backfill_highlight_clips(connection)

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
            recorded_at=row["recorded_at"],
            recorded_at_source=row["recorded_at_source"],
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

    @staticmethod
    def _highlight_clip(row: sqlite3.Row | None) -> HighlightClipRecord | None:
        if row is None:
            return None
        return HighlightClipRecord(
            id=row["id"],
            job_id=row["job_id"],
            upload_id=row["upload_id"],
            clip_filename=row["clip_filename"],
            source_name=row["source_name"],
            source_created_at=row["source_created_at"],
            source_date=row["source_date"],
            source_date_source=row["source_date_source"],
            start=float(row["start"]),
            end=float(row["end"]),
            rally_start=float(row["rally_start"]),
            rally_end=float(row["rally_end"]),
            score=float(row["score"]),
            relative_score=float(row["relative_score"]),
            source_rank=int(row["source_rank"]),
            reason=row["reason"],
            library_version=row["library_version"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _compilation(row: sqlite3.Row | None) -> CompilationRecord | None:
        if row is None:
            return None
        return CompilationRecord(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            file_name=row["file_name"],
            duration=float(row["duration"]) if row["duration"] is not None else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _replace_highlight_clips(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        upload_id: str,
        result: dict[str, Any],
        timestamp: str,
        library_version: str,
        active: bool,
        file_prefix: str = "",
    ) -> None:
        files = [
            item
            for item in result.get("files", [])
            if item.get("kind") in {"highlight", "point", "clip"}
            and isinstance(item.get("name"), str)
        ]
        points = result.get("points", [])
        if not isinstance(points, list):
            points = []

        candidate_scores = [
            float(candidate["score"])
            for candidate in result.get("candidates", [])
            if isinstance(candidate, dict)
            and isinstance(candidate.get("score"), (int, float))
        ]
        point_scores = [
            float(point["score"])
            for point in points
            if isinstance(point, dict) and isinstance(point.get("score"), (int, float))
        ]
        best_score = max(candidate_scores or point_scores or [0.0])
        for index, item in enumerate(files):
            if index >= len(points) or not isinstance(points[index], dict):
                continue
            point = points[index]
            try:
                start = float(point.get("clip_start", point.get("start")))
                end = float(point.get("clip_end", point.get("end")))
                rally_start = float(point.get("rally_start", start))
                rally_end = float(point.get("rally_end", end))
                score = float(point.get("score", 0.0))
                source_rank = int(point.get("rank") or index + 1)
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= start < end and 0 <= rally_start < rally_end):
                continue

            relative_clip = Path(file_prefix) / item["name"]
            if relative_clip.is_absolute() or ".." in relative_clip.parts:
                continue
            clip_filename = relative_clip.as_posix()
            clip_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"highlightcraft:{job_id}:{clip_filename}",
            ).hex
            relative_score = score / best_score if best_score > 0 else 0.0
            relative_score = max(0.0, min(1.0, relative_score))
            connection.execute(
                """
                INSERT INTO highlight_clips (
                    id, job_id, upload_id, clip_filename, start, end,
                    rally_start, rally_end, score, relative_score, source_rank,
                    reason, library_version, active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, clip_filename) DO UPDATE SET
                    start = excluded.start,
                    end = excluded.end,
                    rally_start = excluded.rally_start,
                    rally_end = excluded.rally_end,
                    score = excluded.score,
                    relative_score = excluded.relative_score,
                    source_rank = excluded.source_rank,
                    reason = excluded.reason,
                    library_version = excluded.library_version,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                (
                    clip_id,
                    job_id,
                    upload_id,
                    clip_filename,
                    start,
                    end,
                    rally_start,
                    rally_end,
                    score,
                    relative_score,
                    source_rank,
                    str(point.get("reason", "")),
                    library_version,
                    int(active),
                    timestamp,
                    timestamp,
                ),
            )

    def _backfill_highlight_clips(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT jobs.id, jobs.upload_id, jobs.result_json, jobs.updated_at
            FROM jobs
            WHERE jobs.status = 'completed' AND jobs.result_json IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            try:
                result = json.loads(row["result_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            result_version = str(result.get("algorithm_version") or "")
            is_library_result = result_version.startswith("highlight-library-")
            has_rebuilt_active = connection.execute(
                """
                SELECT 1 FROM highlight_clips
                WHERE job_id = ? AND active = 1 AND clip_filename LIKE 'clip-sets/%'
                LIMIT 1
                """,
                (row["id"],),
            ).fetchone()
            self._replace_highlight_clips(
                connection,
                job_id=row["id"],
                upload_id=row["upload_id"],
                result=result,
                timestamp=row["updated_at"],
                library_version=result_version if is_library_result else "legacy-result",
                active=has_rebuilt_active is None,
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
        recorded_at = _recorded_at_from_filename(filename)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO uploads
                    (id, filename, size, offset, content_type, status, path,
                     recorded_at, recorded_at_source, created_at, updated_at)
                VALUES (?, ?, ?, 0, ?, 'uploading', ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    filename,
                    size,
                    content_type,
                    str(path),
                    recorded_at,
                    "filename" if recorded_at is not None else None,
                    timestamp,
                    timestamp,
                ),
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
        recorded_at = _recorded_at_from_filename(filename)
        job_id = uuid.uuid4().hex
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO uploads
                    (id, filename, size, offset, content_type, status, path, job_id,
                     recorded_at, recorded_at_source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    filename,
                    size,
                    size,
                    content_type,
                    str(path),
                    job_id,
                    recorded_at,
                    "filename" if recorded_at is not None else None,
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
            row = connection.execute(
                "SELECT upload_id FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE highlight_clips SET active = 0 WHERE job_id = ?",
                    (job_id,),
                )
                self._replace_highlight_clips(
                    connection,
                    job_id=job_id,
                    upload_id=row["upload_id"],
                    result=result,
                    timestamp=timestamp,
                    library_version=str(
                        result.get("algorithm_version") or "job-result"
                    ),
                    active=True,
                )

    def activate_highlight_result(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        file_prefix: str,
        library_version: str,
    ) -> int:
        prefix = Path(file_prefix)
        if prefix.is_absolute() or ".." in prefix.parts:
            raise StateConflict("Highlight file prefix must stay inside the job output")
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT upload_id FROM jobs WHERE id = ? AND status = 'completed'",
                (job_id,),
            ).fetchone()
            if row is None:
                raise StateConflict("Completed source job not found")
            connection.execute(
                "UPDATE highlight_clips SET active = 0 WHERE job_id = ?",
                (job_id,),
            )
            self._replace_highlight_clips(
                connection,
                job_id=job_id,
                upload_id=row["upload_id"],
                result=result,
                timestamp=timestamp,
                library_version=library_version,
                active=True,
                file_prefix=file_prefix,
            )
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM highlight_clips
                    WHERE job_id = ? AND active = 1
                    """,
                    (job_id,),
                ).fetchone()[0]
            )
            if count == 0:
                raise StateConflict("The analysis produced no highlight clips to activate")
        return count

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

    def list_highlight_clips(self) -> list[HighlightClipRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT highlight_clips.*, uploads.filename AS source_name,
                       uploads.created_at AS source_created_at,
                       COALESCE(uploads.recorded_at, uploads.created_at) AS source_date,
                       COALESCE(uploads.recorded_at_source, 'uploaded') AS source_date_source
                FROM highlight_clips
                JOIN uploads ON uploads.id = highlight_clips.upload_id
                JOIN jobs ON jobs.id = highlight_clips.job_id
                WHERE jobs.status = 'completed' AND highlight_clips.active = 1
                ORDER BY highlight_clips.relative_score DESC,
                         highlight_clips.score DESC,
                         uploads.created_at DESC,
                         highlight_clips.start
                """
            ).fetchall()
        return [
            record
            for row in rows
            if (record := self._highlight_clip(row)) is not None
        ]

    def get_highlight_clip(self, highlight_id: str) -> HighlightClipRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT highlight_clips.*, uploads.filename AS source_name,
                       uploads.created_at AS source_created_at,
                       COALESCE(uploads.recorded_at, uploads.created_at) AS source_date,
                       COALESCE(uploads.recorded_at_source, 'uploaded') AS source_date_source
                FROM highlight_clips
                JOIN uploads ON uploads.id = highlight_clips.upload_id
                WHERE highlight_clips.id = ?
                """,
                (highlight_id,),
            ).fetchone()
        return self._highlight_clip(row)

    def create_compilation(
        self,
        *,
        name: str,
        highlight_ids: list[str],
    ) -> CompilationRecord:
        ordered_ids = list(dict.fromkeys(highlight_ids))
        if not ordered_ids:
            raise StateConflict("A compilation needs at least one highlight")
        compilation_id = uuid.uuid4().hex
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in ordered_ids)
            rows = connection.execute(
                f"SELECT id FROM highlight_clips WHERE id IN ({placeholders})",
                ordered_ids,
            ).fetchall()
            available = {row["id"] for row in rows}
            missing = [
                highlight_id
                for highlight_id in ordered_ids
                if highlight_id not in available
            ]
            if missing:
                raise StateConflict("One or more selected highlights no longer exist")
            connection.execute(
                """
                INSERT INTO compilations
                    (id, name, status, created_at, updated_at)
                VALUES (?, ?, 'queued', ?, ?)
                """,
                (compilation_id, name, timestamp, timestamp),
            )
            connection.executemany(
                """
                INSERT INTO compilation_items
                    (compilation_id, highlight_id, position)
                VALUES (?, ?, ?)
                """,
                [
                    (compilation_id, highlight_id, position)
                    for position, highlight_id in enumerate(ordered_ids)
                ],
            )
            row = connection.execute(
                "SELECT * FROM compilations WHERE id = ?",
                (compilation_id,),
            ).fetchone()
        record = self._compilation(row)
        assert record is not None
        return record

    def get_compilation(self, compilation_id: str) -> CompilationRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM compilations WHERE id = ?",
                (compilation_id,),
            ).fetchone()
        return self._compilation(row)

    def list_compilations(self, limit: int = 50) -> list[CompilationRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM compilations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            record for row in rows if (record := self._compilation(row)) is not None
        ]

    def list_compilation_highlights(
        self,
        compilation_id: str,
    ) -> list[HighlightClipRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT highlight_clips.*, uploads.filename AS source_name,
                       uploads.created_at AS source_created_at,
                       COALESCE(uploads.recorded_at, uploads.created_at) AS source_date,
                       COALESCE(uploads.recorded_at_source, 'uploaded') AS source_date_source
                FROM compilation_items
                JOIN highlight_clips
                    ON highlight_clips.id = compilation_items.highlight_id
                JOIN uploads ON uploads.id = highlight_clips.upload_id
                WHERE compilation_items.compilation_id = ?
                ORDER BY compilation_items.position
                """,
                (compilation_id,),
            ).fetchall()
        return [
            record
            for row in rows
            if (record := self._highlight_clip(row)) is not None
        ]

    def list_queued_compilations(self) -> list[CompilationRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM compilations WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
        return [
            record for row in rows if (record := self._compilation(row)) is not None
        ]

    def requeue_interrupted_compilations(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE compilations
                SET status = 'queued', error = NULL, updated_at = ?
                WHERE status = 'processing'
                """,
                (_now(),),
            )

    def claim_compilation(self, compilation_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE compilations
                SET status = 'processing', error = NULL, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (_now(), compilation_id),
            )
        return cursor.rowcount == 1

    def finish_compilation(
        self,
        compilation_id: str,
        *,
        file_name: str,
        duration: float,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE compilations
                SET status = 'completed', file_name = ?, duration = ?, error = NULL,
                    updated_at = ?
                WHERE id = ? AND status = 'processing'
                """,
                (file_name, duration, _now(), compilation_id),
            )

    def fail_compilation(self, compilation_id: str, error: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE compilations
                SET status = 'failed', error = ?, updated_at = ?
                WHERE id = ?
                """,
                (error[:1000], _now(), compilation_id),
            )
