from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pingpong_highlight.auth import normalize_username


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp(value: str | datetime) -> str:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _page(limit: int, offset: int) -> tuple[int, int]:
    if limit <= 0 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    if offset < 0:
        raise ValueError("offset must be zero or positive")
    return limit, offset


def _token_hash(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError("token_hash must be a SHA-256 hexadecimal digest")
    return value.casefold()


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
    user_id: str | None = None


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
    user_id: str | None = None


@dataclass(frozen=True, slots=True)
class UserRecord:
    id: str
    username: str
    display_name: str
    role: str
    password_hash: str
    active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    user_id: str
    token_hash: str
    expires_at: str
    revoked_at: str | None
    created_at: str
    last_seen_at: str


@dataclass(frozen=True, slots=True)
class CleanupRecord:
    id: str
    path: Path
    kind: str
    attempts: int
    last_error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class StorageSummary:
    upload_count: int
    source_bytes: int
    uploading_count: int
    queued_count: int
    processing_count: int
    completed_count: int
    failed_count: int


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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
                    password_hash TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    size INTEGER NOT NULL CHECK (size >= 0),
                    offset INTEGER NOT NULL DEFAULT 0 CHECK (offset >= 0),
                    content_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    path TEXT NOT NULL,
                    job_id TEXT,
                    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    upload_id TEXT NOT NULL UNIQUE REFERENCES uploads(id) ON DELETE CASCADE,
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
                    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
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

                CREATE TABLE IF NOT EXISTS cleanup_queue (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('file', 'tree')),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS drive_imports_status_idx
                    ON drive_imports(status, created_at);
                CREATE INDEX IF NOT EXISTS annotations_upload_time_idx
                    ON annotations(upload_id, start, created_at);
                CREATE INDEX IF NOT EXISTS sessions_user_active_idx
                    ON sessions(user_id, revoked_at, expires_at);
                CREATE INDEX IF NOT EXISTS cleanup_queue_created_idx
                    ON cleanup_queue(created_at, id);
                CREATE UNIQUE INDEX IF NOT EXISTS cleanup_queue_target_idx
                    ON cleanup_queue(path, kind);
                """
            )
            self._ensure_column(
                connection,
                "uploads",
                "user_id",
                "TEXT REFERENCES users(id) ON DELETE SET NULL",
            )
            self._ensure_column(
                connection,
                "drive_imports",
                "user_id",
                "TEXT REFERENCES users(id) ON DELETE SET NULL",
            )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS uploads_user_created_idx
                    ON uploads(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS drive_imports_user_created_idx
                    ON drive_imports(user_id, created_at DESC);
                """
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

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
            user_id=row["user_id"],
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
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _cleanup(row: sqlite3.Row | None) -> CleanupRecord | None:
        if row is None:
            return None
        return CleanupRecord(
            id=row["id"],
            path=Path(row["path"]),
            kind=row["kind"],
            attempts=int(row["attempts"]),
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _enqueue_cleanup(
        connection: sqlite3.Connection,
        targets: Iterable[tuple[Path, str]],
    ) -> None:
        timestamp = _now()
        for path, kind in targets:
            if kind not in {"file", "tree"}:
                raise ValueError("cleanup kind must be 'file' or 'tree'")
            connection.execute(
                """
                INSERT OR IGNORE INTO cleanup_queue
                    (id, path, kind, attempts, last_error, created_at, updated_at)
                VALUES (?, ?, ?, 0, NULL, ?, ?)
                """,
                (uuid.uuid4().hex, str(path.absolute()), kind, timestamp, timestamp),
            )

    @staticmethod
    def _user(row: sqlite3.Row | None) -> UserRecord | None:
        if row is None:
            return None
        return UserRecord(
            id=row["id"],
            username=row["username"],
            display_name=row["display_name"],
            role=row["role"],
            password_hash=row["password_hash"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _session(row: sqlite3.Row | None) -> SessionRecord | None:
        if row is None:
            return None
        return SessionRecord(
            id=row["id"],
            user_id=row["user_id"],
            token_hash=row["token_hash"],
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
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

    def create_user(
        self,
        username: str,
        password_hash: str,
        *,
        display_name: str | None = None,
        role: str = "user",
        active: bool = True,
        user_id: str | None = None,
    ) -> UserRecord:
        username = normalize_username(username)
        if role not in {"admin", "user"}:
            raise ValueError("role must be 'admin' or 'user'")
        if not password_hash:
            raise ValueError("password_hash must not be empty")
        clean_display_name = (display_name or username).strip()
        if not clean_display_name:
            raise ValueError("display_name must not be blank")
        timestamp = _now()
        user_id = user_id or uuid.uuid4().hex
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users
                        (id, username, display_name, role, password_hash, active,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        username,
                        clean_display_name,
                        role,
                        password_hash,
                        int(active),
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise StateConflict("Username already exists") from exc
        record = self.get_user(user_id)
        assert record is not None
        return record

    def get_user(self, user_id: str) -> UserRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user(row)

    def get_user_by_username(self, username: str) -> UserRecord | None:
        try:
            username = normalize_username(username)
        except ValueError:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
        return self._user(row)

    def list_users(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_inactive: bool = True,
    ) -> list[UserRecord]:
        limit, offset = _page(limit, offset)
        where = "" if include_inactive else "WHERE active = 1"
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM users {where}
                ORDER BY username COLLATE NOCASE, id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [record for row in rows if (record := self._user(row)) is not None]

    def count_users(self, *, include_inactive: bool = True) -> int:
        where = "" if include_inactive else "WHERE active = 1"
        with self._lock, self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS total FROM users {where}").fetchone()
        return int(row["total"])

    def update_user(
        self,
        user_id: str,
        *,
        username: str | None = None,
        display_name: str | None = None,
        role: str | None = None,
        password_hash: str | None = None,
        active: bool | None = None,
    ) -> UserRecord | None:
        assignments: list[str] = []
        values: list[Any] = []
        if username is not None:
            assignments.append("username = ?")
            values.append(normalize_username(username))
        if display_name is not None:
            display_name = display_name.strip()
            if not display_name:
                raise ValueError("display_name must not be blank")
            assignments.append("display_name = ?")
            values.append(display_name)
        if role is not None:
            if role not in {"admin", "user"}:
                raise ValueError("role must be 'admin' or 'user'")
            assignments.append("role = ?")
            values.append(role)
        if password_hash is not None:
            if not password_hash:
                raise ValueError("password_hash must not be empty")
            assignments.append("password_hash = ?")
            values.append(password_hash)
        if active is not None:
            assignments.append("active = ?")
            values.append(int(active))
        if not assignments:
            return self.get_user(user_id)
        timestamp = _now()
        assignments.append("updated_at = ?")
        values.extend((timestamp, user_id))
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current_user = connection.execute(
                    "SELECT role, active FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
                if current_user is None:
                    return None
                role_changed = role is not None and current_user["role"] != role
                final_role = role if role is not None else current_user["role"]
                final_active = int(active) if active is not None else current_user["active"]
                if (
                    current_user["role"] == "admin"
                    and current_user["active"]
                    and (final_role != "admin" or not final_active)
                ):
                    remaining_admin = connection.execute(
                        """
                        SELECT 1 FROM users
                        WHERE id != ? AND role = 'admin' AND active = 1
                        LIMIT 1
                        """,
                        (user_id,),
                    ).fetchone()
                    if remaining_admin is None:
                        raise StateConflict("At least one active admin is required")
                cursor = connection.execute(
                    f"UPDATE users SET {', '.join(assignments)} WHERE id = ?",
                    values,
                )
                if cursor.rowcount and (
                    active is False or password_hash is not None or role_changed
                ):
                    connection.execute(
                        """
                        UPDATE sessions SET revoked_at = ?
                        WHERE user_id = ? AND revoked_at IS NULL
                        """,
                        (timestamp, user_id),
                    )
        except sqlite3.IntegrityError as exc:
            raise StateConflict("Username already exists") from exc
        if cursor.rowcount != 1:
            return None
        return self.get_user(user_id)

    def deactivate_user(self, user_id: str) -> bool:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current_user = connection.execute(
                "SELECT role, active FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if current_user is None or not current_user["active"]:
                return False
            if current_user["role"] == "admin":
                remaining_admin = connection.execute(
                    """
                    SELECT 1 FROM users
                    WHERE id != ? AND role = 'admin' AND active = 1
                    LIMIT 1
                    """,
                    (user_id,),
                ).fetchone()
                if remaining_admin is None:
                    raise StateConflict("At least one active admin is required")
            cursor = connection.execute(
                """
                UPDATE users SET active = 0, updated_at = ?
                WHERE id = ? AND active = 1
                """,
                (timestamp, user_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE sessions SET revoked_at = ?
                    WHERE user_id = ? AND revoked_at IS NULL
                    """,
                    (timestamp, user_id),
                )
        return cursor.rowcount == 1

    def create_session(
        self,
        user_id: str,
        token_hash: str,
        expires_at: str | datetime,
        *,
        session_id: str | None = None,
        expected_password_hash: str | None = None,
        expected_role: str | None = None,
    ) -> SessionRecord:
        token_hash = _token_hash(token_hash)
        session_id = session_id or uuid.uuid4().hex
        timestamp = _now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                conditions = ["id = ?", "active = 1"]
                parameters: list[Any] = [user_id]
                if expected_password_hash is not None:
                    conditions.append("password_hash = ?")
                    parameters.append(expected_password_hash)
                if expected_role is not None:
                    conditions.append("role = ?")
                    parameters.append(expected_role)
                if (
                    connection.execute(
                        f"SELECT 1 FROM users WHERE {' AND '.join(conditions)}",
                        parameters,
                    ).fetchone()
                    is None
                ):
                    raise StateConflict(
                        "Cannot create a session for changed credentials or an inactive user"
                    )
                connection.execute(
                    """
                    INSERT INTO sessions
                        (id, user_id, token_hash, expires_at, revoked_at,
                         created_at, last_seen_at)
                    VALUES (?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        session_id,
                        user_id,
                        token_hash,
                        _timestamp(expires_at),
                        timestamp,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise StateConflict("Session token hash already exists") from exc
        record = self._session(row)
        assert record is not None
        return record

    def change_password_and_create_session(
        self,
        user_id: str,
        *,
        expected_password_hash: str,
        new_password_hash: str,
        token_hash: str,
        expires_at: str | datetime,
        session_id: str | None = None,
    ) -> tuple[UserRecord, SessionRecord] | None:
        """Atomically rotate a current password, revoke sessions, and issue one replacement."""

        if not expected_password_hash or not new_password_hash:
            raise ValueError("password hashes must not be empty")
        token_hash = _token_hash(token_hash)
        session_id = session_id or uuid.uuid4().hex
        expiration = _timestamp(expires_at)
        timestamp = _now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    """
                    SELECT 1 FROM users
                    WHERE id = ? AND active = 1 AND password_hash = ?
                    """,
                    (user_id, expected_password_hash),
                ).fetchone()
                if current is None:
                    return None
                connection.execute(
                    "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                    (new_password_hash, timestamp, user_id),
                )
                connection.execute(
                    """
                    UPDATE sessions SET revoked_at = ?
                    WHERE user_id = ? AND revoked_at IS NULL
                    """,
                    (timestamp, user_id),
                )
                connection.execute(
                    """
                    INSERT INTO sessions
                        (id, user_id, token_hash, expires_at, revoked_at,
                         created_at, last_seen_at)
                    VALUES (?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (session_id, user_id, token_hash, expiration, timestamp, timestamp),
                )
                user_row = connection.execute(
                    "SELECT * FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
                session_row = connection.execute(
                    "SELECT * FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise StateConflict("Session token hash already exists") from exc
        user = self._user(user_row)
        session = self._session(session_row)
        assert user is not None and session is not None
        return user, session

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._session(row)

    def resolve_session(
        self,
        token_hash: str,
        *,
        at: str | datetime | None = None,
    ) -> tuple[SessionRecord, UserRecord] | None:
        token_hash = _token_hash(token_hash)
        current = _timestamp(at) if at is not None else _now()
        with self._lock, self._connect() as connection:
            session_row = connection.execute(
                """
                SELECT * FROM sessions
                WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?
                """,
                (token_hash, current),
            ).fetchone()
            session = self._session(session_row)
            if session is None:
                return None
            user_row = connection.execute(
                "SELECT * FROM users WHERE id = ? AND active = 1",
                (session.user_id,),
            ).fetchone()
            user = self._user(user_row)
        if user is None:
            return None
        return session, user

    def touch_session(self, session_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET last_seen_at = ?
                WHERE id = ? AND revoked_at IS NULL AND expires_at > ?
                """,
                (_now(), session_id, _now()),
            )
        return cursor.rowcount == 1

    def revoke_session(self, session_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET revoked_at = ?
                WHERE id = ? AND revoked_at IS NULL
                """,
                (_now(), session_id),
            )
        return cursor.rowcount == 1

    def revoke_session_by_token_hash(self, token_hash: str) -> bool:
        token_hash = _token_hash(token_hash)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (_now(), token_hash),
            )
        return cursor.rowcount == 1

    def revoke_user_sessions(self, user_id: str) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (_now(), user_id),
            )
        return cursor.rowcount

    def delete_expired_sessions(self, *, at: str | datetime | None = None) -> int:
        current = _timestamp(at) if at is not None else _now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE expires_at <= ? OR revoked_at IS NOT NULL",
                (current,),
            )
        return cursor.rowcount

    def enqueue_cleanup(self, targets: Iterable[tuple[Path, str]]) -> None:
        targets = tuple(targets)
        if not targets:
            return
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._enqueue_cleanup(connection, targets)

    def list_cleanup_records(self) -> list[CleanupRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM cleanup_queue
                ORDER BY created_at, id
                """
            ).fetchall()
        return [record for row in rows if (record := self._cleanup(row)) is not None]

    def complete_cleanup(self, cleanup_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM cleanup_queue WHERE id = ?", (cleanup_id,))
        return cursor.rowcount == 1

    def record_cleanup_failure(self, cleanup_id: str, error: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cleanup_queue
                SET attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (error[:1000], _now(), cleanup_id),
            )
        return cursor.rowcount == 1

    def claim_unowned_data(self, user_id: str) -> dict[str, int]:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
                is None
            ):
                raise ValueError("user does not exist")
            uploads = connection.execute(
                "UPDATE uploads SET user_id = ?, updated_at = ? WHERE user_id IS NULL",
                (user_id, timestamp),
            ).rowcount
            drive_imports = connection.execute(
                """
                UPDATE drive_imports SET user_id = ?, updated_at = ?
                WHERE user_id IS NULL
                """,
                (user_id, timestamp),
            ).rowcount
        return {"uploads": uploads, "drive_imports": drive_imports}

    def create_upload(
        self,
        upload_id: str,
        filename: str,
        size: int,
        content_type: str,
        path: Path,
        *,
        user_id: str | None = None,
    ) -> UploadRecord:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO uploads
                    (id, filename, size, offset, content_type, status, path, user_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, 0, ?, 'uploading', ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    filename,
                    size,
                    content_type,
                    str(path),
                    user_id,
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
        user_id: str | None = None,
    ) -> tuple[UploadRecord, JobRecord]:
        timestamp = _now()
        job_id = uuid.uuid4().hex
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if drive_import_id is not None:
                import_row = connection.execute(
                    "SELECT user_id FROM drive_imports WHERE id = ?",
                    (drive_import_id,),
                ).fetchone()
                if import_row is None:
                    raise StateConflict("Drive import does not exist")
                import_user_id = import_row["user_id"]
                if user_id is None:
                    user_id = import_user_id
                elif import_user_id is not None and import_user_id != user_id:
                    raise StateConflict("Drive import belongs to another user")
            connection.execute(
                """
                INSERT INTO uploads
                    (id, filename, size, offset, content_type, status, path, job_id,
                     user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    filename,
                    size,
                    size,
                    content_type,
                    str(path),
                    job_id,
                    user_id,
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
                        user_id = COALESCE(user_id, ?), error = NULL, updated_at = ?
                    WHERE id = ? AND status = 'downloading'
                    """,
                    (upload_id, size, size, user_id, timestamp, drive_import_id),
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
        *,
        user_id: str | None = None,
    ) -> DriveImportRecord:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM drive_imports
                WHERE file_id = ? AND user_id IS ? AND status != 'completed'
                ORDER BY created_at DESC LIMIT 1
                """,
                (file_id, user_id),
            ).fetchone()
            record = self._drive_import(row)
            if record is None:
                import_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO drive_imports
                        (id, file_id, resource_key, status, user_id, created_at, updated_at)
                    VALUES (?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (import_id, file_id, resource_key, user_id, timestamp, timestamp),
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

    def get_drive_import(
        self,
        import_id: str,
        *,
        user_id: str | None = None,
    ) -> DriveImportRecord | None:
        owner_clause = "" if user_id is None else " AND user_id = ?"
        parameters: tuple[str, ...] = (import_id,) if user_id is None else (import_id, user_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM drive_imports WHERE id = ?{owner_clause}",
                parameters,
            ).fetchone()
        return self._drive_import(row)

    def list_drive_imports(
        self,
        *,
        include_completed: bool = False,
        user_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DriveImportRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if not include_completed:
            clauses.append("status != 'completed'")
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        pagination = ""
        if limit is not None:
            limit, offset = _page(limit, offset)
            pagination = " LIMIT ? OFFSET ?"
            parameters.extend((limit, offset))
        elif offset:
            raise ValueError("offset requires a limit")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM drive_imports {where}
                ORDER BY created_at DESC, id DESC{pagination}
                """,
                parameters,
            ).fetchall()
        return [record for row in rows if (record := self._drive_import(row)) is not None]

    def count_drive_imports(
        self,
        *,
        include_completed: bool = True,
        user_id: str | None = None,
    ) -> int:
        clauses: list[str] = []
        parameters: list[Any] = []
        if not include_completed:
            clauses.append("status != 'completed'")
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM drive_imports {where}",
                parameters,
            ).fetchone()
        return int(row["total"])

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

    def retry_drive_import(
        self,
        import_id: str,
        *,
        user_id: str | None = None,
    ) -> bool:
        owner_clause = "" if user_id is None else " AND user_id = ?"
        parameters: list[Any] = [_now(), import_id]
        if user_id is not None:
            parameters.append(user_id)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE drive_imports
                SET status = 'queued', error = NULL, updated_at = ?
                WHERE id = ? AND status = 'failed'{owner_clause}
                """,
                parameters,
            )
        return cursor.rowcount == 1

    def delete_drive_import(
        self,
        import_id: str,
        *,
        user_id: str | None = None,
        cleanup_targets: Iterable[tuple[Path, str]] = (),
    ) -> bool:
        cleanup_targets = tuple(cleanup_targets)
        owner_clause = "" if user_id is None else " AND user_id = ?"
        parameters: tuple[str, ...] = (import_id,) if user_id is None else (import_id, user_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"""
                DELETE FROM drive_imports
                WHERE id = ? AND status IN ('queued', 'failed'){owner_clause}
                """,
                parameters,
            )
            if cursor.rowcount:
                self._enqueue_cleanup(connection, cleanup_targets)
        return cursor.rowcount == 1

    def get_upload(
        self,
        upload_id: str,
        *,
        user_id: str | None = None,
    ) -> UploadRecord | None:
        owner_clause = "" if user_id is None else " AND user_id = ?"
        parameters: tuple[str, ...] = (upload_id,) if user_id is None else (upload_id, user_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM uploads WHERE id = ?{owner_clause}",
                parameters,
            ).fetchone()
        return self._upload(row)

    def list_uploads(
        self,
        *,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[UploadRecord]:
        limit, offset = _page(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend((limit, offset))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM uploads {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
        return [record for row in rows if (record := self._upload(row)) is not None]

    def count_uploads(
        self,
        *,
        user_id: str | None = None,
        status: str | None = None,
    ) -> int:
        clauses: list[str] = []
        parameters: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM uploads {where}", parameters
            ).fetchone()
        return int(row["total"])

    def list_incomplete_uploads(self) -> list[UploadRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM uploads WHERE status = 'uploading' ORDER BY created_at"
            ).fetchall()
        return [record for row in rows if (record := self._upload(row)) is not None]

    def delete_incomplete_upload(
        self,
        upload_id: str,
        *,
        cleanup_targets: Iterable[tuple[Path, str]] = (),
    ) -> bool:
        cleanup_targets = tuple(cleanup_targets)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM uploads WHERE id = ? AND status = 'uploading'",
                (upload_id,),
            )
            if cursor.rowcount:
                self._enqueue_cleanup(connection, cleanup_targets)
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

    def get_job(
        self,
        job_id: str,
        *,
        user_id: str | None = None,
    ) -> JobRecord | None:
        owner_join = "" if user_id is None else " JOIN uploads AS u ON u.id = jobs.upload_id"
        owner_clause = "" if user_id is None else " AND u.user_id = ?"
        parameters: tuple[str, ...] = (job_id,) if user_id is None else (job_id, user_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT jobs.* FROM jobs{owner_join} WHERE jobs.id = ?{owner_clause}",
                parameters,
            ).fetchone()
        return self._job(row)

    def list_jobs(
        self,
        limit: int = 50,
        *,
        offset: int = 0,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[JobRecord]:
        limit, offset = _page(limit, offset)
        clauses: list[str] = []
        parameters: list[Any] = []
        join = ""
        if user_id is not None:
            join = " JOIN uploads AS u ON u.id = jobs.upload_id"
            clauses.append("u.user_id = ?")
            parameters.append(user_id)
        if status is not None:
            clauses.append("jobs.status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend((limit, offset))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT jobs.* FROM jobs{join} {where}
                ORDER BY jobs.created_at DESC, jobs.id DESC
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
        return [record for row in rows if (record := self._job(row)) is not None]

    def count_jobs(
        self,
        *,
        user_id: str | None = None,
        status: str | None = None,
    ) -> int:
        clauses: list[str] = []
        parameters: list[Any] = []
        join = ""
        if user_id is not None:
            join = " JOIN uploads AS u ON u.id = jobs.upload_id"
            clauses.append("u.user_id = ?")
            parameters.append(user_id)
        if status is not None:
            clauses.append("jobs.status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM jobs{join} {where}", parameters
            ).fetchone()
        return int(row["total"])

    def get_storage_summary(self, *, user_id: str | None = None) -> StorageSummary:
        upload_owner_clause = "" if user_id is None else "WHERE user_id = ?"
        job_owner_join = "" if user_id is None else " JOIN uploads AS u ON u.id = jobs.upload_id"
        job_owner_clause = "" if user_id is None else "WHERE u.user_id = ?"
        parameters: tuple[str, ...] = () if user_id is None else (user_id,)
        with self._lock, self._connect() as connection:
            upload_row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS upload_count,
                    COALESCE(SUM(
                        CASE WHEN status = 'uploading' THEN offset ELSE size END
                    ), 0) AS source_bytes,
                    COALESCE(SUM(status = 'uploading'), 0) AS uploading_count
                FROM uploads {upload_owner_clause}
                """,
                parameters,
            ).fetchone()
            job_row = connection.execute(
                f"""
                SELECT
                    COALESCE(SUM(jobs.status = 'queued'), 0) AS queued_count,
                    COALESCE(SUM(jobs.status = 'processing'), 0) AS processing_count,
                    COALESCE(SUM(jobs.status = 'completed'), 0) AS completed_count,
                    COALESCE(SUM(jobs.status = 'failed'), 0) AS failed_count
                FROM jobs{job_owner_join} {job_owner_clause}
                """,
                parameters,
            ).fetchone()
        return StorageSummary(
            upload_count=int(upload_row["upload_count"]),
            source_bytes=int(upload_row["source_bytes"]),
            uploading_count=int(upload_row["uploading_count"]),
            queued_count=int(job_row["queued_count"]),
            processing_count=int(job_row["processing_count"]),
            completed_count=int(job_row["completed_count"]),
            failed_count=int(job_row["failed_count"]),
        )

    def retry_job(
        self,
        job_id: str,
        *,
        user_id: str | None = None,
    ) -> JobRecord | None:
        return self._reset_job(job_id, user_id=user_id, allowed_statuses={"failed"})

    def reprocess_job(
        self,
        job_id: str,
        *,
        user_id: str | None = None,
    ) -> JobRecord | None:
        return self._reset_job(
            job_id,
            user_id=user_id,
            allowed_statuses={"completed", "failed"},
        )

    def _reset_job(
        self,
        job_id: str,
        *,
        user_id: str | None,
        allowed_statuses: set[str],
    ) -> JobRecord | None:
        timestamp = _now()
        owner_clause = "" if user_id is None else " AND u.user_id = ?"
        parameters: tuple[str, ...] = (job_id,) if user_id is None else (job_id, user_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""
                SELECT jobs.* FROM jobs
                JOIN uploads AS u ON u.id = jobs.upload_id
                WHERE jobs.id = ?{owner_clause}
                """,
                parameters,
            ).fetchone()
            record = self._job(row)
            if record is None:
                return None
            if record.status not in allowed_statuses:
                raise StateConflict(f"Job in '{record.status}' state cannot be queued")
            connection.execute(
                """
                UPDATE jobs
                SET status = 'queued', progress = 0, stage = 'queued', error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (timestamp, job_id),
            )
            connection.execute(
                """
                UPDATE uploads SET status = 'queued', updated_at = ?
                WHERE job_id = ?
                """,
                (timestamp, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        result = self._job(updated)
        assert result is not None
        return result

    def delete_job(
        self,
        job_id: str,
        *,
        user_id: str | None = None,
        cleanup_targets: Iterable[tuple[Path, str]] = (),
    ) -> tuple[JobRecord, UploadRecord] | None:
        cleanup_targets = tuple(cleanup_targets)
        owner_clause = "" if user_id is None else " AND uploads.user_id = ?"
        parameters: tuple[str, ...] = (job_id,) if user_id is None else (job_id, user_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""
                SELECT jobs.*, uploads.id AS owner_upload_id
                FROM jobs JOIN uploads ON uploads.id = jobs.upload_id
                WHERE jobs.id = ?{owner_clause}
                """,
                parameters,
            ).fetchone()
            job = self._job(row)
            if job is None:
                return None
            if job.status == "processing":
                raise StateConflict("A processing job cannot be deleted")
            upload_row = connection.execute(
                "SELECT * FROM uploads WHERE id = ?", (job.upload_id,)
            ).fetchone()
            upload = self._upload(upload_row)
            assert upload is not None
            self._enqueue_cleanup(connection, cleanup_targets)
            connection.execute(
                "UPDATE drive_imports SET upload_id = NULL WHERE upload_id = ?",
                (upload.id,),
            )
            connection.execute("DELETE FROM annotations WHERE upload_id = ?", (upload.id,))
            connection.execute("DELETE FROM jobs WHERE id = ?", (job.id,))
            connection.execute("DELETE FROM uploads WHERE id = ?", (upload.id,))
        return job, upload

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
