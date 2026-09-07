from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pingpong_highlight.auth import hash_password, hash_session_token
from pingpong_highlight.db import Database, StateConflict


def _user(database: Database, username: str, *, role: str = "user"):
    return database.create_user(
        username,
        hash_password(f"{username}-password"),
        display_name=username.title(),
        role=role,
    )


def _queued_job(database: Database, root: Path, user_id: str, name: str):
    upload = database.create_upload(
        f"upload-{name}",
        f"{name}.mp4",
        10,
        "video/mp4",
        root / f"{name}.mp4",
        user_id=user_id,
    )
    database.force_upload_offset(upload.id, upload.size)
    return database.complete_upload(upload.id)


def test_user_admin_crud_and_session_ttl_revoke(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    user = _user(database, "Alice", role="admin")

    assert database.get_user_by_username(" ALICE ") == user
    assert database.count_users() == 1
    assert database.list_users()[0].display_name == "Alice"
    updated = database.update_user(user.id, display_name="Alice Admin")
    assert updated is not None and updated.display_name == "Alice Admin"

    with pytest.raises(StateConflict, match="At least one active admin"):
        database.deactivate_user(user.id)
    assert database.get_user(user.id).active  # type: ignore[union-attr]

    token_hash = hash_session_token("opaque-client-token")
    session = database.create_session(
        user.id,
        token_hash,
        datetime.now(UTC) + timedelta(hours=1),
    )
    resolved = database.resolve_session(token_hash)
    assert resolved is not None and resolved[0].id == session.id
    assert resolved[1].id == user.id

    database.update_user(user.id, password_hash=hash_password("new-password"))
    assert database.resolve_session(token_hash) is None

    with pytest.raises(StateConflict, match="changed credentials"):
        database.create_session(
            user.id,
            hash_session_token("stale-credential-token"),
            datetime.now(UTC) + timedelta(hours=1),
            expected_password_hash=user.password_hash,
            expected_role=user.role,
        )

    with pytest.raises(StateConflict, match="At least one active admin"):
        database.update_user(user.id, role="user")

    _user(database, "backup-admin", role="admin")
    role_token_hash = hash_session_token("role-change-client-token")
    database.create_session(
        user.id,
        role_token_hash,
        datetime.now(UTC) + timedelta(hours=1),
    )
    database.update_user(user.id, role="user")
    assert database.resolve_session(role_token_hash) is None

    unchanged_role_token_hash = hash_session_token("unchanged-role-client-token")
    database.create_session(
        user.id,
        unchanged_role_token_hash,
        datetime.now(UTC) + timedelta(hours=1),
    )
    database.update_user(user.id, role="user")
    assert database.resolve_session(unchanged_role_token_hash) is not None

    token_hash = hash_session_token("second-opaque-client-token")
    session = database.create_session(
        user.id,
        token_hash,
        datetime.now(UTC) + timedelta(hours=1),
    )

    expired_hash = hash_session_token("expired-client-token")
    database.create_session(
        user.id,
        expired_hash,
        datetime.now(UTC) - timedelta(seconds=1),
    )
    assert database.resolve_session(expired_hash) is None
    assert database.deactivate_user(user.id)
    assert database.resolve_session(token_hash) is None
    assert not database.get_user(user.id).active  # type: ignore[union-attr]
    assert database.get_session(session.id).revoked_at is not None  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="SHA-256"):
        database.create_session(
            user.id,
            "raw-session-token",
            datetime.now(UTC) + timedelta(hours=1),
        )


def test_password_change_and_replacement_session_are_atomic(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    user = _user(database, "password-user")
    old_token_hash = hash_session_token("old-password-session-token")
    database.create_session(
        user.id,
        old_token_hash,
        datetime.now(UTC) + timedelta(hours=1),
    )
    replacement_hash = hash_password("replacement-password")
    replacement_token_hash = hash_session_token("replacement-session-token")

    changed = database.change_password_and_create_session(
        user.id,
        expected_password_hash=user.password_hash,
        new_password_hash=replacement_hash,
        token_hash=replacement_token_hash,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    assert changed is not None
    updated, replacement = changed
    assert updated.password_hash == replacement_hash
    assert replacement.token_hash == replacement_token_hash
    assert database.resolve_session(old_token_hash) is None
    assert database.resolve_session(replacement_token_hash) is not None
    assert (
        database.change_password_and_create_session(
            user.id,
            expected_password_hash=user.password_hash,
            new_password_hash=hash_password("should-not-apply"),
            token_hash=hash_session_token("stale-change-session-token"),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        is None
    )


def test_username_uniqueness_is_case_insensitive(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    _user(database, "alice")

    with pytest.raises(StateConflict, match="Username"):
        _user(database, "ALICE")


def test_owner_scoped_upload_drive_and_job_queries(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    alice = _user(database, "alice")
    bob = _user(database, "bob")
    alice_job = _queued_job(database, tmp_path, alice.id, "alice")
    bob_job = _queued_job(database, tmp_path, bob.id, "bob")
    alice_import = database.create_or_requeue_drive_import(
        "shared-file",
        None,
        user_id=alice.id,
    )
    bob_import = database.create_or_requeue_drive_import(
        "shared-file",
        None,
        user_id=bob.id,
    )

    assert alice_import.id != bob_import.id
    assert database.get_upload(alice_job.upload_id, user_id=bob.id) is None
    assert database.get_job(alice_job.id, user_id=bob.id) is None
    assert [job.id for job in database.list_jobs(user_id=alice.id)] == [alice_job.id]
    assert database.count_jobs(user_id=bob.id) == 1
    assert database.count_uploads(user_id=alice.id) == 1
    assert database.get_drive_import(alice_import.id, user_id=bob.id) is None
    assert database.count_drive_imports(user_id=alice.id) == 1
    alice_summary = database.get_storage_summary(user_id=alice.id)
    assert alice_summary.source_bytes == 10
    assert alice_summary.queued_count == 1
    assert alice_summary.processing_count == 0
    assert database.get_storage_summary().queued_count == 2
    assert {job.id for job in database.list_jobs()} == {alice_job.id, bob_job.id}


def test_retry_reprocess_and_delete_are_atomic_and_owner_scoped(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    alice = _user(database, "alice")
    bob = _user(database, "bob")
    job = _queued_job(database, tmp_path, alice.id, "match")

    database.finish_job(job.id, {"files": []})
    assert database.reprocess_job(job.id, user_id=bob.id) is None
    reprocessed = database.reprocess_job(job.id, user_id=alice.id)
    assert reprocessed is not None and reprocessed.status == "queued"
    assert reprocessed.result == {"files": []}
    assert database.claim_job(job.id)
    with pytest.raises(StateConflict, match="processing"):
        database.delete_job(job.id, user_id=alice.id)
    database.fail_job(job.id, "test failure")
    retried = database.retry_job(job.id, user_id=alice.id)
    assert retried is not None and retried.status == "queued"

    deleted = database.delete_job(job.id, user_id=alice.id)
    assert deleted is not None
    assert deleted[0].id == job.id
    assert database.get_job(job.id) is None
    assert database.get_upload(job.upload_id) is None


def test_additive_migration_preserves_then_claims_legacy_records(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE uploads (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, size INTEGER NOT NULL,
                offset INTEGER NOT NULL, content_type TEXT NOT NULL, status TEXT NOT NULL,
                path TEXT NOT NULL, job_id TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE drive_imports (
                id TEXT PRIMARY KEY, file_id TEXT NOT NULL, resource_key TEXT,
                filename TEXT, size INTEGER, offset INTEGER NOT NULL, status TEXT NOT NULL,
                error TEXT, upload_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            INSERT INTO uploads VALUES
                ('legacy-upload', 'legacy.mp4', 12, 12, 'video/mp4', 'completed',
                 '/tmp/legacy.mp4', NULL, '2025-01-01T00:00:00+00:00',
                 '2025-01-01T00:00:00+00:00');
            INSERT INTO drive_imports VALUES
                ('legacy-drive', 'file', NULL, 'legacy.mp4', 12, 12, 'completed', NULL,
                 'legacy-upload', '2025-01-01T00:00:00+00:00',
                 '2025-01-01T00:00:00+00:00');
            """
        )

    database = Database(path)
    assert database.get_upload("legacy-upload").user_id is None  # type: ignore[union-attr]
    admin = _user(database, "admin", role="admin")

    assert database.claim_unowned_data(admin.id) == {"uploads": 1, "drive_imports": 1}
    assert database.get_upload("legacy-upload").user_id == admin.id  # type: ignore[union-attr]
    assert database.get_drive_import("legacy-drive").user_id == admin.id  # type: ignore[union-attr]
    assert database.claim_unowned_data(admin.id) == {"uploads": 0, "drive_imports": 0}
