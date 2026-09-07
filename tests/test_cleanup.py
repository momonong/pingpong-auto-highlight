from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pingpong_highlight.auth import hash_password
from pingpong_highlight.cleanup import FilesystemCleanup
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.drive import DriveImportManager
from pingpong_highlight.jobs import JobManager
from pingpong_highlight.uploads import UploadStore
from pingpong_highlight.web import create_app


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        upload_token="cleanup-test-token",
        download_min_free_bytes=0,
    )
    settings.ensure_directories()
    return settings


def _user(database: Database) -> str:
    return database.create_user(
        "cleanup-user",
        hash_password("cleanup-password"),
    ).id


def _queued_job(database: Database, settings: Settings):
    source = settings.uploads_dir / "source.mp4"
    source.write_bytes(b"source")
    upload = database.create_upload(
        "source-upload",
        "source.mp4",
        source.stat().st_size,
        "video/mp4",
        source,
        user_id=_user(database),
    )
    database.force_upload_offset(upload.id, upload.size)
    job = database.complete_upload(upload.id)
    updated = database.get_upload(upload.id)
    assert updated is not None
    return updated, job


def test_job_delete_tombstone_survives_crash_and_drains_after_restart(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    upload, job = _queued_job(database, settings)
    output = settings.outputs_dir / job.id
    work = settings.work_dir / job.id
    output.mkdir()
    work.mkdir()
    (output / "reel.mp4").write_bytes(b"reel")
    (work / "scratch.bin").write_bytes(b"scratch")

    deleted = database.delete_job(
        job.id,
        user_id=upload.user_id,
        cleanup_targets=[
            (upload.path, "file"),
            (output, "tree"),
            (work, "tree"),
        ],
    )

    assert deleted == (job, upload)
    assert database.get_job(job.id) is None
    assert database.get_upload(upload.id) is None
    assert upload.path.exists() and output.exists() and work.exists()
    assert len(database.list_cleanup_records()) == 3

    # A new manager/database instance models a crash after SQLite commit and restart.
    restarted = Database(settings.database_path)
    result = FilesystemCleanup(settings, restarted).drain()

    assert result.removed == 3
    assert result.failed == 0
    assert not upload.path.exists() and not output.exists() and not work.exists()
    assert restarted.list_cleanup_records() == []


def test_job_delete_rolls_back_metadata_when_tombstone_enqueue_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    upload, job = _queued_job(database, settings)

    def reject_enqueue(_connection, _targets) -> None:
        raise RuntimeError("simulated SQLite enqueue failure")

    monkeypatch.setattr(database, "_enqueue_cleanup", reject_enqueue)
    with pytest.raises(RuntimeError, match="enqueue failure"):
        database.delete_job(
            job.id,
            user_id=upload.user_id,
            cleanup_targets=[(upload.path, "file")],
        )

    assert database.get_job(job.id) is not None
    assert database.get_upload(upload.id) is not None
    assert database.list_cleanup_records() == []


def test_incomplete_upload_delete_succeeds_when_cleanup_must_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    cleanup = FilesystemCleanup(settings, database)
    store = UploadStore(settings, database, cleanup)
    upload = store.create("partial.mp4", 10, "video/mp4", user_id=_user(database))
    part = store.part_path(upload)
    part.write_bytes(b"part")

    def fail_remove(_record) -> None:
        raise OSError("simulated busy file")

    monkeypatch.setattr(cleanup, "_remove", fail_remove)
    deleted = asyncio.run(store.delete(upload.id))

    assert deleted.id == upload.id
    assert database.get_upload(upload.id) is None
    assert part.exists()
    queued = database.list_cleanup_records()
    assert queued
    assert all(record.attempts == 1 for record in queued)

    result = FilesystemCleanup(settings, database).drain()
    assert result.failed == 0
    assert not part.exists()
    assert database.list_cleanup_records() == []


def test_drive_delete_enqueues_metadata_and_files_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    cleanup = FilesystemCleanup(settings, database)
    uploads = UploadStore(settings, database, cleanup)
    manager = DriveImportManager(settings, database, uploads, lambda _job_id: None, cleanup=cleanup)
    record = database.create_or_requeue_drive_import(
        "1AbCdEfGhIjKlMnOpQrStUvWxYz",
        None,
        user_id=_user(database),
    )
    database.fail_drive_import(record.id, "expected test failure")
    partial = settings.drive_imports_dir / f"{record.id}.mp4.part"
    partial.write_bytes(b"partial")

    def fail_remove(_record) -> None:
        raise OSError("simulated busy file")

    monkeypatch.setattr(cleanup, "_remove", fail_remove)

    try:
        manager.delete(record.id, user_id=record.user_id)
    finally:
        manager.shutdown()

    assert database.get_drive_import(record.id) is None
    assert partial.exists()
    queued = database.list_cleanup_records()
    assert len(queued) == 1 and queued[0].attempts == 1

    result = FilesystemCleanup(settings, database).drain()
    assert result.failed == 0
    assert not partial.exists()
    assert database.list_cleanup_records() == []


def test_cleanup_rejects_outside_paths_and_unlinks_symlink_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    cleanup = FilesystemCleanup(settings, database)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"keep")
    link = settings.uploads_dir / "outside-link.mp4"
    link.symlink_to(outside)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    nested_outside = outside_dir / "nested.mp4"
    nested_outside.write_bytes(b"also-keep")
    linked_parent = settings.work_dir / "linked-parent"
    linked_parent.symlink_to(outside_dir, target_is_directory=True)
    database.enqueue_cleanup(
        [
            (outside, "file"),
            (link, "file"),
            (linked_parent / nested_outside.name, "file"),
        ]
    )

    result = cleanup.drain()

    assert result.removed == 1
    assert result.failed == 2
    assert outside.read_bytes() == b"keep"
    assert nested_outside.read_bytes() == b"also-keep"
    assert not link.exists() and not link.is_symlink()
    queued = database.list_cleanup_records()
    assert len(queued) == 2
    assert {record.path for record in queued} == {outside, linked_parent / nested_outside.name}
    assert all(record.attempts == 1 for record in queued)
    assert all("outside managed" in (record.last_error or "") for record in queued)


def test_startup_only_removes_previous_generation_with_complete_current_result(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    _upload, job = _queued_job(database, settings)
    output = settings.outputs_dir / job.id
    output.mkdir()
    (output / "reel.mp4").write_bytes(b"current")
    database.finish_job(job.id, {"files": [{"name": "reel.mp4", "kind": "reel"}]})
    previous = settings.work_dir / f".{job.id}.previous"
    previous.mkdir()
    (previous / "reel.mp4").write_bytes(b"previous")

    cleanup = FilesystemCleanup(settings, database)
    assert cleanup.discard_obsolete_previous_artifacts().removed == 1
    assert not previous.exists()

    # If the current output cannot be verified, the previous generation is retained.
    previous.mkdir()
    (output / "reel.mp4").unlink()
    assert cleanup.discard_obsolete_previous_artifacts().removed == 0
    assert previous.exists()


@pytest.mark.parametrize("has_uncommitted_output", [False, True])
def test_job_start_restores_previous_generation_before_requeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_uncommitted_output: bool,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    _upload, job = _queued_job(database, settings)
    assert database.claim_job(job.id)
    previous = settings.work_dir / f".{job.id}.previous"
    output = settings.outputs_dir / job.id
    previous.mkdir()
    (previous / "reel.mp4").write_bytes(b"last-committed")
    if has_uncommitted_output:
        output.mkdir()
        (output / "reel.mp4").write_bytes(b"uncommitted")
    manager = JobManager(settings, database)
    monkeypatch.setattr(manager, "enqueue", lambda _job_id: None)

    try:
        manager.start()
    finally:
        manager.shutdown()

    assert not previous.exists()
    assert (output / "reel.mp4").read_bytes() == b"last-committed"
    recovered = database.get_job(job.id)
    assert recovered is not None and recovered.status == "queued"


def test_previous_cleanup_failure_does_not_reverse_completed_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    _upload, job = _queued_job(database, settings)
    output = settings.outputs_dir / job.id
    output.mkdir()
    (output / "reel.mp4").write_bytes(b"old")
    database.finish_job(job.id, {"files": [{"name": "reel.mp4", "kind": "reel"}]})
    assert database.reprocess_job(job.id) is not None

    class ReplacementProcessor:
        def run(self, _source, output_dir, progress=None, *, source_name=None):
            output_dir.mkdir()
            (output_dir / "reel.mp4").write_bytes(b"new")
            return {"files": [{"name": "reel.mp4", "kind": "reel"}]}

    manager = JobManager(settings, database, ReplacementProcessor())
    previous = settings.work_dir / f".{job.id}.previous"
    real_rmtree = shutil.rmtree

    def fail_post_commit_previous_cleanup(path) -> None:
        current = database.get_job(job.id)
        if Path(path) == previous and current is not None and current.status == "completed":
            raise OSError("simulated cleanup failure")
        real_rmtree(path)

    monkeypatch.setattr("pingpong_highlight.jobs.shutil.rmtree", fail_post_commit_previous_cleanup)
    try:
        manager._run(job.id)
    finally:
        manager.shutdown()

    completed = database.get_job(job.id)
    assert completed is not None and completed.status == "completed"
    assert (output / "reel.mp4").read_bytes() == b"new"
    assert (previous / "reel.mp4").read_bytes() == b"old"


def test_lifespan_drains_cleanup_before_starting_job_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    pending = settings.uploads_dir / "pending-delete.mp4"
    pending.write_bytes(b"pending")
    database.enqueue_cleanup([(pending, "file")])

    def assert_cleanup_finished(_manager) -> None:
        assert not pending.exists()

    monkeypatch.setattr("pingpong_highlight.jobs.JobManager.start", assert_cleanup_finished)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").status_code == 200

    assert database.list_cleanup_records() == []
