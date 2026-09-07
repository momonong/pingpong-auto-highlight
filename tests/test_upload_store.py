from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.uploads import UploadError, UploadStore


def test_create_reserves_declared_upload_size_and_minimum_free_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        upload_token="test",
        download_min_free_bytes=100,
    )
    settings.ensure_directories()
    database = Database(settings.database_path)
    store = UploadStore(settings, database)
    monkeypatch.setattr(
        "pingpong_highlight.uploads.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=104),
    )

    with pytest.raises(UploadError) as error:
        store.create("match.mov", 5, "video/quicktime")

    assert error.value.status_code == 507


def test_reconcile_finishes_atomic_rename_crash_window(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, upload_token="test")
    settings.ensure_directories()
    database = Database(settings.database_path)
    store = UploadStore(settings, database)
    record = store.create("match.mov", 5, "video/quicktime")

    part = store.part_path(record)
    part.write_bytes(b"video")
    database.force_upload_offset(record.id, 5)
    os.replace(part, record.path)

    store.reconcile()

    recovered = database.get_upload(record.id)
    assert recovered is not None
    assert recovered.status == "queued"
    assert recovered.job_id is not None


def test_reconcile_finishes_full_part_before_atomic_rename_crash_window(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path, upload_token="test")
    settings.ensure_directories()
    database = Database(settings.database_path)
    store = UploadStore(settings, database)
    record = store.create("match.mov", 5, "video/quicktime")

    part = store.part_path(record)
    part.write_bytes(b"video")
    database.force_upload_offset(record.id, 5)

    store.reconcile()

    recovered = database.get_upload(record.id)
    assert recovered is not None
    assert recovered.status == "queued"
    assert recovered.job_id is not None
    assert recovered.path.read_bytes() == b"video"
    assert not part.exists()
