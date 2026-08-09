from __future__ import annotations

import os
from pathlib import Path

from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.uploads import UploadStore


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
