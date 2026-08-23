from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import pingpong_highlight.compilations as compilation_module
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.media_work import media_work_lock
from pingpong_highlight.web import create_app


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path, upload_token="test-secret")
    settings.ensure_directories()
    return settings


def test_media_work_lock_serializes_separate_processes(tmp_path: Path) -> None:
    ready = tmp_path / "child-ready"
    acquired = tmp_path / "child-acquired"
    script = """
from pathlib import Path
import sys
from pingpong_highlight.media_work import media_work_lock

data_dir, ready, acquired = map(Path, sys.argv[1:])
ready.write_text("ready", encoding="utf-8")
with media_work_lock(data_dir):
    acquired.write_text("acquired", encoding="utf-8")
"""
    process: subprocess.Popen[str] | None = None
    try:
        with media_work_lock(tmp_path):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(tmp_path),
                    str(ready),
                    str(acquired),
                ],
                text=True,
            )
            for _ in range(100):
                if ready.exists() or process.poll() is not None:
                    break
                time.sleep(0.05)
            assert ready.exists()
            time.sleep(0.2)
            assert process.poll() is None
            assert not acquired.exists()
        assert process.wait(timeout=10) == 0
        assert acquired.read_text(encoding="utf-8") == "acquired"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def _register_highlight(
    app,
    settings: Settings,
    *,
    upload_id: str,
    filename: str,
    score: float,
) -> tuple[str, str]:
    source = settings.uploads_dir / f"{upload_id}.mp4"
    source.write_bytes(b"source")
    _upload, job = app.state.database.register_completed_upload(
        upload_id,
        filename,
        source.stat().st_size,
        "video/mp4",
        source,
    )
    assert app.state.database.claim_job(job.id)
    output_dir = settings.outputs_dir / job.id
    output_dir.mkdir(parents=True)
    clip_name = "highlight_001_rank_001.mp4"
    (output_dir / clip_name).write_bytes(upload_id.encode())
    result = {
        "algorithm_version": "highlight-library-v1",
        "source_name": filename,
        "media": {"duration": 120.0},
        "summary": {"point_count": 1, "library_duration": 10.0},
        "candidates": [{"score": score, "selection": "selected"}],
        "points": [
            {
                "start": 10.0,
                "end": 20.0,
                "clip_start": 10.0,
                "clip_end": 20.0,
                "rally_start": 11.5,
                "rally_end": 18.5,
                "score": score,
                "rank": 1,
                "reason": "test",
            }
        ],
        "files": [
            {"name": clip_name, "kind": "highlight"},
            {"name": "analysis.json", "kind": "analysis"},
        ],
    }
    app.state.database.finish_job(job.id, result)
    highlight = next(
        item
        for item in app.state.database.list_highlight_clips()
        if item.upload_id == upload_id
    )
    return highlight.id, clip_name


def test_cross_source_library_and_unlimited_compilation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    built_from: list[str] = []

    def fake_build(clips: list[Path], destination: Path) -> None:
        built_from.extend(clip.parent.name for clip in clips)
        destination.write_bytes(b"joined-video")

    monkeypatch.setattr(compilation_module, "build_point_reel", fake_build)
    monkeypatch.setattr(
        compilation_module,
        "probe_media",
        lambda _path: SimpleNamespace(duration=123.0),
    )

    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        first_id, _first_name = _register_highlight(
            app,
            settings,
            upload_id="first-upload",
            filename="PXL_20260514_081118569.mp4",
            score=20.0,
        )
        second_id, _second_name = _register_highlight(
            app,
            settings,
            upload_id="second-upload",
            filename="second_match.mp4",
            score=12.0,
        )

        library = client.get(
            "/api/highlights",
            headers={"X-Upload-Token": "test-secret"},
        )
        assert library.status_code == 200
        highlights = library.json()["highlights"]
        assert len(highlights) == 2
        dated = next(item for item in highlights if item["id"] == first_id)
        assert dated["source_date"] == "2026-05-14T08:11:18"
        assert dated["source_date_source"] == "filename"
        assert dated["relative_score"] == 1.0

        preview = client.get(
            f"/api/highlights/{first_id}/media",
            headers={"X-Upload-Token": "test-secret", "Range": "bytes=0-4"},
        )
        assert preview.status_code == 206
        assert preview.content == b"first"

        created = client.post(
            "/api/compilations",
            headers={"X-Upload-Token": "test-secret"},
            json={
                "name": "兩場自由集錦",
                "highlight_ids": [second_id, first_id],
            },
        )
        assert created.status_code == 202
        compilation_id = created.json()["id"]

        for _ in range(100):
            compilation = client.get(
                f"/api/compilations/{compilation_id}",
                headers={"X-Upload-Token": "test-secret"},
            ).json()
            if compilation["status"] == "completed":
                break
            time.sleep(0.01)

        assert compilation["status"] == "completed"
        assert compilation["duration"] == 123.0
        assert compilation["item_count"] == 2
        assert compilation["source_count"] == 2
        second_job = next(item["job_id"] for item in highlights if item["id"] == second_id)
        first_job = next(item["job_id"] for item in highlights if item["id"] == first_id)
        assert built_from == [second_job, first_job]

        output = client.get(
            f"/api/compilations/{compilation_id}/file?download=true",
            headers={"X-Upload-Token": "test-secret"},
        )
        assert output.status_code == 200
        assert output.content == b"joined-video"
        assert output.headers["content-disposition"].startswith("attachment")


def test_compilation_rejects_clip_path_outside_job_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    build_called = False

    def fake_build(_clips: list[Path], _destination: Path) -> None:
        nonlocal build_called
        build_called = True

    monkeypatch.setattr(compilation_module, "build_point_reel", fake_build)
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        highlight_id, _clip_name = _register_highlight(
            app,
            settings,
            upload_id="unsafe-upload",
            filename="unsafe.mp4",
            score=10.0,
        )
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"must-not-be-read")
        with sqlite3.connect(settings.database_path) as connection:
            connection.execute(
                "UPDATE highlight_clips SET clip_filename = ? WHERE id = ?",
                ("../../outside.mp4", highlight_id),
            )

        created = client.post(
            "/api/compilations",
            headers={"X-Upload-Token": "test-secret"},
            json={"name": "unsafe", "highlight_ids": [highlight_id]},
        )
        assert created.status_code == 202
        compilation_id = created.json()["id"]
        for _ in range(100):
            compilation = client.get(
                f"/api/compilations/{compilation_id}",
                headers={"X-Upload-Token": "test-secret"},
            ).json()
            if compilation["status"] == "failed":
                break
            time.sleep(0.01)

        assert compilation["status"] == "failed"
        assert "missing" in compilation["error"].lower()
        assert build_called is False


def test_rebuilt_library_version_replaces_visible_legacy_clips_without_deleting_them(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    source = settings.uploads_dir / "source.mp4"
    source.write_bytes(b"source")
    _upload, job = database.register_completed_upload(
        "upload-id",
        "PXL_20260502_074602205.mp4",
        source.stat().st_size,
        "video/mp4",
        source,
    )
    assert database.claim_job(job.id)
    legacy = {
        "algorithm_version": "point-reel-v5",
        "candidates": [{"score": 10.0}],
        "points": [
            {
                "start": 1.0,
                "end": 5.0,
                "rally_start": 2.0,
                "rally_end": 4.0,
                "score": 10.0,
                "rank": 1,
            }
        ],
        "files": [{"name": "point_001.mp4", "kind": "point"}],
    }
    database.finish_job(job.id, legacy)
    legacy_clip = database.list_highlight_clips()[0]

    rebuilt = {
        "algorithm_version": "highlight-library-v1",
        "candidates": [{"score": 12.0}, {"score": 11.0}],
        "points": [
            {
                "start": 10.0,
                "end": 15.0,
                "rally_start": 11.0,
                "rally_end": 14.0,
                "score": 12.0,
                "rank": 1,
            },
            {
                "start": 30.0,
                "end": 36.0,
                "rally_start": 31.0,
                "rally_end": 35.0,
                "score": 11.0,
                "rank": 2,
            },
        ],
        "files": [
            {"name": "highlight_001.mp4", "kind": "highlight"},
            {"name": "highlight_002.mp4", "kind": "highlight"},
        ],
    }
    assert database.activate_highlight_result(
        job.id,
        rebuilt,
        file_prefix="clip-sets/run-2",
        library_version="highlight-library-v1",
    ) == 2

    visible = database.list_highlight_clips()
    assert len(visible) == 2
    assert {clip.clip_filename for clip in visible} == {
        "clip-sets/run-2/highlight_001.mp4",
        "clip-sets/run-2/highlight_002.mp4",
    }
    assert all(clip.library_version == "highlight-library-v1" for clip in visible)
    preserved = database.get_highlight_clip(legacy_clip.id)
    assert preserved is not None
    assert preserved.active is False

    reopened = Database(settings.database_path)
    visible_after_restart = reopened.list_highlight_clips()
    assert len(visible_after_restart) == 2
    assert all("clip-sets/run-2/" in clip.clip_filename for clip in visible_after_restart)
