from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import subprocess
import urllib.parse
from pathlib import Path

import pytest

from pingpong_highlight.archive import (
    ARCHIVE_NAMING_VERSION,
    ArchiveError,
    PCloudArchiver,
    RclonePCloudBackend,
    RemoteStat,
    discover_archive_candidates,
)
from pingpong_highlight.cli import build_parser
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database, StateConflict


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path, upload_token="test")
    settings.ensure_directories()
    return settings


def _catalog_with_media(settings: Settings) -> tuple[Database, str]:
    database = Database(settings.database_path)
    upload_id = "a" * 32
    source = settings.uploads_dir / f"{upload_id}.mp4"
    source.write_bytes(b"original-video")
    _upload, job = database.register_completed_upload(
        upload_id,
        "PXL_20260514_081118569.MP4",
        source.stat().st_size,
        "video/mp4",
        source,
    )
    assert database.claim_job(job.id)
    output_dir = settings.outputs_dir / job.id
    output_dir.mkdir(parents=True)
    clip_name = "highlight_001.mp4"
    (output_dir / clip_name).write_bytes(b"highlight-video")
    result = {
        "algorithm_version": "highlight-library-v2",
        "candidates": [{"score": 12.5}],
        "points": [
            {
                "start": 10.0,
                "end": 18.0,
                "rally_start": 11.5,
                "rally_end": 16.5,
                "score": 12.5,
                "rank": 1,
            }
        ],
        "files": [{"name": clip_name, "kind": "highlight"}],
    }
    database.finish_job(job.id, result)
    highlight = database.list_highlight_clips()[0]

    compilation = database.create_compilation(
        name="我的 夏季 精華",
        highlight_ids=[highlight.id],
    )
    assert database.claim_compilation(compilation.id)
    compilation_dir = settings.compilations_dir / compilation.id
    compilation_dir.mkdir(parents=True)
    compilation_file = compilation_dir / "highlight_compilation.mp4"
    compilation_file.write_bytes(b"compilation-video")
    database.finish_compilation(
        compilation.id,
        file_name=compilation_file.name,
        duration=8.0,
    )
    return database, upload_id


def test_archive_plan_uses_versioned_collision_resistant_names(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database, upload_id = _catalog_with_media(settings)

    candidates = discover_archive_candidates(settings, database)

    assert [candidate.media_kind for candidate in candidates] == [
        "original",
        "highlight_clip",
        "compilation",
    ]
    original = candidates[0]
    assert original.remote_path == (
        "HighlightCraft/archive-v1/originals/2026/05/"
        f"20260514T081118--{upload_id}/"
        f"20260514T081118--original--{upload_id[:12]}.mp4"
    )
    assert original.manifest_remote_path.endswith("/manifest.json")
    assert original.local_relative_path == f"uploads/{upload_id}.mp4"

    clip = candidates[1]
    assert f"/highlight-clips/2026/05/{upload_id}/highlight-library-v2/" in (
        clip.remote_path
    )
    assert clip.remote_path.endswith(f"001--clip--{clip.owner_id}.mp4")
    assert clip.manifest_remote_path.endswith(f"001--clip--{clip.owner_id}.json")

    compilation = candidates[2]
    assert "/compilations/" in compilation.remote_path
    assert "我的-夏季-精華" in compilation.remote_path
    assert compilation.manifest_remote_path.endswith("/manifest.json")

    payload = json.loads(
        original.manifest_bytes(
            byte_size=14,
            sha1="1" * 40,
            sha256="2" * 64,
            remote_name="highlightcraft-pcloud",
        )
    )
    assert payload["source"]["original_name"] == "PXL_20260514_081118569.MP4"
    assert payload["archive"]["naming_version"] == ARCHIVE_NAMING_VERSION
    assert payload["content"]["sha256"] == "2" * 64


def test_archive_limit_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pcloud", "archive", "--limit", "0"])


def test_storage_object_identity_and_state_survive_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    values = {
        "media_kind": "original",
        "owner_type": "upload",
        "owner_id": "a" * 32,
        "source_name": "phone.mp4",
        "local_relative_path": "uploads/source.mp4",
        "provider": "pcloud",
        "remote_name": "highlightcraft-pcloud",
        "naming_version": ARCHIVE_NAMING_VERSION,
        "remote_path": "HighlightCraft/archive-v1/originals/source.mp4",
        "manifest_remote_path": "HighlightCraft/archive-v1/originals/manifest.json",
        "byte_size": 10,
    }
    record = database.ensure_storage_object(**values)
    assert database.ensure_storage_object(**values).id == record.id

    database.start_storage_upload(
        record.id,
        local_sha1="1" * 40,
        local_sha256="2" * 64,
        byte_size=10,
        manifest_sha1="3" * 40,
        manifest_byte_size=123,
        manifest_sha256="3" * 64,
    )
    database.mark_storage_verifying(record.id)
    verified = database.finish_storage_verification(
        record.id,
        remote_file_id="f123",
        remote_hash_algorithm="sha1",
        remote_hash="1" * 40,
    )
    assert verified.archive_state == "verified"
    assert verified.attempts == 1

    reopened = Database(settings.database_path)
    persisted = reopened.get_storage_object(record.id)
    assert persisted is not None
    assert persisted.remote_file_id == "f123"
    assert persisted.verified_at is not None

    changed = dict(values, remote_path="HighlightCraft/archive-v2/source.mp4")
    with pytest.raises(StateConflict, match="catalog migration"):
        reopened.ensure_storage_object(**changed)

    changed_source = dict(values, source_name="renamed-after-transfer.mp4")
    with pytest.raises(StateConflict, match="source identity changed"):
        reopened.ensure_storage_object(**changed_source)


class _FakeBackend:
    def __init__(self, *, fail_manifest_once: bool = False):
        self.files: dict[str, bytes] = {}
        self.copy_calls: list[str] = []
        self.fail_manifest_once = fail_manifest_once

    def stat(self, remote_path: str) -> RemoteStat | None:
        content = self.files.get(remote_path)
        if content is None:
            return None
        return RemoteStat(
            size=len(content),
            sha1=hashlib.sha1(content).hexdigest(),
            file_id=f"id-{hashlib.sha256(remote_path.encode()).hexdigest()[:12]}",
        )

    def copy_to(self, source: Path, remote_path: str) -> None:
        self.copy_calls.append(remote_path)
        if self.fail_manifest_once and source.name == "manifest.json":
            self.fail_manifest_once = False
            raise ArchiveError("simulated manifest transfer failure")
        self.files[remote_path] = source.read_bytes()

    def finalize_no_overwrite(
        self,
        source_remote_path: str,
        destination_remote_path: str,
    ) -> None:
        if destination_remote_path not in self.files:
            self.files[destination_remote_path] = self.files[source_remote_path]

    def delete_staging(self, remote_path: str) -> None:
        self.files.pop(remote_path)


def test_archive_retry_reconciles_final_video_and_never_deletes_local(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    backend = _FakeBackend(fail_manifest_once=True)
    archiver = PCloudArchiver(settings, database, backend)

    with pytest.raises(ArchiveError, match="manifest transfer"):
        archiver.archive(candidate)

    failed = database.list_storage_objects()[0]
    assert failed.archive_state == "failed"
    assert candidate.local_path.read_bytes() == b"original-video"
    assert candidate.remote_path in backend.files
    video_uploads = [path for path in backend.copy_calls if path.endswith(".mp4")]
    assert len(video_uploads) == 1

    verified = archiver.archive(candidate)
    assert verified.archive_state == "verified"
    assert verified.local_state == "present"
    assert verified.local_sha256 == hashlib.sha256(b"original-video").hexdigest()
    assert candidate.manifest_remote_path in backend.files
    assert candidate.local_path.read_bytes() == b"original-video"
    video_uploads = [path for path in backend.copy_calls if path.endswith(".mp4")]
    assert len(video_uploads) == 1

    candidate.local_path.unlink()
    remote_only = archiver.archive(candidate)
    assert remote_only.archive_state == "verified"
    assert remote_only.local_state == "missing"

    candidate.local_path.write_bytes(b"original-video")
    restored = archiver.archive(candidate)
    assert restored.archive_state == "verified"
    assert restored.local_state == "present"


def test_archive_refuses_to_overwrite_mismatched_final_object(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    backend = _FakeBackend()
    backend.files[candidate.remote_path] = b"wrong-content"
    archiver = PCloudArchiver(settings, database, backend)

    with pytest.raises(ArchiveError, match="verification failed"):
        archiver.archive(candidate)

    record = database.list_storage_objects()[0]
    assert record.archive_state == "failed"
    assert backend.files[candidate.remote_path] == b"wrong-content"
    assert candidate.local_path.read_bytes() == b"original-video"


@pytest.mark.parametrize(
    ("replacement", "message"),
    [(b"", "source is empty"), (b"short", "source size changed")],
)
def test_archive_rejects_changed_source_before_transfer(
    tmp_path: Path,
    replacement: bytes,
    message: str,
) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    candidate.local_path.write_bytes(replacement)
    backend = _FakeBackend()

    with pytest.raises(ArchiveError, match=message):
        PCloudArchiver(settings, database, backend).archive(candidate)

    assert backend.files == {}
    assert database.list_storage_objects()[0].archive_state == "failed"


@pytest.mark.parametrize(
    ("table", "column", "unsafe_value", "message"),
    [
        ("highlight_clips", "clip_filename", r"..\escape.mp4", "highlight path"),
        (
            "compilations",
            "file_name",
            "../escape.mp4",
            "compilation filename",
        ),
    ],
)
def test_archive_discovery_rejects_unsafe_catalog_paths(
    tmp_path: Path,
    table: str,
    column: str,
    unsafe_value: str,
    message: str,
) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    assert table in {"highlight_clips", "compilations"}
    assert column in {"clip_filename", "file_name"}
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            f"UPDATE {table} SET {column} = ?",
            (unsafe_value,),
        )

    with pytest.raises(ArchiveError, match=message):
        discover_archive_candidates(settings, database)


def test_verified_archive_detects_remote_replacement(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    backend = _FakeBackend()
    archiver = PCloudArchiver(
        settings,
        database,
        backend,
        remote_hostname="api.pcloud.com",
        remote_account_id="123",
        remote_root_folder_id="d0",
    )
    verified = archiver.archive(candidate)
    backend.files[candidate.remote_path] = b"replaced-remotely"

    with pytest.raises(ArchiveError, match="verification failed"):
        archiver.verify(candidate, verified)

    failed = database.get_storage_object(verified.id)
    assert failed is not None
    assert failed.archive_state == "failed"
    assert candidate.local_path.read_bytes() == b"original-video"


def test_remote_verification_can_recover_after_transient_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    backend = _FakeBackend()
    archiver = PCloudArchiver(settings, database, backend)
    verified = archiver.archive(candidate)
    original_stat = backend.stat
    failures_remaining = 1

    def flaky_stat(remote_path: str):
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise ArchiveError("temporary provider error")
        return original_stat(remote_path)

    monkeypatch.setattr(backend, "stat", flaky_stat)
    with pytest.raises(ArchiveError, match="temporary provider error"):
        archiver.verify_record(verified)

    failed = database.get_storage_object(verified.id)
    assert failed is not None
    assert failed.archive_state == "failed"
    recovered = archiver.verify_record(failed)
    assert recovered.archive_state == "verified"
    assert recovered.last_error is None


@pytest.mark.parametrize("changed_account", [None, "456"])
def test_archive_target_account_is_immutable_after_first_attempt(
    tmp_path: Path,
    changed_account: str | None,
) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    backend = _FakeBackend()
    PCloudArchiver(
        settings,
        database,
        backend,
        remote_hostname="api.pcloud.com",
        remote_account_id="123",
        remote_root_folder_id="d0",
    ).archive(candidate)

    with pytest.raises(StateConflict, match="account changed"):
        PCloudArchiver(
            settings,
            database,
            backend,
            remote_hostname="api.pcloud.com",
            remote_account_id=changed_account,
            remote_root_folder_id="d0",
        ).archive(candidate)


def test_archive_root_change_is_rejected_before_registering_another_owner(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidates = discover_archive_candidates(settings, database)
    backend = _FakeBackend()
    PCloudArchiver(
        settings,
        database,
        backend,
        remote_hostname="api.pcloud.com",
        remote_account_id="123",
        remote_root_folder_id="d0",
    ).archive(candidates[0])

    changed_settings = Settings(
        data_dir=tmp_path,
        upload_token="test",
        pcloud_root="Other",
    )
    changed_candidates = discover_archive_candidates(changed_settings, database)
    with pytest.raises(StateConflict, match="archive root changed"):
        PCloudArchiver(
            changed_settings,
            database,
            backend,
            remote_hostname="api.pcloud.com",
            remote_account_id="123",
            remote_root_folder_id="d0",
        ).archive(changed_candidates[1])

    assert len(database.list_storage_objects()) == 1


def test_retry_keeps_content_identity_from_first_transfer_attempt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    backend = _FakeBackend(fail_manifest_once=True)
    archiver = PCloudArchiver(settings, database, backend)

    with pytest.raises(ArchiveError, match="manifest transfer"):
        archiver.archive(candidate)
    first_attempt = database.list_storage_objects()[0]
    original_sha1 = first_attempt.local_sha1
    original_sha256 = first_attempt.local_sha256
    original_remote = backend.files[candidate.remote_path]

    candidate.local_path.write_bytes(b"replaced-video")
    assert candidate.local_path.stat().st_size == candidate.byte_size
    with pytest.raises(StateConflict, match="content identity changed"):
        archiver.archive(candidate)

    rejected = database.get_storage_object(first_attempt.id)
    assert rejected is not None
    assert rejected.local_sha1 == original_sha1
    assert rejected.local_sha256 == original_sha256
    assert backend.files[candidate.remote_path] == original_remote


def test_retry_keeps_content_identity_after_remote_audit_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database, _upload_id = _catalog_with_media(settings)
    candidate = discover_archive_candidates(settings, database)[0]
    backend = _FakeBackend()
    archiver = PCloudArchiver(settings, database, backend)
    verified = archiver.archive(candidate)
    original_sha1 = verified.local_sha1
    original_sha256 = verified.local_sha256
    original_remote = backend.files[candidate.remote_path]

    backend.files.pop(candidate.manifest_remote_path)
    with pytest.raises(ArchiveError, match="missing video or manifest"):
        archiver.verify_record(verified)

    candidate.local_path.write_bytes(b"replaced-video")
    assert candidate.local_path.stat().st_size == candidate.byte_size
    with pytest.raises(StateConflict, match="content identity changed"):
        archiver.archive(candidate)

    rejected = database.get_storage_object(verified.id)
    assert rejected is not None
    assert rejected.local_sha1 == original_sha1
    assert rejected.local_sha256 == original_sha256
    assert backend.files[candidate.remote_path] == original_remote


def test_partial_storage_schema_is_migrated_without_losing_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "state.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE storage_objects (
                id TEXT PRIMARY KEY,
                media_kind TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                local_relative_path TEXT NOT NULL,
                local_state TEXT NOT NULL,
                provider TEXT NOT NULL,
                naming_version TEXT NOT NULL,
                remote_path TEXT NOT NULL,
                manifest_remote_path TEXT NOT NULL,
                archive_state TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                local_sha1 TEXT,
                local_sha256 TEXT,
                remote_hash_algorithm TEXT,
                remote_hash TEXT,
                remote_file_id TEXT,
                manifest_sha256 TEXT,
                attempts INTEGER NOT NULL,
                last_error TEXT,
                uploaded_at TEXT,
                verified_at TEXT,
                last_checked_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(provider, owner_type, owner_id),
                UNIQUE(provider, remote_path)
            );
            INSERT INTO storage_objects (
                id, media_kind, owner_type, owner_id, source_name,
                local_relative_path, local_state, provider, naming_version,
                remote_path, manifest_remote_path, archive_state, byte_size,
                attempts, created_at, updated_at
            ) VALUES (
                'legacy', 'original', 'upload', 'owner', 'phone.mp4',
                'uploads/source.mp4', 'present', 'pcloud', 'archive-v1',
                'HighlightCraft/archive-v1/source.mp4',
                'HighlightCraft/archive-v1/manifest.json', 'pending', 10,
                0, '2026-08-23T00:00:00Z', '2026-08-23T00:00:00Z'
            );
            """
        )

    database = Database(database_path)
    record = database.get_storage_object("legacy")

    assert record is not None
    assert record.remote_name == "highlightcraft-pcloud"
    assert record.remote_account_id is None
    assert record.manifest_sha1 is None
    assert record.manifest_byte_size is None


def test_rclone_staging_copy_uses_ignore_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    RclonePCloudBackend(settings).copy_to(source, "HighlightCraft/_staging/file.mp4")

    assert len(calls) == 1
    assert "--ignore-existing" in calls[0]
    assert "--checksum" in calls[0]
    assert "--immutable" not in calls[0]


def test_rclone_staging_cleanup_treats_already_missing_as_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            4,
            "",
            "object not found",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    RclonePCloudBackend(settings).delete_staging(
        "HighlightCraft/archive-v1/_staging/object/manifest.json"
    )

    assert len(calls) == 1
    assert "deletefile" in calls[0]


def test_directory_id_uses_pcloud_listfolder_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    rclone_calls: list[tuple[str, ...]] = []
    api_calls: list[tuple[str, dict[str, str]]] = []
    secret_section = backend._parse_remote_config(
        "[highlightcraft-pcloud]\nroot_folder_id = d77\n",
        "highlightcraft-pcloud",
    )

    def fake_run(*arguments: str, **_kwargs):
        rclone_calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_request(method: str, parameters: dict[str, str]):
        api_calls.append((method, parameters))
        folder_id = parameters["folderid"]
        children = {
            "77": [
                {
                    "id": "d88",
                    "folderid": 88,
                    "isfolder": True,
                    "name": "Morris",
                }
            ],
            "88": [
                {
                    "id": "d456",
                    "folderid": 456,
                    "isfolder": True,
                    "name": "HighlightCraft",
                }
            ],
        }
        return {
            "result": 0,
            "metadata": {
                "id": f"d{folder_id}",
                "folderid": int(folder_id),
                "isfolder": True,
                "name": "/" if folder_id == "77" else "Morris",
                "contents": children.get(folder_id, []),
            },
        }

    monkeypatch.setattr(backend, "_run", fake_run)
    monkeypatch.setattr(backend, "_pcloud_request", fake_request)
    monkeypatch.setattr(backend, "_secret_remote_config", lambda: secret_section)

    directory_id = backend._directory_id("Morris/HighlightCraft")

    assert directory_id == "456"
    assert rclone_calls == [
        (
            "mkdir",
            "highlightcraft-pcloud:Morris/HighlightCraft",
        )
    ]
    assert [parameters["folderid"] for _method, parameters in api_calls] == [
        "77",
        "77",
        "88",
    ]
    assert all(method == "listfolder" for method, _parameters in api_calls)
    assert all(parameters["nofiles"] == "1" for _method, parameters in api_calls)
    assert all("path" not in parameters for _method, parameters in api_calls)
    assert all("noshares" not in parameters for _method, parameters in api_calls)
    assert not any(call and call[0] == "lsjson" for call in rclone_calls)


@pytest.mark.parametrize("remote_path", ["/absolute", "safe/../escape"])
def test_directory_id_rejects_unsafe_path_before_remote_calls(
    tmp_path: Path,
    monkeypatch,
    remote_path: str,
) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    calls: list[str] = []
    monkeypatch.setattr(backend, "_run", lambda *_args, **_kwargs: calls.append("rclone"))
    monkeypatch.setattr(
        backend,
        "_pcloud_request",
        lambda *_args, **_kwargs: calls.append("api"),
    )

    with pytest.raises(ArchiveError, match="Invalid relative pCloud directory path"):
        backend._directory_id(remote_path)

    assert calls == []


def test_folder_metadata_requires_a_json_boolean_directory_flag() -> None:
    with pytest.raises(ArchiveError, match="valid folder ID"):
        RclonePCloudBackend._folder_id_from_metadata(
            {"id": "d7", "folderid": 7, "isfolder": "true"}
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"result": 2003, "error": "denied"}, "listfolder failed"),
        ({"result": 0}, "Invalid pCloud folder metadata"),
        (
            {
                "result": 0,
                "metadata": {"id": "d1", "folderid": 2, "isfolder": True},
            },
            "valid folder ID",
        ),
    ],
)
def test_list_folder_rejects_invalid_provider_payload(
    tmp_path: Path,
    monkeypatch,
    payload: dict,
    message: str,
) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    monkeypatch.setattr(backend, "_pcloud_request", lambda *_args: payload)

    with pytest.raises(ArchiveError, match=message):
        backend._list_folder("0")


@pytest.mark.parametrize(
    "contents",
    [
        [],
        [{"id": "f8", "fileid": 8, "isfolder": False, "name": "Target"}],
        [
            {"id": "d8", "folderid": 8, "isfolder": True, "name": "Target"},
            {"id": "d9", "folderid": 9, "isfolder": True, "name": "Target"},
        ],
    ],
)
def test_directory_id_fails_closed_on_ambiguous_or_missing_child(
    tmp_path: Path,
    monkeypatch,
    contents: list[dict],
) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    secret_section = backend._parse_remote_config(
        "[highlightcraft-pcloud]\nroot_folder_id = d0\n",
        "highlightcraft-pcloud",
    )
    monkeypatch.setattr(
        backend,
        "_run",
        lambda *arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, "", ""
        ),
    )
    monkeypatch.setattr(backend, "_secret_remote_config", lambda: secret_section)
    monkeypatch.setattr(
        backend,
        "_pcloud_request",
        lambda *_args: {
            "result": 0,
            "metadata": {
                "id": "d0",
                "folderid": 0,
                "isfolder": True,
                "contents": contents,
            },
        },
    )

    with pytest.raises(ArchiveError, match="exactly one directory"):
        backend._directory_id("Target")


def test_pcloud_doctor_reads_custom_root_id_from_secret_config_and_api(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "rclone.conf"
    config_path.write_text("placeholder", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path,
        upload_token="test",
        rclone_config=config_path,
    )
    settings.ensure_directories()
    backend = RclonePCloudBackend(settings)
    rclone_calls: list[tuple[str, ...]] = []
    api_calls: list[tuple[str, dict[str, str]]] = []
    secret_section = backend._parse_remote_config(
        "[highlightcraft-pcloud]\nroot_folder_id = 77\n",
        "highlightcraft-pcloud",
    )

    def fake_run(*arguments: str, **_kwargs):
        rclone_calls.append(arguments)
        stdout = {
            ("version",): "rclone v1.75.0\n",
            ("listremotes",): "highlightcraft-pcloud:\n",
            (
                "config",
                "redacted",
                "highlightcraft-pcloud",
            ): (
                "[highlightcraft-pcloud]\n"
                "type = pcloud\n"
                "hostname = api.pcloud.com\n"
                "root_folder_id = XXX\n"
            ),
            ("lsd", "highlightcraft-pcloud:"): "",
        }[arguments]
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    def fake_request(method: str, parameters: dict[str, str]):
        api_calls.append((method, parameters))
        if method == "userinfo":
            return {"result": 0, "userid": 123}
        assert method == "listfolder"
        return {
            "result": 0,
            "metadata": {
                "id": "d77",
                "folderid": 77,
                "isfolder": True,
                "path": "/",
                "contents": [],
            },
        }

    monkeypatch.setattr("pingpong_highlight.archive.shutil.which", lambda _name: "rclone")
    monkeypatch.setattr(backend, "_run", fake_run)
    monkeypatch.setattr(backend, "_pcloud_request", fake_request)
    monkeypatch.setattr(backend, "_secret_remote_config", lambda: secret_section)

    result = backend.doctor()

    assert result.region == "US"
    assert result.account_id == "123"
    assert result.root_folder_id == "d77"
    assert not any(call and call[0] == "lsjson" for call in rclone_calls)
    assert api_calls == [
        ("userinfo", {}),
        (
            "listfolder",
            {
                "folderid": "77",
                "nofiles": "1",
            },
        ),
    ]


@pytest.mark.parametrize("result", [0, 2004])
def test_pcloud_finalize_uses_provider_noover(
    tmp_path: Path,
    monkeypatch,
    result: int,
) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    source_path = "HighlightCraft/archive-v1/_staging/object/payload.mp4"
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        backend,
        "stat",
        lambda _path: RemoteStat(size=5, sha1="1" * 40, file_id="f123"),
    )
    monkeypatch.setattr(backend, "_directory_id", lambda _path: "456")

    def fake_request(method: str, parameters: dict[str, str]):
        calls.append((method, parameters))
        return {"result": result, "error": "already exists"}

    monkeypatch.setattr(backend, "_pcloud_request", fake_request)

    backend.finalize_no_overwrite(
        source_path,
        "HighlightCraft/archive-v1/originals/final.mp4",
    )

    assert calls == [
        (
            "copyfile",
            {
                "fileid": "123",
                "tofolderid": "456",
                "toname": "final.mp4",
                "noover": "1",
            },
        )
    ]


def test_pcloud_finalize_rejects_provider_error(tmp_path: Path, monkeypatch) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    monkeypatch.setattr(
        backend,
        "stat",
        lambda _path: RemoteStat(size=5, sha1="1" * 40, file_id="f123"),
    )
    monkeypatch.setattr(backend, "_directory_id", lambda _path: "456")
    monkeypatch.setattr(
        backend,
        "_pcloud_request",
        lambda _method, _parameters: {"result": 2003, "error": "denied"},
    )

    with pytest.raises(ArchiveError, match=r"copyfile failed \(2003\): denied"):
        backend.finalize_no_overwrite("staging.mp4", "final.mp4")


def test_pcloud_api_uses_bearer_header_without_putting_token_in_body_or_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    token = "secret-oauth-token"
    requests = []
    monkeypatch.setattr(
        backend,
        "_pcloud_credentials",
        lambda: ("api.pcloud.com", token),
    )

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(b'{"result": 0, "userid": 123}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    payload = backend._pcloud_request("userinfo", {"sample": "value"})

    request, timeout = requests[0]
    assert payload["userid"] == 123
    assert timeout == 60
    assert request.full_url == "https://api.pcloud.com/userinfo"
    assert request.headers["Authorization"] == f"Bearer {token}"
    assert token.encode() not in request.data
    assert urllib.parse.parse_qs(request.data.decode()) == {"sample": ["value"]}


def test_secret_config_failure_is_noninteractive_and_does_not_leak_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = RclonePCloudBackend(_settings(tmp_path))
    secret = "do-not-leak"
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, secret, "failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ArchiveError) as failure:
        backend._secret_remote_config()

    assert "--ask-password=false" in calls[0]
    assert secret not in str(failure.value)
