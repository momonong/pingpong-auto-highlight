from __future__ import annotations

import configparser
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database, StateConflict, StorageObjectRecord

ARCHIVE_PROVIDER = "pcloud"
ARCHIVE_NAMING_VERSION = "archive-v1"
ARCHIVE_ROOTS = (
    "inbox",
    f"{ARCHIVE_NAMING_VERSION}/originals",
    f"{ARCHIVE_NAMING_VERSION}/highlight-clips",
    f"{ARCHIVE_NAMING_VERSION}/compilations",
    f"{ARCHIVE_NAMING_VERSION}/database-snapshots",
    f"{ARCHIVE_NAMING_VERSION}/_staging",
    f"{ARCHIVE_NAMING_VERSION}/_quarantine",
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_SAFE_EXTENSION = re.compile(r"\.[a-z0-9]{1,8}")


class ArchiveError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArchiveCandidate:
    media_kind: str
    owner_type: str
    owner_id: str
    source_name: str
    local_path: Path
    local_relative_path: str
    remote_path: str
    manifest_remote_path: str
    byte_size: int
    metadata: dict[str, Any]

    def manifest_bytes(
        self,
        *,
        byte_size: int,
        sha1: str,
        sha256: str,
        remote_name: str,
    ) -> bytes:
        payload = {
            "schema": "highlightcraft.archive-manifest.v1",
            "media_kind": self.media_kind,
            "owner": {"type": self.owner_type, "id": self.owner_id},
            "source": {
                "original_name": self.source_name,
                "local_relative_path": self.local_relative_path,
            },
            "archive": {
                "provider": ARCHIVE_PROVIDER,
                "remote_name": remote_name,
                "naming_version": ARCHIVE_NAMING_VERSION,
                "remote_path": self.remote_path,
                "manifest_remote_path": self.manifest_remote_path,
            },
            "content": {
                "bytes": byte_size,
                "sha1": sha1,
                "sha256": sha256,
            },
            "metadata": self.metadata,
        }
        return (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()


@dataclass(frozen=True, slots=True)
class RemoteStat:
    size: int
    sha1: str
    file_id: str | None


@dataclass(frozen=True, slots=True)
class PCloudDoctorResult:
    rclone_version: str
    remote: str
    hostname: str
    region: str
    account_id: str
    root_folder_id: str


class ArchiveBackend(Protocol):
    def stat(self, remote_path: str) -> RemoteStat | None: ...

    def copy_to(self, source: Path, remote_path: str) -> None: ...

    def finalize_no_overwrite(
        self,
        source_remote_path: str,
        destination_remote_path: str,
    ) -> None: ...

    def delete_staging(self, remote_path: str) -> None: ...


def _safe_id(value: str, label: str) -> str:
    if _SAFE_ID.fullmatch(value) is None:
        raise ArchiveError(f"Unsafe {label}: {value!r}")
    return value


def _slug(value: str, *, fallback: str, limit: int = 48) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    output: list[str] = []
    pending_dash = False
    for character in normalized:
        if character.isalnum():
            if pending_dash and output:
                output.append("-")
            output.append(character)
            pending_dash = False
        else:
            pending_dash = True
    result = "".join(output).strip("-")[:limit].rstrip("-")
    return result or fallback


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArchiveError(f"Invalid catalog timestamp: {value!r}") from exc


def _timestamp_parts(value: str) -> tuple[str, str, str]:
    parsed = _parse_timestamp(value)
    suffix = ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC)
        suffix = "Z"
    return parsed.strftime("%Y"), parsed.strftime("%m"), parsed.strftime(
        f"%Y%m%dT%H%M%S{suffix}"
    )


def _extension(path: Path, source_name: str) -> str:
    extension = (path.suffix or Path(source_name).suffix).lower()
    return extension if _SAFE_EXTENSION.fullmatch(extension) else ".mp4"


def _remote_path(settings: Settings, *parts: str) -> str:
    candidate = PurePosixPath(settings.pcloud_root, *parts)
    if candidate.is_absolute() or ".." in candidate.parts or ":" in candidate.as_posix():
        raise ArchiveError("Unsafe generated pCloud path")
    return candidate.as_posix()


def _local_relative_path(settings: Settings, local_path: Path) -> str:
    resolved_data = settings.data_dir.resolve()
    resolved_local = local_path.resolve()
    if not resolved_local.is_relative_to(resolved_data):
        raise ArchiveError(f"Archive source escapes the data directory: {local_path}")
    return resolved_local.relative_to(resolved_data).as_posix()


def _safe_catalog_path(value: str, label: str, *, basename_only: bool = False) -> Path:
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        not candidate.parts
        or candidate.is_absolute()
        or ".." in candidate.parts
        or ":" in normalized
    ):
        raise ArchiveError(f"Unsafe {label}: {value!r}")
    if basename_only and len(candidate.parts) != 1:
        raise ArchiveError(f"Unsafe {label}: {value!r}")
    return Path(*candidate.parts)


def discover_archive_candidates(
    settings: Settings,
    database: Database,
) -> list[ArchiveCandidate]:
    candidates: list[ArchiveCandidate] = []

    for upload in database.list_archivable_uploads():
        upload_id = _safe_id(upload.id, "upload id")
        recorded_at = upload.recorded_at or upload.created_at
        year, month, timestamp = _timestamp_parts(recorded_at)
        local_path = settings.uploads_dir / upload.path.name
        extension = _extension(local_path, upload.filename)
        folder = (
            ARCHIVE_NAMING_VERSION,
            "originals",
            year,
            month,
            f"{timestamp}--{upload_id}",
        )
        remote_path = _remote_path(
            settings,
            *folder,
            f"{timestamp}--original--{upload_id[:12]}{extension}",
        )
        candidates.append(
            ArchiveCandidate(
                media_kind="original",
                owner_type="upload",
                owner_id=upload_id,
                source_name=upload.filename,
                local_path=local_path,
                local_relative_path=_local_relative_path(settings, local_path),
                remote_path=remote_path,
                manifest_remote_path=_remote_path(settings, *folder, "manifest.json"),
                byte_size=upload.size,
                metadata={
                    "content_type": upload.content_type,
                    "recorded_at": upload.recorded_at,
                    "recorded_at_source": upload.recorded_at_source,
                    "uploaded_at": upload.created_at,
                },
            )
        )

    for clip in database.list_highlight_clips():
        clip_id = _safe_id(clip.id, "highlight id")
        upload_id = _safe_id(clip.upload_id, "upload id")
        job_id = _safe_id(clip.job_id, "job id")
        year, month, timestamp = _timestamp_parts(clip.source_date)
        version = _slug(clip.library_version, fallback="unknown-version")
        clip_relative = _safe_catalog_path(clip.clip_filename, "highlight path")
        local_path = settings.outputs_dir / job_id / clip_relative
        extension = _extension(local_path, clip.clip_filename)
        folder = (
            ARCHIVE_NAMING_VERSION,
            "highlight-clips",
            year,
            month,
            upload_id,
            version,
        )
        stem = f"{clip.source_rank:03d}--clip--{clip_id}"
        candidates.append(
            ArchiveCandidate(
                media_kind="highlight_clip",
                owner_type="highlight_clip",
                owner_id=clip_id,
                source_name=clip.source_name,
                local_path=local_path,
                local_relative_path=_local_relative_path(settings, local_path),
                remote_path=_remote_path(settings, *folder, f"{stem}{extension}"),
                manifest_remote_path=_remote_path(settings, *folder, f"{stem}.json"),
                byte_size=local_path.stat().st_size if local_path.is_file() else 0,
                metadata={
                    "upload_id": upload_id,
                    "job_id": job_id,
                    "library_version": clip.library_version,
                    "source_date": clip.source_date,
                    "source_date_source": clip.source_date_source,
                    "source_rank": clip.source_rank,
                    "clip_start_seconds": clip.start,
                    "clip_end_seconds": clip.end,
                    "rally_start_seconds": clip.rally_start,
                    "rally_end_seconds": clip.rally_end,
                    "score": clip.score,
                    "relative_score": clip.relative_score,
                },
            )
        )

    for compilation in database.list_archivable_compilations():
        compilation_id = _safe_id(compilation.id, "compilation id")
        year, month, timestamp = _timestamp_parts(compilation.created_at)
        compilation_name = str(compilation.file_name)
        compilation_relative = _safe_catalog_path(
            compilation_name,
            "compilation filename",
            basename_only=True,
        )
        local_path = settings.compilations_dir / compilation_id / compilation_relative
        extension = _extension(local_path, compilation_name)
        folder = (
            ARCHIVE_NAMING_VERSION,
            "compilations",
            year,
            month,
            f"{timestamp}--{compilation_id}",
        )
        name = _slug(compilation.name, fallback="highlight")
        candidates.append(
            ArchiveCandidate(
                media_kind="compilation",
                owner_type="compilation",
                owner_id=compilation_id,
                source_name=compilation_name,
                local_path=local_path,
                local_relative_path=_local_relative_path(settings, local_path),
                remote_path=_remote_path(
                    settings,
                    *folder,
                    f"{timestamp}--{name}--c-{compilation_id[:12]}{extension}",
                ),
                manifest_remote_path=_remote_path(settings, *folder, "manifest.json"),
                byte_size=local_path.stat().st_size if local_path.is_file() else 0,
                metadata={
                    "name": compilation.name,
                    "duration_seconds": compilation.duration,
                    "created_at": compilation.created_at,
                },
            )
        )

    return candidates


def hash_file(path: Path) -> tuple[int, str, str]:
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            byte_size += len(chunk)
            sha1.update(chunk)
            sha256.update(chunk)
    return byte_size, sha1.hexdigest(), sha256.hexdigest()


class RclonePCloudBackend:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _command(self, *arguments: str) -> list[str]:
        command = [self.settings.rclone_binary]
        if self.settings.rclone_config is not None:
            command.extend(("--config", str(self.settings.rclone_config)))
        command.extend(arguments)
        return command

    def _run(
        self,
        *arguments: str,
        live: bool = False,
        missing_ok: bool = False,
    ) -> subprocess.CompletedProcess[str] | None:
        result = subprocess.run(
            self._command(*arguments),
            check=False,
            text=True,
            capture_output=not live,
        )
        if result.returncode == 0:
            return result
        if missing_ok and result.returncode in {3, 4}:
            return None
        detail = (result.stderr or result.stdout or "unknown rclone error").strip()
        raise ArchiveError(f"rclone failed ({result.returncode}): {detail[:1000]}")

    def _remote(self, remote_path: str = "") -> str:
        suffix = remote_path.lstrip("/")
        return f"{self.settings.pcloud_remote}:{suffix}"

    @staticmethod
    def _parse_remote_config(contents: str, remote_name: str) -> configparser.SectionProxy:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(contents)
            return parser[remote_name]
        except (configparser.Error, KeyError) as exc:
            raise ArchiveError(
                f"Could not parse rclone remote configuration: {remote_name}"
            ) from exc

    def _secret_remote_config(self) -> configparser.SectionProxy:
        result = subprocess.run(
            self._command(
                "config",
                "show",
                self.settings.pcloud_remote,
                "--ask-password=false",
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ArchiveError("rclone could not decrypt the pCloud remote configuration")
        return self._parse_remote_config(result.stdout, self.settings.pcloud_remote)

    def _pcloud_credentials(self) -> tuple[str, str]:
        section = self._secret_remote_config()
        hostname = section.get("hostname", "api.pcloud.com").strip()
        if hostname not in {"api.pcloud.com", "eapi.pcloud.com"}:
            raise ArchiveError(f"Unexpected pCloud API hostname: {hostname!r}")
        try:
            token_payload = json.loads(section["token"])
            access_token = str(token_payload["access_token"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ArchiveError("Could not read the pCloud OAuth token from rclone config") from exc
        if not access_token:
            raise ArchiveError("The pCloud OAuth token in rclone config is empty")
        return hostname, access_token

    def _pcloud_request(
        self,
        method: str,
        parameters: dict[str, str],
    ) -> dict[str, Any]:
        hostname, access_token = self._pcloud_credentials()
        body = urllib.parse.urlencode(parameters).encode()
        request = urllib.request.Request(
            f"https://{hostname}/{method}",
            data=body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "HighlightCraft/pcloud-archive-v1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ArchiveError(f"pCloud {method} request failed") from exc
        if not isinstance(payload, dict):
            raise ArchiveError(f"Invalid pCloud {method} response")
        return payload

    @staticmethod
    def _folder_id_from_metadata(metadata: object) -> str:
        if not isinstance(metadata, dict):
            raise ArchiveError("Invalid pCloud folder metadata")
        try:
            metadata_id = str(metadata["id"])
            numeric_id = str(metadata["folderid"])
            is_directory = metadata["isfolder"]
        except (KeyError, TypeError) as exc:
            raise ArchiveError("Invalid pCloud folder metadata") from exc
        if (
            is_directory is not True
            or re.fullmatch(r"d[0-9]+", metadata_id) is None
            or re.fullmatch(r"[0-9]+", numeric_id) is None
            or metadata_id != f"d{numeric_id}"
        ):
            raise ArchiveError("pCloud did not return a valid folder ID")
        return numeric_id

    def _list_folder(self, folder_id: str) -> dict[str, Any]:
        if re.fullmatch(r"[0-9]+", folder_id) is None:
            raise ArchiveError(f"Invalid pCloud folder ID: {folder_id!r}")
        payload = self._pcloud_request(
            "listfolder",
            {"folderid": folder_id, "nofiles": "1"},
        )
        try:
            result = int(payload["result"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveError("Invalid pCloud listfolder response") from exc
        if result != 0:
            detail = str(payload.get("error") or "unknown pCloud API error")
            raise ArchiveError(f"pCloud listfolder failed ({result}): {detail[:500]}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ArchiveError("Invalid pCloud folder metadata")
        returned_id = self._folder_id_from_metadata(metadata)
        canonical_requested_id = folder_id.lstrip("0") or "0"
        if returned_id != canonical_requested_id:
            raise ArchiveError("pCloud returned metadata for a different folder")
        return metadata

    def _configured_root_directory_id(self) -> str:
        section = self._secret_remote_config()
        configured = section.get("root_folder_id", "d0").strip() or "d0"
        match = re.fullmatch(r"d?([0-9]+)", configured)
        if match is None:
            raise ArchiveError(
                f"Invalid rclone pCloud root_folder_id: {configured!r}"
            )
        numeric_id = match.group(1).lstrip("0") or "0"
        return f"d{numeric_id}"

    def _root_directory_id(self) -> str:
        configured = self._configured_root_directory_id()
        numeric_id = configured[1:]
        metadata = self._list_folder(numeric_id)
        if str(metadata["id"]) != configured:
            raise ArchiveError("pCloud root folder identity does not match rclone config")
        return configured

    def _directory_id(self, remote_path: str) -> str:
        path = PurePosixPath(remote_path)
        if path.is_absolute() or ".." in path.parts:
            raise ArchiveError(f"Invalid relative pCloud directory path: {remote_path!r}")
        parts = tuple(part for part in path.parts if part not in {"", "."})
        if parts:
            self._run("mkdir", self._remote(remote_path))
        current_id = self._root_directory_id()[1:]
        for part in parts:
            metadata = self._list_folder(current_id)
            contents = metadata.get("contents")
            if not isinstance(contents, list):
                raise ArchiveError("Invalid pCloud folder contents")
            matches = [
                child
                for child in contents
                if isinstance(child, dict)
                and child.get("isfolder") is True
                and child.get("name") == part
            ]
            if len(matches) != 1:
                raise ArchiveError(
                    f"pCloud did not return exactly one directory named {part!r}"
                )
            current_id = self._folder_id_from_metadata(matches[0])
        return current_id

    def doctor(self) -> PCloudDoctorResult:
        binary = self.settings.rclone_binary
        if shutil.which(binary) is None and not Path(binary).is_file():
            raise ArchiveError(f"rclone executable not found: {binary}")
        if (
            self.settings.rclone_config is not None
            and not self.settings.rclone_config.is_file()
        ):
            raise ArchiveError(f"rclone config not found: {self.settings.rclone_config}")

        version_result = self._run("version")
        assert version_result is not None
        version = version_result.stdout.splitlines()[0].strip()
        remotes_result = self._run("listremotes")
        assert remotes_result is not None
        expected = f"{self.settings.pcloud_remote}:"
        if expected not in remotes_result.stdout.splitlines():
            raise ArchiveError(f"rclone remote is not configured: {expected}")

        redacted = self._run("config", "redacted", self.settings.pcloud_remote)
        assert redacted is not None
        section = self._parse_remote_config(
            redacted.stdout,
            self.settings.pcloud_remote,
        )
        if section.get("type") != "pcloud":
            raise ArchiveError(f"rclone remote {expected} is not a pCloud backend")
        hostname = section.get("hostname", "api.pcloud.com").strip()
        regions = {"api.pcloud.com": "US", "eapi.pcloud.com": "EU"}
        if hostname not in regions:
            raise ArchiveError(f"Unexpected pCloud API hostname: {hostname!r}")
        region = regions[hostname]
        self._run("lsd", self._remote())
        account = self._pcloud_request("userinfo", {})
        try:
            account_result = int(account["result"])
            account_id = str(account["userid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveError("Invalid pCloud userinfo response") from exc
        if account_result != 0 or not account_id.isdigit():
            raise ArchiveError("pCloud userinfo did not return a valid account ID")
        root_folder_id = self._root_directory_id()
        return PCloudDoctorResult(
            rclone_version=version,
            remote=self.settings.pcloud_remote,
            hostname=hostname,
            region=region,
            account_id=account_id,
            root_folder_id=root_folder_id,
        )

    def bootstrap(self, *, dry_run: bool) -> list[str]:
        created: list[str] = []
        for relative in ARCHIVE_ROOTS:
            remote_path = _remote_path(self.settings, relative)
            arguments = ["mkdir", self._remote(remote_path)]
            if dry_run:
                arguments.append("--dry-run")
            self._run(*arguments)
            created.append(remote_path)
        return created

    def stat(self, remote_path: str) -> RemoteStat | None:
        result = self._run(
            "lsjson",
            self._remote(remote_path),
            "--stat",
            "--hash-type",
            "SHA-1",
            missing_ok=True,
        )
        if result is None:
            return None
        try:
            payload = json.loads(result.stdout)
            hashes = payload.get("Hashes") or {}
            sha1 = str(hashes.get("SHA-1") or hashes.get("sha1") or "").lower()
            size = int(payload["Size"])
            file_id = str(payload["ID"]) if payload.get("ID") is not None else None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArchiveError(f"Invalid rclone metadata for {remote_path}") from exc
        if not sha1:
            raise ArchiveError(f"pCloud did not return a SHA-1 hash for {remote_path}")
        return RemoteStat(size=size, sha1=sha1, file_id=file_id)

    def copy_to(self, source: Path, remote_path: str) -> None:
        arguments = [
            "copyto",
            str(source),
            self._remote(remote_path),
            "--ignore-existing",
            "--checksum",
            "--progress",
        ]
        if self.settings.pcloud_bwlimit:
            arguments.extend(("--bwlimit", self.settings.pcloud_bwlimit))
        self._run(*arguments, live=True)

    def finalize_no_overwrite(
        self,
        source_remote_path: str,
        destination_remote_path: str,
    ) -> None:
        source = self.stat(source_remote_path)
        if source is None or source.file_id is None:
            raise ArchiveError(f"Staging file ID is unavailable: {source_remote_path}")
        if re.fullmatch(r"f\d+", source.file_id) is None:
            raise ArchiveError(f"Invalid pCloud staging file ID: {source.file_id!r}")

        destination = PurePosixPath(destination_remote_path)
        destination_name = destination.name
        if not destination_name or destination_name in {".", ".."}:
            raise ArchiveError(f"Invalid final pCloud path: {destination_remote_path}")
        destination_folder_id = self._directory_id(destination.parent.as_posix())
        payload = self._pcloud_request(
            "copyfile",
            {
                "fileid": source.file_id[1:],
                "tofolderid": destination_folder_id,
                "toname": destination_name,
                "noover": "1",
            },
        )
        try:
            result = int(payload["result"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveError("Invalid pCloud copyfile response") from exc
        if result not in {0, 2004}:
            detail = str(payload.get("error") or "unknown pCloud API error")
            raise ArchiveError(f"pCloud copyfile failed ({result}): {detail[:500]}")

    def delete_staging(self, remote_path: str) -> None:
        # pCloud may make a staging object disappear immediately after the
        # provider-side copy. Cleanup is intentionally idempotent: an object
        # that is already absent is the same successful end state.
        self._run("deletefile", self._remote(remote_path), missing_ok=True)


class PCloudArchiver:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        backend: ArchiveBackend,
        *,
        remote_region: str | None = None,
        remote_hostname: str | None = None,
        remote_account_id: str | None = None,
        remote_root_folder_id: str | None = None,
    ):
        self.settings = settings
        self.database = database
        self.backend = backend
        self.remote_region = remote_region
        self.remote_hostname = remote_hostname
        self.remote_account_id = remote_account_id
        self.remote_root_folder_id = remote_root_folder_id

    def validate_target(self, record: StorageObjectRecord) -> None:
        if record.provider != ARCHIVE_PROVIDER:
            raise StateConflict(f"Unsupported archive provider: {record.provider}")
        if record.remote_name != self.settings.pcloud_remote:
            raise StateConflict(
                "Configured rclone remote changed for an existing archive object"
            )
        expected_root = PurePosixPath(
            self.settings.pcloud_root,
            record.naming_version,
        )
        for label, value in (
            ("video", record.remote_path),
            ("manifest", record.manifest_remote_path),
        ):
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or ".." in path.parts
                or ":" in value
                or not path.is_relative_to(expected_root)
            ):
                raise StateConflict(
                    "Configured pCloud archive root changed for an existing "
                    f"{label} object"
                )
        comparisons = (
            ("region", record.remote_region, self.remote_region),
            ("API hostname", record.remote_hostname, self.remote_hostname),
            ("account", record.remote_account_id, self.remote_account_id),
            (
                "remote root",
                record.remote_root_folder_id,
                self.remote_root_folder_id,
            ),
        )
        for label, existing, configured in comparisons:
            if existing is not None and existing != configured:
                raise StateConflict(
                    f"Configured pCloud {label} changed for an existing archive object"
                )

    def validate_catalog_target(self) -> None:
        for record in self.database.list_storage_objects(provider=ARCHIVE_PROVIDER):
            self.validate_target(record)

    def register(self, candidate: ArchiveCandidate) -> StorageObjectRecord:
        self.validate_catalog_target()
        record = self.database.ensure_storage_object(
            media_kind=candidate.media_kind,
            owner_type=candidate.owner_type,
            owner_id=candidate.owner_id,
            source_name=candidate.source_name,
            local_relative_path=candidate.local_relative_path,
            provider=ARCHIVE_PROVIDER,
            remote_name=self.settings.pcloud_remote,
            naming_version=ARCHIVE_NAMING_VERSION,
            remote_path=candidate.remote_path,
            manifest_remote_path=candidate.manifest_remote_path,
            byte_size=candidate.byte_size,
        )
        self.validate_target(record)
        return record

    @staticmethod
    def _validate_content_identity(
        record: StorageObjectRecord,
        *,
        byte_size: int,
        sha1: str,
        sha256: str,
        manifest_sha1: str,
        manifest_sha256: str,
        manifest_byte_size: int,
    ) -> None:
        if record.attempts == 0:
            return
        stored = (
            record.byte_size,
            record.local_sha1,
            record.local_sha256,
            record.manifest_sha1,
            record.manifest_sha256,
            record.manifest_byte_size,
        )
        current = (
            byte_size,
            sha1,
            sha256,
            manifest_sha1,
            manifest_sha256,
            manifest_byte_size,
        )
        if None in stored or stored != current:
            raise StateConflict(
                "Archive content identity changed after transfer started; "
                "the existing remote object and catalog were not modified"
            )

    @staticmethod
    def _verify(stat: RemoteStat, *, size: int, sha1: str, label: str) -> None:
        if stat.size != size or stat.sha1.lower() != sha1.lower():
            raise ArchiveError(
                f"{label} verification failed: expected {size} bytes/{sha1}, "
                f"got {stat.size} bytes/{stat.sha1}"
            )

    def _copy_with_staging(
        self,
        *,
        source: Path,
        final_path: str,
        staging_path: str,
        size: int,
        sha1: str,
    ) -> RemoteStat:
        final_stat = self.backend.stat(final_path)
        if final_stat is not None:
            self._verify(final_stat, size=size, sha1=sha1, label=final_path)
            self._remove_matching_staging(
                staging_path=staging_path,
                size=size,
                sha1=sha1,
            )
            return final_stat

        staging_stat = self.backend.stat(staging_path)
        if staging_stat is None:
            self.backend.copy_to(source, staging_path)
            staging_stat = self.backend.stat(staging_path)
            if staging_stat is None:
                raise ArchiveError(f"Uploaded staging object is missing: {staging_path}")
        self._verify(staging_stat, size=size, sha1=sha1, label=staging_path)
        self.backend.finalize_no_overwrite(staging_path, final_path)
        final_stat = self.backend.stat(final_path)
        if final_stat is None:
            raise ArchiveError(f"Final archive object is missing: {final_path}")
        self._verify(final_stat, size=size, sha1=sha1, label=final_path)
        self._remove_matching_staging(
            staging_path=staging_path,
            size=size,
            sha1=sha1,
        )
        return final_stat

    def _remove_matching_staging(
        self,
        *,
        staging_path: str,
        size: int,
        sha1: str,
    ) -> None:
        staging_stat = self.backend.stat(staging_path)
        if staging_stat is None:
            return
        self._verify(staging_stat, size=size, sha1=sha1, label=staging_path)
        self.backend.delete_staging(staging_path)
        if self.backend.stat(staging_path) is not None:
            raise ArchiveError(f"Verified staging object could not be removed: {staging_path}")

    def verify(
        self,
        candidate: ArchiveCandidate,
        record: StorageObjectRecord | None = None,
    ) -> StorageObjectRecord:
        registered = self.register(candidate)
        if record is not None and record.id != registered.id:
            raise StateConflict("Archive candidate no longer matches its catalog record")
        return self.verify_record(registered)

    def verify_record(self, record: StorageObjectRecord) -> StorageObjectRecord:
        self.validate_target(record)
        if record.archive_state not in {"verified", "failed"} or record.verified_at is None:
            raise StateConflict("Archive object has never completed remote verification")
        if (
            record.local_sha1 is None
            or record.manifest_sha1 is None
            or record.manifest_byte_size is None
        ):
            raise StateConflict("Verified archive object is missing checksum metadata")
        try:
            video = self.backend.stat(record.remote_path)
            manifest = self.backend.stat(record.manifest_remote_path)
            if video is None or manifest is None:
                raise ArchiveError("Verified pCloud archive is missing video or manifest")
            self._verify(
                video,
                size=record.byte_size,
                sha1=record.local_sha1,
                label=record.remote_path,
            )
            self._verify(
                manifest,
                size=record.manifest_byte_size,
                sha1=record.manifest_sha1,
                label=record.manifest_remote_path,
            )
            relative = _safe_catalog_path(
                record.local_relative_path,
                "stored local archive path",
            )
            local_path = self.settings.data_dir / relative
            resolved_data = self.settings.data_dir.resolve()
            if not local_path.resolve().is_relative_to(resolved_data):
                raise ArchiveError("Stored local archive path escapes the data directory")
            if local_path.is_file():
                self.database.mark_storage_present(record.id)
            else:
                self.database.mark_storage_missing(
                    record.id,
                    f"Local source is missing: {record.local_relative_path}",
                )
            return self.database.finish_storage_check(record.id)
        except Exception as exc:
            self.database.fail_storage_check(record.id, str(exc))
            raise

    def archive(self, candidate: ArchiveCandidate) -> StorageObjectRecord:
        record = self.register(candidate)
        if record.archive_state == "verified":
            return self.verify(candidate, record)
        if not candidate.local_path.is_file():
            self.database.mark_storage_missing(
                record.id,
                f"Local source is missing: {candidate.local_relative_path}",
            )
            raise ArchiveError(f"Local source is missing: {candidate.local_path}")

        try:
            byte_size, sha1, sha256 = hash_file(candidate.local_path)
            if byte_size <= 0:
                raise ArchiveError(f"Archive source is empty: {candidate.local_path}")
            if candidate.byte_size > 0 and byte_size != candidate.byte_size:
                raise ArchiveError(
                    "Archive source size changed since it was cataloged: "
                    f"expected {candidate.byte_size}, got {byte_size}"
                )
            manifest = candidate.manifest_bytes(
                byte_size=byte_size,
                sha1=sha1,
                sha256=sha256,
                remote_name=self.settings.pcloud_remote,
            )
            manifest_sha1 = hashlib.sha1(manifest).hexdigest()
            manifest_sha256 = hashlib.sha256(manifest).hexdigest()
            self._validate_content_identity(
                record,
                byte_size=byte_size,
                sha1=sha1,
                sha256=sha256,
                manifest_sha1=manifest_sha1,
                manifest_sha256=manifest_sha256,
                manifest_byte_size=len(manifest),
            )
            record = self.database.start_storage_upload(
                record.id,
                local_sha1=sha1,
                local_sha256=sha256,
                byte_size=byte_size,
                manifest_sha1=manifest_sha1,
                manifest_byte_size=len(manifest),
                manifest_sha256=manifest_sha256,
                remote_region=self.remote_region,
                remote_hostname=self.remote_hostname,
                remote_account_id=self.remote_account_id,
                remote_root_folder_id=self.remote_root_folder_id,
            )
            staging_root = _remote_path(
                self.settings,
                ARCHIVE_NAMING_VERSION,
                "_staging",
                record.id,
            )
            video_staging = f"{staging_root}/payload{candidate.local_path.suffix.lower()}"
            manifest_staging = f"{staging_root}/manifest.json"
            final_stat = self._copy_with_staging(
                source=candidate.local_path,
                final_path=candidate.remote_path,
                staging_path=video_staging,
                size=byte_size,
                sha1=sha1,
            )

            manifest_dir = self.settings.work_dir / "archive-manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=manifest_dir, prefix=f"{record.id}-") as work:
                manifest_path = Path(work) / "manifest.json"
                manifest_path.write_bytes(manifest)
                self._copy_with_staging(
                    source=manifest_path,
                    final_path=candidate.manifest_remote_path,
                    staging_path=manifest_staging,
                    size=len(manifest),
                    sha1=manifest_sha1,
                )

            self.database.mark_storage_verifying(record.id)
            verified_video = self.backend.stat(candidate.remote_path)
            verified_manifest = self.backend.stat(candidate.manifest_remote_path)
            if verified_video is None or verified_manifest is None:
                raise ArchiveError("Final pCloud verification could not find both objects")
            self._verify(verified_video, size=byte_size, sha1=sha1, label="video")
            self._verify(
                verified_manifest,
                size=len(manifest),
                sha1=manifest_sha1,
                label="manifest",
            )
            return self.database.finish_storage_verification(
                record.id,
                remote_file_id=final_stat.file_id,
                remote_hash_algorithm="sha1",
                remote_hash=final_stat.sha1,
                remote_region=self.remote_region,
                remote_hostname=self.remote_hostname,
                remote_account_id=self.remote_account_id,
                remote_root_folder_id=self.remote_root_folder_id,
            )
        except Exception as exc:
            self.database.fail_storage_object(record.id, str(exc))
            raise
