from __future__ import annotations

import mimetypes
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlsplit

import gdown
from gdown.exceptions import DownloadError, FileURLRetrievalError

from pingpong_highlight.cleanup import FilesystemCleanup
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database, DriveImportRecord, StateConflict
from pingpong_highlight.uploads import ALLOWED_SUFFIXES, UploadError, UploadStore, clean_filename

ALLOWED_DRIVE_HOSTS = {
    "drive.google.com",
    "www.drive.google.com",
    "drive.usercontent.google.com",
}
FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,200}$")
RESOURCE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,300}$")
FILE_PATH_PATTERN = re.compile(r"^/file/(?:u/\d+/)?d/([^/]+)(?:/|$)")


class DriveImportError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class DriveLinkError(DriveImportError):
    def __init__(self, detail: str):
        super().__init__(400, detail)


class DriveImportPolicyError(RuntimeError):
    pass


class DriveImportCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DriveLink:
    file_id: str
    resource_key: str | None


def parse_drive_link(value: str) -> DriveLink:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise DriveLinkError("Google Drive 網址格式不正確") from exc

    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or hostname not in ALLOWED_DRIVE_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise DriveLinkError("只接受 https://drive.google.com 的影片檔案連結")

    path = unquote(parsed.path)
    if "/folders/" in path:
        raise DriveLinkError("請貼單一影片的連結，不是 Google Drive 資料夾連結")

    query = parse_qs(parsed.query)
    match = FILE_PATH_PATTERN.match(path)
    file_id = match.group(1) if match else (query.get("id") or [""])[0]
    if not FILE_ID_PATTERN.fullmatch(file_id):
        raise DriveLinkError("找不到 Google Drive 檔案 ID，請使用影片的共用連結")

    resource_key = (query.get("resourcekey") or [None])[0]
    if resource_key is not None and not RESOURCE_KEY_PATTERN.fullmatch(resource_key):
        raise DriveLinkError("Google Drive resource key 格式不正確")
    return DriveLink(file_id=file_id, resource_key=resource_key)


class DriveDownloader(Protocol):
    def resolve(self, link: DriveLink) -> str: ...

    def download(
        self,
        link: DriveLink,
        output: Path,
        progress: Callable[[int, int | None], None],
    ) -> Path: ...


class GDownDriveDownloader:
    @staticmethod
    def _url(link: DriveLink) -> str:
        url = f"https://drive.google.com/uc?id={link.file_id}"
        if link.resource_key:
            url += f"&resourcekey={link.resource_key}"
        return url

    def resolve(self, link: DriveLink) -> str:
        resolved = gdown.download(
            url=self._url(link),
            quiet=True,
            use_cookies=False,
            skip_download=True,
        )
        path = getattr(resolved, "path", None)
        if not path:
            raise FileURLRetrievalError("Google Drive did not return a filename")
        return str(path)

    def download(
        self,
        link: DriveLink,
        output: Path,
        progress: Callable[[int, int | None], None],
    ) -> Path:
        result = gdown.download(
            url=self._url(link),
            output=str(output),
            quiet=True,
            use_cookies=False,
            resume=True,
            progress=progress,
        )
        if not isinstance(result, str):
            raise DownloadError("Google Drive download did not return a file path")
        return Path(result)


class DriveImportManager:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        uploads: UploadStore,
        enqueue_job: Callable[[str], None],
        downloader: DriveDownloader | None = None,
        cleanup: FilesystemCleanup | None = None,
    ):
        self.settings = settings
        self.database = database
        self.uploads = uploads
        self.enqueue_job = enqueue_job
        self.downloader = downloader or GDownDriveDownloader()
        self.cleanup = cleanup or FilesystemCleanup(settings, database)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="drive-import",
        )
        self._active: set[str] = set()
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

    def start(self) -> None:
        self.database.requeue_interrupted_drive_imports()
        for record in self.database.list_drive_imports():
            if record.status == "queued":
                self._enqueue(record.id)

    def submit(self, url: str, *, user_id: str | None = None) -> DriveImportRecord:
        link = parse_drive_link(url)
        record = self.database.create_or_requeue_drive_import(
            link.file_id,
            link.resource_key,
            user_id=user_id,
        )
        if record.status == "queued":
            self._enqueue(record.id)
        return record

    def retry(
        self,
        import_id: str,
        *,
        user_id: str | None = None,
    ) -> DriveImportRecord:
        record = self.database.get_drive_import(import_id, user_id=user_id)
        if record is None:
            raise DriveImportError(404, "找不到這筆 Google Drive 匯入")
        if not self.database.retry_drive_import(import_id, user_id=user_id):
            raise DriveImportError(409, "只有失敗的 Google Drive 匯入可以重試")
        self._enqueue(import_id)
        updated = self.database.get_drive_import(import_id, user_id=user_id)
        assert updated is not None
        return updated

    def delete(
        self,
        import_id: str,
        *,
        user_id: str | None = None,
    ) -> None:
        record = self.database.get_drive_import(import_id, user_id=user_id)
        if record is None:
            raise DriveImportError(404, "找不到這筆 Google Drive 匯入")
        targets = [
            (path, "file")
            for path in self.settings.drive_imports_dir.iterdir()
            if path.name.startswith(record.id) and (path.is_file() or path.is_symlink())
        ]
        if not self.database.delete_drive_import(
            import_id,
            user_id=user_id,
            cleanup_targets=targets,
        ):
            raise DriveImportError(409, "下載中的 Google Drive 匯入不能刪除")
        self.cleanup.drain()

    def _enqueue(self, import_id: str) -> None:
        with self._lock:
            if self._shutdown.is_set() or import_id in self._active:
                return
            self._active.add(import_id)
        future = self._executor.submit(self._run, import_id)
        future.add_done_callback(lambda completed: self._finished(import_id, completed))

    def _finished(self, import_id: str, _future: Future[None]) -> None:
        with self._lock:
            self._active.discard(import_id)

    def _current_offset(self, output: Path) -> int:
        if output.is_file():
            return output.stat().st_size
        parts = [path for path in output.parent.glob(f"{output.name}*.part") if path.is_file()]
        return parts[0].stat().st_size if len(parts) == 1 else 0

    def _check_policy(self, offset: int, total: int | None) -> None:
        if self._shutdown.is_set():
            raise DriveImportCancelled
        if total is not None and total > self.settings.max_upload_bytes:
            raise DriveImportPolicyError("影片超過目前設定的大小上限")
        if offset > self.settings.max_upload_bytes:
            raise DriveImportPolicyError("影片超過目前設定的大小上限")

        free = shutil.disk_usage(self.settings.drive_imports_dir).free
        required = self.settings.download_min_free_bytes
        if total is not None:
            required += max(0, total - offset)
        if free < required:
            raise DriveImportPolicyError("電腦可用空間不足，已保留目前下載進度")

    def _run(self, import_id: str) -> None:
        if not self.database.claim_drive_import(import_id):
            return
        record = self.database.get_drive_import(import_id)
        if record is None:
            return
        link = DriveLink(record.file_id, record.resource_key)

        try:
            if self._shutdown.is_set():
                raise DriveImportCancelled
            filename = clean_filename(self.downloader.resolve(link))
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise DriveImportPolicyError(
                    "這個 Google Drive 檔案不是支援的影片格式（MOV、MP4、M4V、MKV、AVI、WEBM）"
                )

            self.database.start_drive_import_download(import_id, filename)
            output = self.settings.drive_imports_dir / f"{import_id}{suffix}"
            initial_offset = self._current_offset(output)
            self._check_policy(initial_offset, None)
            self.database.update_drive_import_progress(import_id, initial_offset, None)
            last_saved_at = 0.0
            last_saved_offset = initial_offset

            def progress(offset: int, total: int | None) -> None:
                nonlocal last_saved_at, last_saved_offset
                self._check_policy(offset, total)
                now = time.monotonic()
                if (
                    now - last_saved_at >= 0.5
                    or offset - last_saved_offset >= 8 * 1024**2
                    or (total is not None and offset >= total)
                ):
                    self.database.update_drive_import_progress(import_id, offset, total)
                    last_saved_at = now
                    last_saved_offset = offset

            downloaded = self.downloader.download(link, output, progress)
            size = downloaded.stat().st_size
            self._check_policy(size, size)
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            _upload, job = self.uploads.import_completed_file(
                filename,
                content_type,
                downloaded,
                drive_import_id=import_id,
                user_id=record.user_id,
            )
            self.enqueue_job(job.id)
        except DriveImportCancelled:
            return
        except (DriveImportPolicyError, UploadError) as exc:
            detail = exc.detail if isinstance(exc, UploadError) else str(exc)
            self.database.fail_drive_import(import_id, detail)
        except (FileURLRetrievalError, DownloadError):
            self.database.fail_drive_import(
                import_id,
                "Google Drive 無法下載這個檔案。請確認共用設定為"
                "「知道連結的任何人可檢視」，且檔案允許下載。",
            )
        except (OSError, StateConflict):
            self.database.fail_drive_import(
                import_id,
                "匯入影片時發生本機儲存錯誤，目前進度已保留，可稍後重試。",
            )
        except Exception:
            self.database.fail_drive_import(
                import_id,
                "Google Drive 下載中斷，目前進度已保留，請按重試繼續。",
            )

    def shutdown(self) -> None:
        self._shutdown.set()
        self._executor.shutdown(wait=True, cancel_futures=False)
