from __future__ import annotations

import logging
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from pingpong_highlight.config import Settings
from pingpong_highlight.db import CleanupRecord, Database

LOGGER = logging.getLogger(__name__)


class UnsafeCleanupPath(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CleanupResult:
    removed: int = 0
    failed: int = 0


class FilesystemCleanup:
    """Drain durable, explicit deletion targets without escaping managed data roots."""

    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self._roots = tuple(
            path.resolve()
            for path in (
                settings.uploads_dir,
                settings.outputs_dir,
                settings.work_dir,
                settings.drive_imports_dir,
            )
        )

    def drain(self) -> CleanupResult:
        """Best-effort cleanup; failed targets remain durable for a later retry."""

        removed = 0
        failed = 0
        try:
            records = self.database.list_cleanup_records()
        except Exception:
            LOGGER.exception("Could not read the filesystem cleanup queue")
            return CleanupResult(failed=1)

        for record in records:
            try:
                self._remove(record)
            except Exception as exc:
                failed += 1
                LOGGER.warning("Could not remove queued path %s: %s", record.path, exc)
                try:
                    self.database.record_cleanup_failure(record.id, str(exc))
                except Exception:
                    LOGGER.exception("Could not record a filesystem cleanup failure")
                continue

            try:
                self.database.complete_cleanup(record.id)
            except Exception:
                # Removing a missing target is idempotent, so retaining the row is safe.
                failed += 1
                LOGGER.exception("Could not acknowledge filesystem cleanup for %s", record.path)
                continue
            removed += 1
        return CleanupResult(removed=removed, failed=failed)

    def discard_obsolete_previous_artifacts(self) -> CleanupResult:
        """Remove old generations only while the current committed result is provably intact.

        This deliberately does not create a durable tombstone: the ``.previous``
        name is reused by later reprocess attempts, so a stale tombstone could
        otherwise delete a newer recovery copy.
        """

        removed = 0
        failed = 0
        try:
            candidates = list(self.settings.work_dir.iterdir())
        except FileNotFoundError:
            return CleanupResult()
        except OSError as exc:
            LOGGER.warning("Could not inspect previous output directories: %s", exc)
            return CleanupResult(failed=1)

        for candidate in candidates:
            name = candidate.name
            if not (name.startswith(".") and name.endswith(".previous")):
                continue
            job_id = name[1 : -len(".previous")]
            if len(job_id) != 32 or any(
                character not in "0123456789abcdef" for character in job_id
            ):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                continue

            job = self.database.get_job(job_id)
            if job is None or job.status != "completed" or not job.result:
                continue
            output_dir = self.settings.outputs_dir / job_id
            if output_dir.is_symlink() or not output_dir.is_dir():
                continue
            files = job.result.get("files")
            if not isinstance(files, list) or not files:
                continue
            try:
                output_root = output_dir.resolve()
            except OSError:
                failed += 1
                continue
            current_is_complete = True
            for item in files:
                filename = item.get("name") if isinstance(item, dict) else None
                if not isinstance(filename, str) or not filename:
                    current_is_complete = False
                    break
                current = output_dir / filename
                try:
                    if current.resolve().parent != output_root or not current.is_file():
                        current_is_complete = False
                        break
                except OSError:
                    current_is_complete = False
                    break
            if current_is_complete:
                try:
                    self._remove_target(candidate, "tree")
                except Exception as exc:
                    failed += 1
                    LOGGER.warning(
                        "Could not remove obsolete previous output %s: %s",
                        candidate,
                        exc,
                    )
                else:
                    removed += 1
        return CleanupResult(removed=removed, failed=failed)

    def _remove(self, record: CleanupRecord) -> None:
        self._remove_target(record.path, record.kind)

    def _remove_target(self, path: Path, kind: str) -> None:
        target = self._validated_target(path)
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            return

        if stat.S_ISLNK(mode):
            target.unlink()
            return
        if kind == "file":
            if not stat.S_ISREG(mode):
                raise UnsafeCleanupPath("queued file target is not a regular file")
            target.unlink()
            return
        if kind == "tree":
            if not stat.S_ISDIR(mode):
                raise UnsafeCleanupPath("queued tree target is not a directory")
            shutil.rmtree(target)
            return
        raise UnsafeCleanupPath("unknown cleanup target kind")

    def _validated_target(self, path: Path) -> Path:
        if not path.is_absolute() or path.name in {"", ".", ".."}:
            raise UnsafeCleanupPath("cleanup target must be an absolute child path")
        try:
            parent = path.parent.resolve(strict=False)
        except OSError as exc:
            raise UnsafeCleanupPath("cleanup target parent cannot be resolved") from exc

        for root in self._roots:
            try:
                relative_parent = parent.relative_to(root)
            except ValueError:
                continue
            target = parent / path.name
            if not relative_parent.parts and target == root:
                break
            return target
        raise UnsafeCleanupPath("cleanup target is outside managed data directories")
