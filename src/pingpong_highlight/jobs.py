from __future__ import annotations

import logging
import shutil
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database, StateConflict
from pingpong_highlight.pipeline.processor import HighlightProcessor

LOGGER = logging.getLogger(__name__)


class JobManager:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        processor: HighlightProcessor | None = None,
    ):
        self.settings = settings
        self.database = database
        self.processor = processor or HighlightProcessor(settings)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_count,
            thread_name_prefix="highlight-worker",
        )
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        self._restore_interrupted_outputs()
        self.database.requeue_interrupted_jobs()
        for job in self.database.list_queued_jobs():
            self.enqueue(job.id)

    def _restore_interrupted_outputs(self) -> None:
        """Restore the last committed output if a swap stopped between renames."""

        offset = 0
        while True:
            records = self.database.list_jobs(500, offset=offset, status="processing")
            if not records:
                return
            for job in records:
                if len(job.id) != 32 or any(
                    character not in "0123456789abcdef" for character in job.id
                ):
                    continue
                output_dir = self.settings.outputs_dir / job.id
                previous_dir = self.settings.work_dir / f".{job.id}.previous"
                if previous_dir.is_symlink():
                    raise StateConflict("Previous output recovery path is a symbolic link")
                if not previous_dir.is_dir():
                    continue
                if output_dir.is_symlink():
                    raise StateConflict("Output recovery path is a symbolic link")
                if output_dir.exists():
                    if not output_dir.is_dir():
                        raise StateConflict("Output recovery path is not a directory")
                    shutil.rmtree(output_dir)
                previous_dir.replace(output_dir)
            offset += len(records)

    def _prepare_previous_for_queue(
        self,
        job_id: str,
        *,
        user_id: str | None,
        allowed_statuses: set[str],
    ) -> None:
        """Ensure a queued run starts without an ambiguous previous generation."""

        job = self.database.get_job(job_id, user_id=user_id)
        if job is None or job.status not in allowed_statuses:
            return
        output_dir = self.settings.outputs_dir / job_id
        previous_dir = self.settings.work_dir / f".{job_id}.previous"
        if previous_dir.is_symlink():
            raise StateConflict("Previous output recovery path is a symbolic link")
        if not previous_dir.exists():
            return
        if not previous_dir.is_dir():
            raise StateConflict("Previous output recovery path is not a directory")
        if output_dir.is_symlink():
            raise StateConflict("Output recovery path is a symbolic link")
        if output_dir.exists():
            if not output_dir.is_dir():
                raise StateConflict("Output recovery path is not a directory")
            # The database is not processing, so output_dir is the committed
            # generation and an older .previous directory is obsolete.
            shutil.rmtree(previous_dir)
        else:
            previous_dir.replace(output_dir)

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._active:
                return
            self._active.add(job_id)
        future = self._executor.submit(self._run, job_id)
        future.add_done_callback(lambda completed: self._finished(job_id, completed))

    def _clear_staging(self, job_id: str) -> None:
        directory = self.settings.work_dir / job_id
        if directory.exists():
            shutil.rmtree(directory)

    def retry(self, job_id: str, *, user_id: str | None = None) -> bool:
        self._prepare_previous_for_queue(
            job_id,
            user_id=user_id,
            allowed_statuses={"failed"},
        )
        if not self.database.retry_job(job_id, user_id=user_id):
            return False
        try:
            self._clear_staging(job_id)
        except OSError:
            self.database.fail_job(job_id, "Could not clear staging files before retry")
            raise
        self.enqueue(job_id)
        return True

    def reprocess(self, job_id: str, *, user_id: str | None = None) -> bool:
        self._prepare_previous_for_queue(
            job_id,
            user_id=user_id,
            allowed_statuses={"completed", "failed"},
        )
        if not self.database.reprocess_job(job_id, user_id=user_id):
            return False
        try:
            self._clear_staging(job_id)
        except OSError:
            self.database.fail_job(job_id, "Could not clear staging files before reprocessing")
            raise
        self.enqueue(job_id)
        return True

    def _finished(self, job_id: str, _future: Future[None]) -> None:
        with self._lock:
            self._active.discard(job_id)

    def _run(self, job_id: str) -> None:
        if not self.database.claim_job(job_id):
            return
        job = self.database.get_job(job_id)
        if job is None:
            return
        upload = self.database.get_upload(job.upload_id)
        if upload is None:
            self.database.fail_job(job_id, "The uploaded video record is missing")
            return

        output_dir = self.settings.outputs_dir / job_id
        staging_dir = self.settings.work_dir / job_id
        previous_dir = self.settings.work_dir / f".{job_id}.previous"
        try:
            self._clear_staging(job_id)
            result = self.processor.run(
                upload.path,
                staging_dir,
                progress=lambda value, stage: self.database.update_job(job_id, value, stage),
                source_name=upload.filename,
            )
            if previous_dir.exists():
                shutil.rmtree(previous_dir)
            had_previous = output_dir.exists()
            if had_previous:
                output_dir.replace(previous_dir)
            try:
                staging_dir.replace(output_dir)
            except Exception:
                if had_previous and previous_dir.exists():
                    previous_dir.replace(output_dir)
                raise
            try:
                self.database.finish_job(job_id, result)
            except Exception:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                if had_previous and previous_dir.exists():
                    previous_dir.replace(output_dir)
                raise
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.database.fail_job(job_id, detail)
            return
        if previous_dir.exists():
            try:
                shutil.rmtree(previous_dir)
            except Exception as exc:
                # The output and database result are already committed. Startup
                # reconciliation can safely retry without changing job status.
                LOGGER.warning("Could not remove previous output for job %s: %s", job_id, exc)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
