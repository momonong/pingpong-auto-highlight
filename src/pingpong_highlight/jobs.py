from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.media_work import media_work_lock
from pingpong_highlight.pipeline.processor import HighlightProcessor


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
        self.database.requeue_interrupted_jobs()
        for job in self.database.list_queued_jobs():
            self.enqueue(job.id)

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._active:
                return
            self._active.add(job_id)
        future = self._executor.submit(self._run, job_id)
        future.add_done_callback(lambda completed: self._finished(job_id, completed))

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
        try:
            with media_work_lock(self.settings.data_dir):
                result = self.processor.run(
                    upload.path,
                    output_dir,
                    progress=lambda value, stage: self.database.update_job(
                        job_id,
                        value,
                        stage,
                    ),
                    source_name=upload.filename,
                )
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.database.fail_job(job_id, detail)
            return
        self.database.finish_job(job_id, result)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
