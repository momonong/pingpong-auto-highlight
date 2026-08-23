from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

from pingpong_highlight.config import Settings
from pingpong_highlight.db import CompilationRecord, Database
from pingpong_highlight.media_work import media_work_lock
from pingpong_highlight.pipeline.media import build_point_reel, probe_media

logger = logging.getLogger(__name__)


class CompilationManager:
    """Build user-selected, cross-source reels without a duration quota."""

    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="compilation-worker",
        )
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        self.database.requeue_interrupted_compilations()
        for compilation in self.database.list_queued_compilations():
            self.enqueue(compilation.id)

    def submit(self, *, name: str, highlight_ids: list[str]) -> CompilationRecord:
        compilation = self.database.create_compilation(
            name=name,
            highlight_ids=highlight_ids,
        )
        self.enqueue(compilation.id)
        return compilation

    def enqueue(self, compilation_id: str) -> None:
        with self._lock:
            if compilation_id in self._active:
                return
            self._active.add(compilation_id)
        future = self._executor.submit(self._run, compilation_id)
        future.add_done_callback(lambda completed: self._finished(compilation_id, completed))

    def _finished(self, compilation_id: str, future: Future[None]) -> None:
        with self._lock:
            self._active.discard(compilation_id)
        if error := future.exception():
            logger.error(
                "Compilation worker %s exited unexpectedly",
                compilation_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    def _run(self, compilation_id: str) -> None:
        try:
            if not self.database.claim_compilation(compilation_id):
                return
            highlights = self.database.list_compilation_highlights(compilation_id)
            if not highlights:
                raise RuntimeError("The compilation has no highlight clips")
            clips = []
            missing = []
            for highlight in highlights:
                job_dir = (self.settings.outputs_dir / highlight.job_id).resolve()
                clip = (job_dir / highlight.clip_filename).resolve()
                if not clip.is_relative_to(job_dir) or not clip.is_file():
                    missing.append(highlight.clip_filename)
                    continue
                clips.append(clip)
            if missing:
                raise RuntimeError(
                    "Some selected highlight files are missing: " + ", ".join(missing)
                )

            output_dir = self.settings.compilations_dir / compilation_id
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / "highlight_compilation.mp4"
            with media_work_lock(self.settings.data_dir):
                build_point_reel(clips, destination)
                duration = probe_media(destination).duration
            self.database.finish_compilation(
                compilation_id,
                file_name=destination.name,
                duration=duration,
            )
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.database.fail_compilation(compilation_id, detail)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
