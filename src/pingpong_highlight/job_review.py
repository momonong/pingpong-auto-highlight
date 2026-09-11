"""Authenticated adapter for the existing desktop annotation entry point."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading

from fastapi import HTTPException, Request

from pingpong_highlight.rally_review import (
    Conflict,
    Proposal,
    ReviewCommand,
    ReviewStore,
    source_identity,
)
from pingpong_highlight.review_media import full_preview, prepare_review
from pingpong_highlight.web import AuthenticatedUser


def install(app, settings, job_and_upload):
    identities = {}
    preparing = threading.Lock()

    def payload(store, sid):
        value = store.export(sid, blind=True)
        value["full_preview_available"] = full_preview(store, sid) is not None
        return value

    def context(job_id, user):
        job, upload = job_and_upload(job_id, user)
        path = upload.path.resolve()
        if not path.is_relative_to(settings.uploads_dir.resolve()) or not path.is_file():
            raise HTTPException(404, "Source unavailable")
        duration = (job.result or {}).get("media", {}).get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise HTTPException(409, "Source duration unavailable")
        namespace = hashlib.sha256((upload.user_id or "legacy").encode()).hexdigest()
        store = ReviewStore(settings.data_dir / "rally-review" / f"{namespace}.sqlite3")
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = identities.get(upload.id)
        if cached and cached[0] == signature:
            return store, cached[1], job
        source = source_identity(path)
        source.update(
            duration_ms=round(duration * 1000),
            group="development",
            session_id="UNKNOWN",
            scope=[{"start_ms": 0, "end_ms": round(duration * 1000)}],
            blind_intervals=[{"start_ms": 0, "end_ms": min(30000, round(duration * 1000))}],
        )
        if not any(s["id"] == source["id"] for s in store.sources()):
            store.register(source)
        identities[upload.id] = (signature, source["id"])
        return store, source["id"], job

    @app.get("/api/jobs/{job_id}/rally-review")
    def get_review(job_id: str, user: AuthenticatedUser):
        store, sid, _ = context(job_id, user)
        return payload(store, sid)

    @app.get("/api/jobs/{job_id}/rally-review/export")
    def export_review(job_id: str, user: AuthenticatedUser):
        store, sid, _ = context(job_id, user)
        return store.export(sid, blind=True)

    @app.post("/api/jobs/{job_id}/rally-review")
    def update(job_id: str, command: ReviewCommand, user: AuthenticatedUser):
        store, sid, _ = context(job_id, user)
        try:
            store.apply(sid, command, actor=user.id)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return payload(store, sid)

    @app.get("/api/jobs/{job_id}/rally-review/full-preview")
    def full_video(job_id: str, request: Request, user: AuthenticatedUser):
        from pingpong_highlight.web import MediaFileResponse

        store, sid, _ = context(job_id, user)
        spec = full_preview(store, sid)
        if spec is None:
            raise HTTPException(404, "Whole-video preview not prepared")
        from pathlib import Path

        return MediaFileResponse(Path(spec["path"]), request, media_type="video/mp4")

    @app.post("/api/jobs/{job_id}/rally-review/full-preview")
    def prepare_video(job_id: str, user: AuthenticatedUser):
        store, sid, _ = context(job_id, user)
        if not preparing.acquire(blocking=False):
            raise HTTPException(409, "另一支影片正在準備，請稍後重試")
        try:
            prepare_review(store, sid)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            preparing.release()
        return payload(store, sid)

    @app.post("/api/jobs/{job_id}/rally-review/baseline")
    def baseline(job_id: str, user: AuthenticatedUser):
        store, sid, job = context(job_id, user)
        result = job.result or {}
        encoded = json.dumps(result, sort_keys=True).encode()
        run_id = hashlib.sha256(sid.encode() + encoded).hexdigest()
        duration = store.source(sid)["duration_ms"]
        rows = []
        for i, p in enumerate(result.get("candidates", [])):
            a, b = round(p["rally_start"] * 1000), round(p["rally_end"] * 1000)
            selected = next(
                (
                    x
                    for x in result.get("points", [])
                    if round(x["rally_start"] * 1000) == a and round(x["rally_end"] * 1000) == b
                ),
                None,
            )
            rows.append(
                Proposal(
                    id=f"{run_id}-{i}",
                    start_ms=a,
                    end_ms=b,
                    clip_start_ms=round(selected["start"] * 1000) if selected else a,
                    clip_end_ms=min(duration, round(selected["end"] * 1000)) if selected else b,
                    selected=selected is not None,
                    reason=p.get("reason", ""),
                    evidence=["Existing analysis; original commit/configuration may be unknown"],
                ).model_dump()
            )
        store.add_run(
            {
                "id": run_id,
                "source_id": sid,
                "commit": "UNKNOWN-existing-analysis",
                "code_sha256": "UNKNOWN",
                "model_id": result.get("algorithm_version", "baseline"),
                "model_revision": "UNKNOWN",
                "prompt": None,
                "parameters": result.get("selection", {}),
                "sampling": {
                    "provenance": "existing job result",
                    "result_sha256": hashlib.sha256(encoded).hexdigest(),
                },
                "elapsed_seconds": None,
                "windows": [{"start_ms": 0, "end_ms": duration}],
                "proposals": rows,
            }
        )
        return payload(store, sid)
