"""Shared desktop workbench, usable from authenticated app or isolated offline experiments."""

from __future__ import annotations

import secrets
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from pingpong_highlight.deployment import html_page
from pingpong_highlight.rally_review import Conflict, ReviewCommand, ReviewStore
from pingpong_highlight.review_media import full_preview, prepare_review

ASSETS = Path(__file__).parent / "static" / "review"


def create_review_app(store: ReviewStore) -> FastAPI:
    # Loopback-only CLI; a random per-process key prevents foreign-origin mutation/read.
    app = FastAPI()
    key = secrets.token_urlsafe(32)
    preparing = threading.Lock()

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.url.hostname not in ("127.0.0.1", "localhost", "testserver"):
            return FileResponse(ASSETS / "denied.txt", status_code=403)
        if request.url.path.startswith("/api/") and request.cookies.get("hc_review") != key:
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "Open the local review page first"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        response = html_page(ASSETS / "index.html")
        response.set_cookie("hc_review", key, httponly=True, samesite="strict")
        return response

    @app.get("/static/paths.js")
    def paths():
        return FileResponse(ASSETS.parent / "paths.js")

    @app.get("/review.js")
    def script():
        return FileResponse(ASSETS / "review.js")

    @app.get("/review.css")
    def style():
        return FileResponse(ASSETS / "review.css")

    @app.get("/api/review/sources")
    def sources():
        return [
            {"id": s["id"], "name": s["name"], "duration_ms": s["duration_ms"]}
            for s in store.sources()
        ]

    def preview_spec(run):
        spec = run.get("preview")
        if spec is None:
            # Compatibility with early v1 receipts; never infer from a filename alone.
            cmd = run.get("sampling", {}).get("decode_command", [])
            if not cmd or "-ss" not in cmd or "-t" not in cmd:
                return None
            start = round(float(cmd[cmd.index("-ss") + 1]) * 1000)
            spec = {
                "path": cmd[-1],
                "start_ms": start,
                "end_ms": start + round(float(cmd[cmd.index("-t") + 1]) * 1000),
            }
        path = Path(spec["path"]).resolve()
        if not path.is_relative_to(store.path.resolve().parent) or not path.is_file():
            return None
        if path.suffix.casefold() != ".mp4":
            return None
        return spec | {"path": str(path)}

    def payload(source_id):
        data = store.export(source_id, blind=True)
        data["full_preview_available"] = full_preview(store, source_id) is not None
        for run in data["runs"]:
            spec = preview_spec(run)
            run["preview_available"] = spec is not None
            if spec:
                run["preview_range"] = {k: spec[k] for k in ("start_ms", "end_ms")}
        return data

    def require_source(source_id):
        try:
            return store.source(source_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown source") from exc

    @app.get("/api/review/{source_id}")
    def review(source_id: str):
        require_source(source_id)
        return payload(source_id)

    @app.post("/api/review/{source_id}")
    def update(source_id: str, command: ReviewCommand):
        require_source(source_id)
        try:
            store.apply(source_id, command, actor="local-reviewer")
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return payload(source_id)

    @app.get("/api/review/{source_id}/preview/{run_id}")
    def preview(source_id: str, run_id: str, request: Request):
        from pingpong_highlight.web import MediaFileResponse

        require_source(source_id)
        run = next((r for r in store.export(source_id)["runs"] if r["id"] == run_id), None)
        spec = preview_spec(run) if run else None
        if spec is None:
            raise HTTPException(404, "No compatible preview for this run")
        return MediaFileResponse(Path(spec["path"]), request, media_type="video/mp4")

    @app.get("/api/review/{source_id}/source")
    def video(source_id: str, request: Request):
        from pingpong_highlight.web import MediaFileResponse

        source = require_source(source_id)
        path = Path(source["path"])
        if not path.is_file() or path.stat().st_size != source["size"]:
            raise HTTPException(409, "Source unavailable or changed")
        return MediaFileResponse(path, request, media_type="video/mp4")

    @app.get("/api/review/{source_id}/full-preview")
    def full_video(source_id: str, request: Request):
        from pingpong_highlight.web import MediaFileResponse

        require_source(source_id)
        spec = full_preview(store, source_id)
        if spec is None:
            raise HTTPException(404, "Whole-video preview not prepared")
        return MediaFileResponse(Path(spec["path"]), request, media_type="video/mp4")

    @app.post("/api/review/{source_id}/full-preview")
    def prepare_video(source_id: str):
        require_source(source_id)
        if not preparing.acquire(blocking=False):
            raise HTTPException(409, "另一支影片正在準備，請稍後重試")
        try:
            prepare_review(store, source_id)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            preparing.release()
        return payload(source_id)

    @app.get("/api/review/{source_id}/export")
    def export(source_id: str):
        require_source(source_id)
        # Blind export keeps suggestions hidden until judged; CLI evaluator has full receipt.
        return store.export(source_id, blind=True)

    return app
