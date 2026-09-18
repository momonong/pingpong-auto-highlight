"""Loopback review of an immutable edit pack, with separate append-only feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from pingpong_highlight.edit_comparison import sha

ASSETS = Path(__file__).parent / "static" / "edit-review"


class ReviewEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9-]{8,80}$")
    token: str = Field(max_length=40)
    field: Literal["complete", "rhythm", "note", "transition", "order", "presentation_order"]
    value: str | None = Field(default=None, max_length=2000)
    presentation_sequence: Literal["0", "1"]


def create_app(pack: Path, feedback: Path, *, fixture: bool = False) -> FastAPI:
    pack, feedback = pack.resolve(), feedback.resolve()
    packet = json.loads((pack / "packet.json").read_text())
    plan = json.loads((pack / "plan.json").read_text())
    if sha(pack / "plan.json") != packet["plan_sha256"]:
        raise ValueError("Plan checksum mismatch")
    # Never overwrite the previous pack's feedback or any input file.
    if feedback.is_relative_to(pack):
        raise ValueError("Feedback must be outside the immutable review pack")
    media = {}
    for pair in packet["pairs"]:
        for row in pair["media"]:
            media[row["token"]] = row
    for group, prefix in (("contexts", "context"), ("selected_clips", "point")):
        for i, row in enumerate(packet[group], 1):
            media[f"{prefix}-{i}"] = row
    media.update({r["token"]: r for r in packet["reels"]})
    for row in media.values():
        path = (pack / row["file"]).resolve()
        if path.parent != pack or path.suffix != ".mp4" or sha(path) != row["sha256"]:
            raise ValueError("Media path or checksum mismatch")
    binding = {
        "schema": "rally-edit-feedback-v2",
        "plan_sha256": packet["plan_sha256"],
        "source_sha256": plan["source_sha256"],
        "media": {token: row["sha256"] for token, row in media.items()},
        "actor": "automation-fixture" if fixture else "human-browser-input",
    }
    feedback.parent.mkdir(parents=True, exist_ok=True)
    if feedback.exists():
        state = json.loads(feedback.read_text())
        if any(state.get(k) != v for k, v in binding.items()):
            raise ValueError("Feedback binding/actor mismatch")
    else:
        state = binding | {"records": []}
        with feedback.open("x") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
    lock = threading.Lock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    key = secrets.token_urlsafe(32)
    cookie_name = "hc_edit_" + hashlib.sha256(str(feedback).encode()).hexdigest()[:12]
    pair_tokens = {m["token"] for p in packet["pairs"] for m in p["media"]}
    point_tokens = {f"point-{i}" for i in range(1, len(packet["selected_clips"]) + 1)}

    def gate():
        latest = {r["token"]: r["value"] for r in state["records"] if r["field"] == "complete"}
        return all(latest.get(t) == "完整" for t in point_tokens)

    def snapshot():
        return state | {
            "revision": len(state["records"]),
            "compilation_eligible": gate(),
            "human_status": "fixture-only"
            if fixture
            else ("individual-ratings-recorded" if state["records"] else "pending"),
        }

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.url.hostname not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"detail": "Loopback only"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Same-origin only"}, status_code=403)
        if (
            request.url.path.startswith(("/api/", "/media/"))
            and request.cookies.get(cookie_name) != key
        ):
            return JSONResponse({"detail": "Open review page first"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/")
    def index():
        response = HTMLResponse((ASSETS / "index.html").read_text())
        response.set_cookie(cookie_name, key, httponly=True, samesite="strict")
        return response

    @app.get("/review.js")
    def script():
        return FileResponse(ASSETS / "review.js", media_type="text/javascript")

    @app.get("/api/pack")
    def public_pack():
        # Only blind tokens and source disclosure. No variant/method-key endpoint.
        return {
            "pairs": packet["pairs"],
            "contexts": packet["contexts"],
            "selected_clips": packet["selected_clips"],
            "reels": packet["reels"],
            "fixture": fixture,
        }

    @app.get("/media/{token}")
    def video(token: str):
        row = media.get(token)
        if row is None:
            raise HTTPException(404, "Unknown media")
        # Starlette FileResponse supports byte ranges and seeking.
        return FileResponse(pack / row["file"], media_type="video/mp4")

    @app.get("/api/feedback")
    def current():
        with lock:
            return snapshot()

    @app.get("/api/export")
    def export():
        with lock:
            return JSONResponse(
                snapshot(),
                headers={"Content-Disposition": 'attachment; filename="rally-edit-feedback.json"'},
            )

    @app.post("/api/feedback")
    def update(entry: ReviewEntry):
        data = entry.model_dump()
        with lock:
            existing = next((r for r in state["records"] if r["id"] == entry.id), None)
            if existing:
                if any(existing[k] != v for k, v in data.items()):
                    raise HTTPException(409, "Entry ID already used")
                return snapshot()
            allowed = {
                "complete": {None, "完整", "不完整", "無法判定"},
                "rhythm": {None, "太急", "合適", "太拖"},
                "transition": {None, "K", "M", "無差異", "無法判定"},
                "order": {None, "K", "N", "無差異", "無法判定"},
                "presentation_order": {"0", "1"},
            }
            if entry.field != "note" and entry.value not in allowed[entry.field]:
                raise HTTPException(422, "Invalid rating")
            if entry.field in {"transition", "order"}:
                valid = entry.token == "reel"
                if not gate():
                    raise HTTPException(409, "Confirm each point's completeness first")
            elif entry.field == "presentation_order":
                valid = entry.token == "review"
            else:
                valid = entry.token in pair_tokens | point_tokens
            if not valid:
                raise HTTPException(422, "Invalid token/field")
            record = data | {"actor": binding["actor"], "created_at": datetime.now(UTC).isoformat()}
            updated = state | {"records": [*state["records"], record]}
            temporary = feedback.with_name(feedback.name + ".tmp")
            with temporary.open("w") as stream:
                json.dump(updated, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(feedback)
            state.update(updated)
            return snapshot()

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument(
        "--fixture", action="store_true", help="Label ALL input as automation fixture"
    )
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.pack, args.feedback, fixture=args.fixture), host="127.0.0.1", port=args.port
    )


if __name__ == "__main__":
    main()
