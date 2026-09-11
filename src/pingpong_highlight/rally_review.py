"""Independent proposal and human stores; half-open source timestamps in integer ms."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> str:
    return datetime.now(UTC).isoformat()


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.start_ms >= self.end_ms:
            raise ValueError("Invalid interval")
        return self


class Judgment(Interval):
    rally: Literal["yes", "no", "uncertain", "unable"] = "uncertain"
    complete: Literal["yes", "no", "uncertain", "unable"] = "uncertain"
    highlight: Literal["omit", "include", "must", "unrated"] = "unrated"
    reason: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def meaningful(self):
        if self.rally != "yes" and self.highlight != "unrated":
            raise ValueError("Only a judged rally may receive a highlight rating")
        return self


class Proposal(Judgment):
    id: str = Field(min_length=1, max_length=100)
    selected: bool = False
    clip_start_ms: int = Field(ge=0)
    clip_end_ms: int = Field(gt=0)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def padding(self):
        if not self.clip_start_ms <= self.start_ms < self.end_ms <= self.clip_end_ms:
            raise ValueError("Padding must contain core")
        return self


class ReviewCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=0)
    request_id: str = Field(min_length=8, max_length=100)
    action: Literal["save", "split", "merge", "coverage", "coverage_remove", "timer"]
    point_id: str | None = None
    other_id: str | None = None
    proposal_id: str | None = None
    point: Judgment | None = None
    split_ms: int | None = None
    interval: Interval | None = None
    elapsed_ms: int = Field(default=0, ge=0, le=60000)


class Conflict(ValueError):
    pass


def union(intervals: list[dict]) -> list[dict]:
    result = []
    for item in sorted(intervals, key=lambda x: x["start_ms"]):
        a, b = item["start_ms"], item["end_ms"]
        if result and a <= result[-1]["end_ms"]:
            result[-1]["end_ms"] = max(b, result[-1]["end_ms"])
        else:
            result.append({"start_ms": a, "end_ms": b})
    return result


def gaps(intervals: list[dict], start: int, end: int) -> list[dict]:
    result = []
    cursor = start
    for item in union(intervals):
        a, b = max(start, item["start_ms"]), min(end, item["end_ms"])
        if b <= start or a >= end:
            continue
        if cursor < a:
            result.append({"start_ms": cursor, "end_ms": a})
        cursor = max(cursor, b)
    if cursor < end:
        result.append({"start_ms": cursor, "end_ms": end})
    return result


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables and tables != {"sources", "runs", "reviews", "commands"}:
                raise ValueError("Refusing non-review database; use an isolated review.sqlite3")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, source_id TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS reviews(source_id TEXT PRIMARY KEY, payload TEXT);
                CREATE TABLE IF NOT EXISTS commands(source_id TEXT, request_id TEXT,
                    command TEXT, PRIMARY KEY(source_id,request_id));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, source: dict):
        # Content hash identity, not filename or job identity.
        if source["id"] != source["sha256"] or len(source["id"]) != 64:
            raise ValueError("Source requires SHA-256 identity")
        if source["group"] not in ("development", "held-out"):
            raise ValueError("Unknown split")
        if source["duration_ms"] <= 0:
            raise ValueError("Missing duration")
        for interval in source.get("scope", []):
            if Interval.model_validate(interval).end_ms > source["duration_ms"]:
                raise ValueError("Scope outside source")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT payload FROM sources WHERE id=?", (source["id"],)
            ).fetchone()
            if existing:
                if json.loads(existing[0]) != source:
                    raise Conflict("Source metadata is immutable; use another experiment store")
            else:
                db.execute("INSERT INTO sources VALUES (?,?)", (source["id"], json.dumps(source)))

    def sources(self) -> list[dict]:
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM sources")]

    def source(self, source_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(source_id)
        return json.loads(row[0])

    def add_run(self, run: dict):
        source = self.source(run["source_id"])
        for key in (
            "id",
            "commit",
            "code_sha256",
            "model_id",
            "model_revision",
            "prompt",
            "parameters",
            "sampling",
            "elapsed_seconds",
            "proposals",
            "windows",
        ):
            if key not in run:
                raise ValueError(f"Missing run receipt: {key}")
        ids = set()
        for value in run["windows"]:
            if Interval.model_validate(value).end_ms > source["duration_ms"]:
                raise ValueError("Analyzed window outside source")
        for value in run["proposals"]:
            item = Proposal.model_validate(value)
            if item.id in ids or item.clip_end_ms > source["duration_ms"]:
                raise ValueError("Duplicate ID or proposal outside source")
            if not any(
                w["start_ms"] <= item.start_ms < item.end_ms <= w["end_ms"] for w in run["windows"]
            ):
                raise ValueError("Proposal outside analyzed windows")
            ids.add(item.id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT payload FROM runs WHERE id=?", (run["id"],)).fetchone()
            payload = json.dumps(run, sort_keys=True)
            if existing:
                if existing[0] != payload:
                    raise Conflict("Proposal runs are immutable")
            else:
                old_ids = {
                    p["id"]
                    for row in db.execute(
                        "SELECT payload FROM runs WHERE source_id=?", (source["id"],)
                    )
                    for p in json.loads(row[0])["proposals"]
                }
                if old_ids & ids:
                    raise Conflict("Proposal IDs must be unique across source runs")
                db.execute("INSERT INTO runs VALUES (?,?,?)", (run["id"], source["id"], payload))

    def _state(self, db, source_id):
        row = db.execute("SELECT payload FROM reviews WHERE source_id=?", (source_id,)).fetchone()
        return (
            json.loads(row[0])
            if row
            else {
                "revision": 0,
                "points": [],
                "coverage": [],
                "active_ms": 0,
                "modification_count": 0,
                "history": [],
                "semantics": "serve-to-point-end; all rallies; unreviewed=UNKNOWN; integer-ms",
            }
        )

    def export(self, source_id: str, *, blind: bool = False) -> dict:
        source = self.source(source_id)
        with self.connect() as db:
            state = self._state(db, source_id)
            runs = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM runs WHERE source_id=? ORDER BY rowid", (source_id,)
                )
            ]
        if blind:
            judged = {p for x in state["points"] for p in x["proposal_ids"]}
            for run in runs:
                # Same fixed source-time blind intervals across providers and reruns.
                for p in run["proposals"]:
                    hidden = any(
                        b["start_ms"] < p["end_ms"] and p["start_ms"] < b["end_ms"]
                        for b in source.get("blind_intervals", [])
                    )
                    p["blind"] = hidden and p["id"] not in judged
                    if p["blind"]:
                        for key in (
                            "rally",
                            "complete",
                            "highlight",
                            "reason",
                            "selected",
                            "evidence",
                        ):
                            p.pop(key, None)
                # Raw responses also contain the hidden advice; never send to blind UI.
                for key in ("raw", "audio_events", "errors", "duplicates"):
                    run.pop(key, None)
        intervals = [p for r in runs for p in r["proposals"]]
        state["unreviewed"] = gaps(state["coverage"], 0, source["duration_ms"])
        state["proposal_gaps"] = gaps(intervals, 0, source["duration_ms"])
        return {"source": source, "runs": runs, "review": state}

    def apply(self, source_id: str, command: ReviewCommand, *, actor: str) -> dict:
        source = self.source(source_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._state(db, source_id)
            encoded = command.model_dump_json()
            old = db.execute(
                "SELECT command FROM commands WHERE source_id=? AND request_id=?",
                (source_id, command.request_id),
            ).fetchone()
            if old:
                if old[0] != encoded:
                    raise Conflict("Request ID reused with different content")
                return state
            if command.revision != state["revision"]:
                raise Conflict("Review changed; reload before saving")
            points = {p["id"]: p for p in state["points"]}
            proposals = {
                p["id"]: p
                for r in db.execute("SELECT payload FROM runs WHERE source_id=?", (source_id,))
                for p in json.loads(r[0])["proposals"]
            }

            def make(fields, parents, proposal_ids):
                point = Judgment.model_validate(fields).model_dump()
                if point["end_ms"] > source["duration_ms"]:
                    raise ValueError("Boundary outside source")
                point.update(
                    id=uuid.uuid4().hex,
                    parents=parents,
                    proposal_ids=proposal_ids,
                    actor=actor,
                    reviewed_at=now(),
                )
                return point

            before = state["points"]
            if command.action == "save":
                if command.point is None:
                    raise ValueError("Missing point")
                prior = points.get(command.point_id) if command.point_id else None
                if command.point_id and prior is None:
                    raise ValueError("Unknown point")
                linked = prior["proposal_ids"] if prior else []
                if command.proposal_id:
                    if command.proposal_id not in proposals:
                        raise ValueError("Unknown proposal")
                    if any(
                        command.proposal_id in p["proposal_ids"] and p != prior
                        for p in points.values()
                    ):
                        raise Conflict("Proposal already reviewed; edit its human annotation")
                    linked = sorted(set([*linked, command.proposal_id]))
                item = make(command.point.model_dump(), [prior["id"]] if prior else [], linked)
                if prior:
                    del points[prior["id"]]
                points[item["id"]] = item
            elif command.action in ("split", "merge"):
                if command.point_id not in points:
                    raise ValueError("Select a human annotation first")
                first = points.pop(command.point_id)
                if command.action == "split":
                    cut = command.split_ms
                    if cut is None or not first["start_ms"] < cut < first["end_ms"]:
                        raise ValueError("Split must be within the point")
                    spans = [(first["start_ms"], cut), (cut, first["end_ms"])]
                    parents, linked = [first["id"]], first["proposal_ids"]
                else:
                    if command.other_id not in points:
                        raise ValueError("Choose another human point to merge")
                    other = points.pop(command.other_id)
                    spans = [
                        (
                            min(first["start_ms"], other["start_ms"]),
                            max(first["end_ms"], other["end_ms"]),
                        )
                    ]
                    parents = [first["id"], other["id"]]
                    linked = sorted(set(first["proposal_ids"] + other["proposal_ids"]))
                for a, b in spans:
                    # Structural changes require fresh validity, boundary and rating judgment.
                    item = make({"start_ms": a, "end_ms": b}, parents, linked)
                    points[item["id"]] = item
            elif command.action in ("coverage", "coverage_remove"):
                if command.interval is None or command.interval.end_ms > source["duration_ms"]:
                    raise ValueError("Invalid coverage")
                interval = command.interval.model_dump()
                if command.action == "coverage":
                    state["coverage"] = union(state["coverage"] + [interval])
                else:
                    remainder = []
                    for x in state["coverage"]:
                        remainder.extend(gaps([interval], x["start_ms"], x["end_ms"]))
                    state["coverage"] = union(remainder)
            state["points"] = sorted(points.values(), key=lambda p: p["start_ms"])
            state["active_ms"] += command.elapsed_ms
            state["modification_count"] += int(command.action != "timer")
            state["revision"] += 1
            state["history"].append(
                {
                    "at": now(),
                    "actor": actor,
                    "command": command.model_dump(),
                    "before": before if command.action != "timer" else None,
                }
            )
            db.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?,?)", (source_id, json.dumps(state))
            )
            db.execute(
                "INSERT INTO commands VALUES (?,?,?)", (source_id, command.request_id, encoded)
            )
        return state


def source_identity(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while hashing")
    return {
        "id": digest.hexdigest(),
        "sha256": digest.hexdigest(),
        "size": after.st_size,
        "path": str(path.resolve()),
        "name": path.name,
    }
