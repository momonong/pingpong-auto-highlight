"""Source-bound human point review. No media I/O and no inference from missing labels."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REASONS = {
    "rally": ["相持", "Rally"],
    "attack": ["搶攻", "Attack"],
    "counterloop": ["反拉", "Counterloop"],
    "counter": ["反壓", "Counter"],
    "placement": ["落點", "Placement"],
    "placement_control": ["落點控制", "Placement control"],
    "block": ["擋球", "Block"],
    "defense": ["防守", "Defense"],
    "save": ["救球", "Great save"],
    "turnaround": ["攻守轉換", "Attack-defense transition"],
}
QUALITY = {
    "occlusion": ["遮擋", "Occlusion"],
    "nearby_table": ["鄰桌干擾", "Nearby table"],
    "incomplete": ["切分不完整", "Incomplete segment"],
    "unclear": ["看不清", "Unclear video"],
}
RULES = {
    "schema_version": "highlightcraft-point-review/1",
    "scale_version": "excitement-0-3/1",
    "time_unit": "integer_ms",
    "interval": "[start_ms,end_ms)",
    "point_definition": "one complete point from serve to scoring end, not one hit",
    "ratings": {
        "0": ["普通，不會收錄", "Ordinary; omit"],
        "1": ["有看點，通常不優先", "Interesting; low priority"],
        "2": ["精彩，願意收錄", "Exciting; include"],
        "3": ["特別精彩，優先保留", "Exceptional; prioritize"],
    },
    "rating_states": ["unrated", "rated", "unable"],
    "validity_states": ["pending", "valid", "not_rally", "unclear"],
    "boundary_states": ["pending", "confirmed", "needs_adjustment", "unclear"],
    "reason_tags": REASONS,
    "quality_tags": QUALITY,
    "unreviewed_intervals": "unknown; never implicit negative samples",
    "coverage_meaning": "human explicitly checked this source interval for all points",
    "completion_rule": "not_rally OR (valid AND confirmed AND rated/unable)",
    "legacy_meaning": "original highlight/exclude preserved; no inferred rating or coverage",
}


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must follow start_ms")
        return self


class PointFields(Interval):
    validity: Literal["pending", "valid", "not_rally", "unclear"] = "pending"
    boundary_status: Literal["pending", "confirmed", "needs_adjustment", "unclear"] = "pending"
    rating_status: Literal["unrated", "rated", "unable"] = "unrated"
    excitement: int | None = Field(default=None, ge=0, le=3)
    reason_tags: list[str] = Field(default_factory=list, max_length=10)
    quality_tags: list[str] = Field(default_factory=list, max_length=4)
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def semantics(self):
        if (self.rating_status == "rated") != (self.excitement is not None):
            raise ValueError("Only rated points have an excitement value")
        if self.validity == "not_rally" and self.rating_status != "unrated":
            raise ValueError("Non-rallies cannot have a point rating")
        if set(self.reason_tags) - REASONS.keys() or set(self.quality_tags) - QUALITY.keys():
            raise ValueError("Unknown tag code")
        if len(set(self.reason_tags)) != len(self.reason_tags):
            raise ValueError("Duplicate reason tag")
        if len(set(self.quality_tags)) != len(self.quality_tags):
            raise ValueError("Duplicate quality tag")
        return self


class ReviewCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=0)
    request_id: str = Field(min_length=16, max_length=80, pattern=r"^[a-zA-Z0-9-]+$")
    action: Literal["save", "delete", "split", "merge", "import", "coverage_add", "coverage_delete"]
    point_id: str | None = None
    other_id: str | None = None
    point: PointFields | None = None
    split_ms: int | None = Field(default=None, ge=0)
    interval: Interval | None = None
    coverage_id: str | None = None
    proposal_batch: str | None = None


def stamp() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def candidate_batch(job) -> dict:
    """Explicit allowlist: never return arbitrary analysis metadata, paths or model scores."""
    result = job.result or {}
    candidates = []
    skipped = 0
    duration = source_duration(job)
    for index, item in enumerate(result.get("candidates", [])):
        try:
            # Core rally boundaries only. No fallback to padded clip boundaries.
            start, end = item["rally_start"], item["rally_end"]
            if (
                type(start) not in (int, float)
                or type(end) not in (int, float)
                or not math.isfinite(start)
                or not math.isfinite(end)
            ):
                raise ValueError("invalid times")
            fields = PointFields(start_ms=round(start * 1000), end_ms=round(end * 1000))
            if duration is None or fields.end_ms > duration:
                raise ValueError("outside source")
            candidates.append(
                {"index": index, "start_ms": fields.start_ms, "end_ms": fields.end_ms}
            )
        except (KeyError, TypeError, ValueError):
            skipped += 1
    return {
        "id": digest({"job_id": job.id, "result": result}),
        "job_id": job.id,
        "algorithm_version": result.get("algorithm_version")
        if isinstance(result.get("algorithm_version"), str)
        else None,
        "generated_at": job.updated_at,
        "candidates": candidates,
        "skipped": skipped,
    }


def source_duration(job) -> int | None:
    duration = (job.result or {}).get("media", {}).get("duration")
    if type(duration) in (int, float) and math.isfinite(duration) and duration > 0:
        return round(duration * 1000)
    return None


def empty_review() -> dict:
    return {"points": [], "coverage": [], "imports": []}


def read_review(connection, upload, job) -> dict:
    row = connection.execute(
        "SELECT * FROM point_reviews WHERE source_id=?", (upload.id,)
    ).fetchone()
    content = json.loads(row["content_json"]) if row else empty_review()
    legacy = [
        dict(r)
        for r in connection.execute(
            "SELECT id,label,start,end,note,created_at,updated_at FROM annotations "
            "WHERE upload_id=? "
            "ORDER BY start,id",
            (upload.id,),
        )
    ]
    duration = row["duration_ms"] if row else source_duration(job)
    covered = sorted((c["start_ms"], c["end_ms"]) for c in content["coverage"] if c["active"])
    union = []
    for start, end in covered:
        if union and start <= union[-1][1]:
            union[-1][1] = max(end, union[-1][1])
        else:
            union.append([start, end])
    unknown, cursor = [], 0
    if duration is not None:
        for start, end in union:
            if cursor < start:
                unknown.append({"start_ms": cursor, "end_ms": start})
            cursor = max(cursor, end)
        if cursor < duration:
            unknown.append({"start_ms": cursor, "end_ms": duration})
    return {
        "rules": RULES,
        "source": {
            "id": upload.id,
            "identity_scheme": "persistent-upload-id",
            "filename": upload.filename,
            "size_bytes": upload.size,
            "owner_id": upload.user_id,
            "duration_ms": duration,
            "created_at": upload.created_at,
            "content_hash": None,
        },
        "revision": row["revision"] if row else 0,
        "updated_at": row["updated_at"] if row else None,
        **content,
        "legacy_annotations": legacy,
        "unknown_intervals": unknown if duration is not None else None,
        "fully_reviewed": duration is not None and not unknown,
        "available_proposals": candidate_batch(job),
    }


def apply_command(connection, upload, job, command: ReviewCommand, actor_id: str) -> dict:
    """Caller holds BEGIN IMMEDIATE; receipts and entire source revision commit atomically."""
    from pingpong_highlight.db import StateConflict

    request_hash = digest(command.model_dump() | {"actor_id": actor_id})
    receipt = connection.execute(
        "SELECT * FROM point_review_requests WHERE source_id=? AND request_id=?",
        (upload.id, command.request_id),
    ).fetchone()
    if receipt:
        if receipt["request_hash"] != request_hash:
            raise StateConflict("Request ID already used for another change")
        return read_review(connection, upload, job) | json.loads(receipt["response_json"])
    workspace = read_review(connection, upload, job)
    if command.revision != workspace["revision"]:
        raise StateConflict("Review changed in another window; reload before saving")
    duration = workspace["source"]["duration_ms"]
    if duration is None:
        raise StateConflict("Source duration is not available yet")
    now, revision = stamp(), workspace["revision"] + 1
    points, coverage, imports = workspace["points"], workspace["coverage"], workspace["imports"]

    def point_by_id(point_id):
        point = next((p for p in points if p["id"] == point_id and p["active"]), None)
        if point is None:
            raise StateConflict("Point no longer active or does not belong to this source")
        return point

    def check_range(value):
        if value.end_ms > duration:
            raise ValueError("Interval exceeds source duration")

    def touch(value):
        value.update(updated_at=now, annotator_id=actor_id, version=revision)

    def new_point(fields, origin, parents=None, point_id=None):
        check_range(fields)
        point = {
            "id": point_id or uuid.uuid4().hex,
            "source_id": upload.id,
            **fields.model_dump(),
            "active": True,
            "origin": origin,
            "parent_ids": parents or [],
            "superseded_by": [],
            "created_at": now,
            "created_by": actor_id,
            "human_reviewed": False,
        }
        touch(point)
        points.append(point)
        return point

    if command.action == "save":
        if command.point is None:
            raise ValueError("point is required")
        check_range(command.point)
        if command.point_id:
            point = point_by_id(command.point_id)
            point.update(command.point.model_dump())
            touch(point)
        else:
            point = new_point(command.point, {"kind": "manual", "version": "point-review/1"})
        point["human_reviewed"] = True
        workspace["selected_id"] = point["id"]
    elif command.action in {"delete", "split", "merge"}:
        parent = point_by_id(command.point_id)
        parents = [parent]
        children = []
        if command.action == "split":
            at = command.split_ms
            if at is None or not parent["start_ms"] < at < parent["end_ms"]:
                raise ValueError("Split must be strictly inside the point")
            ranges = [(parent["start_ms"], at), (at, parent["end_ms"])]
        elif command.action == "merge":
            other = point_by_id(command.other_id)
            if other["id"] == parent["id"]:
                raise ValueError("Select two distinct points")
            parents.append(other)
            ordered = sorted(
                (p for p in points if p["active"]),
                key=lambda p: (p["start_ms"], p["end_ms"], p["id"]),
            )
            if abs(ordered.index(parent) - ordered.index(other)) != 1:
                raise ValueError("Only adjacent points may be merged")
            ranges = [(min(p["start_ms"] for p in parents), max(p["end_ms"] for p in parents))]
        else:
            ranges = []
        for start, end in ranges:
            children.append(
                new_point(
                    PointFields(start_ms=start, end_ms=end),
                    {"kind": command.action, "version": "point-review/1"},
                    [p["id"] for p in parents],
                )
            )
        for parent in parents:
            parent.update(
                active=False,
                superseded_by=[c["id"] for c in children],
                retired_reason=command.action,
            )
            touch(parent)
        workspace["selected_id"] = children[0]["id"] if children else None
    elif command.action == "import":
        batch = workspace["available_proposals"]
        if command.proposal_batch != batch["id"]:
            raise StateConflict("Analysis changed; reload proposals before importing")
        if batch["id"] not in {b["id"] for b in imports}:
            imports.append(batch | {"imported_at": now, "imported_by": actor_id})
            for candidate in batch["candidates"]:
                new_point(
                    PointFields(start_ms=candidate["start_ms"], end_ms=candidate["end_ms"]),
                    {
                        "kind": "automatic",
                        "batch_id": batch["id"],
                        "candidate_index": candidate["index"],
                        "version": batch["algorithm_version"],
                    },
                )
        existing_legacy = {p["origin"].get("annotation_id") for p in points}
        for legacy in workspace["legacy_annotations"]:
            if legacy["id"] in existing_legacy:
                continue
            try:
                fields = PointFields(
                    start_ms=round(legacy["start"] * 1000),
                    end_ms=round(legacy["end"] * 1000),
                    note=legacy["note"],
                )
                check_range(fields)
            except ValueError:
                continue  # Original remains exported verbatim, even if outside current duration.
            new_point(
                fields,
                {
                    "kind": "legacy",
                    "annotation_id": legacy["id"],
                    "legacy_label": legacy["label"],
                    "version": "annotations/1",
                },
            )
    elif command.action == "coverage_add":
        if command.interval is None:
            raise ValueError("interval is required")
        check_range(command.interval)
        item = {
            "id": uuid.uuid4().hex,
            **command.interval.model_dump(),
            "active": True,
            "created_at": now,
            "created_by": actor_id,
        }
        touch(item)
        coverage.append(item)
    elif command.action == "coverage_delete":
        item = next((c for c in coverage if c["id"] == command.coverage_id and c["active"]), None)
        if item is None:
            raise StateConflict("Coverage no longer active")
        item["active"] = False
        touch(item)
    content = {"points": points, "coverage": coverage, "imports": imports}
    connection.execute(
        "INSERT INTO point_reviews VALUES (?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET "
        "content_json=excluded.content_json,revision=excluded.revision,updated_at=excluded.updated_at",
        (upload.id, duration, json.dumps(content), revision, now, now),
    )
    response = read_review(connection, upload, job)
    response["selected_id"] = workspace.get("selected_id")
    receipt_data = {"selected_id": response["selected_id"], "applied_revision": revision}
    connection.execute(
        "INSERT INTO point_review_requests VALUES (?,?,?,?)",
        (upload.id, command.request_id, request_hash, json.dumps(receipt_data)),
    )
    return response
