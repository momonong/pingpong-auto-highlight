from __future__ import annotations

import json
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from pingpong_highlight.auth import hash_password
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.web import create_app


@pytest.fixture
def review(tmp_path):
    settings = Settings(
        data_dir=tmp_path, upload_token="unused", bootstrap_admin_password="test-password"
    )
    app = create_app(settings)
    db = app.state.database
    source = settings.uploads_dir / "synthetic.mp4"
    source.write_bytes(b"metadata fixture, not decoded")
    upload, job = db.register_completed_upload(
        uuid.uuid4().hex,
        "synthetic.mp4",
        source.stat().st_size,
        "video/mp4",
        source,
        user_id=app.state.bootstrap_admin.id,
    )
    result = {
        "media": {"duration": 20},
        "algorithm_version": "fixture-v1",
        "candidates": [
            {"rally_start": 1.001, "rally_end": 8, "score": 999, "token": "must-not-export"},
            {"rally_start": 11, "rally_end": 15},
            {"clip_start": 0, "clip_end": 9},  # Padded-only boundaries are not points.
        ],
        "password": "must-not-export",
    }
    db.finish_job(job.id, result)
    # No lifespan: workers never start and no fixture media is processed.
    client = TestClient(app)
    assert (
        client.post(
            "/api/auth/login", json={"username": "admin", "password": "test-password"}
        ).status_code
        == 200
    )
    url = f"/api/jobs/{job.id}/point-review"
    yield client, db, upload, job, result, url, settings
    client.close()


def change(client, url, action, **fields):
    revision = client.get(url).json()["revision"]
    command = {"revision": revision, "request_id": str(uuid.uuid4()), "action": action, **fields}
    response = client.post(url, json=command)
    assert response.status_code == 200, response.text
    return response.json()


def active(payload):
    return sorted((p for p in payload["points"] if p["active"]), key=lambda p: p["start_ms"])


def fields(start=1000, end=9000, **kwargs):
    return {"start_ms": start, "end_ms": end, **kwargs}


def test_roundtrip_split_merge_delete_and_raw_scores(review):
    c, db, upload, job, _result, url, settings = review
    created = change(
        c,
        url,
        "save",
        point=fields(
            validity="valid",
            boundary_status="confirmed",
            rating_status="rated",
            excitement=0,
            reason_tags=["rally", "save"],
            quality_tags=["nearby_table"],
            note="ordinary complete point",
        ),
    )
    point = active(created)[0]
    assert point["excitement"] == 0 and point["human_reviewed"] is True
    assert point["source_id"] == upload.id and point["annotator_id"]
    for value in (1, 2, 3):
        updated = change(
            c,
            url,
            "save",
            point_id=point["id"],
            point=fields(1100, 9100, validity="valid", rating_status="rated", excitement=value),
        )
        assert active(updated)[0]["id"] == point["id"]
        assert active(updated)[0]["excitement"] == value
    split = change(c, url, "split", point_id=point["id"], split_ms=4500)
    children = active(split)
    assert [(p["start_ms"], p["end_ms"]) for p in children] == [(1100, 4500), (4500, 9100)]
    assert all(p["parent_ids"] == [point["id"]] and p["excitement"] is None for p in children)
    retired = next(p for p in split["points"] if p["id"] == point["id"])
    assert not retired["active"] and retired["excitement"] == 3
    assert set(retired["superseded_by"]) == {p["id"] for p in children}
    merged = change(c, url, "merge", point_id=children[0]["id"], other_id=children[1]["id"])
    merged_point = active(merged)[0]
    assert (merged_point["start_ms"], merged_point["end_ms"]) == (1100, 9100)
    assert set(merged_point["parent_ids"]) == {p["id"] for p in children}
    assert merged_point["boundary_status"] == "pending"
    assert c.post("/api/auth/logout").status_code == 204
    assert c.get(url).status_code == 401
    c.post("/api/auth/login", json={"username": "admin", "password": "test-password"})
    assert c.get(url).json()["points"] == merged["points"]
    reopened = Database(settings.database_path)
    assert reopened.get_point_review(upload, db.get_job(job.id))["points"] == merged["points"]
    deleted = change(c, url, "delete", point_id=merged_point["id"])
    assert active(deleted) == [] and len(deleted["points"]) == 4


def test_legacy_and_reprocessing_never_infer_truth_or_overwrite(review):
    c, db, upload, job, result, url, _settings = review
    first = db.create_annotation(
        upload.id, label="highlight", start=2.5, end=6.2, note="相持、搶攻"
    )
    db.create_annotation(
        upload.id, label="exclude", start=16, end=19, note="old negative selection"
    )
    original = db.list_annotations(upload.id)
    empty = c.get(url).json()
    assert empty["revision"] == 0 and empty["points"] == []
    assert len(empty["legacy_annotations"]) == 2 and not empty["fully_reviewed"]
    imported = change(c, url, "import", proposal_batch=empty["available_proposals"]["id"])
    assert len(active(imported)) == 4 and imported["available_proposals"]["skipped"] == 1
    assert all(
        p["rating_status"] == "unrated"
        and p["excitement"] is None
        and p["boundary_status"] == "pending"
        for p in imported["points"]
    )
    assert imported["coverage"] == [] and db.list_annotations(upload.id) == original
    legacy = next(p for p in imported["points"] if p["origin"].get("annotation_id") == first.id)
    assert legacy["note"] == first.note and legacy["origin"]["legacy_label"] == "highlight"
    twice = change(c, url, "import", proposal_batch=empty["available_proposals"]["id"])
    assert twice["points"] == imported["points"]
    db.reprocess_job(job.id)
    during = c.get(url).json()
    assert during["points"] == imported["points"] and during["source"]["duration_ms"] == 20000
    result["algorithm_version"] = "fixture-v2"
    db.finish_job(job.id, result)
    rerun = c.get(url).json()
    assert rerun["points"] == imported["points"]
    assert rerun["available_proposals"]["id"] != empty["available_proposals"]["id"]
    assert rerun["unknown_intervals"] == [{"start_ms": 0, "end_ms": 20000}]


def test_unknown_coverage_export_and_value_distinctions(review):
    c, _db, _upload, _job, _result, url, _settings = review
    for index, options in enumerate(
        [
            {},
            {"rating_status": "unable"},
            {"validity": "not_rally"},
            {"rating_status": "rated", "excitement": 0},
        ]
    ):
        change(c, url, "save", point=fields(index * 2000, index * 2000 + 1000, **options))
    payload = change(c, url, "coverage_add", interval={"start_ms": 2000, "end_ms": 8000})
    assert payload["unknown_intervals"] == [
        {"start_ms": 0, "end_ms": 2000},
        {"start_ms": 8000, "end_ms": 20000},
    ]
    payload = change(c, url, "coverage_add", interval={"start_ms": 5000, "end_ms": 20000})
    payload = change(c, url, "coverage_add", interval={"start_ms": 0, "end_ms": 2000})
    assert payload["fully_reviewed"] is True
    payload = change(c, url, "coverage_delete", coverage_id=payload["coverage"][-1]["id"])
    assert payload["unknown_intervals"] == [{"start_ms": 0, "end_ms": 2000}]
    exported = c.get(url + "/export").json()
    records = [json.loads(line) for line in c.get(url + "/export?format=jsonl").text.splitlines()]
    assert exported == c.get(url).json()
    assert [
        {k: v for k, v in p.items() if k != "type"} for p in records if p["type"] == "point"
    ] == exported["points"]
    assert [p["rating_status"] for p in active(exported)] == [
        "unrated",
        "unable",
        "unrated",
        "rated",
    ]
    assert [p["excitement"] for p in active(exported)] == [None, None, None, 0]
    assert active(exported)[2]["validity"] == "not_rally"
    assert "must-not-export" not in json.dumps(exported)
    assert "session" not in exported and "path" not in exported["source"]


def test_cas_atomicity_idempotency_and_validation(review):
    c, _db, _upload, _job, _result, url, settings = review
    command = {"action": "save", "revision": 0, "request_id": str(uuid.uuid4()), "point": fields()}
    one = c.post(url, json=command).json()
    again = c.post(url, json=command).json()
    assert len(again["points"]) == 1 and again["revision"] == 1
    assert again["points"] == one["points"]
    assert c.post(url, json=command | {"point": fields(2000, 3000)}).status_code == 409
    assert c.post(url, json=command | {"request_id": str(uuid.uuid4())}).status_code == 409
    for point in [
        fields(end=999999),
        fields(start=-1),
        fields(start=1.5),
        fields(start=True),
        fields(excitement=0),
        fields(rating_status="rated"),
        fields(rating_status="unable", excitement=1),
        fields(validity="not_rally", rating_status="rated", excitement=0),
        fields(reason_tags=["occlusion"]),
        fields(quality_tags=["rally"]),
        fields(excitement=4),
        fields(note="x" * 4001),
    ]:
        response = c.post(
            url, json=command | {"revision": 1, "request_id": str(uuid.uuid4()), "point": point}
        )
        assert response.status_code == 422, response.text
    point_id = one["points"][0]["id"]
    invalid = {
        "revision": 1,
        "request_id": str(uuid.uuid4()),
        "action": "split",
        "point_id": point_id,
        "split_ms": 1000,
    }
    assert c.post(url, json=invalid).status_code == 422
    assert c.get(url).json()["revision"] == 1
    assert c.get(url).json()["points"] == one["points"]
    with sqlite3.connect(settings.database_path) as conn:
        assert conn.execute("select count(*) from point_review_requests").fetchone()[0] == 1


def test_owner_read_admin_write_and_cross_source_ids(review):
    c, db, upload, _job, result, url, _settings = review
    user = db.create_user(
        username="viewer",
        display_name="Viewer",
        role="user",
        password_hash=hash_password("viewer-password"),
    )
    _other_upload, other_job = db.register_completed_upload(
        uuid.uuid4().hex, "other.mp4", 1, "video/mp4", upload.path, user_id=user.id
    )
    db.finish_job(other_job.id, result)
    other_url = f"/api/jobs/{other_job.id}/point-review"
    point = active(change(c, url, "save", point=fields()))[0]
    attack = {
        "revision": 0,
        "request_id": str(uuid.uuid4()),
        "action": "delete",
        "point_id": point["id"],
    }
    assert c.post(other_url, json=attack).status_code == 409
    assert c.get(other_url).json()["points"] == []
    c.post("/api/auth/logout")
    c.post("/api/auth/login", json={"username": "viewer", "password": "viewer-password"})
    assert c.get(url).status_code == 404
    assert c.get(url + "/export").status_code == 404
    assert c.get(other_url).status_code == 200
    assert c.get(other_url + "/export").status_code == 200
    assert c.post(other_url, json=attack).status_code == 403


def test_old_schema_upgrade_only_adds_review_tables(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    # An independent metadata fixture, never a connection to the real state database.
    db = Database(path)
    upload, _job = db.register_completed_upload(
        "source", "legacy.mp4", 10, "video/mp4", tmp_path / "not-read.mp4"
    )
    annotation = db.create_annotation(upload.id, label="highlight", start=1, end=2, note="legacy")
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE point_review_requests")
        conn.execute("DROP TABLE point_reviews")
        conn.execute("CREATE TABLE storage_objects (id TEXT, credential TEXT)")
        conn.execute("INSERT INTO storage_objects VALUES ('archive', 'private-fixture')")
    upgraded = Database(path)
    assert upgraded.list_annotations(upload.id) == [annotation]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM storage_objects").fetchall() == [
            ("archive", "private-fixture")
        ]
        assert conn.execute("SELECT count(*) FROM point_reviews").fetchone()[0] == 0
