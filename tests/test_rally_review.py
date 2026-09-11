from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from pingpong_highlight.preannotate import deduplicate, normalize_response, windows
from pingpong_highlight.rally_review import (
    Conflict,
    Judgment,
    Proposal,
    ReviewCommand,
    ReviewStore,
    gaps,
)
from pingpong_highlight.review_evaluation import evaluate_review, match
from pingpong_highlight.review_web import create_review_app

SID = "a" * 64


def source():
    return {
        "id": SID,
        "sha256": SID,
        "path": "unavailable.mp4",
        "size": 1,
        "name": "sample",
        "duration_ms": 10000,
        "group": "development",
        "scope": [{"start_ms": 0, "end_ms": 10000}],
        "blind_intervals": [{"start_ms": 0, "end_ms": 2000}],
    }


def proposal(pid="p1", start=1000, end=3000):
    return Proposal(
        id=pid,
        start_ms=start,
        end_ms=end,
        clip_start_ms=max(0, start - 500),
        clip_end_ms=end + 500,
        rally="yes",
        complete="yes",
        highlight="must",
        reason="observable movement",
        selected=True,
    ).model_dump()


def run(rid="r1", proposals=None):
    return {
        "id": rid,
        "source_id": SID,
        "commit": "fixture",
        "code_sha256": "fixture",
        "model_id": "fixture",
        "model_revision": "fixture",
        "prompt": "fixture",
        "parameters": {},
        "sampling": {},
        "elapsed_seconds": 1,
        "windows": [{"start_ms": 0, "end_ms": 10000}],
        "proposals": proposals if proposals is not None else [proposal()],
        "raw": ["must include"],
        "duplicates": [{"highlight": "must"}],
    }


@pytest.fixture
def store(tmp_path):
    s = ReviewStore(tmp_path / "review.sqlite3")
    s.register(source())
    s.add_run(run())
    return s


def apply(store, action, **kwargs):
    revision = store.export(SID)["review"]["revision"]
    return store.apply(
        SID,
        ReviewCommand(revision=revision, request_id=f"request-{revision}", action=action, **kwargs),
        actor="tester",
    )


def test_persistence_rerun_idempotency_and_conflict(store):
    fields = Judgment(start_ms=1200, end_ms=2800, rally="yes", complete="yes", highlight="include")
    c = ReviewCommand(
        revision=0, request_id="repeat-001", action="save", point=fields, proposal_id="p1"
    )
    first = store.apply(SID, c, actor="tester")
    store.apply(SID, c, actor="tester")
    store.add_run(run("r2", [proposal("p2")]))
    fresh = ReviewStore(store.path).export(SID)
    assert fresh["review"]["points"] == first["points"]
    assert fresh["review"]["revision"] == 1
    with pytest.raises(Conflict):
        store.apply(SID, c.model_copy(update={"request_id": "different"}), actor="tester")
    with pytest.raises(Conflict):
        store.apply(SID, c.model_copy(update={"action": "timer"}), actor="tester")
    with pytest.raises(Conflict):
        store.add_run(run("r1", [proposal("changed")]))


def test_split_merge_lineage_and_unknown_ratings(store):
    saved = apply(
        store,
        "save",
        point=Judgment(start_ms=1000, end_ms=6000, rally="yes", highlight="must"),
        proposal_id="p1",
    )
    split = apply(store, "split", point_id=saved["points"][0]["id"], split_ms=3000)
    assert len(split["points"]) == 2
    assert all(p["highlight"] == "unrated" and p["rally"] == "uncertain" for p in split["points"])
    merged = apply(
        store, "merge", point_id=split["points"][0]["id"], other_id=split["points"][1]["id"]
    )
    assert len(merged["points"]) == 1
    assert merged["points"][0]["parents"] == [p["id"] for p in split["points"]]
    assert merged["points"][0]["proposal_ids"] == ["p1"]
    with pytest.raises(ValueError):
        apply(store, "split", point_id=merged["points"][0]["id"], split_ms=7000)
    assert store.export(SID)["review"]["revision"] == 3


def test_unknown_precision_coverage_retraction_and_timer(store):
    apply(
        store,
        "save",
        point=Judgment(
            start_ms=1000, end_ms=3000, rally="yes", complete="yes", highlight="include"
        ),
    )
    assert evaluate_review(store.export(SID))["runs"][0]["candidate_precision"] == "UNKNOWN"
    from pingpong_highlight.rally_review import Interval

    apply(store, "coverage", interval=Interval(start_ms=0, end_ms=10000))
    report = evaluate_review(store.export(SID))
    assert report["runs"][0]["candidate_precision"] == 1
    apply(store, "coverage_remove", interval=Interval(start_ms=3000, end_ms=5000))
    apply(store, "timer", elapsed_ms=1250)
    state = store.export(SID)["review"]
    assert state["active_ms"] == 1250
    assert state["modification_count"] == 3
    assert state["unreviewed"] == [{"start_ms": 3000, "end_ms": 5000}]
    assert evaluate_review(store.export(SID))["runs"][0]["candidate_precision"] == "UNKNOWN"


def test_blind_payload_does_not_leak_raw_or_duplicates(store):
    hidden = store.export(SID, blind=True)["runs"][0]
    assert "highlight" not in hidden["proposals"][0]
    assert "raw" not in hidden and "duplicates" not in hidden
    apply(store, "save", proposal_id="p1", point=Judgment(start_ms=1000, end_ms=3000))
    assert store.export(SID, blind=True)["runs"][0]["proposals"][0]["highlight"] == "must"


def test_duplicate_matching_cannot_inflate_recall_and_source_isolation():
    preds = [proposal("a") | {"source_id": SID}, proposal("b") | {"source_id": SID}]
    truth = [dict(id="t", start_ms=1000, end_ms=3000, source_id=SID)]
    assert match(preds, truth)["matched"] == 1
    preds[0]["source_id"] = "other"
    preds[1]["source_id"] = "other"
    assert match(preds, truth)["matched"] == 0
    # A greedy matcher loses a match; augmenting paths must recover it.
    pred = [dict(id="wide", start_ms=0, end_ms=4000), dict(id="short", start_ms=0, end_ms=1000)]
    actual = [dict(id="a", start_ms=0, end_ms=2000), dict(id="b", start_ms=2000, end_ms=4000)]
    assert (
        match([p | {"source_id": SID} for p in pred], [p | {"source_id": SID} for p in actual])[
            "matched"
        ]
        == 2
    )
    with pytest.raises(ValueError):
        match(pred, actual)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 6, 10001, True])
def test_output_time_validation(bad):
    fields = Judgment(start_ms=0, end_ms=1000).model_dump() | {"end_ms": bad}
    with pytest.raises(ValueError):
        normalize_response(json.dumps({"rallies": [fields]}), {"start_ms": 50000, "end_ms": 60000})


def test_manifest_source_hash_mismatch_fails_before_decode(tmp_path):
    from pingpong_highlight.preannotate import run_experiment

    media = tmp_path / "fake.mp4"
    media.write_bytes(b"not the expected source")
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "path": str(media),
                        "expected_sha256": "0" * 64,
                        "start_ms": 0,
                        "end_ms": 1000,
                        "group": "development",
                        "session_id": "fixture",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="manifest SHA-256"):
        run_experiment(manifest, tmp_path / "output", "baseline", tmp_path / "unused")


def test_absolute_window_times_and_duplicates():
    raw = json.dumps({"rallies": [Judgment(start_ms=100, end_ms=700).model_dump()]})
    assert normalize_response(raw, {"start_ms": 50000, "end_ms": 60000})[0]["start_ms"] == 50100
    assert windows(0, 35000, 16000, 2000) == [
        dict(start_ms=0, end_ms=16000),
        dict(start_ms=14000, end_ms=30000),
        dict(start_ms=28000, end_ms=35000),
    ]
    kept, removed = deduplicate(
        [proposal("a"), proposal("b", 1100, 3050), proposal("c", 3100, 5000)]
    )
    assert len(kept) == 2 and len(removed) == 1
    assert gaps([dict(start_ms=1000, end_ms=3000)], 0, 5000) == [
        dict(start_ms=0, end_ms=1000),
        dict(start_ms=3000, end_ms=5000),
    ]


def test_store_validation_and_rollbacks(store):
    with pytest.raises(Conflict):
        store.add_run(run("other-batch-with-same-proposal-id"))
    with pytest.raises(ValueError):
        store.add_run(run("invalid", [proposal(), proposal()]))
    invalid = run("invalid-window", [])
    invalid["windows"] = [{"start_ms": 0, "end_ms": 10001}]
    with pytest.raises(ValueError):
        store.add_run(invalid)
    with pytest.raises(ValueError):
        apply(store, "save", point=Judgment(start_ms=9999, end_ms=10001))
    assert store.export(SID)["review"]["revision"] == 0
    with pytest.raises(ValueError):
        Judgment(start_ms=0, end_ms=100, rally="no", highlight="must")


def test_boundary_crossing_human_rally_keeps_precision_unknown(store):
    from pingpong_highlight.rally_review import Interval

    apply(store, "save", point=Judgment(start_ms=500, end_ms=3000, rally="yes", complete="yes"))
    apply(store, "coverage", interval=Interval(start_ms=0, end_ms=10000))
    data = store.export(SID)
    data["runs"][0]["windows"] = [{"start_ms": 1000, "end_ms": 10000}]
    result = evaluate_review(data)["runs"][0]
    assert result["candidate_precision"] == "UNKNOWN"
    assert len(result["boundary_crossing_truth"]) == 1


def test_foreign_database_is_not_migrated(tmp_path):
    import sqlite3

    file = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(file) as db:
        db.execute("CREATE TABLE uploads(id TEXT)")
        db.execute("INSERT INTO uploads VALUES ('protected')")
    before = file.read_bytes()
    with pytest.raises(ValueError, match="non-review"):
        ReviewStore(file)
    assert file.read_bytes() == before


def test_local_api_and_reload(store):
    client = TestClient(create_review_app(store))
    assert client.get(f"/api/review/{SID}").status_code == 403
    assert client.get("/").status_code == 200
    assert client.get(f"/api/review/{SID}").status_code == 200
    payload = ReviewCommand(
        revision=0, request_id="browser-1", action="save", point=Judgment(start_ms=500, end_ms=1000)
    ).model_dump()
    assert client.post(f"/api/review/{SID}", json=payload).status_code == 200
    assert client.get(f"/api/review/{SID}").json()["review"]["revision"] == 1
    payload["request_id"] = "browser-2"
    assert client.post(f"/api/review/{SID}", json=payload).status_code == 409
    assert client.get("/api/review/unknown").status_code == 404
    assert client.get("http://evil.example/").status_code == 403


def test_preview_is_run_scoped_and_stays_inside_experiment(store):
    preview = store.path.parent / "preview.mp4"
    preview.write_bytes(b"test-preview")
    batch = run("preview-batch", [proposal("preview-p")])
    batch["preview"] = {"path": str(preview), "start_ms": 1000, "end_ms": 5000}
    store.add_run(batch)
    client = TestClient(create_review_app(store))
    client.get("/")
    data = client.get(f"/api/review/{SID}").json()
    assert data["runs"][-1]["preview_range"] == {"start_ms": 1000, "end_ms": 5000}
    assert client.get(f"/api/review/{SID}/preview/preview-batch").content == b"test-preview"
    assert client.get(f"/api/review/{SID}/preview/missing").status_code == 404
    outside = store.path.parent.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    batch = run("outside-batch", [proposal("outside-p")])
    batch["preview"] = {"path": str(outside), "start_ms": 0, "end_ms": 5000}
    store.add_run(batch)
    assert client.get(f"/api/review/{SID}/preview/outside-batch").status_code == 404


def test_cache_is_optional_and_fail_closed(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pingpong_highlight.web,sys; assert 'transformers' not in sys.modules; "
            "assert 'huggingface_hub' not in sys.modules",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    import os

    from pingpong_highlight.model_cache import configure_cache

    if os.name == "nt":
        with pytest.raises(ValueError):
            configure_cache(tmp_path)
