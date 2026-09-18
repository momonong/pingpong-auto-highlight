"""Transport and persistence checks only; synthetic ratings are never human evidence."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pingpong_highlight.edit_comparison import sha
from pingpong_highlight.edit_review import create_app


@pytest.fixture
def review(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "plan.json").write_text(json.dumps({"source_sha256": "source-fixture"}))
    (pack / "fixture.mp4").write_bytes(b"0123456789")
    row = {"file": "fixture.mp4", "sha256": sha(pack / "fixture.mp4")}
    packet = {
        "plan_sha256": sha(pack / "plan.json"),
        "pairs": [{"media": [row | {"token": "306"}, row | {"token": "778"}]}],
        "contexts": [row],
        "selected_clips": [row],
        "reels": [row | {"token": x} for x in ["K", "M", "N"]],
    }
    (pack / "packet.json").write_text(json.dumps(packet))
    return pack, tmp_path / "fixture-feedback.json"


def client(review):
    c = TestClient(create_app(*review, fixture=True))
    assert c.get("/").status_code == 200
    return c


def entry(token="306", field="complete", value="不完整", id="fixture-0001"):
    return dict(id=id, token=token, field=field, value=value, presentation_sequence="0")


def test_range_and_loopback_guards(review):
    c = TestClient(create_app(*review, fixture=True))
    assert c.get("/api/feedback").status_code == 403
    assert c.get("/", headers={"host": "foreign.example"}).status_code == 403
    c.get("/")
    assert (
        c.post(
            "/api/feedback", json=entry(), headers={"origin": "https://foreign.example"}
        ).status_code
        == 403
    )
    r = c.get("/media/306", headers={"range": "bytes=2-5"})
    assert r.status_code == 206 and r.content == b"2345"
    assert r.headers["content-range"] == "bytes 2-5/10"
    assert c.get("/media/method-key.json").status_code == 404
    assert c.get("/media/306", headers={"range": "bytes=50-60"}).status_code == 416


def test_append_save_reload_retry_export_and_actor(review):
    c = client(review)
    first = entry()
    assert c.post("/api/feedback", json=first).json()["revision"] == 1
    assert c.post("/api/feedback", json=first).json()["revision"] == 1
    assert c.post("/api/feedback", json=first | {"value": "完整"}).status_code == 409
    c.post("/api/feedback", json=entry(field="rhythm", value="合適", id="fixture-0002"))
    c.post(
        "/api/feedback",
        json=entry(token="review", field="presentation_order", value="1", id="fixture-0003"),
    )
    c = client(review)  # A new server instance and browser session.
    state = c.get("/api/export").json()
    assert state["revision"] == 3
    assert [r["value"] for r in state["records"]] == ["不完整", "合適", "1"]
    assert state["actor"] == "automation-fixture"
    assert state["human_status"] == "fixture-only"
    assert state["plan_sha256"] and state["media"]["306"]
    assert json.loads(review[1].read_text())["records"] == state["records"]
    with pytest.raises(ValueError, match="actor mismatch"):
        create_app(*review, fixture=False)


def test_gate_and_invalid_input(review):
    c = client(review)
    rating = entry(token="reel", field="transition", value="M")
    assert c.post("/api/feedback", json=rating).status_code == 409
    assert c.post("/api/feedback", json=entry(value="model-confirmed")).status_code == 422
    assert c.post("/api/feedback", json=entry(token="missing")).status_code == 422
    assert c.post("/api/feedback", json=entry() | {"actor": "human"}).status_code == 422
    c.post("/api/feedback", json=entry(token="point-1", value="完整", id="fixture-gate"))
    assert c.post("/api/feedback", json=rating).json()["compilation_eligible"] is True
    state = c.post(
        "/api/feedback", json=entry(token="point-1", value="無法判定", id="fixture-reset")
    ).json()
    assert state["compilation_eligible"] is False
    assert len(state["records"]) == 3  # Historical transition preference is retained, not erased.


def test_protect_pack_and_binding(review):
    with pytest.raises(ValueError, match="outside"):
        create_app(review[0], review[0] / "human-feedback.json")
    (review[0] / "fixture.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        create_app(*review)


def test_page_uses_one_player():
    page = Path("src/pingpong_highlight/static/edit-review/index.html").read_text()
    assert page.count("<video ") == 1


def test_failed_disk_save_keeps_last_success_and_allows_retry(review, monkeypatch):
    c = TestClient(create_app(*review, fixture=True), raise_server_exceptions=False)
    c.get("/")
    original = Path.replace

    def unavailable(self, target):
        raise OSError("synthetic disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", unavailable)
        assert c.post("/api/feedback", json=entry()).status_code == 500
    assert c.get("/api/feedback").json()["revision"] == 0
    assert json.loads(review[1].read_text())["records"] == []
    assert Path.replace is original
    assert c.post("/api/feedback", json=entry()).json()["revision"] == 1


def test_distinct_feedback_files_have_independent_cookie_names(review):
    one = client(review)
    two = client((review[0], review[1].with_name("separate-fixture.json")))
    assert set(one.cookies.keys()).isdisjoint(two.cookies.keys())
