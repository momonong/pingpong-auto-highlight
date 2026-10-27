"""Decision transport fixtures are not human acceptance evidence."""

import copy
import json
from pathlib import Path

import pytest

from pingpong_highlight.edit_comparison import sha
from pingpong_highlight.edit_decisions import apply_decisions, load_reviewed_clip


@pytest.fixture
def inputs():
    plan = json.loads(Path("examples/rally-edit-v1.json").read_text())
    decisions = json.loads(Path("examples/rally-edit-decisions-v2.json").read_text())
    return plan, decisions


def test_single_accepted_clip_preserves_identity_events_and_unreviewed_rows(inputs):
    plan, decisions = inputs
    original = copy.deepcopy(plan)
    result = apply_decisions(plan, decisions["parent_plan_sha256"], decisions)
    assert plan == original
    assert result["frozen_order"] == original["frozen_order"]
    assert result["candidates"][1:] == original["candidates"][1:]
    assert result["candidates"][0]["event"] == original["candidates"][0]["event"]
    assert (
        result["candidates"][0]["original_candidate_range"]
        == original["candidates"][0]["original_candidate_range"]
    )
    assert result["candidates"][0]["selected_render"] == [26.75, 32.7]
    assert result["candidates"][0]["human_acceptable_render"]["out"] is None
    assert result["candidates"][0]["event"]["human_complete"] is None


@pytest.mark.parametrize(
    "field,value", [("parent_plan_sha256", "stale"), ("source_sha256", "wrong")]
)
def test_stale_or_wrong_source_decisions_rejected(inputs, field, value):
    plan, decisions = inputs
    parent = decisions["parent_plan_sha256"]
    decisions[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        apply_decisions(plan, parent, decisions)


def test_duplicate_or_nonhuman_decisions_rejected(inputs):
    plan, decisions = inputs
    parent = decisions["parent_plan_sha256"]
    decisions["decisions"].append(copy.deepcopy(decisions["decisions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        apply_decisions(plan, parent, decisions)
    decisions["decisions"].pop()
    decisions["decisions"][0]["actor"] = "automation-fixture"
    with pytest.raises(ValueError, match="human acceptance"):
        apply_decisions(plan, parent, decisions)


def test_accepted_pacing_cannot_override_event_envelope(inputs):
    plan, decisions = inputs
    d = decisions["decisions"][0]
    d["requested_source_range"] = [28, 32.7]
    d["render_source_range"] = [28, 32.7]
    with pytest.raises(ValueError, match="truncates"):
        apply_decisions(plan, decisions["parent_plan_sha256"], decisions)


def test_exact_reviewed_artifact_and_feedback_are_required(tmp_path):
    # All metadata here is synthetic, not a real user's decision.
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"synthetic artifact for binding tests")
    digest = sha(video)
    bounds = [1, 3]
    decision = {
        "candidate_id": "fixture",
        "feedback_record": "feedback.json",
        "render_manifest": "manifest.json",
        "media_sha256": digest,
        "requested_source_range": bounds,
        "render_source_range": bounds,
    }
    media = {
        "file": "fixture.mp4",
        "sha256": digest,
        "requested_source_range": bounds,
        "render_source_range": bounds,
    }
    feedback = {
        "actor": "human-user",
        "channel": "user-chat",
        "candidate_id": "fixture",
        "plan_sha256": "parent",
        "source_sha256": "source",
        "presentation_accepted": True,
        "verbatim": "SYNTHETIC TEST, never human evidence",
        "media": media,
    }
    manifest = {
        "candidate_id": "fixture",
        "source_sha256": "source",
        "parent_plan_sha256": "parent",
        "actual_render": media,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "feedback.json").write_text(json.dumps(feedback))
    path, _, receipt = load_reviewed_clip(decision, tmp_path, "parent", "source")
    assert path == video and receipt["feedback_sha256"]
    feedback["media"]["render_source_range"] = [1, 4]
    (tmp_path / "feedback.json").write_text(json.dumps(feedback))
    with pytest.raises(ValueError, match="Feedback range"):
        load_reviewed_clip(decision, tmp_path, "parent", "source")
    feedback["media"]["render_source_range"] = bounds
    (tmp_path / "feedback.json").write_text(json.dumps(feedback))
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_reviewed_clip(decision, tmp_path, "parent", "source")
