"""Exact-byte plan binding checks; all media and decisions here are synthetic fixtures."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from pingpong_highlight import edit_comparison
from pingpong_highlight.edit_decisions import apply_decisions
from pingpong_highlight.edit_review import create_app


@pytest.fixture
def generated_pack(tmp_path, monkeypatch):
    source = tmp_path / "source.fixture"
    source.write_bytes(b"SYNTHETIC TEST SOURCE, not audiovisual content")
    plan = json.loads(Path("examples/rally-edit-v1.json").read_text())
    plan["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    plan["presentation_hypothesis"] = "SYNTHETIC TEST: Unicode 排版與原始 bytes 綁定"
    for row in plan["candidates"]:
        row["source_sha256"] = plan["source_sha256"]

    def fixture_media(target, duration):
        # Plain bytes stand in for external codecs; never claim a successful decode.
        target.write_bytes(b"SYNTHETIC TEST PLACEHOLDER, not an MP4")
        return {
            "file": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "duration": duration,
            "audio_duration": str(duration),
            "video_start": "0",
            "audio_start": "0",
            "width": 640,
            "height": 360,
            "decoded_frames": 0,
            "full_av_decode": "not-run-synthetic-fixture",
            "command": [],
        }

    def encode_fixture(source, target, bounds):
        start, end = edit_comparison.quantize(bounds)
        return fixture_media(target, end - start) | {
            "requested_source_range": bounds,
            "render_source_range": [start, end],
            "wall_seconds": 0,
        }

    def reel_fixture(clips, rows, target, overlap=0):
        return fixture_media(target, 1) | {
            "timeline": [],
            "overlap_seconds": overlap,
            "transition_guards": [],
            "budget_overrun_seconds": 0,
        }

    monkeypatch.setattr(edit_comparison, "encode", encode_fixture)
    monkeypatch.setattr(edit_comparison, "reel", reel_fixture)

    def generate(layout):
        if layout == "compact":
            raw = json.dumps(plan, separators=(",", ":")).encode()
        else:
            indent = "\t" if layout == "unicode-tabs" else 2
            raw = (json.dumps(plan, ensure_ascii=False, indent=indent) + "\n").encode()
            if layout == "crlf":
                raw = raw.replace(b"\n", b"\r\n")
        input_plan = tmp_path / "input-plan.json"
        input_plan.write_bytes(raw)
        pack = tmp_path / "pack"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "edit-comparison", "--plan", str(input_plan), "--source", str(source),
                "--output", str(pack),
            ],
        )
        edit_comparison.main()
        decisions = json.loads(Path("examples/rally-edit-decisions-v2.json").read_text())
        decisions["parent_plan_sha256"] = hashlib.sha256(raw).hexdigest()
        decisions["source_sha256"] = plan["source_sha256"]
        return pack, raw, decisions

    return generate


@pytest.mark.parametrize("layout", ["canonical", "compact", "crlf", "unicode-tabs"])
def test_generated_plan_bytes_remain_reviewable_and_decision_bound(
    generated_pack, tmp_path, layout
):
    pack, original_bytes, decisions = generated_pack(layout)
    feedback = tmp_path / "synthetic-feedback.json"
    create_app(pack, feedback, fixture=True)
    expected_sha = hashlib.sha256(original_bytes).hexdigest()
    assert (pack / "plan.json").read_bytes() == original_bytes
    assert json.loads((pack / "packet.json").read_text())["plan_sha256"] == expected_sha
    assert json.loads(feedback.read_text())["plan_sha256"] == expected_sha
    saved_plan = json.loads((pack / "plan.json").read_bytes())
    updated = apply_decisions(saved_plan, expected_sha, decisions)
    assert updated["parent_plan_sha256"] == expected_sha
    assert updated["candidates"][0]["selected_render"] == [26.75, 32.7]
    assert updated["candidates"][1:] == saved_plan["candidates"][1:]


def test_changed_plan_still_fails_review_and_decision_binding(generated_pack, tmp_path):
    pack, _, decisions = generated_pack("canonical")
    plan_path = pack / "plan.json"
    changed = json.loads(plan_path.read_bytes())
    changed["blind_seed"] += 1
    changed_bytes = json.dumps(changed).encode()
    plan_path.write_bytes(changed_bytes)
    with pytest.raises(ValueError, match="Plan checksum mismatch"):
        create_app(pack, tmp_path / "synthetic-feedback.json", fixture=True)
    with pytest.raises(ValueError, match="Decision parent plan mismatch"):
        apply_decisions(changed, hashlib.sha256(changed_bytes).hexdigest(), decisions)
