import copy
import json
from pathlib import Path

import pytest

from pingpong_highlight.edit_comparison import (
    encode,
    quantize,
    reel,
    run,
    transition_guard,
    validate,
)


@pytest.fixture
def plan():
    return json.loads(Path("examples/rally-edit-v1.json").read_text())


def test_plan_preserves_candidates_and_human_unknowns(plan):
    before = copy.deepcopy(plan)
    result = validate(plan)
    assert list(result) == ["b2", "v4-3", "v4-6", "b15", "v4-9", "v4-11"]
    assert plan == before
    assert all(c["event"]["human_complete"] is None for c in result.values())
    assert result["v4-11"]["human_original_annotations"][0]["start_ms"] == 140500


def test_event_and_next_point_are_protected(plan):
    plan["candidates"][0]["selected_render"][0] = 28.25
    with pytest.raises(ValueError, match="truncates"):
        validate(plan)
    plan["candidates"][0]["selected_render"][0] = 26.75
    plan["candidates"][4]["selected_render"][1] = 133.875
    with pytest.raises(ValueError, match="next event"):
        validate(plan)


def test_budget_never_truncates_event_to_55_seconds(plan):
    before = copy.deepcopy(plan)
    plan["max_output_seconds"] = 1
    with pytest.raises(ValueError, match="budget"):
        validate(plan)
    assert plan["candidates"] == before["candidates"]


def test_single_variable_and_fixed_identity(plan):
    plan["comparisons"][0]["variants"][1]["range"][1] = 34
    with pytest.raises(ValueError, match="exactly one"):
        validate(plan)
    plan["candidates"].reverse()
    with pytest.raises(ValueError, match="identity"):
        validate(plan)


def test_rhythm_cannot_hide_missing_event(plan):
    plan["comparisons"][-1]["variants"][0]["range"][1] = 150.8
    with pytest.raises(ValueError, match="truncate event"):
        validate(plan)


def test_dissolve_cannot_touch_uncertain_event(plan):
    rows = plan["candidates"]
    assert len(transition_guard(rows, 0.2)) == 5
    rows[0]["selected_render"][1] = rows[0]["event"]["end_uncertainty"][1]
    with pytest.raises(ValueError, match="overlaps"):
        transition_guard(rows, 0.2)


@pytest.mark.parametrize("bounds", [[float("nan"), 1], [0, float("inf")], [2, 1]])
def test_invalid_clock_rejected(bounds):
    with pytest.raises(ValueError):
        quantize(bounds)


def test_outward_frame_rounding_preserves_requested_content():
    a, b = quantize([89.892, 93.644])
    assert 0 <= 89.892 - a < 1 / 30
    assert 0 <= b - 93.644 < 1 / 30


def test_real_media_roundtrip_and_dissolve_av_clock(tmp_path):
    # Explicit synthetic fixture; never a human review decision.
    source = tmp_path / "fixture.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:sample_rate=48000:duration=6",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    first = encode(source, tmp_path / "a.mp4", [0.5, 2.5])
    second = encode(source, tmp_path / "b.mp4", [3, 5])
    rows = [
        {
            "candidate_id": "fixture-a",
            "selected_render": [0.5, 2.5],
            "event": {"start_uncertainty": [1, 1.1], "end_uncertainty": [2, 2.1]},
        },
        {
            "candidate_id": "fixture-b",
            "selected_render": [3, 5],
            "event": {"start_uncertainty": [3.5, 3.6], "end_uncertainty": [4, 4.2]},
        },
    ]
    for overlap, name, expected in [(0, "direct.mp4", 4), (0.2, "dissolve.mp4", 3.8)]:
        result = reel([first, second], rows, tmp_path / name, overlap)
        assert result["duration"] == pytest.approx(expected, abs=0.034)
        assert result["full_av_decode"] == "passed"
        assert abs(float(result["audio_duration"]) - result["duration"]) <= 0.04


def test_aac_padding_cannot_shift_later_cuts_or_truncate_last_frames(tmp_path):
    import runpy

    # Six non-AAC-frame-aligned clips expose accumulated concat padding.
    source = tmp_path / "padding-fixture.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:sample_rate=48000:duration=6",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    clips, rows = [], []
    for i in range(6):
        start = i * 0.7
        bounds = [start, start + 1.1]
        clip = encode(source, tmp_path / f"fixture-{i}.mp4", bounds)
        clip["candidate_id"] = f"fixture-{i}"
        clips.append(clip)
        rows.append(
            {
                "candidate_id": clip["candidate_id"],
                "selected_render": bounds,
                "event": {
                    "start_uncertainty": [start + 0.2, start + 0.3],
                    "end_uncertainty": [start + 0.8, start + 0.9],
                },
            }
        )
    result = reel(clips, rows, tmp_path / "joined.mp4")
    check = runpy.run_path("scripts/verify-edit-media.py")["direct_reel_endpoints"]
    report = check(tmp_path, result, clips)
    assert report["passed"], report
    assert result["decoded_frames"] == sum(c["decoded_frames"] for c in clips)
