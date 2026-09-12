from __future__ import annotations

import shutil
import subprocess

import pytest

from pingpong_highlight.pipeline.media import probe_media
from pingpong_highlight.preannotate import extract


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
def test_vfr_rotation_and_source_offset(tmp_path):
    original = tmp_path / "vfr.mp4"
    rotated = tmp_path / "rotated.mp4"
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=12:duration=5",
            "-vf",
            "select='not(eq(mod(n,3),1))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(original),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(original),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
        capture_output=True,
    )
    assert probe_media(rotated).rotation == 90
    command = extract(rotated, clip, 1000, 3000, fps=2, width=96)
    info = probe_media(clip)
    assert (info.width, info.height, info.rotation) == (96, 160, 0)
    assert abs(info.duration - 2) < 0.05
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(clip),
        ],
        text=True,
    )
    # Frame CSV avoids depending on optional SEI side-data JSON formatting.
    pts = [float(line.split(",")[0]) for line in output.splitlines() if line.strip()]
    assert pts == [0, 0.5, 1, 1.5]
    assert [round(t * 1000) + 1000 for t in pts] == [1000, 1500, 2000, 2500]
    assert command[command.index("-ss") + 1] == "1.0"
    with pytest.raises(FileExistsError):
        extract(rotated, clip, 1000, 3000)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
def test_full_review_proxy_ignores_model_scope_and_preserves_humans(tmp_path):
    from fastapi.testclient import TestClient

    from pingpong_highlight.rally_review import (
        Judgment,
        ReviewCommand,
        ReviewStore,
        source_identity,
    )
    from pingpong_highlight.review_media import full_preview, prepare_review
    from pingpong_highlight.review_web import create_review_app

    original = tmp_path / "original.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=10:duration=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p10le",
            str(original),
        ],
        check=True,
        capture_output=True,
    )
    store = ReviewStore(tmp_path / "experiment" / "review.sqlite3")
    source = source_identity(original)
    source.update(duration_ms=5000, group="development", scope=[{"start_ms": 2000, "end_ms": 3000}])
    store.register(source)
    store.apply(
        source["id"],
        ReviewCommand(
            revision=0,
            request_id="existing-human",
            action="save",
            point=Judgment(start_ms=1000, end_ms=2000, rally="yes", complete="yes"),
        ),
        actor="test-human",
    )
    before = store.path.read_bytes()
    spec = prepare_review(store, source["id"])
    assert spec["start_ms"] == 0 and spec["end_ms"] == 5000
    assert spec["media"]["pixel_format"] == "yuv420p"
    assert spec["media"]["duration"] == pytest.approx(5, abs=0.1)
    assert prepare_review(store, source["id"]) == spec  # no duplicate transcode
    assert full_preview(store, source["id"]) == spec
    assert store.path.read_bytes() == before
    assert source_identity(original)["id"] == source["id"]
    client = TestClient(create_review_app(store))
    client.get("/")
    assert client.get(f"/api/review/{source['id']}").json()["full_preview_available"]
    response = client.get(
        f"/api/review/{source['id']}/full-preview", headers={"Range": "bytes=0-15"}
    )
    assert response.status_code == 206 and len(response.content) == 16
    assert client.post(f"/api/review/{source['id']}/full-preview").status_code == 200
    assert store.export(source["id"])["review"]["revision"] == 1
