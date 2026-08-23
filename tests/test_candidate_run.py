from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import pingpong_highlight.candidate_run as candidate_run_module
from pingpong_highlight.candidate_run import CandidateRunError, run_candidate_analysis
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database
from pingpong_highlight.pipeline.models import (
    AudioFeatures,
    ImpactEvent,
    MediaInfo,
    MotionFeatures,
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _dataset(settings: Settings, source_count: int = 1) -> tuple[Path, list[Path]]:
    database = Database(settings.database_path)
    sources: list[dict[str, object]] = []
    paths: list[Path] = []
    for index in range(source_count):
        upload_id = f"upload-{index}"
        filename = f"PXL_2026010{index + 1}_120000000.mp4"
        source_path = settings.uploads_dir / f"{upload_id}.mp4"
        source_path.write_bytes(f"video-{index}".encode())
        _upload, job = database.register_completed_upload(
            upload_id,
            filename,
            source_path.stat().st_size,
            "video/mp4",
            source_path,
        )
        assert database.claim_job(job.id)
        database.finish_job(
            job.id,
            {
                "algorithm_version": "test",
                "media": {"duration": 10.0},
                "summary": {"point_count": 0},
                "candidates": [],
                "points": [],
                "files": [{"name": "analysis.json", "kind": "analysis"}],
            },
        )
        with sqlite3.connect(settings.database_path) as connection:
            connection.execute(
                "UPDATE uploads SET path = ? WHERE id = ?",
                (f"/data/uploads/{source_path.name}", upload_id),
            )
        sources.append(
            {
                "upload_id": upload_id,
                "job_id": job.id,
                "filename": filename,
                "session_id": f"session-{index}",
                "split": "development",
                "recorded_at": f"2026-01-0{index + 1}",
                "duration_us": 10_000_000,
                "byte_size": source_path.stat().st_size,
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "review": {"review_complete": True},
                "annotations": [],
            }
        )
        paths.append(source_path)
    core = {
        "schema_version": 1,
        "interval_contract": {
            "unit": "integer-millisecond",
            "semantics": "half-open [start_ms, end_ms)",
        },
        "sources": sources,
    }
    dataset = core | {
        "created_at": "2026-08-24T00:00:00+00:00",
        "annotation_snapshot_sha256": hashlib.sha256(_json_bytes(core)).hexdigest(),
    }
    path = settings.data_dir / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path, paths


def _patch_analysis(monkeypatch, *, fail_name: str | None = None) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(
        candidate_run_module,
        "_git_receipt",
        lambda _path: {
            "repository_root_name": "test",
            "commit": "a" * 40,
            "clean": True,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )
    monkeypatch.setattr(
        candidate_run_module,
        "_gpu_receipt",
        lambda: {"nvdec_available": True, "device": "test GPU", "driver": "test"},
    )

    def fake_probe(path: Path) -> MediaInfo:
        return MediaInfo(
            path=path,
            duration=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            video_codec="h264",
            has_audio=True,
            audio_codec="aac",
        )

    def fake_audio(
        path: Path,
        _media: MediaInfo,
        *,
        sample_rate: int,
        progress,
    ) -> AudioFeatures:
        assert sample_rate == 16_000
        calls.append(f"audio:{path.name}")
        progress(1.0)
        return AudioFeatures(
            times=np.array([1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]),
            scores=np.array([0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0]),
            events=[
                ImpactEvent(time=2.0, strength=1.0),
                ImpactEvent(time=2.5, strength=1.0),
                ImpactEvent(time=3.0, strength=1.0),
            ],
        )

    def fake_motion(
        path: Path,
        _media: MediaInfo,
        *,
        fps: float,
        frame_size: int,
        progress,
        require_nvdec: bool,
    ) -> MotionFeatures:
        calls.append(f"motion:{path.name}")
        if path.name == fail_name:
            raise RuntimeError("interrupted test run")
        assert fps == 8.0
        assert frame_size == 320
        assert require_nvdec is True
        progress(1.0)
        return MotionFeatures(
            times=np.arange(0, 10, 0.125),
            scores=np.ones(80),
        )

    monkeypatch.setattr(candidate_run_module, "probe_media", fake_probe)
    monkeypatch.setattr(candidate_run_module, "analyze_audio", fake_audio)
    monkeypatch.setattr(candidate_run_module, "analyze_motion", fake_motion)
    return calls


def test_candidate_only_run_persists_receipt_signals_and_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", upload_token="test")
    settings.ensure_directories()
    dataset, _paths = _dataset(settings)
    calls = _patch_analysis(monkeypatch)

    destination = run_candidate_analysis(
        settings,
        dataset_path=dataset,
        run_id="candidate-test",
        output_root=tmp_path / "runs",
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["sources"][0]
    artifact = json.loads((destination / descriptor["artifact"]).read_text(encoding="utf-8"))
    signals = np.load(destination / descriptor["signals"])
    assert manifest["generation_receipt_valid"] is True
    assert manifest["gpu"]["nvdec_available"] is True
    assert artifact["summary"]["candidate_count"] == 1
    assert artifact["audio_groups"][0]["decision"] == "candidate"
    assert artifact["candidates"][0]["score_components"]
    assert signals["audio_scores"].tolist() == [0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0]
    assert calls == ["audio:upload-0.mp4", "motion:upload-0.mp4"]
    assert list(settings.outputs_dir.rglob("*.mp4")) == []
    assert not (tmp_path / "runs" / ".candidate-test.partial").exists()


def test_candidate_run_resumes_completed_sources_after_interruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", upload_token="test")
    settings.ensure_directories()
    dataset, paths = _dataset(settings, source_count=2)
    first_calls = _patch_analysis(monkeypatch, fail_name=paths[1].name)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_candidate_analysis(
            settings,
            dataset_path=dataset,
            run_id="resume-test",
            output_root=tmp_path / "runs",
        )
    assert first_calls.count(f"audio:{paths[0].name}") == 1
    assert (tmp_path / "runs" / ".resume-test.partial").is_dir()

    second_calls = _patch_analysis(monkeypatch)
    destination = run_candidate_analysis(
        settings,
        dataset_path=dataset,
        run_id="resume-test",
        output_root=tmp_path / "runs",
    )

    assert f"audio:{paths[0].name}" not in second_calls
    assert f"motion:{paths[0].name}" not in second_calls
    assert f"audio:{paths[1].name}" in second_calls
    assert destination.is_dir()


def test_candidate_run_rejects_dirty_worktree_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", upload_token="test")
    settings.ensure_directories()
    dataset, _paths = _dataset(settings)
    monkeypatch.setattr(
        candidate_run_module,
        "_git_receipt",
        lambda _path: {
            "repository_root_name": "test",
            "commit": "a" * 40,
            "clean": False,
            "status_sha256": "b" * 64,
        },
    )

    with pytest.raises(CandidateRunError, match="clean worktree"):
        run_candidate_analysis(
            settings,
            dataset_path=dataset,
            run_id="dirty-test",
            output_root=tmp_path / "runs",
        )


def test_resume_invariant_binds_gpu_execution_receipt() -> None:
    common = {
        "dataset_sha256": "a" * 64,
        "annotation_snapshot_sha256": "b" * 64,
        "configuration_sha256": "c" * 64,
        "git_receipt": {
            "commit": "d" * 40,
            "status_sha256": "e" * 64,
        },
    }

    first = candidate_run_module._state_invariant(
        **common,
        gpu_receipt={"nvdec_available": True, "device": "GPU A", "driver": "1"},
    )
    second = candidate_run_module._state_invariant(
        **common,
        gpu_receipt={"nvdec_available": True, "device": "GPU B", "driver": "2"},
    )

    assert first != second
