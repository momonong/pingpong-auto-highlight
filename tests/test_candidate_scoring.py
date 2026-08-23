from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pingpong_highlight.candidate_evaluation import CandidateEvaluationError
from pingpong_highlight.candidate_scoring import score_candidate_run


def _compact(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _formal_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_sha256 = "1" * 64
    source = {
        "upload_id": "upload-one",
        "job_id": "job-one",
        "filename": "match.mp4",
        "session_id": "session-one",
        "split": "development",
        "recorded_at": "2026-01-01",
        "duration_us": 20_000_000,
        "byte_size": 123,
        "source_sha256": source_sha256,
        "review": {"review_complete": True},
        "annotations": [
            {
                "id": "highlight-one",
                "upload_id": "upload-one",
                "label": "highlight",
                "start_ms": 0,
                "end_ms": 10_000,
                "note": "test",
                "created_at": "now",
                "updated_at": "now",
            }
        ],
    }
    dataset_core = {
        "schema_version": 1,
        "interval_contract": {
            "unit": "integer-millisecond",
            "semantics": "half-open [start_ms, end_ms)",
        },
        "sources": [source],
    }
    dataset = dataset_core | {
        "created_at": "2026-08-24T00:00:00+00:00",
        "annotation_snapshot_sha256": hashlib.sha256(_compact(dataset_core)).hexdigest(),
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_sha256 = _write_json(dataset_path, dataset)

    run_root = tmp_path / "candidate-run"
    signal_path = run_root / "sources" / "upload-one" / "signals.npz"
    signal_path.parent.mkdir(parents=True)
    signal_path.write_bytes(b"frozen-signals")
    configuration = {"algorithm_version": "candidate-generation-test"}
    configuration_sha256 = hashlib.sha256(_compact(configuration)).hexdigest()
    artifact = {
        "schema_version": 1,
        "artifact_type": "candidate-generation-source",
        "algorithm_version": "candidate-generation-test",
        "source": {
            "upload_id": "upload-one",
            "job_id": "job-one",
            "filename": "match.mp4",
            "byte_size": 123,
            "source_sha256": source_sha256,
            "duration": 20.0,
        },
        "receipt": {
            "dataset_sha256": dataset_sha256,
            "annotation_snapshot_sha256": dataset["annotation_snapshot_sha256"],
            "configuration_sha256": configuration_sha256,
        },
        "candidates": [
            {
                "rally_start": 5.0,
                "rally_end": 10.0,
                "score": 12.0,
                "selection": "selected",
            }
        ],
    }
    artifact_path = signal_path.with_name("candidates.json")
    artifact_sha256 = _write_json(artifact_path, artifact)
    manifest = {
        "schema_version": 1,
        "artifact_type": "candidate-generation-run",
        "algorithm_version": "candidate-generation-test",
        "run_id": "formal-test",
        "status": "completed",
        "dataset": {
            "sha256": dataset_sha256,
            "annotation_snapshot_sha256": dataset["annotation_snapshot_sha256"],
        },
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "git": {"commit": "a" * 40, "clean": True},
        "gpu": {"nvdec_available": True, "device": "test GPU"},
        "generation_receipt_valid": True,
        "sources": [
            {
                "upload_id": "upload-one",
                "artifact": artifact_path.relative_to(run_root).as_posix(),
                "artifact_sha256": artifact_sha256,
                "signals": signal_path.relative_to(run_root).as_posix(),
                "signals_sha256": hashlib.sha256(signal_path.read_bytes()).hexdigest(),
                "candidate_count": 1,
            }
        ],
    }
    _write_json(run_root / "manifest.json", manifest)
    return dataset_path, run_root


def test_formal_candidate_scoring_validates_receipts_and_writes_immutable_report(
    tmp_path: Path,
) -> None:
    dataset, candidate_run = _formal_fixture(tmp_path)

    destination, metrics = score_candidate_run(
        dataset_path=dataset,
        candidate_run=candidate_run,
        run_id="scored-test",
        output_root=tmp_path / "scores",
    )

    assert metrics["provenance"]["evidence_valid"] is True
    assert metrics["aggregate"]["strict_candidate_recall"]["micro_recall"] == 1.0
    assert metrics["aggregate"]["candidate_burden"] == {
        "candidate_count": 1,
        "source_minutes": 0.333333,
        "candidates_per_minute": 3.0,
    }
    assert metrics["gate"]["decision"] == "GO_RANKING"
    assert metrics["definitions"]["precision"] is None
    assert (destination / "metrics.json").is_file()
    assert (destination / "checksums.sha256").is_file()

    with pytest.raises(CandidateEvaluationError, match="already exists"):
        score_candidate_run(
            dataset_path=dataset,
            candidate_run=candidate_run,
            run_id="scored-test",
            output_root=tmp_path / "scores",
        )


def test_formal_candidate_scoring_fails_closed_on_signal_drift(tmp_path: Path) -> None:
    dataset, candidate_run = _formal_fixture(tmp_path)
    signal = candidate_run / "sources" / "upload-one" / "signals.npz"
    signal.write_bytes(b"tampered")

    with pytest.raises(CandidateEvaluationError, match="Signal artifact checksum"):
        score_candidate_run(
            dataset_path=dataset,
            candidate_run=candidate_run,
            run_id="tampered-test",
            output_root=tmp_path / "scores",
        )
