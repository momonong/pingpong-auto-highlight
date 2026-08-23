from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pingpong_highlight.candidate_evaluation import CandidateEvaluationError
from pingpong_highlight.candidate_scoring import (
    _aggregate_candidate_burden,
    _burden_gate,
    _measure_candidate_burden,
    score_candidate_run,
)


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
        "precision_eligible": False,
        "precision_ineligible_reason": "No explicit exclude annotations exist.",
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_sha256 = _write_json(dataset_path, dataset)

    run_root = tmp_path / "candidate-run"
    signal_path = run_root / "sources" / "upload-one" / "signals.npz"
    signal_path.parent.mkdir(parents=True)
    signal_path.write_bytes(b"frozen-signals")
    configuration = {
        "algorithm_version": "candidate-generation-test",
        "require_nvdec": True,
    }
    configuration_sha256 = hashlib.sha256(_compact(configuration)).hexdigest()
    artifact = {
        "schema_version": 1,
        "artifact_type": "candidate-generation-source",
        "algorithm_version": "candidate-generation-test",
        "run_id": "formal-test",
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
            "git": {"commit": "a" * 40, "clean": True},
            "gpu": {"nvdec_available": True, "device": "test GPU"},
        },
        "configuration": configuration,
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
    burden = metrics["aggregate"]["candidate_burden"]
    assert burden["candidate_count"] == 1
    assert burden["source_minutes"] == 0.333333
    assert burden["candidates_per_minute"] == 3.0
    assert burden["union_core_coverage"] == 0.25
    assert burden["duplicate_overlap"]["overlapping_pair_count"] == 0
    assert metrics["gate"]["decision"] == "GO_RANKING"
    assert metrics["gate"]["burden_threshold_met"] is True
    assert metrics["definitions"]["precision"] is None
    assert (destination / "metrics.json").is_file()
    assert (destination / "checksums.sha256").is_file()
    report = (destination / "report.md").read_text(encoding="utf-8")
    assert "receipt" in report
    assert "active v2" not in report
    assert metrics["warnings"][0].startswith("All 1 sources")

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


def test_formal_candidate_scoring_fails_closed_on_source_receipt_drift(
    tmp_path: Path,
) -> None:
    dataset, candidate_run = _formal_fixture(tmp_path)
    artifact_path = candidate_run / "sources" / "upload-one" / "candidates.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["receipt"]["git"]["commit"] = "b" * 40
    artifact_sha256 = _write_json(artifact_path, artifact)
    manifest_path = candidate_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["artifact_sha256"] = artifact_sha256
    _write_json(manifest_path, manifest)

    with pytest.raises(CandidateEvaluationError, match="Git receipt"):
        score_candidate_run(
            dataset_path=dataset,
            candidate_run=candidate_run,
            run_id="receipt-drift",
            output_root=tmp_path / "scores",
        )


def test_formal_candidate_scoring_rejects_cpu_generation_receipt(tmp_path: Path) -> None:
    dataset, candidate_run = _formal_fixture(tmp_path)
    artifact_path = candidate_run / "sources" / "upload-one" / "candidates.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["receipt"]["gpu"] = {"nvdec_available": False, "device": None}
    artifact_sha256 = _write_json(artifact_path, artifact)
    manifest_path = candidate_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gpu"] = {"nvdec_available": False, "device": None}
    manifest["sources"][0]["artifact_sha256"] = artifact_sha256
    _write_json(manifest_path, manifest)

    with pytest.raises(CandidateEvaluationError, match="GPU/NVDEC"):
        score_candidate_run(
            dataset_path=dataset,
            candidate_run=candidate_run,
            run_id="cpu-receipt",
            output_root=tmp_path / "scores",
        )


def test_formal_candidate_scoring_v1_rejects_non_development_split(
    tmp_path: Path,
) -> None:
    dataset_path, candidate_run = _formal_fixture(tmp_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["sources"][0]["split"] = "held-out"
    core = {
        "schema_version": dataset["schema_version"],
        "interval_contract": dataset["interval_contract"],
        "sources": dataset["sources"],
    }
    dataset["annotation_snapshot_sha256"] = hashlib.sha256(_compact(core)).hexdigest()
    _write_json(dataset_path, dataset)

    with pytest.raises(CandidateEvaluationError, match="development sources only"):
        score_candidate_run(
            dataset_path=dataset_path,
            candidate_run=candidate_run,
            run_id="held-out-v1",
            output_root=tmp_path / "scores",
        )


def test_formal_candidate_scoring_v1_rejects_non_positive_label(tmp_path: Path) -> None:
    dataset_path, candidate_run = _formal_fixture(tmp_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["sources"][0]["annotations"][0]["label"] = "exclude"
    core = {
        "schema_version": dataset["schema_version"],
        "interval_contract": dataset["interval_contract"],
        "sources": dataset["sources"],
    }
    dataset["annotation_snapshot_sha256"] = hashlib.sha256(_compact(core)).hexdigest()
    _write_json(dataset_path, dataset)

    with pytest.raises(CandidateEvaluationError, match="positive-only"):
        score_candidate_run(
            dataset_path=dataset_path,
            candidate_run=candidate_run,
            run_id="exclude-v1",
            output_root=tmp_path / "scores",
        )


def test_candidate_burden_uses_half_open_union_and_overlap_excess() -> None:
    candidates = [
        {"start_ms": 0, "end_ms": 1000},
        {"start_ms": 500, "end_ms": 1500},
        {"start_ms": 1500, "end_ms": 2000},
    ]

    burden, raw = _measure_candidate_burden(candidates, duration_us=4_000_000)

    assert burden["total_core_seconds"] == 2.5
    assert burden["union_core_seconds"] == 2.0
    assert burden["union_core_coverage"] == 0.5
    assert burden["duplicate_overlap"] == {
        "overlapping_pair_count": 1,
        "overlap_excess_seconds": 0.5,
        "overlap_excess_fraction_of_total_core": 0.2,
    }
    assert raw["overlap_excess_ms"] == 500


def test_candidate_burden_counts_triple_overlap_without_double_counting_excess() -> None:
    burden, _raw = _measure_candidate_burden(
        [
            {"start_ms": 0, "end_ms": 3000},
            {"start_ms": 1000, "end_ms": 2500},
            {"start_ms": 2000, "end_ms": 4000},
        ],
        duration_us=5_000_000,
    )

    assert burden["union_core_seconds"] == 4.0
    assert burden["duplicate_overlap"]["overlapping_pair_count"] == 3
    assert burden["duplicate_overlap"]["overlap_excess_seconds"] == 2.5


def test_aggregate_burden_is_duration_weighted() -> None:
    _short_display, short = _measure_candidate_burden(
        [{"start_ms": 0, "end_ms": 500}],
        duration_us=1_000_000,
    )
    _long_display, long = _measure_candidate_burden(
        [{"start_ms": 0, "end_ms": 900}],
        duration_us=9_000_000,
    )

    aggregate, _raw = _aggregate_candidate_burden([short, long])

    assert aggregate["union_core_coverage"] == 0.14


def test_burden_gate_accepts_exact_limits_and_rejects_epsilon_over() -> None:
    aggregate = {
        "candidates_per_minute": 6.0,
        "union_core_coverage": 0.5,
        "max_core_ms": 20_000,
        "overlap_excess_ms": 0,
        "overlapping_pair_count": 0,
    }
    source = {
        "candidates_per_minute": 8.0,
        "union_core_coverage": 0.75,
    }

    assert _burden_gate(aggregate, [("source", source)])["threshold_met"] is True

    source_over = source | {"union_core_coverage": 0.75000001}
    assert _burden_gate(aggregate, [("source", source_over)])["threshold_met"] is False


def _replace_fixture_candidates(
    candidate_run: Path,
    candidates: list[dict[str, float | str]],
) -> None:
    artifact_path = candidate_run / "sources" / "upload-one" / "candidates.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["candidates"] = candidates
    artifact_sha256 = _write_json(artifact_path, artifact)
    manifest_path = candidate_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["artifact_sha256"] = artifact_sha256
    manifest["sources"][0]["candidate_count"] = len(candidates)
    _write_json(manifest_path, manifest)


def test_recall_pass_with_overlap_stops_candidate_burden(tmp_path: Path) -> None:
    dataset, candidate_run = _formal_fixture(tmp_path)
    candidate = {
        "rally_start": 5.0,
        "rally_end": 10.0,
        "score": 12.0,
        "selection": "selected",
    }
    _replace_fixture_candidates(candidate_run, [candidate, candidate | {"score": 11.0}])

    _destination, metrics = score_candidate_run(
        dataset_path=dataset,
        candidate_run=candidate_run,
        run_id="burden-stop",
        output_root=tmp_path / "scores",
    )

    assert metrics["gate"]["recall_threshold_met"] is True
    assert metrics["gate"]["burden_threshold_met"] is False
    assert metrics["gate"]["decision"] == "STOP_CANDIDATE_BURDEN"
    assert metrics["gate"]["ranker_authorized"] is False
    assert metrics["definitions"]["precision_status"] == (
        "abstained_missing_explicit_negatives"
    )


def test_recall_failure_takes_priority_over_burden_failure(tmp_path: Path) -> None:
    dataset, candidate_run = _formal_fixture(tmp_path)
    candidate = {
        "rally_start": 15.0,
        "rally_end": 17.0,
        "score": 12.0,
        "selection": "selected",
    }
    _replace_fixture_candidates(candidate_run, [candidate, candidate | {"score": 11.0}])

    _destination, metrics = score_candidate_run(
        dataset_path=dataset,
        candidate_run=candidate_run,
        run_id="detector-stop",
        output_root=tmp_path / "scores",
    )

    assert metrics["gate"]["recall_threshold_met"] is False
    assert metrics["gate"]["burden_threshold_met"] is False
    assert metrics["gate"]["decision"] == "STOP_DETECTOR"
