from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pingpong_highlight.candidate_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    CandidateEvaluationError,
    _aggregate_metrics,
    _json_bytes,
    _metric_source,
    _report_markdown,
)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CandidateEvaluationError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    try:
        payload = resolved.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateEvaluationError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateEvaluationError(f"{label} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_dataset(path: Path) -> tuple[dict[str, Any], str]:
    dataset, file_sha256 = _load_json(path, "dataset manifest")
    if dataset.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise CandidateEvaluationError("Unsupported dataset schema version")
    if not isinstance(dataset.get("sources"), list):
        raise CandidateEvaluationError("Dataset has no sources")
    core = {
        "schema_version": dataset.get("schema_version"),
        "interval_contract": dataset.get("interval_contract"),
        "sources": dataset.get("sources"),
    }
    if dataset.get("annotation_snapshot_sha256") != hashlib.sha256(_json_bytes(core)).hexdigest():
        raise CandidateEvaluationError("Dataset annotation snapshot checksum is invalid")
    return dataset, file_sha256


def _safe_child(root: Path, relative_value: str, label: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CandidateEvaluationError(f"{label} path escapes the candidate run")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise CandidateEvaluationError(f"{label} file is missing")
    return path


def _normalize_candidates(
    upload_id: str,
    rows: Any,
    *,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise CandidateEvaluationError("Candidate source artifact has no candidate array")
    normalized: list[dict[str, Any]] = []
    for original_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CandidateEvaluationError("Candidate entries must be JSON objects")
        try:
            start = float(row["rally_start"])
            end = float(row["rally_end"])
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CandidateEvaluationError("Candidate interval or score is invalid") from exc
        if not all(math.isfinite(value) for value in (start, end, score)):
            raise CandidateEvaluationError("Candidate interval or score is not finite")
        if start < 0 or end <= start or end > duration_seconds + 0.001:
            raise CandidateEvaluationError("Candidate interval is outside its source")
        normalized.append(
            {
                "original_index": original_index,
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "score": score,
                "selection": str(row.get("selection") or "candidate"),
            }
        )
    ranked = sorted(
        range(len(normalized)),
        key=lambda index: (
            -normalized[index]["score"],
            normalized[index]["start_ms"],
            normalized[index]["end_ms"],
            index,
        ),
    )
    rank_by_index = {index: rank for rank, index in enumerate(ranked, start=1)}
    chronological = sorted(
        range(len(normalized)),
        key=lambda index: (
            normalized[index]["start_ms"],
            normalized[index]["end_ms"],
            index,
        ),
    )
    chronological_index = {
        candidate_index: index for index, candidate_index in enumerate(chronological, start=1)
    }
    best_score = max((row["score"] for row in normalized), default=0.0)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(normalized):
        output.append(
            {
                "id": f"{upload_id}:candidate-{chronological_index[index]:04d}",
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "score": round(row["score"], 6),
                "score_rank_all": rank_by_index[index],
                "relative_score": (round(row["score"] / best_score, 8) if best_score > 0 else 0.0),
                "selection": row["selection"],
            }
        )
    return sorted(output, key=lambda row: (row["start_ms"], row["end_ms"], row["id"]))


def _load_candidate_sources(
    run_root: Path,
    manifest: dict[str, Any],
    dataset: dict[str, Any],
    dataset_sha256: str,
) -> dict[str, dict[str, Any]]:
    if manifest.get("artifact_type") != "candidate-generation-run":
        raise CandidateEvaluationError("Candidate run manifest has the wrong artifact type")
    if manifest.get("status") != "completed":
        raise CandidateEvaluationError("Candidate run is not complete")
    if manifest.get("generation_receipt_valid") is not True:
        raise CandidateEvaluationError("Candidate run generation receipt is invalid")
    if manifest.get("git", {}).get("clean") is not True:
        raise CandidateEvaluationError("Candidate run was not generated from a clean worktree")
    run_dataset = manifest.get("dataset")
    if not isinstance(run_dataset, dict):
        raise CandidateEvaluationError("Candidate run has no dataset receipt")
    if run_dataset.get("sha256") != dataset_sha256:
        raise CandidateEvaluationError("Candidate run dataset checksum does not match")
    if run_dataset.get("annotation_snapshot_sha256") != dataset.get("annotation_snapshot_sha256"):
        raise CandidateEvaluationError("Candidate run annotation snapshot does not match")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise CandidateEvaluationError("Candidate run has no configuration")
    if (
        manifest.get("configuration_sha256")
        != hashlib.sha256(_json_bytes(configuration)).hexdigest()
    ):
        raise CandidateEvaluationError("Candidate run configuration checksum is invalid")
    descriptors = manifest.get("sources")
    if not isinstance(descriptors, list):
        raise CandidateEvaluationError("Candidate run has no source descriptors")
    descriptors_by_upload = {
        str(descriptor.get("upload_id")): descriptor
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    expected_uploads = {str(source["upload_id"]) for source in dataset["sources"]}
    if set(descriptors_by_upload) != expected_uploads:
        raise CandidateEvaluationError("Candidate run source set does not match the dataset")

    results: dict[str, dict[str, Any]] = {}
    dataset_by_upload = {str(source["upload_id"]): source for source in dataset["sources"]}
    for upload_id in sorted(expected_uploads):
        descriptor = descriptors_by_upload[upload_id]
        artifact_path = _safe_child(
            run_root,
            str(descriptor.get("artifact") or ""),
            "Candidate artifact",
        )
        signals_path = _safe_child(
            run_root,
            str(descriptor.get("signals") or ""),
            "Signal artifact",
        )
        if _sha256_file(artifact_path) != descriptor.get("artifact_sha256"):
            raise CandidateEvaluationError("Candidate artifact checksum is invalid")
        if _sha256_file(signals_path) != descriptor.get("signals_sha256"):
            raise CandidateEvaluationError("Signal artifact checksum is invalid")
        artifact, _artifact_file_hash = _load_json(
            artifact_path,
            "candidate source artifact",
        )
        source_receipt = artifact.get("source")
        receipt = artifact.get("receipt")
        dataset_source = dataset_by_upload[upload_id]
        if not isinstance(source_receipt, dict) or not isinstance(receipt, dict):
            raise CandidateEvaluationError("Candidate source receipt is incomplete")
        if str(source_receipt.get("upload_id")) != upload_id:
            raise CandidateEvaluationError("Candidate source upload identity mismatch")
        if source_receipt.get("filename") != dataset_source["filename"]:
            raise CandidateEvaluationError("Candidate source filename mismatch")
        if source_receipt.get("source_sha256") != dataset_source["source_sha256"]:
            raise CandidateEvaluationError("Candidate source SHA-256 mismatch")
        if int(source_receipt.get("byte_size", -1)) != int(dataset_source["byte_size"]):
            raise CandidateEvaluationError("Candidate source size mismatch")
        if receipt.get("dataset_sha256") != dataset_sha256:
            raise CandidateEvaluationError("Candidate source dataset receipt mismatch")
        if receipt.get("configuration_sha256") != manifest["configuration_sha256"]:
            raise CandidateEvaluationError("Candidate source configuration receipt mismatch")
        duration = float(source_receipt.get("duration", 0))
        expected_duration = int(dataset_source["duration_us"]) / 1_000_000
        if not math.isclose(duration, expected_duration, abs_tol=0.001):
            raise CandidateEvaluationError("Candidate source duration mismatch")
        candidates = _normalize_candidates(
            upload_id,
            artifact.get("candidates"),
            duration_seconds=duration,
        )
        if len(candidates) != int(descriptor.get("candidate_count", -1)):
            raise CandidateEvaluationError("Candidate source count mismatch")
        results[upload_id] = {
            "artifact": artifact,
            "candidates": candidates,
            "descriptor": descriptor,
        }
    return results


def _safe_run_id(value: str) -> str:
    if (
        not value
        or len(value) > 100
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for character in value
        )
    ):
        raise CandidateEvaluationError(
            "Run ID must contain only letters, numbers, dot, dash, and underscore"
        )
    return value


def score_candidate_run(
    *,
    dataset_path: Path,
    candidate_run: Path,
    run_id: str,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate and score an immutable candidate-only run."""

    dataset, dataset_sha256 = _validate_dataset(dataset_path)
    run_root = candidate_run.expanduser().resolve()
    manifest_path = run_root / "manifest.json" if run_root.is_dir() else run_root
    manifest, candidate_manifest_sha256 = _load_json(
        manifest_path,
        "candidate run manifest",
    )
    run_root = manifest_path.parent.resolve()
    candidate_sources = _load_candidate_sources(
        run_root,
        manifest,
        dataset,
        dataset_sha256,
    )

    source_metrics: list[dict[str, Any]] = []
    total_duration_minutes = 0.0
    for source in dataset["sources"]:
        upload_id = str(source["upload_id"])
        positives = [
            annotation for annotation in source["annotations"] if annotation["label"] == "highlight"
        ]
        duration_seconds = int(source["duration_us"]) / 1_000_000
        metric_source = {
            "upload_id": upload_id,
            "job_id": source["job_id"],
            "session_id": source["session_id"],
            "filename": source["filename"],
            "duration_seconds": duration_seconds,
        }
        source_metric = _metric_source(
            metric_source,
            positives,
            candidate_sources[upload_id]["candidates"],
        )
        source_metric["candidate_burden_per_minute"] = round(
            source_metric["candidate_count"] / (duration_seconds / 60),
            6,
        )
        source_metrics.append(source_metric)
        total_duration_minutes += duration_seconds / 60

    aggregate = _aggregate_metrics(source_metrics)
    aggregate["candidate_burden"] = {
        "candidate_count": sum(row["candidate_count"] for row in source_metrics),
        "source_minutes": round(total_duration_minutes, 6),
        "candidates_per_minute": round(
            sum(row["candidate_count"] for row in source_metrics) / total_duration_minutes,
            6,
        ),
    }
    observed = aggregate["strict_candidate_recall"]["micro_recall"]
    target_met = observed >= 0.80
    created_at = datetime.now(UTC).isoformat()
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at": created_at,
        "definitions": {
            "strict_candidate_recall": (
                "source-local chronological one-to-one matches where candidate core covers "
                "at least 50% of the human highlight interval"
            ),
            "matching_ties": (
                "maximum cardinality, maximum summed annotation coverage, maximum summed IoU, "
                "minimum summed absolute boundary error, then lexicographic IDs"
            ),
            "ranking_scope": "score ranks are recomputed across all candidates within each source",
            "precision": None,
            "precision_reason": (
                "Positive-only annotations cannot identify false positives; unmatched candidates "
                "remain unknown."
            ),
        },
        "provenance": {
            "evidence_valid": True,
            "dataset_sha256": dataset_sha256,
            "annotation_snapshot_sha256": dataset["annotation_snapshot_sha256"],
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "candidate_algorithm_version": manifest["algorithm_version"],
            "configuration_sha256": manifest["configuration_sha256"],
            "git": manifest["git"],
            "gpu": manifest["gpu"],
        },
        "gate": {
            "metric": "strict_candidate_recall",
            "target": 0.80,
            "observed": observed,
            "threshold_met": target_met,
            "evidence_status": "valid-development-baseline",
            "decision": "GO_RANKING" if target_met else "STOP_DETECTOR",
            "next_component": (
                "ranking" if target_met else "impact-detection-point-grouping-and-core-boundaries"
            ),
            "ranker_authorized": target_met,
        },
        "aggregate": aggregate,
        "sources": source_metrics,
        "warnings": [
            "All five sources are development data; this report is not held-out model accuracy.",
            "Precision, AP, AUROC, NDCG, FPR, and point purity are unavailable "
            "without explicit negatives.",
        ],
    }

    run_id = _safe_run_id(run_id)
    output_root = output_root.expanduser().resolve()
    destination = output_root / run_id
    if destination.exists():
        raise CandidateEvaluationError(f"Evaluation run already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        payloads = {
            "metrics.json": _json_bytes(metrics, pretty=True),
            "report.md": _report_markdown(metrics).encode("utf-8"),
        }
        hashes: dict[str, str] = {}
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
            hashes[filename] = hashlib.sha256(payload).hexdigest()
        evaluation_manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": created_at,
            "immutable": True,
            "dataset_sha256": dataset_sha256,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "files": hashes,
        }
        manifest_payload = _json_bytes(evaluation_manifest, pretty=True)
        (staging / "manifest.json").write_bytes(manifest_payload)
        hashes["manifest.json"] = hashlib.sha256(manifest_payload).hexdigest()
        checksums = "".join(
            f"{digest}  {filename}\n" for filename, digest in sorted(hashes.items())
        ).encode("ascii")
        (staging / "checksums.sha256").write_bytes(checksums)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    return destination, metrics
