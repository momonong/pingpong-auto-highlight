from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pingpong_highlight.candidate_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    CandidateEvaluationError,
    _aggregate_metrics,
    _json_bytes,
    _metric_source,
)

CANDIDATE_BURDEN_GUARDRAILS = {
    "aggregate_candidates_per_minute": 6.0,
    "source_candidates_per_minute": 8.0,
    "aggregate_union_core_coverage": 0.50,
    "source_union_core_coverage": 0.75,
    "maximum_core_duration_seconds": 20.0,
    "maximum_unresolved_overlap_ms": 0,
}


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


def _measure_candidate_burden(
    candidates: Sequence[Mapping[str, Any]],
    *,
    duration_us: int,
) -> tuple[dict[str, Any], dict[str, int | float]]:
    if duration_us <= 0:
        raise CandidateEvaluationError("Candidate burden requires a positive source duration")
    intervals = sorted(
        (int(row["start_ms"]), int(row["end_ms"])) for row in candidates
    )
    for start_ms, end_ms in intervals:
        if start_ms < 0 or end_ms <= start_ms or end_ms * 1000 > duration_us + 1000:
            raise CandidateEvaluationError("Candidate burden interval is outside its source")

    total_core_ms = sum(end_ms - start_ms for start_ms, end_ms in intervals)
    union_core_ms = 0
    if intervals:
        union_start, union_end = intervals[0]
        for start_ms, end_ms in intervals[1:]:
            if start_ms <= union_end:
                union_end = max(union_end, end_ms)
            else:
                union_core_ms += union_end - union_start
                union_start, union_end = start_ms, end_ms
        union_core_ms += union_end - union_start

    overlapping_pair_count = 0
    for index, (_start_ms, end_ms) in enumerate(intervals):
        for next_start_ms, _next_end_ms in intervals[index + 1 :]:
            if next_start_ms >= end_ms:
                break
            overlapping_pair_count += 1

    duration_minutes = duration_us / 60_000_000
    duration_ms = duration_us / 1000
    count = len(intervals)
    excess_ms = total_core_ms - union_core_ms
    raw: dict[str, int | float] = {
        "candidate_count": count,
        "duration_us": duration_us,
        "candidates_per_minute": count / duration_minutes,
        "total_core_ms": total_core_ms,
        "union_core_ms": union_core_ms,
        "union_core_coverage": union_core_ms / duration_ms,
        "max_core_ms": max((end - start for start, end in intervals), default=0),
        "overlapping_pair_count": overlapping_pair_count,
        "overlap_excess_ms": excess_ms,
    }
    display = {
        "candidate_count": count,
        "source_minutes": round(duration_minutes, 6),
        "candidates_per_minute": round(float(raw["candidates_per_minute"]), 6),
        "total_core_seconds": round(total_core_ms / 1000, 6),
        "union_core_seconds": round(union_core_ms / 1000, 6),
        "union_core_coverage": round(float(raw["union_core_coverage"]), 8),
        "max_core_duration_seconds": round(float(raw["max_core_ms"]) / 1000, 6),
        "duplicate_overlap": {
            "overlapping_pair_count": overlapping_pair_count,
            "overlap_excess_seconds": round(excess_ms / 1000, 6),
            "overlap_excess_fraction_of_total_core": (
                round(excess_ms / total_core_ms, 8) if total_core_ms else 0.0
            ),
        },
    }
    return display, raw


def _aggregate_candidate_burden(
    raw_sources: Sequence[Mapping[str, int | float]],
) -> tuple[dict[str, Any], dict[str, int | float]]:
    duration_us = sum(int(row["duration_us"]) for row in raw_sources)
    if duration_us <= 0:
        raise CandidateEvaluationError("Aggregate candidate burden has no source duration")
    count = sum(int(row["candidate_count"]) for row in raw_sources)
    total_core_ms = sum(int(row["total_core_ms"]) for row in raw_sources)
    union_core_ms = sum(int(row["union_core_ms"]) for row in raw_sources)
    overlap_excess_ms = sum(int(row["overlap_excess_ms"]) for row in raw_sources)
    overlapping_pair_count = sum(int(row["overlapping_pair_count"]) for row in raw_sources)
    duration_minutes = duration_us / 60_000_000
    raw: dict[str, int | float] = {
        "candidate_count": count,
        "duration_us": duration_us,
        "candidates_per_minute": count / duration_minutes,
        "total_core_ms": total_core_ms,
        "union_core_ms": union_core_ms,
        "union_core_coverage": union_core_ms / (duration_us / 1000),
        "max_core_ms": max((int(row["max_core_ms"]) for row in raw_sources), default=0),
        "overlapping_pair_count": overlapping_pair_count,
        "overlap_excess_ms": overlap_excess_ms,
    }
    display = {
        "candidate_count": count,
        "source_minutes": round(duration_minutes, 6),
        "candidates_per_minute": round(float(raw["candidates_per_minute"]), 6),
        "total_core_seconds": round(total_core_ms / 1000, 6),
        "union_core_seconds": round(union_core_ms / 1000, 6),
        "union_core_coverage": round(float(raw["union_core_coverage"]), 8),
        "max_core_duration_seconds": round(float(raw["max_core_ms"]) / 1000, 6),
        "duplicate_overlap": {
            "overlapping_pair_count": overlapping_pair_count,
            "overlap_excess_seconds": round(overlap_excess_ms / 1000, 6),
            "overlap_excess_fraction_of_total_core": (
                round(overlap_excess_ms / total_core_ms, 8) if total_core_ms else 0.0
            ),
        },
    }
    return display, raw


def _burden_gate(
    aggregate: Mapping[str, int | float],
    sources: Sequence[tuple[str, Mapping[str, int | float]]],
) -> dict[str, Any]:
    limits = CANDIDATE_BURDEN_GUARDRAILS
    checks = [
        {
            "name": "aggregate_candidates_per_minute",
            "observed": round(float(aggregate["candidates_per_minute"]), 6),
            "limit": limits["aggregate_candidates_per_minute"],
            "passed": (
                float(aggregate["candidates_per_minute"])
                <= limits["aggregate_candidates_per_minute"]
            ),
        },
        {
            "name": "aggregate_union_core_coverage",
            "observed": round(float(aggregate["union_core_coverage"]), 8),
            "limit": limits["aggregate_union_core_coverage"],
            "passed": (
                float(aggregate["union_core_coverage"])
                <= limits["aggregate_union_core_coverage"]
            ),
        },
        {
            "name": "maximum_core_duration_seconds",
            "observed": round(float(aggregate["max_core_ms"]) / 1000, 6),
            "limit": limits["maximum_core_duration_seconds"],
            "passed": (
                int(aggregate["max_core_ms"])
                <= limits["maximum_core_duration_seconds"] * 1000
            ),
        },
        {
            "name": "unresolved_overlap_ms",
            "observed": int(aggregate["overlap_excess_ms"]),
            "limit": limits["maximum_unresolved_overlap_ms"],
            "passed": (
                int(aggregate["overlap_excess_ms"])
                <= limits["maximum_unresolved_overlap_ms"]
                and int(aggregate["overlapping_pair_count"]) == 0
            ),
        },
    ]
    source_checks: list[dict[str, Any]] = []
    for upload_id, source in sources:
        cpm_passed = (
            float(source["candidates_per_minute"])
            <= limits["source_candidates_per_minute"]
        )
        coverage_passed = (
            float(source["union_core_coverage"])
            <= limits["source_union_core_coverage"]
        )
        source_checks.append(
            {
                "upload_id": upload_id,
                "candidates_per_minute": round(
                    float(source["candidates_per_minute"]),
                    6,
                ),
                "candidates_per_minute_limit": limits["source_candidates_per_minute"],
                "candidates_per_minute_passed": cpm_passed,
                "union_core_coverage": round(float(source["union_core_coverage"]), 8),
                "union_core_coverage_limit": limits["source_union_core_coverage"],
                "union_core_coverage_passed": coverage_passed,
                "passed": cpm_passed and coverage_passed,
            }
        )
    return {
        "contract_version": 1,
        "limits": limits,
        "checks": checks,
        "source_checks": source_checks,
        "threshold_met": all(check["passed"] for check in checks)
        and all(check["passed"] for check in source_checks),
    }


def _formal_report_markdown(metrics: Mapping[str, Any]) -> str:
    strict = metrics["aggregate"]["strict_candidate_recall"]
    burden = metrics["aggregate"]["candidate_burden"]
    gate = metrics["gate"]
    lines = [
        "# Formal candidate evaluation",
        "",
        f"Created: `{metrics['created_at']}`",
        "",
        "## Decision",
        "",
        f"- Decision: **{gate['decision']}**",
        f"- Evidence status: `{gate['evidence_status']}`",
        f"- Strict candidate recall: **{strict['hits']}/{strict['total']} "
        f"({strict['micro_recall']:.2%})**; target **{gate['target']:.0%}**",
        f"- Candidate burden: **{burden['candidates_per_minute']:.3f}/min**, "
        f"**{burden['union_core_coverage']:.2%}** timeline coverage",
        f"- Burden guardrails: **{'PASS' if gate['burden_threshold_met'] else 'FAIL'}**",
        "- Precision: unavailable; positive-only annotations cannot identify false positives.",
        "",
        "## Per source",
        "",
        "| Source | Strict | Candidates | CPM | Core coverage | Max core |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in metrics["sources"]:
        recall = source["strict_candidate_recall"]
        source_burden = source["candidate_burden"]
        lines.append(
            f"| {source['filename']} | {recall['hits']}/{recall['total']} "
            f"({recall['recall']:.2%}) | {source['candidate_count']} | "
            f"{source_burden['candidates_per_minute']:.3f} | "
            f"{source_burden['union_core_coverage']:.2%} | "
            f"{source_burden['max_core_duration_seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "The immutable candidate manifest, source artifacts, raw signal files, dataset, "
            "configuration, clean Git receipt, and GPU receipt were verified before scoring. "
            "Matching is source-local, chronological, one-to-one, and requires a candidate "
            "core to cover at least 50% of a human highlight.",
            "",
            "All sources are development data used during detector iteration. This is a "
            "development regression result, not held-out accuracy or a precision claim.",
            "",
        ]
    )
    return "\n".join(lines)


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
    if manifest.get("gpu", {}).get("nvdec_available") is not True:
        raise CandidateEvaluationError("Formal candidate scoring requires a GPU/NVDEC run")
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
    if configuration.get("require_nvdec") is not True:
        raise CandidateEvaluationError("Formal candidate scoring requires strict NVDEC execution")
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
        if artifact.get("schema_version") != manifest.get("schema_version"):
            raise CandidateEvaluationError("Candidate source schema version mismatch")
        if artifact.get("artifact_type") != "candidate-generation-source":
            raise CandidateEvaluationError("Candidate source artifact type mismatch")
        if artifact.get("algorithm_version") != manifest.get("algorithm_version"):
            raise CandidateEvaluationError("Candidate source algorithm version mismatch")
        if artifact.get("run_id") != manifest.get("run_id"):
            raise CandidateEvaluationError("Candidate source run identity mismatch")
        if artifact.get("configuration") != configuration:
            raise CandidateEvaluationError("Candidate source configuration mismatch")
        source_receipt = artifact.get("source")
        receipt = artifact.get("receipt")
        dataset_source = dataset_by_upload[upload_id]
        if not isinstance(source_receipt, dict) or not isinstance(receipt, dict):
            raise CandidateEvaluationError("Candidate source receipt is incomplete")
        if str(source_receipt.get("upload_id")) != upload_id:
            raise CandidateEvaluationError("Candidate source upload identity mismatch")
        if source_receipt.get("filename") != dataset_source["filename"]:
            raise CandidateEvaluationError("Candidate source filename mismatch")
        if source_receipt.get("job_id") != dataset_source["job_id"]:
            raise CandidateEvaluationError("Candidate source job identity mismatch")
        if source_receipt.get("source_sha256") != dataset_source["source_sha256"]:
            raise CandidateEvaluationError("Candidate source SHA-256 mismatch")
        if int(source_receipt.get("byte_size", -1)) != int(dataset_source["byte_size"]):
            raise CandidateEvaluationError("Candidate source size mismatch")
        if receipt.get("dataset_sha256") != dataset_sha256:
            raise CandidateEvaluationError("Candidate source dataset receipt mismatch")
        if (
            receipt.get("annotation_snapshot_sha256")
            != dataset["annotation_snapshot_sha256"]
        ):
            raise CandidateEvaluationError("Candidate source annotation receipt mismatch")
        if receipt.get("configuration_sha256") != manifest["configuration_sha256"]:
            raise CandidateEvaluationError("Candidate source configuration receipt mismatch")
        if receipt.get("git") != manifest.get("git"):
            raise CandidateEvaluationError("Candidate source Git receipt mismatch")
        if receipt.get("gpu") != manifest.get("gpu"):
            raise CandidateEvaluationError("Candidate source GPU receipt mismatch")
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
    source_count = len(dataset["sources"])
    if source_count == 0:
        raise CandidateEvaluationError("Formal candidate scoring requires at least one source")
    if {source.get("split") for source in dataset["sources"]} != {"development"}:
        raise CandidateEvaluationError(
            "Formal scoring contract v1 accepts development sources only"
        )
    if dataset.get("precision_eligible") is not False or any(
        annotation.get("label") != "highlight"
        for source in dataset["sources"]
        for annotation in source.get("annotations", [])
    ):
        raise CandidateEvaluationError(
            "Formal scoring contract v1 accepts positive-only recall datasets"
        )
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
    source_burden_raw: list[tuple[str, dict[str, int | float]]] = []
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
        burden, burden_raw = _measure_candidate_burden(
            candidate_sources[upload_id]["candidates"],
            duration_us=int(source["duration_us"]),
        )
        source_metric["candidate_burden"] = burden
        source_metrics.append(source_metric)
        source_burden_raw.append((upload_id, burden_raw))

    aggregate = _aggregate_metrics(source_metrics)
    aggregate_burden, aggregate_burden_raw = _aggregate_candidate_burden(
        [row for _upload_id, row in source_burden_raw]
    )
    aggregate["candidate_burden"] = aggregate_burden
    burden_gate = _burden_gate(aggregate_burden_raw, source_burden_raw)
    observed = aggregate["strict_candidate_recall"]["micro_recall"]
    recall_target_met = observed >= 0.80
    burden_target_met = bool(burden_gate["threshold_met"])
    if not recall_target_met:
        decision = "STOP_DETECTOR"
        next_component = "impact-detection-point-grouping-and-core-boundaries"
    elif not burden_target_met:
        decision = "STOP_CANDIDATE_BURDEN"
        next_component = "candidate-consolidation-and-core-boundaries"
    else:
        decision = "GO_RANKING"
        next_component = "ranking"
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
            "precision_status": "abstained_missing_explicit_negatives",
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
            "recall_threshold_met": recall_target_met,
            "burden_threshold_met": burden_target_met,
            "threshold_met": recall_target_met and burden_target_met,
            "evidence_status": "valid-development-regression",
            "decision": decision,
            "next_component": next_component,
            "ranker_authorized": recall_target_met and burden_target_met,
            "candidate_burden": burden_gate,
        },
        "aggregate": aggregate,
        "sources": source_metrics,
        "warnings": [
            f"All {source_count} sources are development data; this report is not held-out "
            "model accuracy.",
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
            "report.md": _formal_report_markdown(metrics).encode("utf-8"),
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
