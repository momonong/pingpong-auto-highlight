from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

EVALUATION_SCHEMA_VERSION = 1
STRICT_COVERAGE = Fraction(1, 2)
RANK_CUTOFFS = (1, 3, 6, 10)
PADDING_SECONDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)

ProgressCallback = Callable[[str], None]


class CandidateEvaluationError(RuntimeError):
    """Raised when a frozen evaluation would be ambiguous or incomplete."""


@dataclass(frozen=True, slots=True)
class _MatchPlan:
    pairs: tuple[tuple[int, int], ...] = ()
    coverage: Fraction = Fraction(0)
    iou: Fraction = Fraction(0)
    boundary_error_ms: int = 0

    @property
    def count(self) -> int:
        return len(self.pairs)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + ("\n" if pretty else "")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _seconds_to_ms(value: float) -> int:
    if not math.isfinite(value):
        raise CandidateEvaluationError("Interval timestamps must be finite")
    return round(value * 1000)


def _seconds_to_us(value: float) -> int:
    if not math.isfinite(value):
        raise CandidateEvaluationError("Media duration must be finite")
    return round(value * 1_000_000)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _median(values: Iterable[float]) -> float | None:
    collected = list(values)
    return round(float(median(collected)), 6) if collected else None


@contextmanager
def _readonly_connection(path: Path) -> Iterator[sqlite3.Connection]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CandidateEvaluationError(f"Database does not exist: {resolved}")
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro",
        uri=True,
        timeout=30.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA query_only = ON")
    try:
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.rollback()
        connection.close()


def canonical_annotations(
    annotations: Iterable[Mapping[str, Any] | sqlite3.Row],
) -> list[dict[str, Any]]:
    """Return the representation used by the existing review snapshot digest."""

    records = [
        {
            "id": str(annotation["id"]),
            "label": str(annotation["label"]),
            "start": float(annotation["start"]),
            "end": float(annotation["end"]),
            "note": str(annotation["note"]),
            "created_at": str(annotation["created_at"]),
            "updated_at": str(annotation["updated_at"]),
        }
        for annotation in annotations
    ]
    return sorted(records, key=lambda item: (item["start"], item["end"], item["id"]))


def annotation_review_digest(
    annotations: Iterable[Mapping[str, Any] | sqlite3.Row],
) -> str:
    return _sha256_bytes(_json_bytes(canonical_annotations(annotations)))


def _intersection_ms(
    annotation: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    pad_ms: int = 0,
    source_duration_ms: int | None = None,
) -> int:
    candidate_start = max(0, int(candidate["start_ms"]) - pad_ms)
    candidate_end = int(candidate["end_ms"]) + pad_ms
    if source_duration_ms is not None:
        candidate_end = min(source_duration_ms, candidate_end)
    return max(
        0,
        min(int(annotation["end_ms"]), candidate_end)
        - max(int(annotation["start_ms"]), candidate_start),
    )


def _edge_values(
    annotation: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    pad_ms: int,
    source_duration_ms: int | None,
) -> tuple[int, Fraction, Fraction, int]:
    intersection = _intersection_ms(
        annotation,
        candidate,
        pad_ms=pad_ms,
        source_duration_ms=source_duration_ms,
    )
    annotation_duration = int(annotation["end_ms"]) - int(annotation["start_ms"])
    candidate_start = max(0, int(candidate["start_ms"]) - pad_ms)
    candidate_end = int(candidate["end_ms"]) + pad_ms
    if source_duration_ms is not None:
        candidate_end = min(source_duration_ms, candidate_end)
    candidate_duration = candidate_end - candidate_start
    union = annotation_duration + candidate_duration - intersection
    coverage = Fraction(intersection, annotation_duration)
    iou = Fraction(intersection, union) if union > 0 else Fraction(0)
    boundary_error = abs(candidate_start - int(annotation["start_ms"])) + abs(
        candidate_end - int(annotation["end_ms"])
    )
    return intersection, coverage, iou, boundary_error


def _better_plan(
    left: _MatchPlan,
    right: _MatchPlan,
    annotations: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> _MatchPlan:
    left_quality = (left.count, left.coverage, left.iou, -left.boundary_error_ms)
    right_quality = (right.count, right.coverage, right.iou, -right.boundary_error_ms)
    if left_quality != right_quality:
        return left if left_quality > right_quality else right
    left_ids = tuple((str(annotations[a]["id"]), str(candidates[c]["id"])) for a, c in left.pairs)
    right_ids = tuple((str(annotations[a]["id"]), str(candidates[c]["id"])) for a, c in right.pairs)
    return left if left_ids <= right_ids else right


def match_intervals(
    annotations: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: Fraction | None = STRICT_COVERAGE,
    minimum_intersection_ms: int = 1,
    pad_ms: int = 0,
    source_duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Match chronological intervals one-to-one without crossing source time."""

    ordered_annotations = sorted(
        annotations,
        key=lambda row: (int(row["start_ms"]), int(row["end_ms"]), str(row["id"])),
    )
    ordered_candidates = sorted(
        candidates,
        key=lambda row: (int(row["start_ms"]), int(row["end_ms"]), str(row["id"])),
    )
    rows = len(ordered_annotations) + 1
    columns = len(ordered_candidates) + 1
    plans = [[_MatchPlan() for _ in range(columns)] for _ in range(rows)]

    for annotation_index in range(1, rows):
        for candidate_index in range(1, columns):
            best = _better_plan(
                plans[annotation_index - 1][candidate_index],
                plans[annotation_index][candidate_index - 1],
                ordered_annotations,
                ordered_candidates,
            )
            annotation = ordered_annotations[annotation_index - 1]
            candidate = ordered_candidates[candidate_index - 1]
            intersection, coverage, iou, boundary_error = _edge_values(
                annotation,
                candidate,
                pad_ms=pad_ms,
                source_duration_ms=source_duration_ms,
            )
            eligible = intersection >= minimum_intersection_ms
            if minimum_coverage is not None:
                eligible = eligible and coverage >= minimum_coverage
            if eligible:
                previous = plans[annotation_index - 1][candidate_index - 1]
                matched = _MatchPlan(
                    pairs=(*previous.pairs, (annotation_index - 1, candidate_index - 1)),
                    coverage=previous.coverage + coverage,
                    iou=previous.iou + iou,
                    boundary_error_ms=previous.boundary_error_ms + boundary_error,
                )
                best = _better_plan(
                    best,
                    matched,
                    ordered_annotations,
                    ordered_candidates,
                )
            plans[annotation_index][candidate_index] = best

    matches: list[dict[str, Any]] = []
    for annotation_index, candidate_index in plans[-1][-1].pairs:
        annotation = ordered_annotations[annotation_index]
        candidate = ordered_candidates[candidate_index]
        intersection, coverage, iou, _boundary_error = _edge_values(
            annotation,
            candidate,
            pad_ms=pad_ms,
            source_duration_ms=source_duration_ms,
        )
        candidate_start = max(0, int(candidate["start_ms"]) - pad_ms)
        candidate_end = int(candidate["end_ms"]) + pad_ms
        if source_duration_ms is not None:
            candidate_end = min(source_duration_ms, candidate_end)
        matches.append(
            {
                "annotation_id": annotation["id"],
                "candidate_id": candidate["id"],
                "intersection_seconds": round(intersection / 1000, 6),
                "annotation_coverage": round(float(coverage), 8),
                "iou": round(float(iou), 8),
                "start_error_seconds": round(
                    (candidate_start - int(annotation["start_ms"])) / 1000,
                    6,
                ),
                "end_error_seconds": round(
                    (candidate_end - int(annotation["end_ms"])) / 1000,
                    6,
                ),
                "score_rank_all": candidate.get("score_rank_all"),
                "score": candidate.get("score"),
                "selection": candidate.get("selection"),
            }
        )
    return matches


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def _live_sources(database_path: Path) -> list[dict[str, Any]]:
    with _readonly_connection(database_path) as connection:
        sources = [
            dict(row)
            for row in connection.execute(
                """
                SELECT j.id AS job_id, j.upload_id, u.filename, u.size, u.path,
                       u.recorded_at, u.created_at, j.updated_at AS job_updated_at
                FROM jobs AS j
                JOIN uploads AS u ON u.id = j.upload_id
                WHERE j.status = 'completed'
                ORDER BY u.filename, j.id
                """
            ).fetchall()
        ]
        if not sources:
            raise CandidateEvaluationError("No completed source jobs were found")
        for source in sources:
            source["annotations"] = canonical_annotations(
                connection.execute(
                    """
                    SELECT id, label, start, end, note, created_at, updated_at
                    FROM annotations
                    WHERE upload_id = ?
                    ORDER BY start, end, id
                    """,
                    (source["upload_id"],),
                ).fetchall()
            )
            source["active_clips"] = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT clip_filename, library_version, source_rank, score
                    FROM highlight_clips
                    WHERE job_id = ? AND active = 1
                    ORDER BY source_rank, clip_filename
                    """,
                    (source["job_id"],),
                ).fetchall()
            ]
    return sources


def _review_evidence(
    review_database: Path,
    sources: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    with _readonly_connection(review_database) as connection:
        if not _table_exists(connection, "source_reviews"):
            raise CandidateEvaluationError("Review database has no source_reviews table")
        if not _table_exists(connection, "annotations"):
            raise CandidateEvaluationError("Review database has no annotations table")
        for source in sources:
            review = connection.execute(
                "SELECT * FROM source_reviews WHERE upload_id = ?",
                (source["upload_id"],),
            ).fetchone()
            if review is None:
                raise CandidateEvaluationError(
                    f"Missing review evidence for upload {source['upload_id']}"
                )
            frozen_annotations = canonical_annotations(
                connection.execute(
                    """
                    SELECT id, label, start, end, note, created_at, updated_at
                    FROM annotations
                    WHERE upload_id = ?
                    ORDER BY start, end, id
                    """,
                    (source["upload_id"],),
                ).fetchall()
            )
            if frozen_annotations != source["annotations"]:
                raise CandidateEvaluationError(
                    f"Live annotations differ from the review snapshot for {source['filename']}"
                )
            digest = annotation_review_digest(frozen_annotations)
            if int(review["review_complete"]) != 1:
                raise CandidateEvaluationError(f"Review is not complete for {source['filename']}")
            if int(review["annotation_count"]) != len(frozen_annotations):
                raise CandidateEvaluationError(
                    f"Review annotation count is stale for {source['filename']}"
                )
            if review["annotation_digest"] != digest:
                raise CandidateEvaluationError(
                    f"Review annotation digest is stale for {source['filename']}"
                )
            source_duration = float(review["source_duration"])
            reviewed_until = review["reviewed_until"]
            if reviewed_until is None or not math.isclose(
                float(reviewed_until), source_duration, abs_tol=0.001
            ):
                raise CandidateEvaluationError(
                    f"Review coverage is incomplete for {source['filename']}"
                )
            reviews[str(source["upload_id"])] = {
                "session_id": str(review["session_id"]),
                "recorded_at": review["recorded_at"],
                "review_complete": True,
                "source_duration": source_duration,
                "reviewed_until": float(reviewed_until),
                "reviewed_at": review["reviewed_at"],
                "annotation_count": int(review["annotation_count"]),
                "annotation_digest": digest,
            }
    return reviews


def _safe_active_analysis(
    data_dir: Path,
    source: Mapping[str, Any],
) -> tuple[Path, str]:
    clips = list(source["active_clips"])
    if not clips:
        raise CandidateEvaluationError(
            f"No active clip-set identifies the candidate artifact for {source['filename']}"
        )
    parents: set[PurePosixPath] = set()
    versions: set[str] = set()
    for clip in clips:
        relative = PurePosixPath(str(clip["clip_filename"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CandidateEvaluationError("Active clip path escapes its job output")
        parents.add(relative.parent)
        versions.add(str(clip["library_version"]))
    if len(parents) != 1 or PurePosixPath(".") in parents:
        raise CandidateEvaluationError(
            f"Active clips do not identify one versioned artifact for {source['filename']}"
        )
    if len(versions) != 1:
        raise CandidateEvaluationError(
            f"Active clips disagree on algorithm version for {source['filename']}"
        )
    job_root = (data_dir / "outputs" / str(source["job_id"])).resolve()
    parent = next(iter(parents))
    analysis = job_root.joinpath(*parent.parts, "analysis.json").resolve()
    if not analysis.is_relative_to(job_root) or not analysis.is_file():
        raise CandidateEvaluationError(
            f"Active analysis artifact is missing for {source['filename']}"
        )
    return analysis, next(iter(versions))


def _load_active_candidates(
    data_dir: Path,
    source: dict[str, Any],
    review: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    analysis_path, library_version = _safe_active_analysis(data_dir, source)
    try:
        result = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateEvaluationError(
            f"Cannot read active analysis for {source['filename']}: {exc}"
        ) from exc
    if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
        raise CandidateEvaluationError(
            f"Active analysis has no complete candidate array for {source['filename']}"
        )
    if result.get("algorithm_version") != library_version:
        raise CandidateEvaluationError(
            f"Active artifact version disagrees with SQLite for {source['filename']}"
        )
    if result.get("source_name") != source["filename"]:
        raise CandidateEvaluationError(
            f"Active artifact source identity mismatch for {source['filename']}"
        )
    media = result.get("media")
    duration = media.get("duration") if isinstance(media, dict) else None
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise CandidateEvaluationError(
            f"Active artifact has no valid duration for {source['filename']}"
        )
    if not math.isclose(float(duration), float(review["source_duration"]), abs_tol=0.001):
        raise CandidateEvaluationError(
            f"Active artifact duration disagrees with review evidence for {source['filename']}"
        )

    raw_candidates: list[dict[str, Any]] = []
    for original_index, row in enumerate(result["candidates"]):
        if not isinstance(row, dict):
            raise CandidateEvaluationError("Candidate entries must be JSON objects")
        try:
            start = float(row["rally_start"])
            end = float(row["rally_end"])
            score = float(row["score"])
            impact_count = int(row["impact_count"])
            motion_score = float(row["motion_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CandidateEvaluationError(
                f"Invalid candidate entry for {source['filename']}"
            ) from exc
        if not all(math.isfinite(value) for value in (start, end, score, motion_score)):
            raise CandidateEvaluationError("Candidate values must be finite")
        if start < 0 or end <= start or end > float(duration) + 0.001:
            raise CandidateEvaluationError(f"Candidate interval is outside {source['filename']}")
        raw_candidates.append(
            {
                "original_index": original_index,
                "start_ms": _seconds_to_ms(start),
                "end_ms": _seconds_to_ms(end),
                "score": score,
                "impact_count": impact_count,
                "motion_score": motion_score,
                "origin": "motion_fallback" if impact_count == 0 else "audio",
                "reason": str(row.get("reason") or ""),
                "selection": str(row.get("selection") or "candidate"),
                "selected_rank": int(row["rank"]) if row.get("rank") is not None else None,
            }
        )
    summary = result.get("summary")
    expected_count = summary.get("candidate_point_count") if isinstance(summary, dict) else None
    if expected_count != len(raw_candidates):
        raise CandidateEvaluationError(f"Candidate summary count mismatch for {source['filename']}")
    selected_count = sum(row["selection"] == "selected" for row in raw_candidates)
    points = result.get("points")
    if not isinstance(points, list) or len(points) != selected_count:
        raise CandidateEvaluationError(
            f"Selected candidates and points disagree for {source['filename']}"
        )
    if selected_count != len(source["active_clips"]):
        raise CandidateEvaluationError(
            f"Active clips and selected candidates disagree for {source['filename']}"
        )

    ranked = sorted(
        range(len(raw_candidates)),
        key=lambda index: (
            -raw_candidates[index]["score"],
            raw_candidates[index]["start_ms"],
            raw_candidates[index]["end_ms"],
            raw_candidates[index]["original_index"],
        ),
    )
    rank_by_index = {candidate_index: rank for rank, candidate_index in enumerate(ranked, 1)}
    best_score = max((candidate["score"] for candidate in raw_candidates), default=0.0)
    chronological = sorted(
        range(len(raw_candidates)),
        key=lambda index: (
            raw_candidates[index]["start_ms"],
            raw_candidates[index]["end_ms"],
            index,
        ),
    )
    chronological_index = {
        candidate_index: index for index, candidate_index in enumerate(chronological, 1)
    }
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw_candidates):
        candidate_id = f"{source['upload_id']}:candidate-{chronological_index[index]:04d}"
        candidates.append(
            {
                "id": candidate_id,
                "upload_id": source["upload_id"],
                "job_id": source["job_id"],
                "start_ms": candidate["start_ms"],
                "end_ms": candidate["end_ms"],
                "score": round(candidate["score"], 6),
                "score_rank_all": rank_by_index[index],
                "relative_score": (
                    round(candidate["score"] / best_score, 8) if best_score > 0 else 0.0
                ),
                "impact_count": candidate["impact_count"],
                "motion_score": round(candidate["motion_score"], 6),
                "origin": candidate["origin"],
                "reason": candidate["reason"],
                "product_policy": {
                    "selection": candidate["selection"],
                    "selected_rank": candidate["selected_rank"],
                },
                "selection": candidate["selection"],
            }
        )
    candidates.sort(key=lambda row: (row["start_ms"], row["end_ms"], row["id"]))
    artifact_relative = analysis_path.relative_to(data_dir.resolve()).as_posix()
    artifact = {
        "path": artifact_relative,
        "sha256": _sha256_file(analysis_path),
        "algorithm_version": result["algorithm_version"],
        "candidate_count": len(candidates),
        "selected_count": selected_count,
        "contract": "active-analysis-v2-legacy-import",
        "provenance_complete": False,
        "missing_provenance": [
            "generation_git_commit",
            "generation_worktree_state",
            "full_detection_and_sampling_config",
            "config_sha256",
            "source_sha256_at_generation",
            "raw_impact_events",
            "rejected_impact_groups",
            "candidate_score_components",
        ],
    }
    return artifact, candidates


def _source_path(value: str, data_dir: Path) -> Path:
    container_path = PurePosixPath(value.replace("\\", "/"))
    if container_path.is_absolute() and container_path.parts[:2] == ("/", "data"):
        path = data_dir.joinpath(*container_path.parts[2:]).resolve()
    else:
        path = Path(value).expanduser()
        path = (data_dir.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise CandidateEvaluationError(f"Source video is not available locally: {path}")
    return path


def _git_receipt(start: Path) -> dict[str, Any]:
    try:
        root_result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(root_result.stdout.strip()).resolve()
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "clean": None, "status_sha256": None}
    return {
        "available": True,
        "commit": commit,
        "clean": not bool(status),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
    }


def _metric_source(
    source: Mapping[str, Any],
    annotations: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    duration_ms = _seconds_to_ms(float(source["duration_seconds"]))
    strict_matches = match_intervals(
        annotations,
        candidates,
        source_duration_ms=duration_ms,
    )
    loose_matches = match_intervals(
        annotations,
        candidates,
        minimum_coverage=None,
        minimum_intersection_ms=500,
        source_duration_ms=duration_ms,
    )
    strict_by_annotation = {match["annotation_id"]: match for match in strict_matches}
    annotation_results: list[dict[str, Any]] = []
    for annotation in annotations:
        intersecting: list[tuple[int, float, int, str]] = []
        annotation_duration = annotation["end_ms"] - annotation["start_ms"]
        for candidate in candidates:
            intersection = _intersection_ms(annotation, candidate)
            if intersection:
                intersecting.append(
                    (
                        intersection,
                        float(candidate["score"]),
                        -int(candidate["score_rank_all"]),
                        str(candidate["id"]),
                    )
                )
        best_overlap = max(intersecting, default=None)
        matched = strict_by_annotation.get(annotation["id"])
        annotation_results.append(
            {
                "annotation_id": annotation["id"],
                "start_ms": annotation["start_ms"],
                "end_ms": annotation["end_ms"],
                "strict_match": matched,
                "best_any_overlap_coverage": (
                    round(best_overlap[0] / annotation_duration, 8)
                    if best_overlap is not None
                    else 0.0
                ),
            }
        )

    ranking: dict[str, Any] = {}
    for cutoff in RANK_CUTOFFS:
        subset = [row for row in candidates if int(row["score_rank_all"]) <= cutoff]
        matches = match_intervals(
            annotations,
            subset,
            source_duration_ms=duration_ms,
        )
        ranking[f"recall_at_{cutoff}"] = {
            "hits": len(matches),
            "total": len(annotations),
            "recall": _ratio(len(matches), len(annotations)),
            "retention_of_detectable": _ratio(len(matches), len(strict_matches)),
        }
    library_subset = [row for row in candidates if row["selection"] == "selected"]
    recommendation_subset = [row for row in candidates if row["relative_score"] >= 0.87]
    threshold_results: dict[str, Any] = {}
    for name, subset in (
        ("library_policy", library_subset),
        ("recommendation_0_87", recommendation_subset),
    ):
        matches = match_intervals(
            annotations,
            subset,
            source_duration_ms=duration_ms,
        )
        threshold_results[name] = {
            "hits": len(matches),
            "total": len(annotations),
            "recall": _ratio(len(matches), len(annotations)),
            "retention_of_detectable": _ratio(len(matches), len(strict_matches)),
        }

    padding = []
    for seconds in PADDING_SECONDS:
        matches = match_intervals(
            annotations,
            candidates,
            pad_ms=_seconds_to_ms(seconds),
            source_duration_ms=duration_ms,
        )
        padding.append(
            {
                "seconds_each_side": seconds,
                "hits": len(matches),
                "total": len(annotations),
                "recall": _ratio(len(matches), len(annotations)),
            }
        )

    return {
        "upload_id": source["upload_id"],
        "job_id": source["job_id"],
        "session_id": source["session_id"],
        "filename": source["filename"],
        "annotation_count": len(annotations),
        "candidate_count": len(candidates),
        "strict_candidate_recall": {
            "hits": len(strict_matches),
            "total": len(annotations),
            "recall": _ratio(len(strict_matches), len(annotations)),
        },
        "loose_core_overlap_0_5s": {
            "hits": len(loose_matches),
            "total": len(annotations),
            "recall": _ratio(len(loose_matches), len(annotations)),
        },
        "ranking": ranking,
        "threshold_retention": threshold_results,
        "padding_sensitivity": padding,
        "matches": strict_matches,
        "annotations": annotation_results,
    }


def _aggregate_metrics(source_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(row["strict_candidate_recall"]["hits"] for row in source_metrics)
    total = sum(row["strict_candidate_recall"]["total"] for row in source_metrics)
    loose_hits = sum(row["loose_core_overlap_0_5s"]["hits"] for row in source_metrics)
    recalls = [row["strict_candidate_recall"]["recall"] for row in source_metrics]
    all_matches = [match for row in source_metrics for match in row["matches"]]

    ranking: dict[str, Any] = {}
    for cutoff in RANK_CUTOFFS:
        key = f"recall_at_{cutoff}"
        cutoff_hits = sum(row["ranking"][key]["hits"] for row in source_metrics)
        ranking[key] = {
            "hits": cutoff_hits,
            "total": total,
            "recall": _ratio(cutoff_hits, total),
            "retention_of_detectable": _ratio(cutoff_hits, hits),
        }

    threshold_retention: dict[str, Any] = {}
    for key in ("library_policy", "recommendation_0_87"):
        threshold_hits = sum(row["threshold_retention"][key]["hits"] for row in source_metrics)
        threshold_retention[key] = {
            "hits": threshold_hits,
            "total": total,
            "recall": _ratio(threshold_hits, total),
            "retention_of_detectable": _ratio(threshold_hits, hits),
        }

    padding = []
    for index, seconds in enumerate(PADDING_SECONDS):
        padding_hits = sum(row["padding_sensitivity"][index]["hits"] for row in source_metrics)
        padding.append(
            {
                "seconds_each_side": seconds,
                "hits": padding_hits,
                "total": total,
                "recall": _ratio(padding_hits, total),
            }
        )

    session_totals: dict[str, dict[str, Any]] = {}
    for row in source_metrics:
        session = session_totals.setdefault(
            row["session_id"],
            {"hits": 0, "total": 0, "sources": []},
        )
        session["hits"] += row["strict_candidate_recall"]["hits"]
        session["total"] += row["strict_candidate_recall"]["total"]
        session["sources"].append(row["upload_id"])
    for session in session_totals.values():
        session["recall"] = _ratio(session["hits"], session["total"])

    start_errors = [float(match["start_error_seconds"]) for match in all_matches]
    end_errors = [float(match["end_error_seconds"]) for match in all_matches]
    return {
        "strict_candidate_recall": {
            "hits": hits,
            "total": total,
            "micro_recall": _ratio(hits, total),
            "macro_source_recall": round(sum(recalls) / len(recalls), 8) if recalls else 0.0,
        },
        "loose_core_overlap_0_5s": {
            "hits": loose_hits,
            "total": total,
            "micro_recall": _ratio(loose_hits, total),
        },
        "ranking": ranking,
        "threshold_retention": threshold_retention,
        "padding_sensitivity": padding,
        "sessions": session_totals,
        "boundary_on_strict_matches": {
            "sample_count": len(all_matches),
            "median_absolute_start_error_seconds": _median(abs(value) for value in start_errors),
            "median_absolute_end_error_seconds": _median(abs(value) for value in end_errors),
            "start_late_fraction": _ratio(
                sum(value > 0 for value in start_errors),
                len(start_errors),
            ),
            "end_early_fraction": _ratio(sum(value < 0 for value in end_errors), len(end_errors)),
            "median_iou": _median(float(match["iou"]) for match in all_matches),
        },
    }


def _report_markdown(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics["aggregate"]
    strict = aggregate["strict_candidate_recall"]
    gate = metrics["gate"]
    lines = [
        "# Candidate recall evaluation",
        "",
        f"Created: `{metrics['created_at']}`",
        "",
        "## Decision",
        "",
        f"- Decision: **{gate['decision']}**",
        f"- Evidence status: `{gate['evidence_status']}`",
        f"- Strict candidate recall: **{strict['hits']}/{strict['total']} "
        f"({strict['micro_recall']:.2%})**",
        f"- Engineering target: **{gate['target']:.0%}**",
        "- Precision: unavailable; unmatched candidates remain unknown because there are no "
        "explicit exclude labels.",
        "",
        "## Per source",
        "",
        "| Source | Session | Strict | Candidates |",
        "|---|---|---:|---:|",
    ]
    for source in metrics["sources"]:
        recall = source["strict_candidate_recall"]
        lines.append(
            f"| {source['filename']} | {source['session_id']} | "
            f"{recall['hits']}/{recall['total']} ({recall['recall']:.2%}) | "
            f"{source['candidate_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The strict rule uses half-open integer-millisecond intervals and requires one "
            "candidate core to cover at least 50% of one human highlight. Matching is "
            "source-local, chronological, one-to-one, and maximum-cardinality.",
            "",
            "This import freezes the current active v2 artifacts for diagnosis. Those older "
            "artifacts did not embed their generating Git/config/source receipt, so the result "
            "is not yet a valid formal model baseline even though the files and source videos "
            "are checksummed now.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_name(value: str) -> str:
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


def _relative_display_path(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.name


def freeze_active_candidate_evaluation(
    data_dir: Path,
    *,
    review_database: Path,
    run_id: str | None = None,
    output_root: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Freeze and score current active candidates without changing runtime state."""

    data_dir = data_dir.expanduser().resolve()
    database_path = data_dir / "state.sqlite3"
    review_database = review_database.expanduser().resolve()
    report_progress = progress or (lambda _message: None)
    sources = _live_sources(database_path)
    reviews = _review_evidence(review_database, sources)
    created_at = _utc_now()

    dataset_sources: list[dict[str, Any]] = []
    candidate_sources: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        review = reviews[str(source["upload_id"])]
        source_path = _source_path(str(source["path"]), data_dir)
        actual_size = source_path.stat().st_size
        if int(source["size"]) != actual_size:
            raise CandidateEvaluationError(
                f"Source size disagrees with SQLite for {source['filename']}"
            )
        report_progress(f"[{index}/{len(sources)}] SHA-256 {source['filename']}")
        source_sha256 = _sha256_file(source_path)
        artifact, candidates = _load_active_candidates(data_dir, source, review)
        annotations = [
            {
                "id": annotation["id"],
                "upload_id": source["upload_id"],
                "label": annotation["label"],
                "start_ms": _seconds_to_ms(annotation["start"]),
                "end_ms": _seconds_to_ms(annotation["end"]),
                "note": annotation["note"],
                "created_at": annotation["created_at"],
                "updated_at": annotation["updated_at"],
            }
            for annotation in source["annotations"]
        ]
        dataset_sources.append(
            {
                "upload_id": source["upload_id"],
                "job_id": source["job_id"],
                "filename": source["filename"],
                "session_id": review["session_id"],
                "split": "development",
                "recorded_at": review["recorded_at"],
                "duration_us": _seconds_to_us(review["source_duration"]),
                "byte_size": actual_size,
                "source_sha256": source_sha256,
                "review": review,
                "annotations": annotations,
            }
        )
        candidate_sources.append(
            {
                "upload_id": source["upload_id"],
                "job_id": source["job_id"],
                "filename": source["filename"],
                "artifact": artifact,
                "candidates": candidates,
            }
        )

    dataset_core = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "interval_contract": {
            "unit": "integer-millisecond",
            "semantics": "half-open [start_ms, end_ms)",
            "annotation_semantics": "human-selected point interval",
        },
        "sources": dataset_sources,
    }
    annotation_snapshot_sha256 = _sha256_bytes(_json_bytes(dataset_core))
    dataset = {
        **dataset_core,
        "created_at": created_at,
        "annotation_snapshot_sha256": annotation_snapshot_sha256,
        "review_database": {
            "path": _relative_display_path(review_database, data_dir),
            "sha256": _sha256_file(review_database),
        },
        "precision_eligible": False,
        "precision_ineligible_reason": (
            "No explicit exclude annotations exist; blank reviewed time is not a negative label."
        ),
    }
    candidate_run_core = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "contract": "active-v2-legacy-diagnostic-import",
        "sources": candidate_sources,
    }
    candidate_snapshot_sha256 = _sha256_bytes(_json_bytes(candidate_run_core))
    candidate_run = {
        **candidate_run_core,
        "created_at": created_at,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "generation_receipt_valid": False,
        "git_at_freeze": _git_receipt(data_dir.parent),
    }

    source_metrics: list[dict[str, Any]] = []
    candidates_by_upload = {
        source["upload_id"]: source["candidates"] for source in candidate_sources
    }
    for source in dataset_sources:
        positives = [row for row in source["annotations"] if row["label"] == "highlight"]
        metric_source = {
            "upload_id": source["upload_id"],
            "job_id": source["job_id"],
            "session_id": source["session_id"],
            "filename": source["filename"],
            "duration_seconds": source["duration_us"] / 1_000_000,
        }
        source_metrics.append(
            _metric_source(
                metric_source,
                positives,
                candidates_by_upload[source["upload_id"]],
            )
        )
    aggregate = _aggregate_metrics(source_metrics)
    observed = aggregate["strict_candidate_recall"]["micro_recall"]
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
            "annotation_snapshot_sha256": annotation_snapshot_sha256,
            "candidate_snapshot_sha256": candidate_snapshot_sha256,
            "generation_receipt_valid": False,
        },
        "gate": {
            "metric": "strict_candidate_recall",
            "target": 0.80,
            "observed": observed,
            "threshold_met": observed >= 0.80,
            "evidence_status": "legacy-diagnostic-only",
            "decision": "STOP_DETECTOR",
            "next_component": "impact-detection-point-grouping-and-core-boundaries",
            "ranker_authorized": False,
        },
        "aggregate": aggregate,
        "sources": source_metrics,
        "warnings": [
            "Current active analysis artifacts lack their generation-time "
            "source/config/code receipt.",
            "All five sources are development data; this report is not held-out model accuracy.",
            "Precision, AP, AUROC, NDCG, FPR, and point purity are unavailable "
            "without explicit negatives.",
        ],
    }

    if run_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"active-v2-baseline-{stamp}-{annotation_snapshot_sha256[:8]}"
    run_id = _safe_name(run_id)
    root = (output_root or data_dir / "evaluations" / "candidate-recall").resolve()
    destination = root / run_id
    if destination.exists():
        raise CandidateEvaluationError(f"Evaluation run already exists: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        payloads = {
            "dataset.json": _json_bytes(dataset, pretty=True),
            "candidate-run.json": _json_bytes(candidate_run, pretty=True),
            "metrics.json": _json_bytes(metrics, pretty=True),
            "report.md": _report_markdown(metrics).encode("utf-8"),
        }
        file_hashes: dict[str, str] = {}
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
            file_hashes[filename] = _sha256_bytes(payload)
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": created_at,
            "immutable": True,
            "files": file_hashes,
            "annotation_snapshot_sha256": annotation_snapshot_sha256,
            "candidate_snapshot_sha256": candidate_snapshot_sha256,
        }
        manifest_payload = _json_bytes(manifest, pretty=True)
        (staging / "manifest.json").write_bytes(manifest_payload)
        file_hashes["manifest.json"] = _sha256_bytes(manifest_payload)
        checksums = "".join(
            f"{digest}  {filename}\n" for filename, digest in sorted(file_hashes.items())
        ).encode("ascii")
        (staging / "checksums.sha256").write_bytes(checksums)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    report_progress(f"Frozen evaluation written to {destination}")
    return destination, metrics
