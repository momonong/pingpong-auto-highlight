from __future__ import annotations

import json
import sqlite3
from fractions import Fraction
from pathlib import Path

import pytest

from pingpong_highlight.candidate_evaluation import (
    CandidateEvaluationError,
    annotation_review_digest,
    freeze_active_candidate_evaluation,
    match_intervals,
)
from pingpong_highlight.config import Settings
from pingpong_highlight.db import Database


def _annotation(annotation_id: str, start_ms: int, end_ms: int) -> dict[str, object]:
    return {"id": annotation_id, "start_ms": start_ms, "end_ms": end_ms}


def _candidate(candidate_id: str, start_ms: int, end_ms: int) -> dict[str, object]:
    return {
        "id": candidate_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "score": 10.0,
        "score_rank_all": 1,
        "selection": "selected",
    }


def test_strict_match_includes_exactly_half_but_not_below_or_touching() -> None:
    annotations = [_annotation("point", 0, 10_000)]

    exact = match_intervals(annotations, [_candidate("exact", 5_000, 10_000)])
    below = match_intervals(annotations, [_candidate("below", 5_001, 10_000)])
    touching = match_intervals(annotations, [_candidate("touch", 10_000, 12_000)])

    assert len(exact) == 1
    assert exact[0]["annotation_coverage"] == 0.5
    assert below == []
    assert touching == []


def test_matching_is_one_to_one_and_deterministic() -> None:
    annotations = [
        _annotation("first", 0, 10_000),
        _annotation("second", 20_000, 30_000),
    ]
    candidate = _candidate("shared", 5_000, 25_000)

    forward = match_intervals(annotations, [candidate])
    reverse = match_intervals(list(reversed(annotations)), [candidate])

    assert len(forward) == 1
    assert forward == reverse
    assert forward[0]["annotation_id"] == "first"


def test_loose_overlap_does_not_change_strict_gate() -> None:
    annotations = [_annotation("point", 0, 10_000)]
    candidates = [_candidate("nearby", 9_000, 12_000)]

    strict = match_intervals(annotations, candidates)
    loose = match_intervals(
        annotations,
        candidates,
        minimum_coverage=None,
        minimum_intersection_ms=500,
    )

    assert strict == []
    assert len(loose) == 1
    assert loose[0]["annotation_coverage"] == 0.1


def _result(filename: str, duration: float) -> dict[str, object]:
    candidate = {
        "rally_start": 5.0,
        "rally_end": 10.0,
        "rally_duration": 5.0,
        "score": 12.0,
        "impact_count": 5,
        "motion_score": 1.0,
        "reason": "test candidate",
        "selection": "selected",
        "rank": 1,
    }
    point = {
        "start": 4.0,
        "end": 11.0,
        "clip_start": 4.0,
        "clip_end": 11.0,
        "rally_start": 5.0,
        "rally_end": 10.0,
        "score": 12.0,
        "rank": 1,
        "reason": "test candidate",
    }
    return {
        "algorithm_version": "highlight-library-v2",
        "source_name": filename,
        "media": {"duration": duration},
        "summary": {"candidate_point_count": 1, "point_count": 1},
        "selection": {
            "library_minimum_point_score_ratio": 0.7,
            "recommendation_score_ratio": 0.87,
        },
        "candidates": [candidate],
        "points": [point],
        "files": [
            {"name": "highlight_001_rank_001.mp4", "kind": "highlight"},
            {"name": "analysis.json", "kind": "analysis"},
        ],
    }


def _review_database(
    path: Path,
    *,
    upload_id: str,
    duration: float,
    annotations: list[object],
) -> None:
    rows = [
        {
            "id": annotation.id,
            "label": annotation.label,
            "start": annotation.start,
            "end": annotation.end,
            "note": annotation.note,
            "created_at": annotation.created_at,
            "updated_at": annotation.updated_at,
        }
        for annotation in annotations
    ]
    digest = annotation_review_digest(rows)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE annotations (
                id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                label TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE source_reviews (
                upload_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                recorded_at TEXT,
                review_complete INTEGER NOT NULL,
                source_duration REAL NOT NULL,
                reviewed_until REAL,
                reviewed_at TEXT,
                annotation_count INTEGER NOT NULL,
                annotation_digest TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO annotations
                (id, upload_id, label, start, end, note, created_at, updated_at)
            VALUES (:id, :upload_id, :label, :start, :end, :note, :created_at, :updated_at)
            """,
            [row | {"upload_id": upload_id} for row in rows],
        )
        connection.execute(
            """
            INSERT INTO source_reviews (
                upload_id, session_id, recorded_at, review_complete,
                source_duration, reviewed_until, reviewed_at, annotation_count,
                annotation_digest, created_at, updated_at
            ) VALUES (?, 'session-one', '2026-01-01', 1, ?, ?,
                      '2026-08-24T00:00:00+00:00', ?, ?,
                      '2026-08-24T00:00:00+00:00', '2026-08-24T00:00:00+00:00')
            """,
            (upload_id, duration, duration, len(rows), digest),
        )


def _evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    settings = Settings(data_dir=tmp_path / "data", upload_token="test")
    settings.ensure_directories()
    database = Database(settings.database_path)
    upload_id = "upload-one"
    filename = "PXL_20260101_120000000.mp4"
    source_path = settings.uploads_dir / f"{upload_id}.mp4"
    source_path.write_bytes(b"small-video-fixture")
    _upload, job = database.register_completed_upload(
        upload_id,
        filename,
        source_path.stat().st_size,
        "video/mp4",
        source_path,
    )
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            "UPDATE uploads SET path = ? WHERE id = ?",
            (f"/data/uploads/{source_path.name}", upload_id),
        )
    assert database.claim_job(job.id)
    result = _result(filename, 20.0)
    database.finish_job(job.id, result)
    clip_set = Path("clip-sets") / "highlight-library-v2-test"
    output = settings.outputs_dir / job.id / clip_set
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    (output / "highlight_001_rank_001.mp4").write_bytes(b"clip")
    database.activate_highlight_result(
        job.id,
        result,
        file_prefix=clip_set.as_posix(),
        library_version="highlight-library-v2",
    )
    annotation = database.create_annotation(
        upload_id,
        label="highlight",
        start=0.0,
        end=10.0,
        note="exact half",
    )
    review_path = settings.data_dir / "state.training-baseline-test.sqlite3"
    _review_database(
        review_path,
        upload_id=upload_id,
        duration=20.0,
        annotations=[annotation],
    )
    return settings.data_dir, review_path, job.id, upload_id


def test_freeze_active_evaluation_is_atomic_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    data_dir, review_path, _job_id, _upload_id = _evaluation_fixture(tmp_path)
    output_root = tmp_path / "evaluations"

    destination, metrics = freeze_active_candidate_evaluation(
        data_dir,
        review_database=review_path,
        run_id="test-run",
        output_root=output_root,
    )

    assert metrics["aggregate"]["strict_candidate_recall"] == {
        "hits": 1,
        "total": 1,
        "micro_recall": 1.0,
        "macro_source_recall": 1.0,
    }
    assert metrics["gate"]["evidence_status"] == "legacy-diagnostic-only"
    assert (destination / "dataset.json").is_file()
    assert (destination / "candidate-run.json").is_file()
    assert (destination / "metrics.json").is_file()
    assert (destination / "report.md").is_file()
    assert (destination / "manifest.json").is_file()
    assert (destination / "checksums.sha256").is_file()
    assert list(output_root.glob(".test-run.tmp-*")) == []

    with pytest.raises(CandidateEvaluationError, match="already exists"):
        freeze_active_candidate_evaluation(
            data_dir,
            review_database=review_path,
            run_id="test-run",
            output_root=output_root,
        )


def test_freeze_fails_closed_when_active_artifact_is_missing(tmp_path: Path) -> None:
    data_dir, review_path, job_id, _upload_id = _evaluation_fixture(tmp_path)
    analysis = (
        data_dir / "outputs" / job_id / "clip-sets" / "highlight-library-v2-test" / "analysis.json"
    )
    analysis.unlink()

    with pytest.raises(CandidateEvaluationError, match="artifact is missing"):
        freeze_active_candidate_evaluation(
            data_dir,
            review_database=review_path,
            run_id="missing",
            output_root=tmp_path / "evaluations",
        )


def test_review_digest_is_stable_to_input_order_and_sensitive_to_note() -> None:
    first = {
        "id": "a",
        "label": "highlight",
        "start": 1.0,
        "end": 2.0,
        "note": "rally",
        "created_at": "now",
        "updated_at": "now",
    }
    second = first | {"id": "b", "start": 3.0, "end": 4.0}

    assert annotation_review_digest([first, second]) == annotation_review_digest([second, first])
    assert annotation_review_digest([first]) != annotation_review_digest(
        [first | {"note": "attack"}]
    )


def test_custom_fraction_threshold_is_supported() -> None:
    annotations = [_annotation("point", 0, 10_000)]
    candidates = [_candidate("quarter", 7_500, 10_000)]

    assert (
        len(
            match_intervals(
                annotations,
                candidates,
                minimum_coverage=Fraction(1, 4),
            )
        )
        == 1
    )
