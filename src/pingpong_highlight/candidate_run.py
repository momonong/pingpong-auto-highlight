from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from pingpong_highlight.config import Settings
from pingpong_highlight.media_work import media_work_lock
from pingpong_highlight.pipeline.audio import analyze_audio
from pingpong_highlight.pipeline.detect import DetectionConfig, detect_points
from pingpong_highlight.pipeline.media import has_nvdec, probe_media
from pingpong_highlight.pipeline.motion import analyze_motion

CANDIDATE_RUN_SCHEMA_VERSION = 1
CANDIDATE_ALGORITHM_VERSION = "candidate-generation-v3"


class CandidateRunError(RuntimeError):
    """Raised when a reproducible candidate-only run cannot continue safely."""


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
        raise CandidateRunError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _safe_name(value: str) -> str:
    if (
        not value
        or len(value) > 100
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for character in value
        )
    ):
        raise CandidateRunError(
            "Run ID must contain only letters, numbers, dot, dash, and underscore"
        )
    return value


def _load_dataset(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    try:
        payload = resolved.read_bytes()
        dataset = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateRunError(f"Cannot read dataset manifest: {exc}") from exc
    if not isinstance(dataset, dict) or not isinstance(dataset.get("sources"), list):
        raise CandidateRunError("Dataset manifest has no sources")
    if dataset.get("schema_version") != 1:
        raise CandidateRunError("Unsupported dataset schema version")
    expected_snapshot = dataset.get("annotation_snapshot_sha256")
    core = {
        "schema_version": dataset.get("schema_version"),
        "interval_contract": dataset.get("interval_contract"),
        "sources": dataset.get("sources"),
    }
    if expected_snapshot != _sha256_bytes(_json_bytes(core)):
        raise CandidateRunError("Dataset annotation snapshot checksum is invalid")
    return dataset, _sha256_bytes(payload)


def _git_receipt(start: Path) -> dict[str, Any]:
    try:
        root = Path(
            subprocess.run(
                ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CandidateRunError("Git receipt is unavailable") from exc
    return {
        "repository_root_name": root.name,
        "commit": commit,
        "clean": not bool(status),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
    }


def _gpu_receipt() -> dict[str, Any]:
    if not has_nvdec():
        return {"nvdec_available": False, "device": None, "driver": None}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first = result.stdout.splitlines()[0]
        device, separator, driver = first.rpartition(",")
        return {
            "nvdec_available": True,
            "device": device.strip() if separator else first.strip(),
            "driver": driver.strip() if separator else None,
        }
    except (OSError, subprocess.CalledProcessError, IndexError):
        return {"nvdec_available": True, "device": None, "driver": None}


def _source_path(value: str, data_dir: Path) -> Path:
    container_path = PurePosixPath(value.replace("\\", "/"))
    if container_path.is_absolute() and container_path.parts[:2] == ("/", "data"):
        path = data_dir.joinpath(*container_path.parts[2:]).resolve()
    else:
        path = Path(value).expanduser()
        path = (data_dir.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise CandidateRunError(f"Source video is not available locally: {path}")
    return path


def _live_source(data_dir: Path, upload_id: str) -> dict[str, Any]:
    database = (data_dir / "state.sqlite3").resolve()
    connection = sqlite3.connect(
        f"file:{database.as_posix()}?mode=ro",
        uri=True,
        timeout=30.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        row = connection.execute(
            """
            SELECT u.id AS upload_id, u.filename, u.size, u.path,
                   j.id AS job_id, j.status
            FROM uploads AS u
            JOIN jobs AS j ON j.upload_id = u.id
            WHERE u.id = ?
            """,
            (upload_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None or row["status"] != "completed":
        raise CandidateRunError(f"Completed live source not found: {upload_id}")
    return dict(row)


def _configuration(settings: Settings) -> tuple[dict[str, Any], DetectionConfig]:
    detection = DetectionConfig(
        minimum_point_score_ratio=settings.library_minimum_point_score_ratio,
        max_points=None,
        target_reel_duration=None,
        pre_roll=settings.clip_pre_roll_seconds,
        post_roll=settings.clip_post_roll_seconds,
    )
    configuration = {
        "algorithm_version": CANDIDATE_ALGORITHM_VERSION,
        "audio_sample_rate": settings.audio_sample_rate,
        "video_sample_fps": settings.video_sample_fps,
        "analysis_frame_size": settings.analysis_frame_size,
        "detection": asdict(detection),
        "candidate_policy": (
            "audio candidates when any accepted audio group exists; otherwise motion fallback"
        ),
    }
    return configuration, detection


def _save_signals(
    path: Path,
    *,
    audio_times: np.ndarray,
    audio_scores: np.ndarray,
    motion_times: np.ndarray,
    motion_scores: np.ndarray,
) -> None:
    temporary = path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}.npz")
    np.savez_compressed(
        temporary,
        audio_times=audio_times,
        audio_scores=audio_scores,
        motion_times=motion_times,
        motion_scores=motion_scores,
    )
    os.replace(temporary, path)


def _state_invariant(
    *,
    dataset_sha256: str,
    annotation_snapshot_sha256: str,
    configuration_sha256: str,
    git_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "annotation_snapshot_sha256": annotation_snapshot_sha256,
        "configuration_sha256": configuration_sha256,
        "git_commit": git_receipt["commit"],
        "git_status_sha256": git_receipt["status_sha256"],
    }


def _load_or_create_state(
    partial: Path,
    *,
    run_id: str,
    created_at: str,
    invariant: dict[str, Any],
    configuration: dict[str, Any],
    git_receipt: dict[str, Any],
    gpu_receipt: dict[str, Any],
) -> dict[str, Any]:
    state_path = partial / "run-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateRunError("Partial run state is unreadable") from exc
        if state.get("invariant") != invariant:
            raise CandidateRunError(
                "Partial run belongs to different data, config, code, or worktree state"
            )
        return state
    partial.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": CANDIDATE_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "status": "running",
        "invariant": invariant,
        "configuration": configuration,
        "git": git_receipt,
        "gpu": gpu_receipt,
        "completed_sources": [],
    }
    _atomic_write(state_path, _json_bytes(state, pretty=True))
    return state


def _valid_completed_source(partial: Path, descriptor: dict[str, Any]) -> bool:
    try:
        artifact = partial / descriptor["artifact"]
        signals = partial / descriptor["signals"]
        return (
            artifact.is_file()
            and signals.is_file()
            and _sha256_file(artifact) == descriptor["artifact_sha256"]
            and _sha256_file(signals) == descriptor["signals_sha256"]
        )
    except (KeyError, OSError):
        return False


def run_candidate_analysis(
    settings: Settings,
    *,
    dataset_path: Path,
    run_id: str,
    output_root: Path | None = None,
    require_gpu: bool = True,
    allow_dirty: bool = False,
    progress: Any = None,
) -> Path:
    """Run detector analysis without exporting clips or mutating the runtime database."""

    report = progress or (lambda _message: None)
    run_id = _safe_name(run_id)
    dataset, dataset_sha256 = _load_dataset(dataset_path)
    git_receipt = _git_receipt(Path.cwd())
    if not git_receipt["clean"] and not allow_dirty:
        raise CandidateRunError(
            "Formal candidate runs require a clean worktree; commit changes first"
        )
    gpu_receipt = _gpu_receipt()
    if require_gpu and not gpu_receipt["nvdec_available"]:
        raise CandidateRunError("NVIDIA NVDEC is required for candidate analysis")
    configuration, detection_config = _configuration(settings)
    configuration_sha256 = _sha256_bytes(_json_bytes(configuration))
    invariant = _state_invariant(
        dataset_sha256=dataset_sha256,
        annotation_snapshot_sha256=dataset["annotation_snapshot_sha256"],
        configuration_sha256=configuration_sha256,
        git_receipt=git_receipt,
    )

    root = (
        (output_root or settings.data_dir / "evaluations" / "candidate-runs").expanduser().resolve()
    )
    destination = root / run_id
    partial = root / f".{run_id}.partial"
    if destination.exists():
        raise CandidateRunError(f"Candidate run already exists: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    state = _load_or_create_state(
        partial,
        run_id=run_id,
        created_at=_utc_now(),
        invariant=invariant,
        configuration=configuration,
        git_receipt=git_receipt,
        gpu_receipt=gpu_receipt,
    )
    completed = {
        descriptor["upload_id"]: descriptor
        for descriptor in state["completed_sources"]
        if _valid_completed_source(partial, descriptor)
    }

    with media_work_lock(settings.data_dir):
        for source_index, dataset_source in enumerate(dataset["sources"], start=1):
            upload_id = str(dataset_source["upload_id"])
            if upload_id in completed:
                report(
                    f"[{source_index}/{len(dataset['sources'])}] resume "
                    f"{dataset_source['filename']}"
                )
                continue
            live = _live_source(settings.data_dir, upload_id)
            if live["filename"] != dataset_source["filename"]:
                raise CandidateRunError(f"Source filename drift for {upload_id}")
            source_path = _source_path(str(live["path"]), settings.data_dir)
            source_size = source_path.stat().st_size
            if source_size != int(dataset_source["byte_size"]):
                raise CandidateRunError(f"Source size drift for {live['filename']}")
            report(f"[{source_index}/{len(dataset['sources'])}] verify SHA-256 {live['filename']}")
            source_sha256 = _sha256_file(source_path)
            if source_sha256 != dataset_source["source_sha256"]:
                raise CandidateRunError(f"Source SHA-256 drift for {live['filename']}")

            media = probe_media(source_path)
            expected_duration = int(dataset_source["duration_us"]) / 1_000_000
            if not math.isclose(media.duration, expected_duration, abs_tol=0.001):
                raise CandidateRunError(f"Source duration drift for {live['filename']}")

            last_audio_percent = -10

            def audio_progress(value: float) -> None:
                nonlocal last_audio_percent
                percent = int(value * 10) * 10
                if percent >= last_audio_percent + 10:
                    report(f"  audio {percent}%")
                    last_audio_percent = percent

            report("  audio analysis")
            audio = analyze_audio(
                source_path,
                media,
                sample_rate=settings.audio_sample_rate,
                progress=audio_progress,
            )

            last_motion_percent = -10

            def motion_progress(value: float) -> None:
                nonlocal last_motion_percent
                percent = int(value * 10) * 10
                if percent >= last_motion_percent + 10:
                    report(f"  GPU motion {percent}%")
                    last_motion_percent = percent

            report("  GPU motion analysis")
            motion = analyze_motion(
                source_path,
                media,
                fps=settings.video_sample_fps,
                frame_size=settings.analysis_frame_size,
                progress=motion_progress,
            )
            detection = detect_points(
                media.duration,
                audio,
                motion,
                detection_config,
            )

            source_directory = partial / "sources" / upload_id
            source_directory.mkdir(parents=True, exist_ok=True)
            signals_path = source_directory / "signals.npz"
            _save_signals(
                signals_path,
                audio_times=audio.times,
                audio_scores=audio.scores,
                motion_times=motion.times,
                motion_scores=motion.scores,
            )
            artifact = {
                "schema_version": CANDIDATE_RUN_SCHEMA_VERSION,
                "artifact_type": "candidate-generation-source",
                "algorithm_version": CANDIDATE_ALGORITHM_VERSION,
                "created_at": _utc_now(),
                "run_id": run_id,
                "source": {
                    "upload_id": upload_id,
                    "job_id": live["job_id"],
                    "filename": live["filename"],
                    "byte_size": source_size,
                    "source_sha256": source_sha256,
                    "duration": media.duration,
                    "media": media.to_dict() | {"path": live["filename"]},
                },
                "receipt": {
                    "dataset_sha256": dataset_sha256,
                    "annotation_snapshot_sha256": dataset["annotation_snapshot_sha256"],
                    "configuration_sha256": configuration_sha256,
                    "git": git_receipt,
                    "gpu": gpu_receipt,
                },
                "configuration": configuration,
                "signals": {
                    "path": "signals.npz",
                    "audio_sample_count": int(audio.scores.size),
                    "motion_sample_count": int(motion.scores.size),
                },
                "audio_events": [
                    {"time": round(event.time, 6), "strength": round(event.strength, 6)}
                    for event in audio.events
                ],
                "audio_groups": [group.to_dict() for group in detection.audio_groups],
                "candidate_mode": detection.candidate_mode,
                "candidates": [candidate.to_dict() for candidate in detection.candidates],
                "motion_candidates": [
                    candidate.to_dict() for candidate in detection.motion_candidates
                ],
                "summary": {
                    "audio_event_count": len(audio.events),
                    "audio_group_count": len(detection.audio_groups),
                    "accepted_audio_group_count": sum(
                        group.accepted for group in detection.audio_groups
                    ),
                    "candidate_count": len(detection.candidates),
                    "motion_candidate_count": len(detection.motion_candidates),
                },
            }
            artifact_path = source_directory / "candidates.json"
            _atomic_write(artifact_path, _json_bytes(artifact, pretty=True))
            descriptor = {
                "upload_id": upload_id,
                "job_id": live["job_id"],
                "filename": live["filename"],
                "artifact": artifact_path.relative_to(partial).as_posix(),
                "artifact_sha256": _sha256_file(artifact_path),
                "signals": signals_path.relative_to(partial).as_posix(),
                "signals_sha256": _sha256_file(signals_path),
                "candidate_count": len(detection.candidates),
                "motion_candidate_count": len(detection.motion_candidates),
            }
            completed[upload_id] = descriptor
            state["completed_sources"] = [completed[key] for key in sorted(completed)]
            _atomic_write(
                partial / "run-state.json",
                _json_bytes(state, pretty=True),
            )

    expected_uploads = {str(source["upload_id"]) for source in dataset["sources"]}
    if set(completed) != expected_uploads:
        raise CandidateRunError("Candidate run did not complete every dataset source")
    manifest = {
        "schema_version": CANDIDATE_RUN_SCHEMA_VERSION,
        "artifact_type": "candidate-generation-run",
        "algorithm_version": CANDIDATE_ALGORITHM_VERSION,
        "run_id": run_id,
        "created_at": state["created_at"],
        "completed_at": _utc_now(),
        "status": "completed",
        "dataset": {
            "path": dataset_path.expanduser().resolve().name,
            "sha256": dataset_sha256,
            "annotation_snapshot_sha256": dataset["annotation_snapshot_sha256"],
        },
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "git": git_receipt,
        "gpu": gpu_receipt,
        "generation_receipt_valid": git_receipt["clean"],
        "sources": [completed[key] for key in sorted(completed)],
    }
    manifest_payload = _json_bytes(manifest, pretty=True)
    _atomic_write(partial / "manifest.json", manifest_payload)
    state["status"] = "completed"
    state["completed_at"] = manifest["completed_at"]
    _atomic_write(partial / "run-state.json", _json_bytes(state, pretty=True))
    os.replace(partial, destination)
    report(f"Candidate-only run written to {destination}")
    return destination
