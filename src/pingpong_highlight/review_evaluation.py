"""Source-local maximum-cardinality matching; missing review is never a negative."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

from pingpong_highlight.rally_review import gaps, union


def match(predictions: list[dict], truth: list[dict], *, padded=False) -> dict:
    if any(not item.get("source_id") for item in [*predictions, *truth]):
        raise ValueError("Matching requires explicit source identity on every interval")
    start_key, end_key = ("clip_start_ms", "clip_end_ms") if padded else ("start_ms", "end_ms")
    edges = []
    for predicted in predictions:
        possible = []
        for j, actual in enumerate(truth):
            # Source identity must match even when callers accidentally mix sources.
            if predicted.get("source_id") != actual.get("source_id"):
                continue
            overlap = max(
                0,
                min(predicted[end_key], actual["end_ms"])
                - max(predicted[start_key], actual["start_ms"]),
            )
            if overlap / (actual["end_ms"] - actual["start_ms"]) >= 0.5:
                possible.append((overlap, j))
        edges.append([j for _, j in sorted(possible, reverse=True)])
    owners = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or augment(owners[j], visited):
                owners[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    pairs = [
        {
            "prediction": predictions[i]["id"],
            "truth": truth[j]["id"],
            "start_error_ms": predictions[i][start_key] - truth[j]["start_ms"],
            "end_error_ms": predictions[i][end_key] - truth[j]["end_ms"],
        }
        for j, i in sorted(owners.items())
    ]
    return {
        "matched": len(pairs),
        "truth_count": len(truth),
        "prediction_count": len(predictions),
        "recall": len(pairs) / len(truth) if truth else "UNKNOWN",
        "pairs": pairs,
        "unmatched_truth": [t["id"] for j, t in enumerate(truth) if j not in owners],
        "unmatched_predictions": [
            p["id"] for i, p in enumerate(predictions) if i not in owners.values()
        ],
        "median_start_absolute_ms": median(abs(p["start_error_ms"]) for p in pairs)
        if pairs
        else None,
        "median_end_absolute_ms": median(abs(p["end_error_ms"]) for p in pairs) if pairs else None,
    }


def evaluate_review(export: dict) -> dict:
    source, review = export["source"], export["review"]
    truth = [p | {"source_id": source["id"]} for p in review["points"] if p["rally"] == "yes"]
    output = {
        "source_id": source["id"],
        "group": source["group"],
        "semantics": review["semantics"],
        "review_revision": review["revision"],
        "review_coverage": review["coverage"],
        "active_review_ms": review["active_ms"],
        "modification_count": review["modification_count"],
        "time_savings": "UNKNOWN",
        "reel_utility": "UNKNOWN - requires watching the actual 55-second Reel",
        "matching": "maximum cardinality, source-local, >=50% human interval overlap",
        "runs": [],
    }
    for run in export["runs"]:
        scope = union(run["windows"])

        def inside(p, intervals=scope):
            return any(
                w["start_ms"] <= p["start_ms"] < p["end_ms"] <= w["end_ms"] for w in intervals
            )

        def intersects(p, intervals=scope):
            return any(
                p["start_ms"] < w["end_ms"] and w["start_ms"] < p["end_ms"] for w in intervals
            )

        local_truth = [p for p in truth if inside(p)]
        selected_truth = [p for p in local_truth if p["highlight"] in ("include", "must")]
        # Positive/uncertain proposals are review candidates. Explicit no proposals are separate.
        proposed = [p | {"source_id": source["id"]} for p in run["proposals"] if p["rally"] != "no"]
        selected = [p for p in proposed if p["selected"]]
        candidates = match(proposed, local_truth)
        selected_core = match(selected, selected_truth)
        selected_padding = match(selected, selected_truth, padded=True)
        coverage_complete = bool(scope) and all(
            not gaps(review["coverage"], w["start_ms"], w["end_ms"]) for w in scope
        )
        unresolved = [
            p
            for p in review["points"]
            if intersects(p)
            and (
                p["rally"] in ("uncertain", "unable")
                or (p["rally"] == "yes" and p["complete"] in ("uncertain", "unable"))
            )
        ]
        crossing = [p["id"] for p in truth if not inside(p) and intersects(p)]
        # A truncated human rally is excluded from this recall denominator, so it cannot
        # safely turn a matching boundary proposal into a false positive either.
        complete = coverage_complete and not unresolved and not crossing
        ratings_complete = complete and all(p["highlight"] != "unrated" for p in local_truth)
        corrections = []
        by_id = {p["id"]: p for p in run["proposals"]}
        for p in review["points"]:
            for pid in p["proposal_ids"]:
                if pid in by_id:
                    corrections.append(
                        {
                            "proposal": pid,
                            "human": p["id"],
                            "start_delta_ms": p["start_ms"] - by_id[pid]["start_ms"],
                            "end_delta_ms": p["end_ms"] - by_id[pid]["end_ms"],
                        }
                    )
        output["runs"].append(
            {
                "run_id": run["id"],
                "model_id": run["model_id"],
                "commit": run["commit"],
                "model_revision": run["model_revision"],
                "candidate_core": candidates,
                "selected_core": selected_core,
                "selected_padding": selected_padding,
                "candidate_precision": candidates["matched"] / len(proposed)
                if complete and proposed
                else "UNKNOWN",
                "selection_precision": selected_core["matched"] / len(selected)
                if ratings_complete and selected
                else "UNKNOWN",
                "precision_scope": "fully reviewed analyzed intervals only; otherwise UNKNOWN",
                "scope": scope,
                "coverage_complete": complete,
                "boundary_crossing_truth": crossing,
                "boundary_corrections": corrections,
                "elapsed_seconds": run["elapsed_seconds"],
                "raw_output_errors": len(run.get("errors", [])),
            }
        )
    return output


def compare_legacy(dataset: Path, reports: Path, *, current_runs: bool = False) -> dict:
    """Read positive-only legacy annotations without importing them as complete rally truth."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    label_bytes = (dataset / "annotations.jsonl").read_bytes()
    if hashlib.sha256(label_bytes).hexdigest() != manifest["files"]["annotations.jsonl"]["sha256"]:
        raise ValueError("Annotation snapshot hash mismatch")
    annotations = [json.loads(line) for line in label_bytes.decode("utf-8").splitlines()]
    runs = json.loads(reports.read_text(encoding="utf-8")) if current_runs else None
    if current_runs:
        for run in runs:
            if len(run.get("commit", "")) != 40 or len(run.get("code_sha256", "")) != 64:
                raise ValueError("Bounded run is missing a complete commit/code receipt")
    sources = []
    for source in manifest["sources"]:
        sid = source["source_sha256"]
        truth = [
            {
                "id": p["annotation_id"],
                "source_id": sid,
                "start_ms": round(p["start"] * 1000),
                "end_ms": round(p["end"] * 1000),
            }
            for p in annotations
            if p["upload_id"] == source["upload_id"] and p["label"] == "highlight"
        ]
        if current_runs:
            local_runs = [r for r in runs if r["source_id"] == sid]
        else:
            path = reports / source["upload_id"] / "analysis.json"
            raw = path.read_bytes()
            report = json.loads(raw)

            def convert(p, index, selected=False, source_id=sid):
                return {
                    "id": str(index),
                    "source_id": source_id,
                    "start_ms": round(p["rally_start"] * 1000),
                    "end_ms": round(p["rally_end"] * 1000),
                    "clip_start_ms": round(p.get("start", p["rally_start"]) * 1000),
                    "clip_end_ms": round(p.get("end", p["rally_end"]) * 1000),
                    "selected": selected,
                }

            local_runs = [
                {
                    "id": source["upload_id"],
                    "model_id": report["algorithm_version"],
                    "proposals": [convert(p, i) for i, p in enumerate(report["candidates"])],
                    "selected_points": [
                        convert(p, i, True) for i, p in enumerate(report["points"])
                    ],
                    "windows": [{"start_ms": 0, "end_ms": round(source["source_duration"] * 1000)}],
                    "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                    "commit": "UNKNOWN-old-artifact",
                }
            ]
        for run in local_runs:
            scope = union(run["windows"])
            local_truth = [
                t
                for t in truth
                if any(w["start_ms"] <= t["start_ms"] < t["end_ms"] <= w["end_ms"] for w in scope)
            ]
            candidates = [
                p | {"source_id": sid} for p in run["proposals"] if p.get("rally") != "no"
            ]
            selected = run.get("selected_points", [p for p in candidates if p["selected"]])
            sources.append(
                {
                    "source_id": sid,
                    "run_id": run["id"],
                    "model_id": run["model_id"],
                    "commit": run["commit"],
                    "artifact_sha256": run.get("artifact_sha256"),
                    "known_positive_count": len(local_truth),
                    "scope": scope,
                    "candidate_core": match(candidates, local_truth),
                    "selected_core": match(selected, local_truth),
                    "selected_padding": match(selected, local_truth, padded=True),
                    "precision": "UNKNOWN",
                    "scope_boundary_positives": [
                        t["id"]
                        for t in truth
                        if t not in local_truth
                        and any(
                            t["start_ms"] < w["end_ms"] and w["start_ms"] < t["end_ms"]
                            for w in scope
                        )
                    ],
                }
            )
    keys = ["candidate_core", "selected_core", "selected_padding"]
    return {
        "evidence": "bounded run artifacts; inspect recorded commit and code hash"
        if current_runs
        else "recomputed historical artifacts, not current HEAD inference",
        "group": "development",
        "current_version_gate": "UNKNOWN; receipt alone does not establish quality",
        "runs_sha256": hashlib.sha256(reports.read_bytes()).hexdigest() if current_runs else None,
        "label_semantics": "legacy positive highlights only; not all rallies",
        "unannotated": "UNKNOWN",
        "precision": "UNKNOWN",
        "reel_utility": "UNKNOWN",
        "annotations_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "sources": sources,
        "totals": {
            "positive_count": sum(s["known_positive_count"] for s in sources),
            **{k: sum(s[k]["matched"] for s in sources) for k in keys},
        },
    }
