"""Apply clip-specific human presentation decisions without changing event observations."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from pingpong_highlight.edit_comparison import quantize, reel, sha, validate, verify_media, write


def apply_decisions(plan, parent_sha, decisions):
    """Return a new plan; neither input nor unreviewed candidates may be altered."""
    if decisions["parent_plan_sha256"] != parent_sha:
        raise ValueError("Decision parent plan mismatch")
    if decisions["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Decision source mismatch")
    updated = copy.deepcopy(plan)
    rows = {r["candidate_id"]: r for r in updated["candidates"]}
    seen = set()
    for decision in decisions["decisions"]:
        cid = decision["candidate_id"]
        if cid not in rows or cid in seen:
            raise ValueError("Unknown or duplicate candidate decision")
        seen.add(cid)
        if decision["kind"] != "presentation_acceptance":
            raise ValueError("Only presentation acceptance is supported")
        if decision["actor"] != "human-user" or decision["accepted"] is not True:
            raise ValueError("Decision needs explicit human acceptance")
        bounds = decision["requested_source_range"]
        actual = decision["render_source_range"]
        if list(quantize(bounds)) != actual:
            raise ValueError("Decision frame mapping mismatch")
        row = rows[cid]
        a, b = actual
        if not row["context"][0] <= a < b <= row["context"][1]:
            raise ValueError("Accepted clip outside inspected context")
        next_start = row["event"].get("next_start_lower_bound")
        if next_start is not None and b >= next_start:
            raise ValueError("Quantized clip includes next event")
        row["selected_render"] = list(bounds)
        row["selected_render_reason"] = (
            "Exact reviewed clip accepted for presentation; see decision receipt."
        )
        row["presentation_acceptance"] = copy.deepcopy(decision)
        # One accepted rendered sample does not establish an interval of acceptable cuts.
        row["human_acceptable_render"]["accepted_samples"] = [
            {
                "requested_source_range": list(bounds),
                "render_source_range": list(actual),
                "media_sha256": decision["media_sha256"],
                "feedback_record": decision["feedback_record"],
            }
        ]
    updated["version"] = decisions["version"]
    updated["parent_plan_sha256"] = parent_sha
    validate(updated)
    return updated


def load_reviewed_clip(decision, decisions_dir, parent_sha, source_sha):
    """Cross-check decision, original user quote and exact rendered artifact."""
    feedback_path = (decisions_dir / decision["feedback_record"]).resolve()
    feedback = json.loads(feedback_path.read_text())
    for key, expected in {
        "actor": "human-user",
        "channel": "user-chat",
        "candidate_id": decision["candidate_id"],
        "plan_sha256": parent_sha,
        "source_sha256": source_sha,
    }.items():
        if feedback.get(key) != expected:
            raise ValueError("Human feedback binding mismatch")
    if not feedback.get("verbatim") or feedback.get("presentation_accepted") is not True:
        raise ValueError("Human feedback does not explicitly accept presentation")
    if feedback["media"]["sha256"] != decision["media_sha256"]:
        raise ValueError("Feedback media mismatch")
    for field in ("requested_source_range", "render_source_range"):
        if feedback["media"][field] != decision[field]:
            raise ValueError("Feedback range mismatch")
    manifest_path = (decisions_dir / decision["render_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["candidate_id"] != decision["candidate_id"]
        or manifest["source_sha256"] != source_sha
        or manifest["parent_plan_sha256"] != parent_sha
    ):
        raise ValueError("Render identity mismatch")
    clip = manifest["actual_render"]
    for field in ("requested_source_range", "render_source_range"):
        if clip[field] != decision[field]:
            raise ValueError("Render range mismatch")
    path = (manifest_path.parent / clip["file"]).resolve()
    if path.parent != manifest_path.parent or sha(path) != decision["media_sha256"]:
        raise ValueError("Reviewed media checksum/path mismatch")
    if clip["sha256"] != decision["media_sha256"]:
        raise ValueError("Render receipt checksum mismatch")
    return (
        path,
        clip,
        {"feedback_sha256": sha(feedback_path), "manifest_sha256": sha(manifest_path)},
    )


def build_preview(pack: Path, decisions_path: Path, output: Path):
    packet = json.loads((pack / "packet.json").read_text())
    plan = json.loads((pack / "plan.json").read_text())
    parent_sha = sha(pack / "plan.json")
    if parent_sha != packet["plan_sha256"] or packet["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Parent pack binding mismatch")
    decisions = json.loads(decisions_path.read_text())
    updated = apply_decisions(plan, parent_sha, decisions)
    accepted = {d["candidate_id"]: d for d in decisions["decisions"]}
    old = {c["candidate_id"]: c for c in packet["selected_clips"]}
    prepared, receipts = [], {}
    for row in updated["candidates"]:
        cid = row["candidate_id"]
        if cid in accepted:
            path, clip, receipt = load_reviewed_clip(
                accepted[cid], decisions_path.parent, parent_sha, plan["source_sha256"]
            )
            receipts[cid] = receipt
        else:
            clip = old[cid]
            path = (pack / clip["file"]).resolve()
            if path.parent != pack.resolve() or sha(path) != clip["sha256"]:
                raise ValueError("Unreviewed media changed")
        if clip["render_source_range"] != list(quantize(row["selected_render"])):
            raise ValueError("Selected media/plan range mismatch")
        prepared.append((cid, path, clip))
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", updated)
    clips = []
    for i, (cid, path, clip) in enumerate(prepared, 1):
        target = output / f"point-{i}.mp4"
        shutil.copyfile(path, target)
        a, b = clip["render_source_range"]
        verified = verify_media(target, b - a)
        if verified["sha256"] != clip["sha256"]:
            raise ValueError("Copied media changed")
        clips.append(
            {k: v for k, v in clip.items() if k not in {"command", "file"}}
            | {
                "file": target.name,
                "candidate_id": cid,
                **verified,
                "presentation_status": "human-accepted" if cid in accepted else "pending-human",
            }
        )
    preview = reel(clips, updated["candidates"], output / "preview.mp4")
    result = {
        "schema": "rally-edit-decision-preview-v1",
        "source_sha256": plan["source_sha256"],
        "parent_plan_sha256": parent_sha,
        "plan_sha256": sha(output / "plan.json"),
        "decisions_sha256": sha(decisions_path),
        "code_sha256": sha(__file__),
        "frozen_order": updated["frozen_order"],
        "selected_clips": clips,
        "pairs": [],
        "reels": [preview],
        "decision_receipts": receipts,
        "presentation_accepted_count": len(accepted),
        "event_completeness_human_confirmed_count": sum(
            r["event"]["human_complete"] is True for r in updated["candidates"]
        ),
        "compilation_status": "preview-pending-human",
        "scope": "Same six candidates/order, direct cuts; no new transition or ranking comparison.",
    }
    write(output / "packet.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_preview(args.pack, args.decisions, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "duration": result["reels"][0]["duration"],
                "accepted": result["presentation_accepted_count"],
                "status": result["compilation_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
