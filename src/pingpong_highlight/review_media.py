"""Full-length human-review proxies, independent of model experiment windows."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from pingpong_highlight.rally_review import ReviewStore, source_identity


def preview_manifest(store: ReviewStore, source_id: str) -> Path:
    # IDs come from registered source identities, never an arbitrary filesystem path.
    store.source(source_id)
    if len(source_id) != 64 or any(c not in "0123456789abcdef" for c in source_id):
        raise ValueError("Invalid source identity")
    directory = store.path.resolve().parent / "review-media" / source_id
    if not directory.resolve().is_relative_to(store.path.resolve().parent):
        raise ValueError("Review media path escaped experiment")
    return directory / "preview.json"


def full_preview(store: ReviewStore, source_id: str) -> dict | None:
    manifest = preview_manifest(store, source_id)
    if not manifest.is_file():
        return None
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if value.get("media", {}).get("pixel_format") != "yuv420p":
        return None  # Regenerate older 10-bit proxies for browser compatibility.
    path = Path(value["path"]).resolve()
    source = store.source(source_id)
    if (
        value["source_id"] != source_id
        or value["start_ms"] != 0
        or value["end_ms"] != source["duration_ms"]
        or not path.is_relative_to(manifest.parent.resolve())
        or not path.is_file()
        or path.stat().st_size != value["size"]
    ):
        raise ValueError("Invalid full-review preview receipt")
    return value


def prepare_review(store: ReviewStore, source_id: str) -> dict:
    """One explicitly selected source, no model inference and no review DB mutation."""
    from pingpong_highlight.pipeline.media import probe_media
    from pingpong_highlight.preannotate import extract

    source = store.source(source_id)
    if source_identity(Path(source["path"]))["id"] != source_id:
        raise ValueError("Source changed; refusing to prepare mismatched video")
    existing = full_preview(store, source_id)
    if existing:
        return existing
    manifest = preview_manifest(store, source_id)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output = manifest.parent / f"full-{uuid.uuid4().hex}.mp4"
    command = extract(
        Path(source["path"]), output, 0, source["duration_ms"], pixel_format="yuv420p"
    )
    media = probe_media(output)
    if abs(round(media.duration * 1000) - source["duration_ms"]) > 100:
        raise ValueError("Full review preview duration mismatch")
    value = {
        "source_id": source_id,
        "start_ms": 0,
        "end_ms": source["duration_ms"],
        "path": str(output.resolve()),
        "size": output.stat().st_size,
        "decode_command": command,
        "media": media.to_dict(),
        "purpose": "full-film human review, not additional model analysis",
    }
    temporary = manifest.with_name(f"preview-{uuid.uuid4().hex}.json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest)
    return value
