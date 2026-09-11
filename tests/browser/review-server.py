"""Disposable review API with deterministic proposals, not model-quality evidence."""
import sys
from pathlib import Path

import uvicorn

from pingpong_highlight.rally_review import Proposal, ReviewStore, source_identity
from pingpong_highlight.review_web import create_review_app

directory, port, media = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
store = ReviewStore(directory / "review.sqlite3")
source = source_identity(media)
source.update(duration_ms=10000, group="development", scope=[{"start_ms":0,"end_ms":10000}],
              blind_intervals=[{"start_ms":0,"end_ms":4000}])
store.register(source)
store.add_run({"id":"browser-fixture", "source_id":source["id"], "commit":"fixture",
               "code_sha256":"fixture", "model_id":"synthetic-fixture", "model_revision":"fixture",
               "prompt":"fixture", "parameters":{},"sampling":{},"elapsed_seconds":0,
               "preview":{"path":str(directory / "preview.mp4"),"start_ms":2000,"end_ms":10000},
               "windows":[{"start_ms":0,"end_ms":10000}],
               "raw":["must include confidential hidden advice"],
               "proposals":[Proposal(id="fixture-1",start_ms=1000,end_ms=3500,clip_start_ms=0,
                   clip_end_ms=5000,rally="yes",complete="yes",highlight="must",reason="model fixture reason").model_dump(),
                   Proposal(id="fixture-2",start_ms=6000,end_ms=8000,clip_start_ms=5000,
                   clip_end_ms=9000).model_dump()]})
uvicorn.run(create_review_app(store),host="127.0.0.1",port=port,log_level="warning")
