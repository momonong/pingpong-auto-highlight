"""Serve ONLY a marked synthetic browser-test fixture on a separate loopback port.

First run tests/browser/point-review.cjs. This script never accepts a production data path.
"""

from __future__ import annotations

import argparse
import runpy
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument(
        "--fixture", help="browser-test-* directory name (default: latest marked fixture)"
    )
    args = parser.parse_args()
    data = (ROOT / "data").resolve()
    if args.fixture:
        candidates = [(data / args.fixture).resolve()]
    else:
        candidates = sorted(
            data.glob("browser-test-*"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    fixture = next(
        (
            p
            for p in candidates
            if p.parent == data
            and p.name.startswith("browser-test-")
            and (p / ".point-review-fixture").is_file()
            and (p / "sample.mp4").is_file()
            and (p / "state/state.sqlite3").is_file()
        ),
        None,
    )
    if fixture is None:
        parser.error("Run tests/browser/point-review.cjs first; no marked synthetic fixture found")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    print(f"Synthetic fixture preview: http://127.0.0.1:{args.port}", flush=True)
    print(
        f"Fixture: {fixture.name}; login admin / browser-test-password; Ctrl+C to stop", flush=True
    )
    sys.path.insert(0, str(ROOT / "src"))
    sys.argv = [
        str(ROOT / "tests/browser/server.py"),
        str(fixture / "state"),
        str(args.port),
        str(fixture / "sample.mp4"),
    ]
    runpy.run_path(str(ROOT / "tests/browser/server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
