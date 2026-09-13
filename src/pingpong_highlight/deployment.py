"""Explicit HTML template slots for the configured external application path."""
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def html_page(path: Path, root_path: str = "") -> HTMLResponse:
    # Only our two HTML entry templates contain this slot. No proxy body rewriting.
    return HTMLResponse(path.read_text(encoding="utf-8").replace("__HC_ROOT_PATH__", root_path))


class AppStaticFiles(StaticFiles):
    def __init__(self, *, directory: Path, root_path: str):
        super().__init__(directory=directory)
        self.root_path = root_path

    async def get_response(self, path, scope):
        if path in {"index.html", "review/index.html"} and scope["method"] in {"GET", "HEAD"}:
            return html_page(Path(self.directory) / path, self.root_path)
        return await super().get_response(path, scope)
