"""Process-local, fail-closed model storage. Import before HF/Transformers."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULT_ROOT = Path("D:/hf/_models")


def configure_cache(root: Path = DEFAULT_ROOT) -> dict[str, str]:
    if any(name in sys.modules for name in ("huggingface_hub", "transformers")):
        raise RuntimeError("Configure model paths before importing Hugging Face libraries")
    root = root.resolve()
    if os.name == "nt" and (
        root.drive.casefold() != "d:" or not root.is_relative_to(DEFAULT_ROOT.resolve())
    ):
        raise ValueError("Model storage must be inside D:/hf/_models")
    paths = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "hub",
        "HF_XET_CACHE": root / "xet",
        "HF_ASSETS_CACHE": root / "assets",
        "HF_MODULES_CACHE": root / "modules",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
        "TMPDIR": root / "tmp",
        "TORCH_HOME": root / "torch",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
        "TRITON_CACHE_DIR": root / "triton",
    }
    for path in [*paths.values(), root / "offload"]:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Cache path resolves outside model root")
        path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=root / "tmp") as handle:
        handle.write(b"storage-check")
    os.environ.update({key: str(value) for key, value in paths.items()})
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "15"
    os.environ.pop("TRANSFORMERS_CACHE", None)
    tempfile.tempdir = str(root / "tmp")
    return {key: str(value.resolve()) for key, value in paths.items()}


def download(root: Path = DEFAULT_ROOT) -> str:
    paths = configure_cache(root)
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=paths["HF_HUB_CACHE"],
        token=False,
        max_workers=1,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*"],
    )
    if not Path(snapshot).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Resolved snapshot escaped model root")
    return snapshot


if __name__ == "__main__":
    import faulthandler

    faulthandler.dump_traceback_later(90, repeat=True)
    print("Starting fixed-revision download to D:/hf/_models", flush=True)
    print(download())
