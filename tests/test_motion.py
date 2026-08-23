from __future__ import annotations

from pathlib import Path

import pytest

import pingpong_highlight.pipeline.motion as motion_module
from pingpong_highlight.pipeline.media import MediaError
from pingpong_highlight.pipeline.models import MediaInfo, MotionFeatures


def test_motion_analysis_retries_on_cpu_after_nvdec_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []
    expected = MotionFeatures.empty()

    def fake_analysis(*_args, use_nvdec: bool, **_kwargs) -> MotionFeatures:
        calls.append(use_nvdec)
        if use_nvdec:
            raise MediaError("unsupported hardware decode")
        return expected

    monkeypatch.setattr(motion_module, "has_nvdec", lambda: True)
    monkeypatch.setattr(motion_module, "_analyze_motion_once", fake_analysis)
    source = tmp_path / "phone.mp4"
    media = MediaInfo(source, 10.0, 1920, 1080, 30.0, "hevc", True, "aac")

    result = motion_module.analyze_motion(source, media)

    assert result is expected
    assert calls == [True, False]


def test_strict_motion_analysis_does_not_fallback_after_nvdec_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []

    def fake_analysis(*_args, use_nvdec: bool, **_kwargs) -> MotionFeatures:
        calls.append(use_nvdec)
        raise MediaError("unsupported hardware decode")

    monkeypatch.setattr(motion_module, "has_nvdec", lambda: True)
    monkeypatch.setattr(motion_module, "_analyze_motion_once", fake_analysis)
    source = tmp_path / "phone.mp4"
    media = MediaInfo(source, 10.0, 1920, 1080, 30.0, "hevc", True, "aac")

    with pytest.raises(MediaError, match="unsupported hardware decode"):
        motion_module.analyze_motion(source, media, require_nvdec=True)

    assert calls == [True]
