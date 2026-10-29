# 實驗性 E0 工程檢查

這兩個獨立、僅使用 Python 標準函式庫的模組提供程序觀察記帳及候選核心區間檢查，尚未接入產品、controller 或自動影片處理。公開回歸測試只使用合成 metadata、小型文字／二進位 fixture 與有界 Python 子程序；不包含私人來源、抽樣結果、人工答案或執行收據。

## 候選核心區間

`research_window_coverage.classify_intervals(core_us, observation_us, source_duration_us, *, canonical_quantization=False)` 接受非負、非空、有序的半開區間 `[start_us, end_us)`。所有端點及 duration 必須為 Python `int`，拒絕 bool、float、字串及非有限值。觀察區間不得超過來源或固定 8 秒上限。

```python
from pingpong_highlight.research_window_coverage import classify_intervals

row = classify_intervals([0, 8_000_001], [0, 8_000_000], 9_000_000)
assert row["full_core_contained"] is False
assert row["missing_end_us"] == 1
assert row["coverage_fraction"] == "8000000/8000001"
```

`coverage_fraction` 為精確約分結果；`coverage_decimal_display_only` 僅供顯示。`full_core_contained`、`full_core_feasible_under_8s_and_source_bounds` 與 `source_clipping_reasons`（legacy adapter）回答不同問題。來源邊界裁切不必然裁到核心；8 秒觀察區間也不必然包含較長核心。

`canonical_quantization=True` 只接受明確已知、依最近毫秒四捨五入而向來源結尾外伸至多 500 µs 的 canonical 核心端點，並記錄 overhang。它不提供容差：外伸部分仍算未覆蓋，完整核心仍不可行。等端點的半開區間可滿足 metadata containment；不代表端點事件可見、整個回合完整、人工偏好或選球品質。

`coverage_for_window(row, source, *, canonical_quantization=False)` 保留 legacy recipe：以 `target_ms` 兩端點的整數毫秒中點向下取整，前後各 4 秒，再裁至 `source["duration_us"]`。若提供 `source_id`／`source_sha256`／`session_id`，必須與來源 metadata 一致；`source_id` 對應來源的 `upload_id`。若提供 `source_duration_us`、`target_seconds_original`、`center_ms`、`requested_window_ms`、`actual_window_us`、`actual_duration_us`、`actual_window_seconds`、`actual_window_timecode`、`core_timecode`，也必須符合這個 recipe 的 canonical 值，不會忽略矛盾欄位。原始 decimal 秒以十進位字串、int 或 Decimal 傳入，依 ROUND_HALF_UP 轉毫秒；輸出秒及 timecode 使用六位小數字串。

`require_full_core(windows, sources, *, canonical_quantization=False)` 要求非空 list、唯一非空 `window_id`，以及以 `source_id` 索引的來源 dict。它檢查整批並保留原順序；合法 metadata 中任一核心未完整包含時，拋出 `StrictCoverageError`，其 `coverage_rows` 保留所有列，`failures` 保留全部未包含列與原因。格式或身分矛盾則拋出 `MetadataError`，不得把未檢查列當成有效結果。此 API 不篩掉失敗列、不重選、不修改輸入，也不授予媒體或資源執行許可。

```python
from pingpong_highlight.research_window_coverage import StrictCoverageError, require_full_core

sources = {"synthetic": {"upload_id": "synthetic", "duration_us": 30_000_000}}
windows = [
    {"window_id": "synthetic-short", "source_id": "synthetic", "target_ms": [1000, 2000]},
    {"window_id": "synthetic-long", "source_id": "synthetic", "target_ms": [10000, 19000]},
]
try:
    require_full_core(windows, sources)
except StrictCoverageError as error:
    assert len(error.coverage_rows) == 2
    assert [row["window_id"] for row in error.failures] == ["synthetic-long"]
```

## Linux 子程序觀察記帳

`research_process_guard.Budget(root, cores, total_seconds=120)` 只適用 Linux `/proc`、CPU affinity、`waitid` 與 `wait4`。在單執行緒 supervisor 中依序呼叫，並提供獨立、已存在的輸出目錄。`Popen` 使用 `preexec_fn` 設定 affinity 與 address-space limit，因此不適合多執行緒父程序；呼叫者負責命令與資料的執行授權，不得將秘密放進 command arguments（會寫入 journal）。

```python
import os
import sys
import tempfile
from pathlib import Path
from pingpong_highlight.research_process_guard import Budget

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    budget = Budget(root, sorted(os.sched_getaffinity(0))[:1], total_seconds=3)
    output = budget.run([sys.executable, "-B", "-c", "print('synthetic')"],
                        root / "example", budget.now() + 1)
    assert output.read_text().strip() == "synthetic"
    assert budget.commands[-1]["accounting_complete"] is True
```

`run(command, prefix, clip_deadline)` 使用 monotonic 絕對 deadline、繼承呼叫者工作目錄，回傳 stdout 路徑。每次先保存 `prefix.command.json`，分開保存 `.stdout`／`.stderr`；既有檔案（含 dangling symlink）拒絕覆寫或重試。`snapshot()` 與 `commands` 提供觀察摘要；`GuardError` 表示拒絕。啟動後的監測、清理及記帳錯誤會進入 terminal journal；若 journal 本身無法保存，I/O 例外會向外傳播，呼叫者須保存並核對現有狀態，不能據此重送。

程序正常退出時，以 `wait4` 後父程序 `/proc/self/io` 差值作保守 read charge，包含監測開銷；不宣稱精確子程序或來源讀取量。只有 kernel 確認該子程序已退出，才調和 EACCES／missing `/proc`／zombie RSS 競態；父程序取樣失敗仍拒絕。清理只針對已核對 pid、ppid、pgrp、session、start ticks 的自有 isolated group；無關 `/proc` census 失敗不阻止自有 leader 清理／reap，但 descendants 與記帳必須標為不完整。偵測到後代、未知記帳或其他錯誤時拒絕成功，且同一 Budget 不再派送。原始 timeout 原因不被後續清理或記帳錯誤覆寫。

沿用的上限為累積 logical-read 2 GiB、取樣父子 RSS 合計 4 GiB、輸出樹 512 MiB、保留可用磁碟 1 GiB、單子程序 address space 2 GiB，以及每命令／總 wall deadline。預設約每 20 ms 取樣，清理等待至多 0.5 秒；輪詢與核心讀取本身可能延遲並 overshoot。RSS 是取樣，`wait4` CPU 記錄只作觀察。極短、已脫離群組或未等待的 descendants 可能漏過 census；這不是 cgroup、沙箱、完整 subtree 帳本或硬性資源界線。既有未知結果須核對原 journal，不能另換 prefix 當成可重試授權。

## 驗證

由 repository root 執行，使用鎖定版本，不安裝 train／vlm extra：

```bash
uv sync --offline --locked --extra dev --no-default-groups
uv run --offline --locked --extra dev --no-default-groups python -B -m pytest tests/test_research_process_guard.py tests/test_research_window_coverage.py
uv run --offline --locked --extra dev --no-default-groups ruff check src/pingpong_highlight/research_process_guard.py src/pingpong_highlight/research_window_coverage.py tests/test_research_process_guard.py tests/test_research_window_coverage.py src/pingpong_highlight/web.py src/pingpong_highlight/__init__.py
uv lock --offline --check
uv build --offline --wheel
```

完整適用 metadata／synthetic suite 包含暫存 SQLite、in-process TestClient 與測試專用的 `127.0.0.1` 動態 port（該 fixture 會關閉 server）。以下具名排除的 11 個測試會實際生成／探測／解碼／render 媒體，超出此輪執行範圍；其餘測試全部執行，沒有用缺少 FFmpeg 當作隱性 skip：

```bash
uv run --offline --locked --extra dev --no-default-groups python -B -m pytest \
  --deselect tests/test_pipeline.py::test_end_to_end_signal_fusion_pipeline \
  --deselect tests/test_media.py::test_probe_and_frame_accurate_export \
  --deselect tests/test_media.py::test_social_reel_is_vertical_and_uses_direct_cuts \
  --deselect tests/test_media.py::test_point_reel_preserves_source_geometry \
  --deselect tests/test_media.py::test_point_reel_normalizes_large_input_timestamps \
  --deselect tests/test_media.py::test_full_range_phone_video_is_normalized_for_browser_playback \
  --deselect tests/test_media.py::test_limited_range_10_bit_video_is_not_squeezed_again \
  --deselect tests/test_preannotation_media.py::test_vfr_rotation_and_source_offset \
  --deselect tests/test_preannotation_media.py::test_full_review_proxy_ignores_model_scope_and_preserves_humans \
  --deselect tests/test_edit_comparison.py::test_real_media_roundtrip_and_dissolve_av_clock \
  --deselect tests/test_edit_comparison.py::test_aac_padding_cannot_shift_later_cuts_or_truncate_last_frames
```

上述工程檢查不包含真影片、codec／PTS 實測、模型／GPU、完整回合或偏好驗證，也不是產品接線或人工接受。本輪 source patch 為 1.4.2；歷史 pull-only image 參照仍為 1.4.0，沒有建立或發布新 image。
