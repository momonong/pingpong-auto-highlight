# Evaluation and model roadmap

## What to label first

先收集 20–30 支你真的會拍的影片，刻意涵蓋直式／橫式、桌側／底線／斜角、遠近、安靜與吵雜球館。第一輪不用畫每一顆球的 bounding box，只需為每支影片標記：

- 每一分的發球、最後一拍與得分結束時間；
- 精彩度原始 0–3 分，另記未評分／無法判斷，不將缺值填 0；
- 失敗原因標籤，例如附近球桌、拍手、鏡頭晃動、球員被遮擋；
- 若有偏好，再記 `long_rally`、`fast_exchange`、`winner_reaction`、`great_save`。

用影片分組切 train／validation／test；同一場球切出的片段不能跨集合，否則背景和拍攝角度會造成資料洩漏。

## Metrics

每次演算法版本至少報告：

1. Point recall：真實精彩得分有多少與輸出片段重疊至少 50%。
2. Point purity：輸出片段有多少只包含一分，沒有混入前後得分。
3. Boundary error：預測開始／結束與真實時間的絕對誤差中位數。
4. Compression ratio：輸出總長度 ÷ 原片長度。
5. Threshold precision／recall：在固定 validation threshold 下，入選中有多少值得保留，以及人工精彩球有多少被選到。
6. Selection volume：每片入選球數、零球率與總長；分別按短／中／長片回報，避免固定 Top-k 掩蓋長度偏差。
7. Reel pacing：成品總長、每分平均長度與直接剪接後是否仍看得懂得分結果。
8. Runtime factor：分析秒數 ÷ 影片秒數，以及 peak RAM／VRAM。

產品初期應優先 point recall，因為漏掉好球無法挽回；ranking precision 可以先透過 review UI 讓人快速刪除。建議 baseline gate：精彩 point recall ≥ 0.90、point purity ≥ 0.85、開始邊界誤差中位數 ≤ 1.5 秒。

## Iteration order

### 1. Calibrate the existing signals

把 `analysis.json` 的所有 candidates 與人工標註比對，分別畫 audio score、motion score、相對門檻決策與錯誤類型。先確認問題來自事件偵測、時序 grouping 或 ranking，避免同時調十個 threshold。未標記區間不能直接當負樣本；必須明確記錄已完整檢查的原片區間，再在相應覆蓋範圍內定義 precision 的分母。

### 2. Train a table-tennis impact classifier

從 audio transient 周圍裁 100–250 ms log-mel patch，將真實擊球、鞋聲、拍手、說話、附近球桌做分類。這個小模型比直接在 4K 畫面找 40 mm 球更便宜，也最能降低吵雜球館 false positive。模型輸出仍可沿用現在的 event／grouping interface。

### 3. Add semantic visual evidence

只有當錯誤分析證明需要時，再加入低頻率的 table／person／pose inference：

- 以多個時間點估計 stable play area，不使用「前 90 幀最大框」。
- 將 pose velocity、兩側球員同時活動、racket-side wrist acceleration 當 evidence，不直接當 rally state。
- 若要偵測球，需以實際手機素材訓練 tiny-object detector，並保留高解析 crop；generic YOLO weight 不足以支撐這項假設。

### 4. Learn personal point ranking

保留使用者「加入 Reel／略過／調整邊界」行為，訓練 ranking model，而不是把精彩定義寫死。ranking 與 point segmentation 分離：前者可以個人化，後者仍追求客觀 recall。

## Reproducible experiment record

每次實驗記錄 Git commit、algorithm version、設定、test video IDs、metrics 與輸出報告。不得用 test set 調 threshold；確認 validation 改善後才跑一次 test。這會讓 side project 從 demo 變成能持續進步的系統。

## 逐分標記資料契約 v1

`schema_version=highlightcraft-point-review/1`、`scale_version=excitement-0-3/1`。一分指發球到得分結束的完整回合，半開區間 `[start_ms,end_ms)` 相對原片，以整數毫秒保存；播放器 padding 不在人工邊界內。時間長度來自處理 metadata，本工具不重新解碼校準。

| 欄位 | 語意 |
| --- | --- |
| `source.id` / point `id` | 穩定 upload identity／回合 UUID；另留 parent_ids、superseded_by 與 active |
| `validity` | pending／valid／not_rally／unclear；普通一分是 valid，非 not_rally |
| `boundary_status` | pending／confirmed／needs_adjustment／unclear；confirmed 必須人工選定 |
| `rating_status`, `excitement` | unrated/null、unable/null 或 rated/0–3；原始四級分數保留 |
| 0 / 1 / 2 / 3 | 普通不收錄／有看點低優先／精彩願收錄／特別精彩優先保留 |
| `reason_tags` | 穩定 code：rally, attack, counterloop, counter, placement, placement_control, block, defense, save, turnaround |
| `quality_tags` | occlusion, nearby_table, incomplete, unclear；與精彩原因分開 |
| `origin`, `imports` | manual／automatic／legacy／split／merge、產生版本、批次核心邊界與 job metadata |
| `human_reviewed` | 是否有人儲存過這筆，不等於有效回合、確認邊界或完整覆蓋 |
| `created_by`, `annotator_id`, `version` | 建立者、最後更新者、最後修改時的 source revision，另有 UTC created_at/updated_at |
| `coverage`, `unknown_intervals` | 人工明確宣告的檢查區間、其聯集之外的未知區間；inactive coverage 為已撤回歷史 |
| `legacy_annotations` | 舊 highlight/exclude、秒制邊界及文字原樣保留，絕不自動換算分級或宣稱全片審核 |

「已完成」篩選定義為 not_rally，或 valid + confirmed + (rated/unable)。這只是單筆整理完成，unable 仍不能作分數樣本，也不代表全片已找到所有回合。

JSON 是整份來源 envelope；JSONL 首行 `type=manifest` 包含 source、rules、revision、imports、unknown_intervals、fully_reviewed 及 available_proposals，後續 `point`、`coverage`、`legacy_annotation` 行可還原對應陣列。`active=false` 是刪除／拆合歷史，分析時須排除。白名單匯出不包含 session、token、密碼、原片路徑、Drive resource key 或封存憑證；使用者自行填入備註的內容會原樣匯出。

離線讀取範例（不呼叫服務或模型）：

```python
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("review.jsonl").read_text(encoding="utf-8").splitlines()]
manifest = rows[0]
assert manifest["rules"]["schema_version"] == "highlightcraft-point-review/1"
points = [r for r in rows if r["type"] == "point" and r["active"]]
scored = [p for p in points if p["validity"] == "valid"
          and p["boundary_status"] == "confirmed" and p["rating_status"] == "rated"]
# Optional analysis threshold; never write the derived binary label back over raw ratings.
positive = [p for p in scored if p["excitement"] >= 2]
ordinary = [p for p in scored if p["excitement"] == 0]
for p in scored:
    start_seconds, end_seconds = p["start_ms"] / 1000, p["end_ms"] / 1000
```

coverage 外也能有明確人工點標記，但 coverage 外沒標記的部分始終 unknown。候選 recall／全片負樣本分母需先限制在明確完整檢查範圍；不要把 unmatched proposal、inactive point、unrated、unable 或 legacy exclude 偷換成分級 0。以原片／場次分組切資料，模型比較另保存來源與版本 receipt。

### 本輪隔離軟體驗收

起點 `main=3c2ff26`，原工作目錄未提交的 2026-09-09 驗收補充已閱讀並保留，沒有帶入或提交本分支。本節僅記本任務結果；整合時須保留該補充。

新增 `tests/test_point_review.py` 驗證資料往返、四級值／未知、拆合 lineage、CAS／冪等／原子回滾、舊 metadata fixture 升級、重製保留、coverage complement、JSON/JSONL 還原、owner read/admin write。完整 Python suite 為 **94 passed, 9 skipped**；7 項 FFmpeg 缺 PATH，2 項為 Windows symlink／POSIX 權限測試。這些略過項目不能算媒體驗收通過。

`tests/browser/point-review.cjs` 使用全新資料目錄、動態 localhost port 與 Chromium canvas 產生的約 2 秒 MP4，實際通過登入／上傳／Range 播放、候選匯入、邊界修改、0 與 3 分／unable、拆合刪、快捷鍵輸入保護、草稿 reload、伺服器已提交但回應遺失的重試、過期版本衝突、coverage、JSONL 下載、中英介面及重新登入。處理器與候選均為 fixture，無使用者影片、GPU 模型運算、FFmpeg 分析或外部模型呼叫；不代表模型邊界準確率、精彩度效度、真實手機或正式 DB 相容性。

既有 `tests/browser/workspace.cjs` 亦通過 desktop/mobile viewport、上傳中斷續傳、播放／下載、fake Drive handoff、admin、斷線／503／health 格式錯誤／逾時恢復、語言切換與帳號隔離。最後修改的 targeted Python 驗證 **15 passed**，Ruff 通過；快速暫停引發的 AbortError 已驗證不再誤報 codec failure。最終逐分 fixture 證據在本 worktree `data/browser-test-bFR6AX`（point-review-zh.png、point-review-en.png、review.jsonl）；固定播放器補充截圖在 `data/browser-test-0IAfO4/point-review-final.png`，既有 workspace 證據在 `data/browser-test-RjftAf`；這些合成產物不進 Git。
