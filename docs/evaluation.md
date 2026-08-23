# Evaluation and model roadmap

## Current evidence boundary (2026-08-24)

目前 runtime database `data/state.sqlite3` 有 5 支來源、56 筆人工 annotation；56 筆全部是 `highlight`，沒有明確的 `exclude`。runtime schema 本身沒有 `source_reviews`，但凍結的 `data/state.training-baseline-20260822.sqlite3` 已記錄 5/5 來源 `review_complete`，且 `reviewed_until` 覆蓋各自完整來源時長。這仍不會把空白時間自動變成負樣本：目前 evaluation contract 只接受明確 `exclude` 作為 negative，未標記 candidate 一律是 unknown。

現役素材庫另有 135 筆 heuristic clip metadata，其中 102 筆是 active `highlight-library-v2`；程式的新 runtime 輸出版本雖已是 `highlight-library-v3`，candidate-only evaluation 不會自動重建或切換 active。人工標記、素材庫與 candidate artifacts 不能混為一談：

- `annotations` 是人工選擇，存在 `data/state.sqlite3`。
- `highlight_clips` 與 `analysis.json` 是演算法預測／診斷輸出，不是 ground truth。
- `data/evaluations/candidate-runs/` 的 JSON／NPZ 是完整候選與 signals，不是可播放素材，也不會寫入 `highlight_clips`。
- 沒有明確 `exclude` 的時間不能視為「不精彩」；即使來源已 review complete，precision 仍然 abstain。

五支來源全部是 detector iteration 使用過的 `development` data，沒有 held-out source。Formal scoring contract v1 也刻意只接受 `development` split；「formal」代表 immutable receipts 完整有效，不代表 held-out。這份資料足以做 receipt-valid 的 development regression、檢查漏抓案例、邊界與來源差異；仍不足以宣稱 held-out accuracy、precision，或訓練可靠的二元分類器。

## What to label next

長期目標是收集 20–30 支你真的會拍的影片，刻意涵蓋直式／橫式、桌側／底線／斜角、遠近、安靜與吵雜球館。第一輪不用畫每一顆球的 bounding box；目前標記介面實際能保存：

- 一個得分的 `start`／`end` 範圍；
- `highlight`（值得收錄）或 `exclude`（不該收錄）；
- 可選的備註字串，包含常用精彩類型或自由文字。

正式 runtime 資料契約下一步應把 frozen DB 已有的 `review_complete`／`reviewed_until` 納入主資料流，並收集明確 `exclude`、結構化 failure reason 與偏好 tags。若需要 separately 評估發球、最後一拍與反應邊界，也要新增明確欄位；目前 schema 不能從單一 start/end 推回這三個時間點。

用影片分組切 train／validation／test；同一場球切出的片段不能跨集合，否則背景和拍攝角度會造成資料洩漏。

## Metric contract

Strict candidate recall 使用整數毫秒、half-open `[start_ms, end_ms)` 區間，在同一來源內做 chronological one-to-one matching。只有 candidate **core** 與人工 `highlight` 的交集除以人工區間長度至少為 50% 才算命中；剛好 50% 通過，同一 candidate 不得重複命中兩筆 annotation。播放器與成品用的前後 1.5 秒 padding 不參與 matching，避免靠展示脈絡灌高 recall。matching 先最大化命中數，再依 annotation coverage、IoU、boundary error 與穩定 ID 決定 ties。

Candidate recall 不能靠密集 sliding windows、超長 core 或重複區間達成，因此正式 gate 同時要求：

| Candidate burden guardrail | 上限 |
|---|---:|
| 全資料 candidates/minute | 6.0 |
| 單一來源 candidates/minute | 8.0 |
| 全資料 candidate-core union coverage | 50% |
| 單一來源 candidate-core union coverage | 75% |
| 單一 candidate core 長度 | 20 秒 |
| unresolved overlap | 0 ms，且 overlapping pair count = 0 |

後續每次演算法版本還應報告 boundary error、library volume、compression ratio、compilation utility 與 runtime factor。Point purity、threshold precision／recall、AP、AUROC、NDCG 與 FPR 必須等明確 negatives 存在後才啟用；現在不能把 unmatched candidates 當 false positive。

產品初期優先 candidate recall，因為偵測層漏掉好球後 ranking 無法挽回。建議未來 held-out baseline target：精彩 point recall ≥ 0.90、point purity ≥ 0.85、開始邊界誤差中位數 ≤ 1.5 秒；在獨立 test split 和負向標註建立前，這些數字只是未來驗收門檻。

## Current objective and GO/STOP gate (2026-08-24)

第一個工程目標是 **strict candidate recall ≥ 0.80**，同時通過 candidate-burden guardrails。決策順序固定如下：

| 條件 | 決策 | 下一個元件 |
|---|---|---|
| recall < 0.80 | `STOP_DETECTOR` | impact detection、point grouping、core boundaries |
| recall ≥ 0.80，但 burden 失敗 | `STOP_CANDIDATE_BURDEN` | candidate consolidation、core boundaries |
| recall 與 burden 都通過 | `GO_RANKING` | ranking |

`candidate-generation-v3` 的正式 GPU baseline 只有 4/56（7.14%），因此是 `STOP_DETECTOR`。commit `7e0881d` 的 `candidate-generation-v4` 已完成 clean-worktree、strict NVIDIA NVDEC run，GPU receipt 為 NVIDIA GeForce RTX 5090 Laptop GPU；結果如下：

| 指標 | v4 結果 |
|---|---:|
| Strict candidate recall | **51/56（91.07%）** |
| Candidates | 481 |
| 來源總長 | 108.658377 分鐘 |
| Candidate density | 4.426718/min |
| Candidate-core union coverage | 46.3734% |
| 最長 core | 18.957 秒 |
| Overlapping pairs／overlap excess | 0／0 ms |
| Gate | **`GO_RANKING`** |

五支來源的 per-source density 與 union coverage 也都在 guardrails 內。這代表 detector candidate gate 已通過，可以開始評估 ranking；它不是整個產品完成，也不是 held-out accuracy。證據狀態是 `valid-development-regression`，precision 狀態仍是 `abstained_missing_explicit_negatives`。

## Iteration order

### 1. Preserve the detector gate

v4 已在目前 development set 通過 detector gate。之後任何 detector、boundary 或 signal 變更都要重跑同一份 frozen dataset，同時維持 recall 與 burden；不能只看 recall 上升，也不能用固定 Top-k 隱藏長影片的候選量。

### 2. Build ranking evidence and explicit negatives

保留使用者「加入集錦／略過／調整邊界」行為，讓 ranking model 學個人偏好，而不是把精彩定義寫死。ranking 與 point segmentation 分離：前者可以個人化，後者仍維持客觀 candidate recall。現在只有送出 compilation 後的 clip IDs／順序保存在 `compilation_items`；下一步要把明確略過／`exclude` 與排序選擇匯出成可稽核 training examples。

### 3. Establish a held-out source split

新增不同日期、鏡位、場館的來源，依整支影片或場次切 validation／test。development 上的 91.07% 只能用來防 regression；threshold 確定後才執行一次 held-out evaluation，不得把 test set 反覆拿來調參。

### 4. Add learned detector evidence only when error analysis requires it

若 held-out miss atlas 顯示 audio false events 是主要瓶頸，可從 transient 周圍裁 100–250 ms log-mel patch，分類真實擊球、鞋聲、拍手、說話與附近球桌。只有錯誤分析證明需要時，才加入低頻率的 table／person／pose inference：

- 以多個時間點估計 stable play area，不使用「前 90 幀最大框」。
- 將 pose velocity、兩側球員同時活動、racket-side wrist acceleration 當 evidence，不直接當 rally state。
- 若要偵測球，需以實際手機素材訓練 tiny-object detector，並保留高解析 crop；generic YOLO weight 不足以支撐這項假設。

## Reproducible experiment record

每次實驗記錄 Git commit、algorithm version、設定、來源 IDs、metrics 與輸出報告。不得用 test set 調 threshold；確認 validation 改善後才跑一次 test。

目前 formal development run 使用已凍結的 dataset：

```powershell
$dataset = "data/evaluations/candidate-recall/exploratory-active-v2-20260824/dataset.json"

uv run --frozen pingpong-highlight evaluation run-candidates `
  --data-dir data `
  --dataset $dataset `
  --run-id candidate-v4-YYYYMMDD

uv run --frozen pingpong-highlight evaluation score-candidates `
  --data-dir data `
  --dataset $dataset `
  --candidate-run data/evaluations/candidate-runs/candidate-v4-YYYYMMDD `
  --run-id formal-v4-YYYYMMDD
```

第一個命令預設要求 clean worktree 與 NVIDIA NVDEC，並在 `data/evaluations/candidate-runs/<run-id>/` 保存 `manifest.json`、每來源 `candidates.json`、`signals.npz` 與 checksummed receipts；它不輸出 MP4、不修改 runtime database 或 active library。`--allow-cpu`／`--allow-dirty` 只供診斷，不能通過目前 formal scorer。第二個命令會重新驗證 dataset、annotation snapshot、來源、設定、Git、GPU 與 signal hashes，再把 `metrics.json`、`report.md`、`manifest.json` 與 `checksums.sha256` 寫到 `data/evaluations/candidate-recall/<run-id>/`。GO 回傳 0；有效的 STOP gate 回傳 3。

`evaluation freeze-active` 只用來從舊 active artifacts 建立 immutable dataset 與 legacy diagnostic。舊 v2 artifacts 沒有 generation-time Git/config/source receipt，因此其 report 即使 checksummed 也不能取代上述 formal run。

不要把 `data/outputs/` 的 active clips 當成 frozen evaluation set。每次評估至少成套保存 annotation snapshot／checksum、來源分組、candidate run、raw signals、生成版本與設定，才能重現結果並避免重建素材庫後悄悄改變測試資料。
