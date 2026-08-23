# Evaluation and model roadmap

## Current evidence boundary (2026-08-23)

目前 runtime database 有 5 支來源、56 筆人工 annotation；56 筆全部是 `highlight`，沒有明確的 `exclude`，schema 也還沒有每支來源的 `review_complete`。素材庫另有 135 筆 heuristic clip metadata，其中 102 筆是目前 active 的 v2 候選。這兩者不能混為一談：

- `annotations` 是人工選擇，存在 `data/state.sqlite3`。
- `highlight_clips` 與 `analysis.json` 是演算法預測／診斷輸出，不是 ground truth。
- 沒有 annotation 的時間不能視為「不精彩」，因為無法知道使用者是看過後否決，還是尚未審閱。

這份資料已足以檢查漏抓案例、邊界與來源差異，也足以驗證素材庫工作流；仍不足以宣稱 accuracy、校準跨來源 threshold，或訓練可靠的二元分類器。以下是下一階段評估設計，而不是目前已達成的結果。

## What to label first

長期目標是收集 20–30 支你真的會拍的影片，刻意涵蓋直式／橫式、桌側／底線／斜角、遠近、安靜與吵雜球館。第一輪不用畫每一顆球的 bounding box；目前標記介面實際能保存：

- 一個得分的 `start`／`end` 範圍；
- `highlight`（值得收錄）或 `exclude`（不該收錄）；
- 可選的備註字串，包含常用精彩類型或自由文字。

下一版資料契約應再加入每支來源的 `review_complete`，並把 failure reason 與偏好 tags 拆成結構化欄位。若需要 separately 評估發球、最後一拍與反應邊界，也要新增明確欄位；目前 schema 不能從單一 start/end 推回這三個時間點。

用影片分組切 train／validation／test；同一場球切出的片段不能跨集合，否則背景和拍攝角度會造成資料洩漏。

## Metrics

每次演算法版本至少報告：

1. Point recall：真實精彩得分有多少與輸出片段重疊至少 50%。
2. Point purity：輸出片段有多少只包含一分，沒有混入前後得分。
3. Boundary error：預測開始／結束與真實時間的絕對誤差中位數。
4. Compression ratio：輸出總長度 ÷ 原片長度。
5. Threshold precision／recall：在固定 validation threshold 下，入選中有多少值得保留，以及人工精彩球有多少被選到。
6. Library volume：每片保存候選數、推薦數、零候選率、素材總長與磁碟量；分別按短／中／長片回報，避免固定 Top-k 掩蓋長度偏差。
7. Compilation utility：使用者實際選入率、跨來源比例、成品總長、每分平均長度與直接剪接後是否仍看得懂得分結果。
8. Runtime factor：分析秒數 ÷ 影片秒數，以及 peak RAM／VRAM。

產品初期應優先 point recall，因為漏掉好球無法挽回；ranking precision 可以先透過 review UI 讓人快速刪除。建議未來 baseline target：精彩 point recall ≥ 0.90、point purity ≥ 0.85、開始邊界誤差中位數 ≤ 1.5 秒；在 test split 和標註完整性建立前，這些數字只是驗收門檻，不是目前成績。

## Iteration order

### 1. Calibrate the existing signals

把 `analysis.json` 的所有 candidates 與人工標註比對，分別畫 audio score、motion score、相對門檻決策與錯誤類型。先確認問題來自事件偵測、時序 grouping 或 ranking，避免同時調十個 threshold。未標記區間不能直接當負樣本；每支影片要先記錄 review complete，才能計算正式 precision。

### 2. Train a table-tennis impact classifier

從 audio transient 周圍裁 100–250 ms log-mel patch，將真實擊球、鞋聲、拍手、說話、附近球桌做分類。這個小模型比直接在 4K 畫面找 40 mm 球更便宜，也最能降低吵雜球館 false positive。模型輸出仍可沿用現在的 event／grouping interface。

### 3. Add semantic visual evidence

只有當錯誤分析證明需要時，再加入低頻率的 table／person／pose inference：

- 以多個時間點估計 stable play area，不使用「前 90 幀最大框」。
- 將 pose velocity、兩側球員同時活動、racket-side wrist acceleration 當 evidence，不直接當 rally state。
- 若要偵測球，需以實際手機素材訓練 tiny-object detector，並保留高解析 crop；generic YOLO weight 不足以支撐這項假設。

### 4. Learn personal point ranking

保留使用者「加入集錦／略過／調整邊界」行為，訓練 ranking model，而不是把精彩定義寫死。ranking 與 point segmentation 分離：前者可以個人化，後者仍追求客觀 recall。目前只有送出 compilation 後的 clip IDs／順序保存在 `compilation_items`；編輯中未送出的選擇只在瀏覽器記憶體，也還沒有把「略過」或排序行為匯出成正式 training examples。

## Reproducible experiment record

每次實驗記錄 Git commit、algorithm version、設定、test video IDs、metrics 與輸出報告。不得用 test set 調 threshold；確認 validation 改善後才跑一次 test。這會讓 side project 從 demo 變成能持續進步的系統。

實驗輸出應放在與 runtime 媒體可區分的位置；不要把 `data/outputs/` 的 active clips 當成 frozen evaluation set。每次評估至少保存 annotation snapshot／checksum、來源分組、候選生成版本與設定，才能重現結果並避免重建素材庫後悄悄改變測試資料。
