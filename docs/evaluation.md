# Evaluation and model roadmap

## What to label first

先收集 20–30 支你真的會拍的影片，刻意涵蓋直式／橫式、桌側／底線／斜角、遠近、安靜與吵雜球館。第一輪不用畫每一顆球的 bounding box，只需為每支影片標記：

- 每一分的發球、最後一拍與得分結束時間；
- 是否值得保留（yes / maybe / no）；
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
6. Library volume：每片保存候選數、推薦數、零候選率、素材總長與磁碟量；分別按短／中／長片回報，避免固定 Top-k 掩蓋長度偏差。
7. Compilation utility：使用者實際選入率、跨來源比例、成品總長、每分平均長度與直接剪接後是否仍看得懂得分結果。
8. Runtime factor：分析秒數 ÷ 影片秒數，以及 peak RAM／VRAM。

產品初期應優先 point recall，因為漏掉好球無法挽回；ranking precision 可以先透過 review UI 讓人快速刪除。建議 baseline gate：精彩 point recall ≥ 0.90、point purity ≥ 0.85、開始邊界誤差中位數 ≤ 1.5 秒。

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

保留使用者「加入 Reel／略過／調整邊界」行為，訓練 ranking model，而不是把精彩定義寫死。ranking 與 point segmentation 分離：前者可以個人化，後者仍追求客觀 recall。

## Reproducible experiment record

每次實驗記錄 Git commit、algorithm version、設定、test video IDs、metrics 與輸出報告。不得用 test set 調 threshold；確認 validation 改善後才跑一次 test。這會讓 side project 從 demo 變成能持續進步的系統。
