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
6. Selection volume：每片入選球數、零球率與總長；分別按短／中／長片回報，避免固定 Top-k 掩蓋長度偏差。
7. Reel pacing：成品總長、每分平均長度與直接剪接後是否仍看得懂得分結果。
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

## 模型輔助預標註 v1（2026-09-11）

任務分支 `codex/model-assisted-review`，起點 `3c2ff26895690c1f9876f6e078d80632858dd63a`，隔離 worktree `D:\projects\pingpong-auto-highlight\data\worktrees\model-assisted-review`。原 checkout 的未提交 2026-09-09 驗收紀錄只讀參考，沒有複製、覆寫或提交；本輪不接管正式 `data/state.sqlite3`、素材庫或封存結構，不合 main、不推送、不部署。

### 既有證據核對

- 原 checkout 的 88 passed／9 skipped 是前次隔離測試，不能作本輪 gate。
- `data/datasets/personal-baseline-20260822` 的五支來源、56 個正標註、零負標註，全部屬已使用的 development 素材。未標註不是不精彩、不是沒有回合。
- 用新的 source-local 一對一 evaluator 重算 `data/evaluations/threshold-v5-gpu-verified-20260823` 舊產物，得到候選核心 4/56、入選核心 4/56、入選 padding 7/56。這不是本輪 main 全片重跑。
- 唯讀查核 `codex/preserve-local-20260907` 的 candidate-generation-v4（`7e0881` 所在歷史）：弱／強音訊 peak、2.75 秒 grouping gap、動作確認、安靜區間拆分及核心擴張，連帶修改 pipeline 資料介面。51/56 是既有 development regression，不是 held-out、不是 Reel 品質。本版沒有整支合併或移植；baseline 直接呼叫起始 main 的 audio/motion/detect_points，方便獨立比較。

### 本輪資料與語意

`examples/preannotation-development.json` 明列原片 SHA-256、session、development、scope 與盲審區間；讀取時重新核對內容 hash。三段總共 **116 秒**，只比較範圍內五個既有精彩正標註。這些範圍是已知正例附近的定向 regression，不是隨機代表樣本。

| 來源檔名（原 checkout data/uploads） | 原片範圍 | 既有正例數 |
| --- | --- | --- |
| 6e7e1019273942d8babb122188d64a30.mp4 | 10–74 秒 | 3 |
| a6ef1ae5b64d4f62b19bf2d850b9282b.mp4 | 132–160 秒 | 1 |
| f470599a5b5e434c8ab809db749436b2.mp4 | 330–354 秒 | 1 |

smoke 是第一支原片 16–24 秒，包含在以上範圍內。第一支 10–28 秒先隱藏模型判斷／精彩建議／理由；盲審本身仍看到候選時間，不宣稱完全盲測。尚無新 held-out 或真人計時樣本；下一輪需要未參與任何調參的新拍攝 session，先封存來源，再由人完整審核全部回合及精彩程度，保留不顯示模型評分的子集。

人工標註的 `rally` 與 `complete` 是 yes/no/uncertain/unable；`highlight` 是 omit/include/must/unrated。只有 rally=yes 才能給精彩等級。保存人工紀錄不會自動完成 coverage，需看完整段後另行標記；拒絕候選不等於把周圍區間全部設為負例。拆合保留來源 lineage 並重設判斷，避免沿用已不適合新區間的結論。計時預設暫停，可開始／暫停，背景頁自動暫停，約每 15 秒保存；突然關閉最多可能遺失尚未送達的短段計時，不把計時當作節省時間的證據。

### 模型、依賴與儲存

依 [官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 與 [官方 Qwen3-VL 安裝說明](https://github.com/QwenLM/Qwen3-VL)，使用 `Qwen/Qwen3-VL-8B-Instruct`、revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`、Apache-2.0，四個 BF16 權重分片合計 17,534,339,512 bytes。Transformers 固定 4.57.6（官方要求至少 4.57.0）；本機重用 torch 2.13.0+cu130，accelerate 1.15.0、PyAV 17.1.0，RTX 5090 Laptop 24 GiB。選用 `vlm` extra 不在一般網站啟動時 import 或下載模型。

唯一模型 root 為 `D:\hf\_models`，snapshot 放在 `hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/<revision>`。`model_cache.configure_cache` 必須在 HF/Transformers import 前呼叫，將 Hub、Xet、assets、modules、TEMP/TMP、torch/kernel cache 與 offload 都導向 root 子目錄；驗證解析路徑和可寫性，失敗即停止，沒有 C 槽 fallback。未更動全系統環境；沒有搬動或刪除其他模型。Windows Hub/Xet transport 本輪曾停滯，改用 `scripts/download-qwen.ps1` 單一官方 revision、可續傳 Range、逐分片 LFS SHA-256 校驗，寫入同一 snapshot，不使用 local_dir 另存權重。

可重現設定（在任務 worktree 的 PowerShell 執行；FFmpeg/ffprobe 需同一個完整可用版本）：

```powershell
$env:PYTHONPATH = 'src'
$env:PIP_CACHE_DIR = 'D:/hf/_models/pip'
$env:UV_CACHE_DIR = 'D:/hf/_models/uv'
$env:TEMP = 'D:/hf/_models/tmp'
$env:TMP = $env:TEMP
# 先確認 D:/hf/_models 可寫並建立上述子目錄，模型絕不回退其他磁碟。
# 新環境：uv sync --extra dev --extra vlm（會安裝套件，但不下載模型）。
# 本輪另建 .venv-vlm，以 .pth 重用原 checkout 已安裝的 torch，避免改動既有 venv。
./scripts/download-qwen.ps1
nvidia-smi
.venv-vlm/Scripts/python.exe -m pingpong_highlight.preannotate run --manifest examples/preannotation-smoke.json --output data/experiments/preannotation-v1/smoke-current/qwen --backend qwen --window-ms 8000 --overlap-ms 1000
```

每次 `--output` 必須是新目錄；既有產物不覆寫。先確認其他 GPU 程序與空閒 VRAM，單模型、單 worker、BF16/SDPA，沒有 CPU 靜默 fallback；8B adapter 要求至少 20 GiB free。短片先 accurate seek／autorotate 成 30 fps、640 寬 H.264，VLM 再以 2 fps、384 寬取樣，核對 PyAV PTS。模型視窗最大 30 秒、總數最多 120；預設 16 秒／重疊 2 秒，完整掃描指定 scope，不依附 baseline 找到的候選。音軌不傳模型，只在 prompt 附 baseline transient 時間。

```powershell
# baseline 與 qwen 目錄共用其上一層的獨立 review.sqlite3；重跑使用新的子目錄名稱。
python -m pingpong_highlight.preannotate run --manifest examples/preannotation-development.json --output data/experiments/preannotation-v1/dev-current/baseline --backend baseline
.venv-vlm/Scripts/python.exe -m pingpong_highlight.preannotate run --manifest examples/preannotation-development.json --output data/experiments/preannotation-v1/dev-current/qwen --backend qwen
python -m pingpong_highlight.preannotate serve --store data/experiments/preannotation-v1/dev-current/review.sqlite3 --port 8799
# 先確認 port 未被使用，再開 http://127.0.0.1:8799。
```

既有網站桌面人工標記頁也可點「回合預標註與人工修正」，手動匯入該 job 的既有 baseline，再播放／確認／拒絕／調整／拆合／補漏。本輪不啟動服務連接正式資料庫。實驗頁預設播放相容 H.264 短片；所有標記仍為原片絕對秒數。查看 scope 外區段可切回原片；若瀏覽器不支援 HEVC，需另產該段相容預覽，不能把只有音訊的播放當通過。瀏覽器 export 與 CLI evaluate 都可用於後續交接，人工資料與模型 runs 分表保存。

```powershell
python -m pingpong_highlight.preannotate compare-legacy --dataset D:/projects/pingpong-auto-highlight/data/datasets/personal-baseline-20260822 --reports data/experiments/preannotation-v1/dev-current/baseline/runs.json --current-runs --output data/experiments/preannotation-v1/baseline-comparison-new.json
python -m pingpong_highlight.preannotate evaluate --store data/experiments/preannotation-v1/dev-current/review.sqlite3 --source 636c4bb3605b97f3a00e4a7787cc518864bd4a2a9c7b57727d5bdb906bd7adc8 --output data/experiments/preannotation-v1/review-evaluation-new.json
```

### 評估解讀與驗收

匹配以 source identity 隔離，最大一對一匹配要求覆蓋至少 50% 人工區間；重複候選無法灌高 recall。分開輸出全部候選核心、入選核心、入選 padding、切點誤差、人工修正差量與未匹配項。已規劃而推論失敗的視窗仍保留分母。positive-only 比較永遠沒有 precision；完整 coverage、沒有跨分析範圍的人工回合，且回合／完整性無未決判斷時才計候選 precision，入選 precision 另需完整精彩評分。區間交集匹配不是 point purity，亦不代表確實完整一分。

目前 baseline 在這 116 秒產生 6 個候選；五個已知正例的候選核心 **0/5**、入選核心 **0/5**、入選 padding **3/5**。三段分析耗時約 **29.4 秒**。這是本版 wrapper 的短段重跑，不是原始影片全片 pipeline；不可與歷史 56 例或 v4 51/56 直接作效果提升比較。

每個 run 保存來源 hash/size、commit、dirty、程式 hash、model ID/revision、prompt、參數、FFmpeg 命令與版本、抽樣、音訊事件、raw、格式錯誤、去重與耗時。55 秒選擇 budget 是「每個分析段」的簡單排序政策，尚未渲染或驗收最終 Reel 效用。真人審核時間改善、held-out、全片召回與最終 Reel 品質保持 UNKNOWN。

新增正規化保護拒絕短於 500 ms 抽樣間距的模型區間；不把疑似秒值偷偷乘以 1000。PyAV 取樣時間、實際輸入影格形狀、video grid、輸入／輸出 token 數也記入 runtime。三次同一 8 秒 smoke 保留在 `docs/evidence/preannotation-v1/smoke-*.json`：初版照抄示例；第二版把秒填成毫秒；第三版出現 500 ms「完整回合」重複敘述，最後超出 768 token 被截斷。這些都是實際模型失敗，不能以 JSON 或載入成功掩蓋。之後凍結設定，僅在已授權 116 秒做一次診斷，沒有繼續調參或新增模型。

### 本輪功能驗證

- 本次完整 pytest：**115 passed、2 skipped**。兩個 skip 為 Windows symlink 權限與 POSIX mode bits，不是跳過模型／回合功能測試。
- 新增測試覆盖：一對一重複與跨來源、VFR/rotation/原片時間、輸出格式與毫秒單位、來源 hash、跨分析邊界 UNKNOWN、未審核 precision、CAS／冪等／重跑保護、拆合 lineage、計時、盲審、不同 owner 同 bytes 的隔離，以及拒絕對外部 SQLite schema 初始化。
- Playwright Chromium + 真實 API/獨立 SQLite：播放、首次盲審、調整保存與重載、拆分／合併、拒絕／補漏、coverage、計時暫停、網路失敗後草稿與重試、匯出、非零起點相容預覽的原片時間保存，全部通過。這些人工操作由自動化執行，屬 fixture 證據。
- 既有 workspace 瀏覽器驗證通過：desktop/mobile、續傳／重載、播放／下載、Drive fixture、管理／標記、連線狀態、語言與帳號隔離。
- 真實 HEVC 原片在 Chromium 曾只有音訊（videoWidth/Height=0）；改用本次已產生的 H.264 proxy 後確認 640×360 實際畫面，短片 1.509 秒保存欄位顯示原片 11.509 秒；真實來源的 review revision 仍為 0。沒有假造人工確認或省時資料。
- 新增模組 Ruff、`git diff --check`、`uv lock --check --offline` 通過。鎖檔僅新增 VLM 相容依賴，網站預設依賴不包含它們。

本輪使用 FFmpeg 8.1.1 完整 WinGet build（程序 PATH），原 conda 8.0 在此主機曾有 ffprobe stdout JSON 不完整，未把那次環境失敗當產品通過。精簡機器可重現命令：

```powershell
$env:PYTHONPATH = 'src'
$env:PATH = 'C:/Users/morris/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin;' + $env:PATH
$python = 'D:/projects/pingpong-auto-highlight/.venv/Scripts/python.exe'
& $python -m pytest -ra
$env:HIGHLIGHTCRAFT_TEST_PYTHON = $python
# Playwright 由本機既有 Codex dependency runtime 提供；其他環境可使用自己的 Playwright 安裝。
$env:NODE_PATH = 'C:/Users/morris/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
node tests/browser/rally-review.cjs
```

### 小樣本結果與決策

| 本次 116 秒 development 比較 | baseline audio/motion | Qwen3-VL-8B |
| --- | ---: | ---: |
| 非否定候選（含待確認） | 6 | 17 |
| 候選核心匹配既有正例 | 0/5 | 0/5 |
| 入選核心匹配既有正例 | 0/5 | 0/5 |
| 入選 padding 匹配既有正例 | 3/5 | 0/5 |
| 格式／時間校驗失敗視窗 | 不適用 | 4/9 |
| 分段處理耗時（不含模型載入） | 29.4 秒 | 276.3 秒 |
| precision、全回合 recall、Reel 效用、真人省時 | UNKNOWN | UNKNOWN |

Qwen 一共回傳 20 個通過格式校驗的提案，其中 3 個判定 rally=no，17 個為非否定候選；選片政策入選 5 個。4 個失敗視窗包含 2 次 768-token 截斷 JSON、2 次小於 500 ms 的時間單位錯誤；沒有修補或隱藏這些失敗，三段 scope 的五個既有正例仍全數納入分母。純 inference 約 257.9 秒，模型載入另約 10.5 秒；本輪峰值 PyTorch allocated VRAM 約 18.83 GiB，不等於整機 VRAM 使用量。沒有 OOM、CPU fallback 或換模型。

語意失敗包括：理由描述「對打中但看不到結尾」，rally 卻填 no；把一段連續對打切成多個短「完整回合」；理由聲稱可見球路／結束但缺乏足夠證據。它們不因 JSON 有效就成為可信標註。上述僅是定向 regression 的負面結果，不能宣稱模型在所有影片都無用，也不能反過來宣稱網站原有全片剪輯已驗收。

**決策：STOP 擴大目前這組 VLM 預標註設定。** 機制與人工修正流程可用，但格式穩定性、回合完整性和正例覆蓋均沒有達到擴大條件。下一個最小改善項是用現有審核頁完整核對這 116 秒（保留盲審區間），保存全回合起訖、精彩判斷與 review coverage，再把這組固定 development 真值用於下一個候選方法比較；不要繼續只靠五個精彩正例調 prompt。

真實 VLM 批次另經瀏覽器讀取，在 **`browser-check.sqlite3` 副本** 保存一筆明示 `BROWSER AUTOMATION FIXTURE` 的 uncertain 紀錄並重載，最後 CLI evaluate 的 precision 仍為 UNKNOWN。原 `dev-current/review.sqlite3` 位元組保持不變，所有三支來源的真人紀錄仍為空；不把副本標註納入模型品質評估。

精簡原始證據隨分支保存在 [docs/evidence/preannotation-v1](evidence/preannotation-v1)：baseline/qwen runs、五例與歷史 56 例評估、三次 smoke 原始回答、權重 checksum、pytest、兩組瀏覽器及副本評估。大型代理影片、逐窗片段、DB、完整 log 和 screenshot 留在任務 worktree `data/experiments/preannotation-v1/`，原片仍只讀引用。模型權重只在 `D:\hf\_models`。

實作起始提交 `100337f`；smoke 與小樣本當時仍有本任務修改，receipt 明列 `dirty=true` 與程式 hash。[experiment-code.patch](evidence/preannotation-v1/experiment-code.patch) 保存由該提交到實驗完成版本的差異。小樣本第一段與後兩段 hash 不同，只因執行中補上審核 API 的失敗視窗計數，VLM 推論程式與 prompt 未再變動；不能把 dirty receipt 說成未修改的 `100337f` 效果。最終分支提交另見 Git log。

### 公開資料適用性（僅查核，未下載）

[Extended OpenTTGames 官方資料庫](https://github.com/moamal01/table_tennis_data) 提供靜態側視 120 fps、stroke/serve/rally-ending events 等標註，授權 CC BY-NC-SA 4.0。可研究回合事件邊界，但攝影域與手機球館影片不同，也沒有個人精彩偏好標籤；非商用及相同方式分享條件需要在後續實際用途確認，不能默認可作任何產品訓練。第一版不依賴外部資料完成流程。

## 整片人工標註流程修正（2026-09-11）

使用者操作後指出播放與使用順序不清楚，原本期待先標完一整支影片。核對後確認前版把模型實驗短片當預設播放入口，造成「播放器局部時間／標註原片時間」並存，還要求直接操作候選批次與 coverage 數字欄位。這些是先前 fixture 通過仍未涵蓋的使用流程問題。

本次將人工審核與模型實驗分開：整片觀看 → I/O 記錄回合 → 保存並續播 → 明確確認已檢查範圍。模型候選預設收合；精彩程度可稍後再評。重載從第一個未檢查區段接續，未完成草稿仍可恢復，新增明確放棄草稿操作。保存一筆回合不自動完成 coverage，播放也不會自動產生負例或已審核真值。

僅為第一支來源 `636c4bb3…` 製作 223.4 秒、640×360、H.264 Constrained Baseline／yuv420p 的完整播放副本，存在既有獨立實驗目錄的 `review-media/`；沒有模型重跑、資料分組變更、正式 DB migration 或原片修改。原模型仍只分析其既有 10–74 秒 scope；整片人工標註可覆蓋 0–223.394 秒。CLI 入口是 `preannotate prepare-review --store … --source <sha256>`，亦可從頁面按「準備整片播放」。

驗證：完整 pytest **116 passed、2 skipped**（原 Windows 平台限制）；新增整片媒體測試驗證 model scope 不限制播放長度、8-bit 格式、Range、可重用副本及人工 DB bytes 不變。`tests/browser/full-review.cjs` 驗證整片預設、單一時間軸、I/O、保存續播、顯式 coverage、重載接續、未填完草稿恢復、明確放棄及保存資料保留。既有 `rally-review.cjs` 的盲審／拆合／補漏／失敗重試／短片映射亦通過。

真實瀏覽器在 0、110、220 秒播放均有 640×360 影像，總長度 223.4 秒，前後 review export 相同。這是播放與自動化操作證據；尚未由真人完成整片標註，也沒有新增模型效果或省時結論。紀錄位於 `docs/evidence/preannotation-v1/full-review-*`，截圖與影片衍生物留在 `data/experiments/preannotation-v1/`。模型擴大實驗的 STOP 結論維持。

## 部署主機人工標註驗收（2026-09-12，Linux RTX 4090）

**初次階段：隔離環境與 fixture 功能驗收通過，當時真實影片尚未搬入。後續真實影片驗收結果見本節末的「真實原片搬入後續驗」。** 初次驗收時使用者確認 Windows 原片、正式 DB、獨立審核 DB 尚未搬入。不能將下列 fixture 標註當作人工真值，模型擴大實驗 STOP 維持，沒有模型下載、推論或演算法修改。

- 起始核對：原 checkout `main=321da1e`、乾淨，`origin/main=213fc18`（PR #5，落後五個提交）；`v1.4.0^{}` 仍為 `321da1e`。另建 `.worktrees/host-annotation-acceptance-20260912`、分支 `codex/host-annotation-acceptance-20260912`，以 `213fc18` 驗收，原 checkout 保留。
- 可用入口：`http://127.0.0.1:18082`，原生 Docker `default` context，Compose project `highlightcraft-annotation-acceptance`；套件版本仍為 1.4.0，映像 revision 為 `213fc1895a50ce2bae131e52ec2159b6e2e8a885`。
- 映像 `highlightcraft:acceptance-213fc18-reused`，image ID `sha256:7a22e7fd8bbfcaddc5ceb712a4136199645c7f60008806b00e5597dc775838d0`。原 Dockerfile 的完整依賴下載約 90 KB/s，該建置由本任務取消；改以原 builder 階段重建最新 source wheel，再安裝至既有 candidate 依賴層。31 個 runtime 套件版本均符合未變更的 requirements.lock，43 個 package 檔案 hash 全部吻合本次 src。既有 base 的 revision label 是 unknown，因此此證據不等於完成冷建置或 release provenance；完整 image receipt 與重用建置檔已保存。
- 新資料僅在本 worktree 的 `data/host-acceptance/`；正式 state 在其 `state.sqlite3`，owner 審核 DB 在 `rally-review/<owner-hash>.sqlite3`，整片播放副本在 `rally-review/review-media/`。這是驗收 instance，包含明示的 AUTOMATION FIXTURE 帳號與測試片，不可整份直接升為正式資料。
- 管理員為 `admin`，隔離密碼保存在本 worktree 的 `data/acceptance-admin-password`（mode 0600）；Compose 設定與相同 bootstrap secret 在 `data/acceptance.env`（mode 0600）。兩檔皆不追蹤，不可隨證據分享。

| 本機驗證 | 結果與界線 |
| --- | --- |
| 鎖檔 dev 環境 | `uv sync --locked --extra dev --no-default-groups` 完成；Python 3.11.15，未安裝 vlm/train |
| 完整 pytest | **118 passed、0 skipped，19.17 秒**；先前既有 dev Python 另有 118 passed，最終結果以獨立環境為準 |
| 靜態檢查 | Ruff、uv lock offline check、Python compileall、變更的 JavaScript syntax、git diff check 通過 |
| 三組既有瀏覽器回歸 | full-review、rally-review、workspace 皆 PASS；測試資料為生成影片、獨立 SQLite |
| 直接容器瀏覽器驗收 | 登入、匿名拒絕、owner 隔離、實際 HTTP 上傳及正式 processor 完成、640×360 H.264 整片影像、Range 206、I/O、保存續播、明確 coverage、草稿恢復／放棄與匯出 PASS |
| 隔離容器重啟 | 有效 session 保留；完整 review export 前後相同，保存一筆 fixture、coverage 0–4 秒 |
| NVENC | doctor 實際試編通過 |
| NVDEC + NVENC | 容器 FFmpeg 強制 CUDA frames、hwdownload/format=nv12、h264_nvenc，兩秒輸出 60 影格，exit 0；沒有軟體解碼 fallback。測試片為 H.264，不代表真實 HEVC/VFR 來源已驗收 |
| 現役服務與資料 | Desktop 18080/18081 仍 healthy；既有兩份 state.sqlite3 SHA-256 未變，沒有重新啟動現役服務 |
| 真實原片首／中／尾播放及整片人工操作 | **BLOCKED：原片尚未搬入** |
| 正式 DB 相容性、完整備份與還原 | **未驗收：開發機資料與備份尚未提供** |
| 真人整片標註／模型品質／省時 | **UNKNOWN** |

本次修正僅在測試及部署文件：workspace 測試明確指定 zh-TW locale（避免主機預設英文導致切換方向相反）；修正原有三項 Ruff 問題；新增 direct-container 瀏覽器驗收與 loopback Compose overlay。沒有修改產品演算法或 runtime 程式。

初次 direct-container 腳本因備註區收合而無法填寫，並在第二次遇到同 bytes 已有整片 proxy，現已先展開備註並接受既有 proxy；仍驗證新 owner 人工紀錄為空。GPU 自訂零拷貝探測初次發生 `No decoder surfaces left`，額外 32 surfaces 又超出解碼器限制；最後以單 thread、extra_hw_frames=8 及系統記憶體 filter 路線成功。這些是探測設定的除錯證據，未據此更改產品 GPU 參數，初次失敗紀錄亦保留。

精簡證據位於 [host-acceptance-20260912](evidence/host-acceptance-20260912/)，包含 pytest、三組瀏覽器、直接容器、restart、GPU、映像 hashes、原 DB 前後與服務 health。截圖與 fixture export 保存在本 worktree 的 `data/deployment-browser-3/`；這些為操作 fixture，不得匯入正式真值集。

### 補齊真實資料與正式切換

Windows 路徑由目前 manifest／實驗 receipt 取得，尚未在 Windows 主機即時核對：

1. 第一支 223.4 秒原片：`D:\projects\pingpong-auto-highlight\data\uploads\6e7e1019273942d8babb122188d64a30.mp4`，SHA-256 應為 `636c4bb3605b97f3a00e4a7787cc518864bd4a2a9c7b57727d5bdb906bd7adc8`。先提供這支即可補做真實播放；Linux 暫存位置為本 worktree 的 `data/incoming/`，不可覆蓋原 checkout 的 data。
2. 審核資料：`D:\projects\pingpong-auto-highlight\data\worktrees\model-assisted-review\data\experiments\preannotation-v1\dev-current\review.sqlite3`；建議保留完整 dev-current 目錄，包含 review-media、runs 與相容短片。運行中的 SQLite 需先取得一致性備份，不能只複製單一主 DB 而忽略 WAL。
3. 開發機正式 state：`D:\projects\pingpong-auto-highlight\data\state.sqlite3`。需原片、outputs、審核與隱藏設定齊全的完整 data 備份；本次未確認備份實際路徑。素材庫保存分支的資料不得直接接入此版本。
4. 補驗原片 SHA、時長與首中尾影像，在副本完成整片 I/O／保存／coverage／恢復／匯出；由真人在獨立於 fixture 的紀錄完成整片標註。任何 Windows 路徑重設及 proxy manifest 重建只作用於副本，既有 runs 與人工內容需前後比較。
5. 正式切換仍需另行授權。先完成真實資料相容性與備份還原演練，建立可追溯的正式映像，確認無 active jobs/uploads/imports，安排備份與維護時間，再切換明確的 context、image、資料路徑及入口。保留舊 image 與完整備份供 rollback；不必停止整個 Docker Desktop 或改全域 context，也不可讓兩個容器同寫一份 data。


### 真實原片搬入後續驗

使用者後續將原片放入主 checkout 的 `data/uploads/`。三支 development 原片 SHA-256 均與既有 manifest 完全一致：`636c4bb3…`（223.393767 秒）、`414c278b…`（221.687256 秒）、`ace52809…`（602.570378 秒）。其餘新原片本輪未分析；僅把第一支透過 18082 的正常 TUS API 匯入，執行既有 processor 一次並準備完整 H.264 播放副本。直接放置 MP4 不會自動建立網站 DB 紀錄；原 checkout 的 DB 關聯仍是原有一筆工作。

**結論：第一支真實影片已具備整片人工標註條件；真人真值仍待使用者操作。** 主來源為 HEVC 10-bit、1920×1080，瀏覽器相容副本為 640×360、H.264 8-bit、223.4 秒。不是只播放模型實驗的 116 秒片段，也沒有新增模型推論。

- 登入入口：`http://127.0.0.1:18082`，使用該隔離 instance 的 admin 帳號。
- 已匯入影片：`2026-04-25-real-film-223s.mp4`，job `0c2f73a1b8154034b89b084569ee4b51`。
- 登入後直接進入：`http://127.0.0.1:18082/static/review/index.html?job=0c2f73a1b8154034b89b084569ee4b51`。
- 本 worktree 的 `data/real-import-receipt.json` 保存匯入資訊；主網站 owner review DB 已註冊真實 SHA-256。操作前後完整 export 相同，目前 **0 筆標註、coverage 空白**，可從頭建立真人紀錄。

實測結果：

| 驗證 | 證據 |
| --- | --- |
| 真實影像 | Chromium 在 0、110、220 秒播放，三處均有 640×360 影像及 decoded frames；已人工查看三張截圖，確認桌球影片畫面 |
| 相容播放 | 全片 duration=223.4；HTTP Range 回應 206／1024 bytes；模型面板預設收合 |
| 真實原片 GPU | 在 110 秒附近強制 CUDA frames 解碼、p010le 下載及轉 yuv420p、h264_nvenc 編碼兩秒；exit 0，未以軟體解碼 fallback 通過 |
| 副本標註 | 在 18083 的獨立資料副本完成 110–113 秒 I/O、保存續播、保存不自動增加 coverage、明確全片 coverage、重載從 120 秒接續、150 秒未完成草稿恢復／放棄及匯出 |
| 權限 | 匿名 API 401；非 owner 讀取原片、審核及匯出，以及準備 proxy／寫 coverage 均回 404 |
| 重啟 | 只重啟測試副本，session 與完整 review export 保持一致 |
| 真人資料隔離 | 18082 的真實來源 export 全程不變；副本標記為 uncertain，備註為 AUTOMATION COPY ONLY，不是人工真值 |
| 現役資料 | 18080／18081 對應的兩份 DB SHA-256 與原驗收前一致；下載原片 bytes 不變 |

副本取得方式：先確認沒有 queued/processing jobs 或未完成 uploads，從 18082 容器以唯讀 SQLite connection 呼叫 backup API，核對每份備份 `integrity_check=ok`；媒體則從唯讀 bind mount 複製。最初直接在只讀 bind mount 開 WAL-mode DB 失敗，未把那次空目錄當成快照。最終副本為本 worktree 的 `data/real-review-copy-v2/`，Compose project `highlightcraft-real-review-copy`；完成後已停止其 18083 容器，保留資料與證據。這是本機驗收副本，不等於開發機完整資料備份或離機備份已驗收。

新增 `tests/browser/real-review.cjs` 支援首中尾唯讀播放及明確副本操作；寫入模式只允許專用 18083 port。實際證據為 `real-source-inventory.json`、`real-playback-browser.txt`、`real-copy-browser.txt`、`real-hevc-gpu.txt`、`real-copy-restart.json`、`real-human-state-final.json`。影片截圖保留在本 worktree 的 `data/real-playback-1/`，不隨 Git 證據提交。

本輪只新增驗收腳本與文件，產品 src／映像未變，因此沿用同主機已完成的 118 passed；另執行新腳本 syntax、禁止對真人入口寫入的 guard 檢查、Ruff、git diff check。正式 DB 遷移、舊審核資料延續、完整備份還原、正式切換與發布仍待後續資料及授權。

## 部署整合驗收（2026-09-13）

**本機子路徑與根路徑驗證通過；正式資料選擇及外網入口仍待確認。**
來源為 `213fc18` 上的人工標註驗收 worktree，未提交 44 檔已逐位元保存於本機快照，
並在獨立 `codex/deployment-integration-20260913` 分支以 `0315009` 留下提交。
不改來源 branch、不清理來源 data、不合併 preserve-local，也未推送、發布或更動正式入口。

| 驗證 | 結果與限制 |
| --- | --- |
| 完整 Python 回歸 | 128 passed，19.08 秒；之後補增停寫 WAL 還原模式，備份專項 3 passed（包含既有 2 項重驗） |
| Ruff／鎖檔／JS syntax | 通過；來源原始 FFmpeg logs 的尾端空白逐位元保留，不為了 diff-check 更改歷史證據 |
| Nginx + TLS 子路徑 | 18443 `/pingpong-highlight/`：登入、靜態資源、API、job artifact URL、同 origin 私人下載、query、308／307、可信 scheme 通過 |
| 根路徑 | 18086：登入、上傳、播放、標註、下載回歸通過 |
| 子路徑前端續傳 | HTTPS 登入後 reload、重新選取原檔，UI 找到相同 upload ID 並續傳成功，下載 bytes 一致 |
| TUS 真斷線 | POST → PATCH 半份 → socket 中途斷線 → 新 client HEAD → stale offset 409 → 正確 offset 續傳，成品處理完成 |
| 分塊上限 | 前端 config 8 MiB；8 MiB 與 32 MiB PATCH 通過；32 MiB+1 被 Nginx 413 拒絕、offset 不变；只刪除該測試未完成上傳 |
| 權限與 cache | 匿名 401、其他 owner 404；媒體與審核 no-store；子路徑 cookie Path/Secure/HttpOnly/SameSite、登出清除與 sibling cookie 保留通過 |
| 代理 header | Nginx 覆寫 client 傳入的偽造 scheme/host；不可信容器直接送 XFP 不被 Uvicorn 信任；allowed host 400 通過 |
| 影片 | 真實 223.4 秒片的還原副本在 0/110/220 秒有 640×360 decoded frames；已查看中段截圖；Range 206 與首中尾拖曳通過 |
| 標註工作流 | fixture I/O、保存續播、保存不代表 coverage、顯式 coverage、草稿恢復／放棄、匯出通過；不當人工真值 |
| 三組既有瀏覽器測試 | full-review、rally-review、workspace 通過，含獨立 review CLI 與 390px 工作區 |
| 副本重啟 | cookie session 保留，完整 review export、1 筆 fixture 與 coverage 0–4 秒不變 |
| 映像 | 本機 ID `160214f083a2…`，runtime commit `b367685`；45 個 src 檔案 hash、31 個鎖定依賴一致；NVDEC/NVENC doctor 可用 |
| 主 DB 還原與 migration | 18080 snapshot 11 檔 hash、SQLite、原片及輸出關聯通過；副本升級後仍 1 job，新增 owner 管理表且無未歸屬 upload |
| 18082 還原 | 31 檔、5 DB integrity ok、foreign key check 空、來源／成品／preview 關聯通過；含 7 users、4 completed jobs；不直接升為正式資料 |
| 停寫演練 | 只停止本輪子路徑副本；唯讀 mount、SQLite/WAL scratch recovery、37 檔備份至新目錄還原通過，再啟動副本 |
| 正式資料保護 | 18080/18082 邏輯 DB hash 與本輪備份一致；18080/18081 DB bytes hash 與上一輪證據相同；來源 44 檔 hash 不變 |
| 外網 | 出站 TCP 7844、443 可連；hostname、DNS、NAT/CGNAT、帳號方案、公開 TLS、實際 tunnel、大檔/慢速外網及外站回歸均未驗證 |

第一輪瀏覽器執行曾因選錯本機 Node module 路徑而未啟動；改用已存在的 Playwright 後完成，
沒有因此下載新套件。上限測試首次清理缺 Tus-Resumable 回 412，已修正並只清除具名 fixture。
最初 pytest 在 sandbox 的 socket 限制下卡住，停止該測試程序後於主機隔離 fixtures 重跑；
子路徑第一轮亦發現 middleware 在 static mount 修改 scope 後才取 route path，已修正在呼叫前取得，
故 static cache headers 現已通過。這些除錯未對真人入口發送写入請求。

證據在 [deployment-integration-20260913](evidence/deployment-integration-20260913/)。
影片、SQLite、secret、自簽憑證與瀏覽器截圖只留本機 ignored data。
切換／rollback 的完整操作見 [部署手冊](deployment.md#可執行切換流程須先取得正式停寫與入口變更授權)。

剩餘 gate：實際 hostname 與現有網站歸屬、Windows 正式 DB／舊審核來源、18082 fixture 與真人資料
的選擇、瀏覽器未保存草稿、離機備份位置，以及影片流量適用方案。正式停機、切換、DNS、推送及發布
均需對具體差異另行授權。沒有任何新模型下載、Qwen 擴大實驗或候選演算法修改。
