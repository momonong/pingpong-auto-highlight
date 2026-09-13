# Architecture

## Design goals

1. 手機只負責錄影與選檔，耗電的解碼、推論和編碼全部留在電腦。
2. 影片可以很長，網路中斷不必重傳，服務重啟不遺失工作狀態。
3. 分析以時間戳而非 frame number 為真實來源，正常處理手機常見的 HEVC、VFR 與 rotation metadata。
4. 第一個 baseline 不依賴固定鏡位、球桌方框或數 GB 模型權重。
5. 每次結果留下可比較的 `analysis.json`，後續模型迭代能量化，而不是憑感覺調參數。
6. 產品輸出以一個 scored point 為剪輯單位，再組成短 Reel；不把多分混成一個長候選段。
7. 每筆上傳、Drive 匯入、處理工作與成品都有明確 owner；一般使用者彼此隔離，管理員才可做全域資料管理。

## Components

### Identity and access

空資料庫第一次啟動會建立 bootstrap administrator；密碼可由部署 secret 注入，未提供時只寫到持久資料目錄的 `.admin-password`。登入後由 HttpOnly session cookie 驗證請求，session 有固定有效期限；HTTPS tunnel／production 必須加上 `Secure`，純 HTTP localhost/LAN 則不能加，否則瀏覽器不會送 cookie。

使用者、password hash、session hash 與 owner 關聯保存在同一個 SQLite state store。一般使用者查詢與修改 upload、Drive import、job、artifact 時都會套用 user scope；administrator 才能管理帳號與檢視全域影片庫。舊 `X-Upload-Token` 僅保留作遷移期相容入口，新頁面與分享網址都不再承載 bearer token。

### Mobile upload UI

登入後首頁以影片清單為主；新增來源可展開，管理與人工標記切換獨立檢視，但不切換帳號 scope 或重啟傳輸。播放器保持按需載入；成品下載可直接操作，來源下載與破壞性操作收在次要選單。

前端保留無 build step 的 classic deferred scripts，依 `index.html` 順序共用 `core.js` 的 session/state bindings。`core.js` 管理 DOM references、帳號世代與 API；`uploads.js` 保留續傳協定；`sources.js` 管理來源卡與 Drive；`results.js` 管理影片卡、播放與下載；`annotations.js` 管理標記；`admin.js` 管理帳號／全域資料；`activity.js` 負責輪詢；`workspace.js` 管理檢視與篩選；`app.js` 綁定事件及啟動。這是功能拆分，尚未改成各模組獨立持有 state 的 ES module 架構。語言測試及靜態資源測試涵蓋全部功能檔案。

`connection.js` 使用獨立、禁止 cache 的 `/api/health` 檢查，初始狀態為 checking；每次檢查最多等待 6 秒，完成後 5 秒再查。僅有效的 `status=ok` JSON 回應可顯示 online。離線事件立即取消舊檢查，重新連線／頁面恢復可見時重新驗證；epoch 防止過期回應覆蓋新狀態。這只確認 HTTP 連線，不代表處理器 readiness。影片 API 的讀取有 15 秒期限，失敗時保留舊資料並標示 stale；上傳 PATCH 不套用短期限，以保留慢速網路傳輸行為。

瀏覽器回歸：安裝 Playwright 與 Chromium 後，以 `HIGHLIGHTCRAFT_TEST_PYTHON` 指向專案 dev Python，執行 `node tests/browser/workspace.cjs`。測試自行取得 localhost port、在 `data/browser-test-*` 建立獨立帳號與資料，生成短 MP4，結束時停止所啟動的服務。API、續傳及播放為實際流程；處理器與 Drive downloader 為 CPU-only fixture，不驗證模型或 Google 服務。螢幕截圖留在該測試目錄；不使用既有使用者影片。

`src/pingpong_highlight/static/` 是無 build step 的 mobile-first 頁面。它把影片切成 8 MiB blob，依序 `PATCH` 到 upload resource。每次 request 都帶 server offset；request 或 response 中斷時，client 先用 `HEAD` 查詢電腦實際收到的位置，再決定是否重送。瀏覽器允許 Web Crypto 時，另帶 SHA-256 checksum。

瀏覽器不允許頁面在背景永久執行，因此 iOS 把 Safari 完全關掉時上傳仍會停下；但 partial file 和 offset 會保留。重新開頁、再選同一個原檔即可續傳。

`GET /api/uploads` 會回傳尚未完成的 upload offset、總大小與最後更新時間。前端把它和 `/api/jobs` 組成一致的活動視圖，因此重新整理或換到另一台已授權裝置仍能監看伺服器實際收到的百分比。另一台裝置沒有手機相簿裡的 `File` bytes，只能監看；續傳仍由來源裝置重新選擇同一原檔後執行。

### Upload store and state

- `uploads.py` 只用隨機 ID 當磁碟檔名，原始 filename 僅作 metadata，避免 path traversal。
- chunk 先寫獨立暫存檔並驗證 checksum，再 append 到 `.part`。完整收到後才以 atomic rename 變成分析輸入。
- `db.py` 以 SQLite WAL 保存使用者歸屬、upload offset、job 狀態、進度與結果。
- `jobs.py` 預設單 worker，避免多份影片互搶 GPU／磁碟頻寬。重啟時會把 `processing` 工作重新排入 `queued`。

目前內建 store 以單一 uvicorn process 為假設。若公開部署或多機擴充，API contract 可保留，傳輸層改接官方 `tusd`，輸入放 S3-compatible object storage，工作佇列改用 Redis／Postgres。

### Google Drive import

`drive.py` 只接受明確的 HTTPS Google Drive 單檔網址，解析並保存 file ID，不會把使用者輸入當作任意下載網址，以避免 SSRF。公開影片由獨立的單 worker 背景下載器寫入 `data/drive-imports`；SQLite 保存 queued、resolving、downloading、failed 與 completed 狀態。下載完成後，檔案以 atomic rename 移入 upload store，並在同一個資料庫 transaction 建立既有 job，因此後續一律走相同的 GPU 優先分析與輸出流程。

下載器會保留 `.part`、回報電腦端 offset、限制單檔大小並預留磁碟空間。服務重啟會把中斷中的匯入重新排隊，再從磁碟上的部分檔案續傳。這條公開連結模式不需要 OAuth，但可讀權限由 Google Drive 連結本身承擔；多人或敏感資料版本應改成 OAuth service account／使用者授權，而不是擴大這個 bearer-link 模式。

### Timestamp-based media layer

`pipeline/media.py` 用 `ffprobe` 取得 duration、codec、audio stream 與 rotation，再啟動兩條 FFmpeg decode pipe：

- audio：16 kHz mono float PCM；
- video：8 fps、320 × 320 letterboxed grayscale raw frames。

Docker 預設掛入 NVIDIA 的 `compute,utility,video` capabilities。NVDEC runtime 可用時，video pipe 以 CUDA 硬體解碼並把畫面傳回系統記憶體，再由 FFmpeg filters 縮小成分析尺寸；不支援的 codec／pixel format 或 GPU 錯誤會自動重新以軟體解碼。FFmpeg 預設會在 filter stage 套用 rotation metadata。固定 fps filter 讓第 `n` 個分析畫面對應 `n / analysis_fps`，不會沿用不可靠的原片 frame count。固定尺寸只供訊號分析，最終輸出仍從原片重編碼。縮放、灰階 motion 與 NumPy 訊號計算仍在 CPU 執行。

### Signal fusion baseline

音訊每 16 ms 計算一次短時頻譜，組合正向 spectral flux、高頻能量與 RMS transient，再以每分鐘 contextual median／MAD 正規化。non-maximum suppression 把局部峰值轉成 impact events。

畫面訊號計算相鄰 sample 的灰階差，切成 8 × 8 blocks。只聚合變化最大的八分之一區塊，並扣除全畫面背景變化，因此不必知道球桌在哪裡。曝光突變或 scene cut 影響大多數 blocks，會被抑制。

相鄰 impact events 依合理回球間隔組成 point candidate；impact count、tempo、節奏一致性、span 與局部 motion 共同形成 ranking score。沒有可靠 audio candidate 時才使用 motion-only fallback。

point candidate 不會再彼此合併。系統先以同片最佳分數為基準套用相對門檻，再依分數由高到低放入 Reel 秒數預算；預設不設固定球數，也不會為了湊數回填。`max_points` 只保留為選用的安全上限。相鄰的已入選得分若 padding 重疊，兩者會平分中間的安靜區域，避免下一次發球或上一分反應同時出現在兩個片段；被門檻或預算淘汰的候選不會縮短入選片段的前後脈絡。最後依原片時間排序播放，rank 仍表示精彩度順序。FFmpeg 會把片段邊界對齊 frame／audio packet，因此成品 probe 長度可能和理論預算有極小差異。

目前相對門檻是 heuristic retention rule，不是校準過的精彩機率。`analysis.json` 會保存所有候選的核心區間、分數、有效門檻，以及 `selected`、`below-score-threshold`、`duration-budget`、`point-cap` 決策，供後續以完整正／負標記校準。

### Export

舊版的 `-c copy` 只能在 keyframe 附近切割。現在每個 point 都經 accurate seek 後重編碼成 H.264/AAC，加 `faststart` 方便手機播放。GPU runtime 可用時，輸入優先經 NVDEC 解碼並以 NVENC 編碼；能力檢查會實際試編一個 frame，避免只因 FFmpeg 列出 `h264_nvenc` 就誤判。任何 GPU 編解碼失敗仍會用 CPU／`libx264` 重試。

`build_point_reel()` 以第一個單分片段的解析度與畫面比例作為成品規格，將正規化後的影音 stream 以 FFmpeg `concat` filter 直接剪接，不重疊或淡化相鄰得分。直式、裁切與字幕屬於發佈衍生版本，不改變核心分析輸出；`build_social_reel()` 保留為後續 renderer，但不在預設流程使用。

完成頁以同一個需要有效登入、且會檢查 owner／管理員權限的檔案端點提供兩種回應：inline response 供 `<video>` range playback，`download=true` 則加入 attachment header。手機可以先預覽，再使用一般下載或 Web Share 儲存；單分片段與分析報告收在次要展開區。

## 模型提案與人工回合審核

`preannotate.py` 是不接觸正式 DB 的有界實驗入口：最多三段 development 素材、總長不超過 600 秒。`Provider.infer(clip,audio_events)` 可替換；第一個 VLM adapter 為 Qwen3-VL-8B-Instruct。baseline 仍呼叫現有 analyze_audio/analyze_motion/detect_points，沒有移植保存分支 v4。兩者共用經 FFmpeg autorotate、30 fps／640 寬的短片代理；這是分段比較，不代表整支原片的 main pipeline 重跑。

VLM 以重疊視窗掃描指定區段全部內容，另抽樣 2 fps／384 寬影格。PyAV 讀取 PTS 並核對抽樣 grid，再傳遞明確 VideoMetadata 給 processor；時間以 `source_ms=segment_start+window_offset+local_ms` 換回原片。模型看不到音軌，只有 baseline 提供的未驗證音訊 transient 時間。prompt 禁止稀疏影格數拍／勝負猜測；JSON 必須包含完整欄位、整數毫秒與列舉值。格式或越界失敗保留原文與錯誤，不修補成看似有效的回合。已規劃但輸出失敗的視窗仍保留在評估 scope，不從 recall 分母移除。跨窗近似重複以 IoU≥0.8 且起點差≤1 秒去重，保存被抑制的提案；部分相鄰回合留給人合併，不自動延長成完整一分。

`rally_review.py` 將 `sources`、不可變 `runs`、人工 `reviews` 及冪等 `commands` 分表保存。人工判斷包含回合有效性、完整性、精彩程度、理由、actor、時間、父標註與 proposal IDs；拆合保留 lineage 並清除原判斷。每次寫入使用 SQLite transaction、revision CAS 與 request ID。新推論只新增 runs，不修改 reviews。人工 coverage 表示人已在該區間檢查所有回合，空白區間維持 UNKNOWN。計時與修改數是觀測記錄，不是省時效果證據。

`review_web.py` 的 loopback 實驗服務與 `job_review.py` 的網站 adapter 共用 `static/review/`。前者只讀 manifest 原片，綁定 127.0.0.1，檢查 Host 與 SameSite session cookie；後者沿用 job owner/admin 權限，並按影片 owner 分開審核 DB，避免同 bytes 的跨使用者資料洩漏。網站審核 DB 位於 `data/rally-review/<owner-hash>.sqlite3`；不修改既有 annotations schema。主網站匯入舊 analysis 時 commit/耗時未知者明列 UNKNOWN，不冒充本次推論。

固定來源時間區間可指定 blind_intervals。首次人工判斷前，API 不傳模型判斷、精彩建議、理由、原始回答或去重原文；人工紀錄保留之後才顯示該提案。它降低建議錨定風險，但候選區間本身仍由模型提出，不能稱為完全盲測。

人工工作區以完整來源影片為單位，優先播放 `review_media.py` 建立的整片 H.264／yuv420p 副本（時間從零開始，與原片一致）。播放 manifest 存於審核 DB 同目錄下的 `review-media/<source-sha>/preview.json`，與不可變模型 run、人工 DB 狀態分離；先重驗來源 SHA-256，再 accurate seek／autorotate／8-bit 轉碼，驗證長度後 atomic 寫入 manifest。相同來源可重用已完成副本。prepare-review CLI 與受權限保護的 full-preview API 均不呼叫模型，不擴大 model scope。

「保存回合，繼續播放」只保存一筆標註並從終點續播；「確認已檢查到此處」才明確保存 coverage。重新載入以第一個未檢查區段起點作為續審位置，草稿可恢復或明確放棄。模型建議、原始檔與實驗短片切換放在選用區；選候選批次不再切換影片。實驗短片仍保留原片時間換算供比較。實際播放驗收需確認 videoWidth/videoHeight 與畫面，只有播放時間前進可能是僅音訊播放。

`review_evaluation.py` 以 source-local 最大一對一匹配（交集≥50% 人工區間）分開評估候選核心、入選核心、padding、切點誤差與未匹配項。候選 precision 需完整 scope coverage 及無未决回合／邊界；入選 precision 另需完整精彩評分。歷史正標註只能評估已知精彩球覆蓋。55 秒 Reel 效用、回合 purity、held-out 準確度與真人省時均需另外取得證據。

## Failure and recovery model

| Failure | Recovery |
| --- | --- |
| 手機 Wi-Fi 短暫中斷 | client `HEAD` offset 後重試 |
| 手機關頁 | 重新選同一檔案後續傳 |
| chunk 損毀 | checksum mismatch，不推進 offset |
| 服務在上傳途中停止 | `.part` 大小與 SQLite offset 在啟動時 reconcile |
| Drive 下載中斷或服務停止 | 保留 `.part`，頁面重試或下次啟動後續傳 |
| Drive 權限或下載政策拒絕 | 匯入標記失敗，修正共用權限後從頁面重試 |
| 服務在分析途中停止 | job 重新排隊並從頭分析；不會重傳原片 |
| NVDEC 不可用或不支援來源格式 | 同一支影片自動改用 CPU 解碼 |
| NVENC 不可用 | 同一 clip 自動改用 `libx264` |
| Reel filter 或編碼失敗 | 保留已輸出的單分片段並在報告記錄 warning |
| 無音軌 | motion-only fallback，報告會標記 |

## Intentional non-goals for this baseline

- 不追蹤 3 px 寬且常 motion-blur 的球；沒有專用訓練資料時，generic object detector 對此不可靠。
- 不把 generic pose ID 當 rally state；人站在畫面裡不代表正在打球。
- 不建立雲端物件儲存、付款流程或跨節點租戶平台。帳號只用來隔離這個單機 instance 上的試用者；Quick Tunnel 只負責傳輸，影片分析與持久儲存仍是單機 local-first。

## 子路徑與反向代理契約

正式部署可設定 `PINGPONG_ROOT_PATH=/pingpong-highlight`；空字串保留根路徑。
`PINGPONG_PUBLIC_URL` 的 path 必須一致，不從 Host 或 forwarded prefix 猜測路由。
Nginx 的 `location /pingpong-highlight/` 配 `proxy_pass http://backend/` 去掉前綴；
CLI 同時給 Uvicorn 與 FastAPI 相同 root_path，讓 ASGI 路徑、靜態掛載 redirect 與 query 保持一致。

只有兩個 HTML entry template 的 `__HC_ROOT_PATH__` slot 在伺服器端填入；
`paths.js` 讀取 meta，再由共用 URL helper 處理 API、媒體與下載。
後端的 TUS Location 與 job artifact URL 自帶前綴，helper 不會重複加前綴。
沒有代理 response body 替換，也沒有把網域的全域 `/api` 或 `/static` 接到本 app。
獨立 preannotate review CLI 仍只服務 loopback 根路徑，不能用作正式公開入口。

cookie 預設名稱依 root_path 分隔，根模式沿用 `pingpong_session`；可指定部署專用名稱。
Path 為 `<root_path>/`，Secure 由部署明確設定，HttpOnly／SameSite=Strict 保持。
登入、換密碼、登出使用同一名稱與 Path；子路徑登出不發會影響整個 origin 的 Clear-Site-Data。
Path 用來避免名稱與送出範圍衝突，不是同 origin 不可信網站間的安全邊界。
續傳 storage key 亦依前綴隔離；舊 origin 草稿不會被刪除，但搬往新 origin 不會自動搬移。

`PINGPONG_FORWARDED_ALLOW_IPS` 空白時不信任 forwarded headers；正式配置必須列出
實際直接代理來源 IP，禁止 `*`。Nginx 覆寫 Host／scheme／client IP，清除另一套
Forwarded／X-Forwarded-Host／Prefix／CF header；`PINGPONG_ALLOWED_HOSTS` 限定正式 hostname。
部署模式設定 `PINGPONG_TRUSTED_PROXY_PROVIDER=none`，由 Uvicorn 的明確來源信任處理 IP，
避免 legacy provider 再次解讀 header。所有 API（包含拒絕回應）、私人媒體及審核 export
均為 private/no-store；入口與任何 CDN cache rule 必須 bypass 此子路徑。
