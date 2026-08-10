# Architecture

## Design goals

1. 手機只負責錄影與選檔，耗電的解碼、推論和編碼全部留在電腦。
2. 影片可以很長，網路中斷不必重傳，服務重啟不遺失工作狀態。
3. 分析以時間戳而非 frame number 為真實來源，正常處理手機常見的 HEVC、VFR 與 rotation metadata。
4. 第一個 baseline 不依賴固定鏡位、球桌方框或數 GB 模型權重。
5. 每次結果留下可比較的 `analysis.json`，後續模型迭代能量化，而不是憑感覺調參數。
6. 產品輸出以一個 scored point 為剪輯單位，再組成短 Reel；不把多分混成一個長候選段。

## Components

### Mobile upload UI

`src/pingpong_highlight/static/` 是無 build step 的 mobile-first 頁面。它把影片切成 8 MiB blob，依序 `PATCH` 到 upload resource。每次 request 都帶 server offset；request 或 response 中斷時，client 先用 `HEAD` 查詢電腦實際收到的位置，再決定是否重送。瀏覽器允許 Web Crypto 時，另帶 SHA-256 checksum。

瀏覽器不允許頁面在背景永久執行，因此 iOS 把 Safari 完全關掉時上傳仍會停下；但 partial file 和 offset 會保留。重新開頁、再選同一個原檔即可續傳。

`GET /api/uploads` 會回傳尚未完成的 upload offset、總大小與最後更新時間。前端把它和 `/api/jobs` 組成一致的活動視圖，因此重新整理或換到另一台已授權裝置仍能監看伺服器實際收到的百分比。另一台裝置沒有手機相簿裡的 `File` bytes，只能監看；續傳仍由來源裝置重新選擇同一原檔後執行。

### Upload store and state

- `uploads.py` 只用隨機 ID 當磁碟檔名，原始 filename 僅作 metadata，避免 path traversal。
- chunk 先寫獨立暫存檔並驗證 checksum，再 append 到 `.part`。完整收到後才以 atomic rename 變成分析輸入。
- `db.py` 以 SQLite WAL 保存 upload offset、job 狀態、進度與結果。
- `jobs.py` 預設單 worker，避免多份影片互搶 GPU／磁碟頻寬。重啟時會把 `processing` 工作重新排入 `queued`。

目前內建 store 以單一 uvicorn process 為假設。若公開部署或多機擴充，API contract 可保留，傳輸層改接官方 `tusd`，輸入放 S3-compatible object storage，工作佇列改用 Redis／Postgres。

### Timestamp-based media layer

`pipeline/media.py` 用 `ffprobe` 取得 duration、codec、audio stream 與 rotation，再啟動兩條 FFmpeg decode pipe：

- audio：16 kHz mono float PCM；
- video：8 fps、320 × 320 letterboxed grayscale raw frames。

FFmpeg 預設會在 filter stage 套用 rotation metadata。固定 fps filter 讓第 `n` 個分析畫面對應 `n / analysis_fps`，不會沿用不可靠的原片 frame count。固定尺寸只供訊號分析，最終輸出仍從原片重編碼。

### Signal fusion baseline

音訊每 16 ms 計算一次短時頻譜，組合正向 spectral flux、高頻能量與 RMS transient，再以每分鐘 contextual median／MAD 正規化。non-maximum suppression 把局部峰值轉成 impact events。

畫面訊號計算相鄰 sample 的灰階差，切成 8 × 8 blocks。只聚合變化最大的八分之一區塊，並扣除全畫面背景變化，因此不必知道球桌在哪裡。曝光突變或 scene cut 影響大多數 blocks，會被抑制。

相鄰 impact events 依合理回球間隔組成 point candidate；impact count、tempo、節奏一致性、span 與局部 motion 共同形成 ranking score。沒有可靠 audio candidate 時才使用 motion-only fallback。

point candidate 不會再彼此合併。相鄰兩分的 padding 若重疊，兩者會平分中間的安靜區域，避免下一次發球或上一分反應同時出現在兩個片段。排名後使用「最多分數 + Reel 目標秒數」雙重預算選片，再依原片時間排序播放。

### Export

舊版的 `-c copy` 只能在 keyframe 附近切割。現在每個 point 都經 accurate seek 後重編碼成 H.264/AAC，加 `faststart` 方便手機播放。

`build_point_reel()` 以第一個單分片段的解析度與畫面比例作為成品規格。FFmpeg `xfade` 與 `acrossfade` 只建立在相鄰 point 的交界，因此 N 個 point 只有 N−1 個 dissolve，最後一分不會 fade-out。直式、裁切與字幕屬於發佈衍生版本，不改變核心分析輸出；`build_social_reel()` 保留為後續 renderer，但不在預設流程使用。

完成頁以同一個受 token 保護的檔案端點提供兩種回應：inline response 供 `<video>` range playback，`download=true` 則加入 attachment header。手機可以先預覽，再使用一般下載或 Web Share 儲存；單分片段與分析報告收在次要展開區。

## Failure and recovery model

| Failure | Recovery |
| --- | --- |
| 手機 Wi-Fi 短暫中斷 | client `HEAD` offset 後重試 |
| 手機關頁 | 重新選同一檔案後續傳 |
| chunk 損毀 | checksum mismatch，不推進 offset |
| 服務在上傳途中停止 | `.part` 大小與 SQLite offset 在啟動時 reconcile |
| 服務在分析途中停止 | job 重新排隊並從頭分析；不會重傳原片 |
| NVENC 不可用 | 同一 clip 自動改用 `libx264` |
| Reel filter 或編碼失敗 | 保留已輸出的單分片段並在報告記錄 warning |
| 無音軌 | motion-only fallback，報告會標記 |

## Intentional non-goals for this baseline

- 不追蹤 3 px 寬且常 motion-blur 的球；沒有專用訓練資料時，generic object detector 對此不可靠。
- 不把 generic pose ID 當 rally state；人站在畫面裡不代表正在打球。
- 不建立雲端物件儲存、付款流程或多人帳號系統。Quick Tunnel 只負責傳輸，影片分析與持久儲存仍是單機 local-first。
