# Ping-Pong Auto Highlight

把一段完整桌球錄影，自動剪成「以得分為單位」的直式精彩集錦。

手機只負責錄影與上傳；電腦在本地完成影片解碼、逐分切點、精彩度排序與 Reel 剪接。原片不會送到雲端。

## 成品形式

- 預設選出最多 6 個精彩得分，而不是輸出一段長時間區間。
- 每一分保留發球前與得分後的短暫脈絡。
- 集錦預設控制在約 55 秒內。
- 輸出 1080 × 1920、30 fps 的 9:16 MP4。
- 橫向原片完整置中，使用模糊背景填滿直式畫布，不裁掉兩側球員。
- 相鄰得分以 0.35 秒 cross-dissolve 連接；最後一分不做 fade-out。
- 同時保留每一分的獨立 MP4，方便人工檢查或重新排序。

## 安裝與啟動

需要 Python 3.11 以上、`ffmpeg` 與 `ffprobe`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pingpong-highlight doctor
pingpong-highlight serve
```

終端機會顯示手機網址與 QR code。手機和電腦連到同一個區域網路後，用手機開啟網址並選擇影片。上傳採用可續傳分塊；中斷後重新選擇同一檔案即可接續。

也可以直接分析電腦上的影片：

```powershell
pingpong-highlight analyze "D:\videos\match.mov"
```

每次工作會輸出：

- `best_points_reel.mp4`：直式得分集錦；
- `point_###_rank_##.mp4`：各個得分片段；
- `analysis.json`：切點、排名、媒體資訊與剪接設定。

## 流程

```mermaid
flowchart LR
    A["手機影片"] -->|"可續傳分塊上傳"| B["電腦本地儲存"]
    B --> C["時間戳式音訊與畫面分析"]
    C --> D["逐分切點"]
    D --> E["精彩度排序與時長預算"]
    E --> F["單分精準重編碼"]
    F --> G["9:16 版面與 cross-dissolve"]
    G --> H["得分 Reel"]
```

音訊瞬變提供擊球節奏，局部畫面動態協助排除缺乏比賽活動的雜音。分析以 FFmpeg 時間戳為準，可處理手機常見的 HEVC、VFR 與 rotation metadata。輸出優先使用 NVIDIA NVENC，失敗時自動退回 `libx264`。

## 主要設定

| 環境變數 | 預設值 | 用途 |
| --- | ---: | --- |
| `PINGPONG_DATA_DIR` | `./data` | 上傳、工作狀態與輸出資料夾 |
| `PINGPONG_HOST` | `0.0.0.0` | LAN 服務位址 |
| `PINGPONG_PORT` | `8000` | LAN 服務連接埠 |
| `PINGPONG_MAX_UPLOAD_BYTES` | 100 GiB | 單檔上限 |
| `PINGPONG_VIDEO_SAMPLE_FPS` | 8 | 畫面分析取樣率 |
| `PINGPONG_MAX_POINTS` | 6 | 集錦最多收錄幾分 |
| `PINGPONG_REEL_TARGET_SECONDS` | 55 | 集錦目標長度 |
| `PINGPONG_REEL_TRANSITION_SECONDS` | 0.35 | 得分間 dissolve 長度 |
| `PINGPONG_REEL_WIDTH` | 1080 | 直式成品寬度 |
| `PINGPONG_REEL_HEIGHT` | 1920 | 直式成品高度 |
| `PINGPONG_REEL_FPS` | 30 | 直式成品幀率 |

舊的 `PINGPONG_MAX_HIGHLIGHTS` 仍可作為 `PINGPONG_MAX_POINTS` 的備援值。

## 驗證

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

架構與評估方式見 [docs/architecture.md](docs/architecture.md) 與 [docs/evaluation.md](docs/evaluation.md)。
