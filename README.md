# Ping-Pong Auto Highlight

把一段完整桌球錄影，自動剪成「以得分為單位」的精彩集錦。

手機只負責錄影與提供影片；可以直接續傳，也可以先放上 Google Drive 再貼公開連結。電腦在本地完成下載、影片解碼、逐分切點、精彩度排序與 Reel 剪接。匯入後的原片副本與成品持久儲存在這台電腦；使用 ngrok 或 Cloudflare Tunnel 時，網頁流量會經過該入口服務，而 Drive 原片由電腦直接向 Google 下載。

## 成品形式

- 預設選出最多 6 個精彩得分，而不是輸出一段長時間區間。
- 每一分預設在實際回合前後各保留 1.5 秒脈絡。
- 集錦預設控制在約 55 秒內。
- 預設保留原片的寬高比例、方向與畫面內容。
- 相鄰得分以 0.35 秒 cross-dissolve 連接；最後一分不做 fade-out。
- 同時保留每一分的獨立 MP4，方便人工檢查或重新排序。

直式、裁切、字幕等社群發佈格式屬於後續輸出，不會在分析階段綁死。

## 啟動系統（平常從手機使用）

以下是電腦重新開機後的標準流程。預設使用 ngrok，因為它的主要 tunnel 連線走 TLS 443，不像 Cloudflare Tunnel 需要目前被學校網路封鎖的 7844 連接埠。第一次使用需要免費 ngrok 帳號與 authtoken。

1. 開啟 Docker Desktop，等到左下角顯示 Docker Engine 正在執行。
2. 開啟 Git Bash，進入專案目錄：

   ```bash
   cd /d/projects/pingpong-auto-highlight
   ```

   如果之後移動了專案資料夾，請把路徑換成新的位置。

3. 第一次使用時，先在 [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) 建立帳號並複製 authtoken。它相當於 ngrok 帳號密碼，請只貼進下一步的本機隱藏提示，不要貼到聊天、README 或公開場所。
4. 啟動剪輯服務與手機外網入口：

   ```bash
   ./scripts/start-ngrok-tunnel.sh
   ```

   第一次執行會顯示 `Paste the ngrok authtoken`；輸入內容不會顯示在畫面上，按 Enter 後會存入 Git 已忽略的 `data/.ngrok-authtoken`，之後不必重貼。腳本預設會把影片解碼與編碼交給 NVIDIA GPU，並啟動或更新 Docker 服務。若電腦暫時沒有可用的 NVIDIA Docker runtime，才改用：

   ```bash
   ./scripts/start-ngrok-tunnel.sh -CpuOnly
   ```

5. 等終端機顯示 `ngrok tunnel is ready.`，把下一行完整 HTTPS 網址傳到自己的手機並開啟。最新網址也可以隨時從電腦讀取：

   ```bash
   cat ./data/remote-access-url.txt
   ```

   免費方案第一次以瀏覽器開啟該網域時，ngrok 會先顯示防濫用提示；確認網址是自己剛產生的，再按一次 `Visit`。ngrok 會為該網域保存 cookie，通常七天內不再顯示。

   完整網址含有私人存取權杖，拿到網址的人就能使用服務，請勿分享或貼在公開場所。

6. 想確認系統是否正常，可執行：

   ```powershell
   docker compose -f compose.yaml -f compose.ngrok.yaml ps
   ```

   `pingpong-highlight` 應顯示 `healthy`，`ngrok` 應顯示 `Up`。

   想確認容器內的 GPU 編解碼真的可用，而不是只有看得到顯示卡，可再執行：

   ```powershell
   docker compose exec pingpong-highlight pingpong-highlight doctor
   ```

   `NVIDIA NVDEC` 與 `NVIDIA NVENC` 都應顯示 `可用`。

使用期間請讓電腦保持喚醒，並維持 Docker Desktop 與網路連線。影片仍在上傳時，不要重啟 Docker、`ngrok` 或電腦；上傳完成後，手機頁面可以關閉，電腦會繼續分析與剪輯。

### 用 Google Drive 加入影片（大檔案建議）

這條路徑通常比手機透過公開 tunnel 把大檔案直接傳到電腦更穩定，而且貼完連結後可以立刻關閉手機頁面：

1. 在手機的 Google Drive 上傳原始影片，等 Drive 顯示上傳完成。
2. 開啟該影片的「管理存取權」或「共用」，把一般存取權改成「知道連結的任何人」，角色選「檢視者」。只要允許下載，不需要給編輯權限。
3. 複製的是單一影片連結，不是資料夾連結。
4. 在 RallyCut 的 Google Drive 區塊貼上連結，按「開始匯入」。
5. 「Drive 下載中」會顯示電腦端實際收到的進度；完成後會自動消失並變成 GPU 剪輯工作，不需要再按一次開始。

系統端不需要 Google 帳號、OAuth、API key 或額外設定。公開連結等同任何拿到連結的人都能讀取，因此不要用在敏感影片。等 RallyCut 狀態已從 Drive 下載切換成「排隊中／分析中」後，就可以把 Google Drive 共用權限改回「受限制」；電腦已下載的本機副本不受影響。

Drive 匯入的狀態與暫存檔都在 `./data`。網路中斷或服務重啟時，會保留已下載部分並在下次啟動續傳；失敗項目也可以直接在頁面重試或刪除。Google 仍可能因下載次數、擁有者禁止下載或組織政策拒絕公開下載，這時頁面會保留進度並顯示權限提示。

### 手動標記精彩球

完成的工作下方有「手動標記精彩球」。在電腦上展開後，按「在電腦開啟標記工作區」會進入大播放器與右側標記清單；原片到這時才會以 HTTP Range 載入，因此可以直接拖曳長影片，不必先下載整支檔案。手機上仍可使用原本的小型標記表單。

1. 播放或拖曳原片，在該回合實際開始的位置按 `I` 設起點。
2. 到該分結束時按 `O` 設終點，再按 `Enter` 儲存；全程不需要抄寫或輸入時間碼。
3. 選「值得收錄」；如果它是模型選到但你不喜歡的球，改選「不該收錄」。備註可填反拉、長回合等原因，也可留白。
4. `Space` 控制播放／暫停，方向鍵前後 1 秒，`Shift` 加方向鍵前後 5 秒，`Esc` 關閉工作區。標記會綁定原片與時間碼，存在 `data/state.sqlite3`，重新整理、更新容器或重跑模型都不會消失。

第一輪建議選 2–3 支不同角度、距離或光線的影片，把其中所有你會放進集錦的球標完；另留 1 支完全不參與調整，最後才用來驗證是否真的改善。先累積約 15–30 個「值得收錄」的球就足以開始比較；不必一次標完整個影片裡所有普通球。

即使頁面重新整理或公開網址改變，電腦已收到的分塊也不會遺失。請回到持有原始影片的手機，重新選擇同一支影片；只要伺服器上剛好有一筆檔名與大小相同的未完成紀錄，系統就會從保存的 offset 續傳。其他裝置可同步查看進度，但無法代替來源手機提供原始檔案。若同一影片不小心留下多筆紀錄，頁面會先要求刪除重複項目，避免再建立第四筆；「刪除這筆上傳」只會刪除電腦上的未完成分塊，不會影響手機原片。

只關閉手機外網入口、保留本機服務與所有資料：

```powershell
docker compose -f compose.yaml -f compose.ngrok.yaml stop ngrok
```

停止全部服務但保留 `data` 裡的原片、進度與成品：

```powershell
docker compose -f compose.yaml -f compose.ngrok.yaml stop
```

下次使用時重新執行啟動腳本即可。如果啟動失敗，先查看兩個服務的狀態與最近紀錄：

```powershell
docker compose -f compose.yaml -f compose.ngrok.yaml ps
docker compose -f compose.yaml -f compose.ngrok.yaml logs --tail 100 pingpong-highlight ngrok
```

若 ngrok 顯示 authtoken 無效，重新執行 `./scripts/start-ngrok-tunnel.sh -ReplaceAuthtoken`；若本機 4040 已被其他程式使用，改用 `./scripts/start-ngrok-tunnel.sh -InspectPort 4041`。

## Docker 常駐服務與區域網路

若只從手機外網使用，照上一節操作即可。以下設定用於同一個 Wi-Fi 的區域網路連線；需要先安裝並啟動 Docker Desktop。第一次設定：

```powershell
Copy-Item .env.example .env
notepad .env
docker compose up -d --build
docker compose logs -f pingpong-highlight
```

把 `.env` 裡的 `PINGPONG_PUBLIC_URL` 改成電腦目前的 Wi‑Fi IP，例如 `http://192.168.1.19:8000`。啟動後，log 會顯示完整手機網址與 QR code；`restart: unless-stopped` 會讓容器在 Docker 重新啟動後自動恢復。

上傳原片、Drive 下載暫存、續傳資訊、工作狀態與成品都掛載在電腦的 `./data`，重新 build 或刪除容器不會遺失。要更新程式時再執行一次：

```powershell
docker compose up -d --build
```

這份預設配置使用 NVIDIA GPU，並掛入 FFmpeg 所需的 NVDEC／NVENC driver capability。這台電腦平常直接使用：

```powershell
docker compose up -d --build
```

只有在 NVIDIA Docker runtime 暫時不可用時，才使用 CPU override：

```powershell
docker compose -f compose.yaml -f compose.cpu.yaml up -d --build
```

常用管理指令：

```powershell
docker compose ps                    # 查看服務與健康狀態
docker compose logs -f pingpong-highlight
docker compose restart               # 重新啟動
docker compose down                  # 停止服務；保留 ./data
```

若希望登入 Windows 後一直可用，請同時開啟 Docker Desktop 的「Start Docker Desktop when you sign in」。

## 在 4090／其他 NVIDIA 電腦使用已發佈 image

Docker Hub 的 public image 是 `docker.io/momonong/pingpong-auto-highlight:1.2.0`。RTX 5090 Laptop 與 RTX 4090 Desktop 都使用同一個 `linux/amd64` image；image 不包含 NVIDIA driver，啟動時由主機的 NVIDIA Container Toolkit 提供 NVDEC／NVENC 所需元件，因此不要建立 `5090` 或 `4090` 專用 tag。

新電腦需要先安裝並啟動 Docker Desktop、使用 Linux containers，並讓 Docker 能存取 NVIDIA GPU。取得這份 repository 後，在 Git Bash 執行：

```bash
cd /d/projects/pingpong-auto-highlight
./scripts/start-ngrok-tunnel.sh -UsePublishedImage
```

這個選項會直接 pull 版本固定的 public image，不會在新電腦重新 build。只在 GPU runtime 暫時不可用、確定願意接受較慢速度時，才同時加上 `-CpuOnly`。

若只需要區域網路、不啟動 ngrok：

```powershell
docker compose -f compose.yaml -f compose.release.yaml up -d
docker compose -f compose.yaml -f compose.release.yaml exec pingpong-highlight pingpong-highlight doctor
```

`doctor` 的 `NVIDIA NVDEC` 與 `NVIDIA NVENC` 都必須顯示 `可用`。目前 5090 Laptop 已實測通過；4090 移機後仍應用同一支測試影片做一次驗收，比較選出的得分時間點、成品長度與是否可播放，不要比較 MP4 檔案 hash，因為不同 GPU／driver 的硬體編碼結果不保證逐 bit 相同。

正式使用請固定版本 tag 或 `data/published-image.txt` 記錄的 digest；`latest` 只供方便查看最新版本，不應作為長期部署鎖定值。所有原片、續傳與成品仍在該電腦的 `./data`，不會存進 Docker Hub image。

### 發佈新版本到 Docker Hub

只有在變更已提交、合併到乾淨的 `main`，且 Docker Desktop 已登入 `momonong` 時才執行：

```bash
./scripts/publish-dockerhub.sh
```

發佈器固定產生 `linux/amd64`，使用 `pyproject.toml` 的版本建立 immutable version tag 與 `latest`，附加 SBOM／provenance，確認 repository 是 public，再從 Docker Hub pull 回來檢查版本與 NVIDIA NVDEC／NVENC。Python base image、production dependencies 與 build dependencies 都以 digest 或 hashes 固定；`data`、`.env`、影片及 token 由 `.dockerignore` 排除。

## ngrok Tunnel 細節

Git Bash 的標準啟動指令：

```bash
./scripts/start-ngrok-tunnel.sh
```

若手機第一次開啟時先看到 ngrok 的「Visit Site」頁面，按下後請再開一次終端顯示的完整網址（必須包含 `#token=...`）。第一次只是在該手機建立 ngrok 的瀏覽器 cookie，跳轉時可能沒有保留 fragment；RallyCut 收不到 token 時仍能顯示首頁，但會無法讀取伺服器端的處理 session。此時可重新開啟完整網址，或把完整網址／存取碼貼進頁面的解鎖欄位。session 與成品都還保存在電腦的 `data`，不會因手機換頁或重新整理而消失。

PowerShell 也可以使用同一套流程：

```powershell
.\scripts\start-ngrok-tunnel.ps1
```

啟動器會確認 Docker、啟動預設 GPU 服務與 ngrok、等待公開 health check 通過，再產生一條可直接在手機開啟的網址。ngrok 的 HTTP 請求檢視預設關閉，避免在本機 Traffic Inspector 保存影片要求與帶有 RallyCut 存取權杖的下載網址；4040 只綁在 `127.0.0.1`，啟動器用它讀取 tunnel 網址，不會對區域網路公開。

免費方案會提供一個開發用網域，而且 endpoint 沒有固定逾時，但目前每月包含 1 GB 對外傳輸與 20,000 個 HTTP requests。把原片先傳到 Google Drive、再讓電腦直接下載，不會讓整支原片經過 ngrok；手機直接上傳與下載完成的集錦則會使用 ngrok 額度。額度可能調整，實際數字以 [ngrok Free plan limits](https://ngrok.com/docs/pricing-limits/free-plan-limits) 為準。

只停止 ngrok、保留本機剪輯服務與資料：

```powershell
docker compose -f compose.yaml -f compose.ngrok.yaml stop ngrok
```

要讓新 token 取代本機保存的舊 token：

```bash
./scripts/start-ngrok-tunnel.sh -ReplaceAuthtoken
```

## Cloudflare Quick Tunnel 細節（備用）

不需要 Cloudflare 帳號或網域。Docker Desktop 啟動後，在專案目錄執行。Git Bash：

```bash
./scripts/start-cloudflare-tunnel.sh
```

PowerShell 也可以使用同一套流程：

```powershell
.\scripts\start-cloudflare-tunnel.ps1
```

腳本會啟動本機服務、保留健康的既有 tunnel、在 tunnel 失效時自動建立新的臨時 HTTPS 網址、確認公開 health check，最後顯示一條可直接在手機開啟、含有存取權杖的專用網址。不要把完整網址轉傳給別人。網址裡的 upload token 放在 `#` 後方，不會隨第一次 HTTP 請求送到 Cloudflare；頁面讀取後也會立刻從網址列移除。

這台電腦與 Docker Desktop 必須保持開啟。Quick Tunnel 是測試用途，沒有固定網址或 uptime SLA；`cloudflared` 容器重建後需重新執行腳本並使用新網址。最新網址也會保存在本機的 `data/remote-access-url.txt`。新網址仍可藉由檔名與檔案大小接回唯一一筆未完成上傳；若之後要固定書籤或避免網址變動，再改用 Cloudflare named tunnel。

這台電腦沒有可用的 NVIDIA Docker runtime 時，才改用 CPU 模式：

```bash
./scripts/start-cloudflare-tunnel.sh -CpuOnly
```

PowerShell：

```powershell
.\scripts\start-cloudflare-tunnel.ps1 -CpuOnly
```

只停止外網入口、保留本機剪輯服務與資料：

```powershell
docker compose -f compose.yaml -f compose.cloudflare.yaml stop cloudflared
```

如果紀錄出現 `CONNECTIVITY PRE-CHECKS`、`QUIC connection failed`，並同時顯示 TCP 與 UDP 失敗，代表目前網路封鎖 Cloudflare Tunnel 對外使用的 `7844` 連接埠；一直重啟不會解決。請改用上方 ngrok，或讓電腦改連家中網路／手機熱點後再重試。公司或學校網路則需要網路管理員允許對外 TCP 或 UDP `7844`。詳情見 [Cloudflare connectivity pre-checks](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/troubleshoot-tunnels/connectivity-prechecks/)。

## 本機 Python 開發

需要 Python 3.11 以上、`ffmpeg` 與 `ffprobe`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pingpong-highlight doctor
pingpong-highlight serve
```

終端機會顯示手機網址與 QR code。手機和電腦連到同一個區域網路後，用手機開啟網址並選擇影片。上傳採用可續傳分塊；中斷後重新選擇同一檔案即可接續。

上傳百分比以電腦實際保存的分塊為準，受 token 保護的網址可在手機或電腦跨裝置監看。重新整理不會遺失已傳資料，但瀏覽器基於安全限制不會自動恢復手機相簿裡的檔案；請在原來源裝置重新選擇同一支影片。系統會先使用該瀏覽器保存的工作階段；若網址已改變，則以完全相同的檔名與檔案大小接回伺服器上唯一的未完成紀錄。其他裝置能監看，無法代替來源裝置送出它沒有的原始檔案。

完整操作流程：

1. 電腦執行 `pingpong-highlight serve`，保持終端機與電腦開啟。
2. 手機掃描 QR code，從相簿選擇原始影片，或貼上公開 Google Drive 影片連結。
3. 手機直傳需等上傳完成才能關頁；Drive 連結送出後即可關頁，電腦會在背景下載並處理。
4. 回到同一網址即可直接預覽成品。
5. 使用「下載 MP4」，或在支援 Web Share 的手機使用「分享／存到相簿」。

成品也會保留在電腦的 `data/outputs/<job-id>/`。LAN 模式不需要雲端帳號或訂閱；Quick Tunnel 模式的傳輸會經過 Cloudflare，但分析、剪輯與持久儲存仍只在這台電腦進行。

也可以直接分析電腦上的影片：

```powershell
pingpong-highlight analyze "D:\videos\match.mov"
```

每次工作會輸出：

- `best_points_reel.mp4`：保留原片比例的得分集錦；
- `point_###_rank_##.mp4`：各個得分片段；
- `analysis.json`：切點、排名、媒體資訊與剪接設定。

## 流程

```mermaid
flowchart LR
    A["手機影片"] -->|"可續傳分塊上傳"| B["電腦本地儲存"]
    D["Google Drive 公開影片"] -->|"電腦背景續傳"| B
    B --> C["時間戳式音訊與畫面分析"]
    C --> D["逐分切點"]
    D --> E["精彩度排序與時長預算"]
    E --> F["單分精準重編碼"]
    F --> G["原片比例與 cross-dissolve"]
    G --> H["得分 Reel"]
```

音訊瞬變提供擊球節奏，局部畫面動態協助排除缺乏比賽活動的雜音。分析以 FFmpeg 時間戳為準，可處理手機常見的 HEVC、VFR 與 rotation metadata。預設容器使用 NVIDIA NVDEC 解碼、NVENC 編碼；不相容的影片或 GPU runtime 失效時會自動退回 CPU／`libx264`。音訊分析、NumPy 訊號計算與部分 FFmpeg filters 仍會使用 CPU，因此 CPU 用量不會完全歸零。

## 主要設定

| 環境變數 | 預設值 | 用途 |
| --- | ---: | --- |
| `PINGPONG_DATA_DIR` | `./data` | 上傳、工作狀態與輸出資料夾 |
| `PINGPONG_HOST` | `0.0.0.0` | LAN 服務位址 |
| `PINGPONG_PORT` | `8000` | LAN 服務連接埠 |
| `PINGPONG_PUBLIC_URL` | 自動偵測 | QR code 與手機要開啟的公開基底網址；Docker 建議明確設定 |
| `PINGPONG_MAX_UPLOAD_BYTES` | 100 GiB | 單檔上限 |
| `PINGPONG_DOWNLOAD_MIN_FREE_BYTES` | 2 GiB | Drive 匯入時保留的最低可用空間 |
| `PINGPONG_VIDEO_SAMPLE_FPS` | 8 | 畫面分析取樣率 |
| `PINGPONG_MAX_POINTS` | 6 | 集錦最多收錄幾分 |
| `PINGPONG_REEL_TARGET_SECONDS` | 55 | 集錦目標長度 |
| `PINGPONG_REEL_TRANSITION_SECONDS` | 0.35 | 得分間 dissolve 長度 |
| `PINGPONG_CLIP_PRE_ROLL_SECONDS` | 1.5 | 每球在實際回合前保留的秒數 |
| `PINGPONG_CLIP_POST_ROLL_SECONDS` | 1.5 | 每球在實際回合後保留的秒數 |

舊的 `PINGPONG_MAX_HIGHLIGHTS` 仍可作為 `PINGPONG_MAX_POINTS` 的備援值。

## 驗證

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

架構與評估方式見 [docs/architecture.md](docs/architecture.md) 與 [docs/evaluation.md](docs/evaluation.md)。
