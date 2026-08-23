# HighlightCraft

把完整桌球錄影整理成「以得分為單位」的精彩球素材庫，再自由組成跨影片集錦。

手機只負責錄影與提供影片；可以直接續傳，也可以先放上 Google Drive 再貼公開連結。電腦在本地完成下載、影片解碼、逐分切點、精彩度排序、素材保存與集錦剪接。匯入後的原片副本、精彩球片段與成品持久儲存在這台電腦；使用 ngrok 或 Cloudflare Tunnel 時，網頁流量會經過該入口服務，而 Drive 原片由電腦直接向 Google 下載。

## 素材與成品形式

- 每支影片會保存分數至少達到同片最佳候選 70% 的候選素材；87% 只標記為「推薦」，不再提前丟掉其他可用球。
- 每一分預設在實際回合前後各保留 1.5 秒脈絡。
- 分析階段沒有六球或 55 秒上限；長影片可以保存更多素材，短影片也不會為了湊數加入弱球。
- 預設保留原片的寬高比例、方向與畫面內容。
- 桌面素材庫可依來源、拍攝日期、相對分數與片段長度篩選排序；可取目前前六，也可讓每個來源各取前六。
- 每支集錦可勾選最多 100 球、調整順序後才輸出；超過一分鐘只提示，不阻止輸出，相鄰得分以 hard cut 直接剪接。

舊版已完成 Reel 仍以收合卡片保留。新版工作完成後會把達 70% 保存門檻的每一球送入素材庫；自訂集錦另列在輸出區，點開時才載入播放器。

直式、裁切、字幕等社群發佈格式屬於後續輸出，不會在分析階段綁死。

## 資料存在哪裡

系統採用「SQLite 存索引與狀態、檔案系統存影片」的混合方式，不會把大型 MP4 塞進資料庫：

- `data/state.sqlite3`：上傳與 Drive 匯入進度、job 狀態、人工標記、逐球片段索引、active 版本、集錦順序與 pCloud archive 狀態。
- `data/uploads/`：手機直傳或 Google Drive 匯入後的原片副本。
- `data/outputs/<job-id>/`：各來源的分析 JSON、逐球 MP4，以及重建產生的版本化 `clip-sets/`。
- `data/compilations/<compilation-id>/highlight_compilation.mp4`：跨來源選取、排序後的最後自訂集錦。

Docker 會把主機的 `./data` bind mount 到容器的 `/data`，因此重建 image 或 container 不會清掉資料；但持久化不等於備份，磁碟損壞或手動刪除仍會失去唯一副本。SQLite 與影片路徑彼此相依，備份或搬機時必須一起複製權威 runtime set；也不要直接從檔案總管刪除 active 影片。完整目錄、資料表關聯、保留策略與備份方式見 [docs/storage.md](docs/storage.md)。

### pCloud 長期影片 archive（第一階段可用）

可以串，而且目標流程改成：

```text
Pixel 手機 → Google Drive 送件箱 → 桌機本地處理/cache → pCloud 影片 archive
```

Google Drive 保留目前已經順手的手機入口；pCloud 負責長期保存原片、逐球影片和最後集錦。桌機仍在本機跑 FFmpeg/GPU。現在已可用明確的 CLI 將影片經 staging 上傳、比對 size + SHA-1、用 pCloud provider-side `copyfile noover=1` finalize、附上 manifest，並把 remote path/file ID/account/root/checksum 寫進 SQLite `storage_objects`。這一步不需要 GPU，也不會刪除本機或 Google Drive 檔案。

[pCloud Android App](https://help.pcloud.com/article/uploading-downloading-organizing) 本身也支援手動、Android 分享選單與 Automatic Upload，所以可保留成備援入口；目前不需要為此放棄既有 Google Drive 匯入流程。不要把正在使用的 SQLite 或 FFmpeg 工作目錄直接放在 pCloud Drive/WebDAV mount：雲端 mount 的鎖定、seek、延遲寫回與斷線語意都不適合 live database 和長影片處理。

不需要申請 pCloud API key 或自己建立 developer app。[rclone pCloud backend](https://rclone.org/pcloud/) 的 `client_id`／`client_secret` 一般留空，只要由帳號本人在瀏覽器完成一次 OAuth 授權。系統沿用這個 OAuth token 呼叫 pCloud 官方的 provider-side no-overwrite finalize；token 存在 Git 忽略的 `secrets/rclone/rclone.conf`，不要貼到 issue、commit、Docker image 或聊天訊息。常駐網站容器不會掛載這個檔案；只有手動啟動且沒有 port 的 `pcloud-admin` container 能讀取。

Git Bash 的一次性設定：

```bash
./scripts/setup-pcloud.sh
docker compose build pingpong-highlight
docker compose run --rm --no-deps pcloud-admin doctor
docker compose run --rm --no-deps pcloud-admin bootstrap --dry-run
docker compose run --rm --no-deps pcloud-admin bootstrap
```

PowerShell 的第一行改為 `./scripts/setup-pcloud.ps1`，後面的 Docker 指令相同；Windows Git Bash 的 `.sh` 會安全轉交給這支 PowerShell script。設定腳本會下載並驗證固定的 rclone 1.75.0 portable binary，自動建立名為 `highlightcraft-pcloud` 的 pCloud remote，再開啟瀏覽器完成一次 OAuth；不需要手動填 client ID/secret，而且完整 OAuth credential 不會輸出到終端。若任何舊版設定流程曾把 credential 顯示在終端，請先到 pCloud 網頁的 `Settings → Linked Accounts → Linked Apps` 移除 rclone，再刪除本機 `secrets/rclone/rclone.conf` 並重新執行設定腳本。

原生 Linux 要讓網站與 admin 使用同一個 host UID/GID，避免 SQLite 或影片變成另一個使用者擁有；在同一個 shell 先執行 `export PINGPONG_UID=$(id -u) PINGPONG_GID=$(id -g)`，再執行 `docker compose up` 或下列 pCloud 指令。Windows 不需要設定。

先只看重新命名與容量，不連 pCloud、不登記 job：

```bash
docker compose run --rm --no-deps pcloud-admin plan
```

第一次建議只傳一個較小的逐球片段驗證完整流程：

```bash
docker compose run --rm --no-deps pcloud-admin \
  archive --kind highlight --limit 1
docker compose run --rm --no-deps pcloud-admin \
  archive --kind highlight --limit 1 --execute
docker compose run --rm --no-deps pcloud-admin status
docker compose run --rm --no-deps pcloud-admin verify --limit 1
```

`archive` 沒有 `--execute` 時也只會預覽。`status` 只讀 SQLite；`verify` 才會重新讀取 pCloud 的影片與 manifest checksum。測試成功後才移除 `--limit 1` 或改成 `--kind original`。pCloud 會建立 `HighlightCraft/inbox/` 與 `HighlightCraft/archive-v1/{originals,highlight-clips,compilations,...}`；原始手機檔名保存在 manifest/SQLite，正式遠端名稱使用拍攝時間、種類、rank 與穩定 ID，避免重名和後續手動整理。

`HighlightCraft` 是預設 archive root；若要以 `PINGPONG_PCLOUD_ROOT` 改名，必須在第一次真正傳輸前決定。開始傳輸後，系統會固定 root、帳號與內容 checksum，避免同一份 catalog 被混寫到另一個位置；未來若要改 root 或升級 `archive-v2`，要先做明確的 catalog migration。

目前仍是 operator-run 第一階段，尚未接到常駐背景 worker，也還沒有 restore/hydration 或安全釋放本機空間。因此即使顯示 `verified`，現在也不要手動刪除 `data/uploads`、active clips 或 compilations；播放器與重跑仍使用本機檔案。完整命名、狀態與故障恢復規則見 [pCloud 長期保存方案](docs/storage.md#pcloud-長期保存方案第一階段已實作)。

## 快速選擇啟動方式

| 需求 | 建議唯一入口 | 誰能連入 |
|---|---|---|
| 只在這台電腦用 | `./scripts/start-localhost.sh` | 本機 |
| 同一個 Wi-Fi 的手機也要用 | `docker compose up -d --build` | LAN |
| 手機從外網使用 | `./scripts/start-ngrok-tunnel.sh` | 持有私人 token 網址的人 |
| 明確使用最後公開版 | 在對應啟動器加 `-UsePublishedImage` | 依所選模式；目前 1.3.0 不含新素材庫 |

這三個入口預設都從目前 source 使用 NVIDIA NVDEC/NVENC；只有 GPU runtime 確實不可用時才加 `-CpuOnly`。切換或重建前先等所有 upload、Drive import、來源分析與 compilation 完成。

## 只在這台電腦使用（localhost，不走 tunnel）

這是 ngrok 額度用完、公開網址失效，或只想在電腦上標記精彩球時的建議模式。若 ngrok 頁面顯示 `Network bandwidth exceeded`，不必繼續重試，直接使用本節。它不啟動 ngrok 或 Cloudflare Tunnel，網站只綁在 `127.0.0.1`，因此瀏覽器播放原片與成品不會消耗 tunnel 流量，也不會讓同一個 Wi-Fi 的其他裝置連入。

1. 開啟 Docker Desktop，等到 Docker Engine 正在執行。
2. 開啟 Git Bash，進入專案並從目前 source 建置、啟動 GPU 版本：

   ```bash
   cd /d/projects/pingpong-auto-highlight
   ./scripts/start-localhost.sh
   ```

   如果只想使用 Docker Hub 上已正式發佈、而且不需要目前 checkout 新功能的版本，才加上 `-UsePublishedImage`：

   ```bash
   ./scripts/start-localhost.sh -UsePublishedImage
   ```

   PowerShell 對應指令是：

   ```powershell
   .\scripts\start-localhost.ps1
   ```

   PowerShell 若要使用已發佈 image，對應指令是：

   ```powershell
   .\scripts\start-localhost.ps1 -UsePublishedImage
   ```

3. 等終端機顯示 `HighlightCraft localhost-only mode is ready.`，在**同一台電腦**的瀏覽器開啟它顯示的完整網址。之後也可以從 Git Bash 讀取：

   ```bash
   cat ./data/local-access-url.txt
   ```

   完整網址包含本機存取權杖，頁面第一次讀取後會把它從網址列移除。`data/local-access-url.txt` 只存在本機且不會進 Git，仍不應貼到公開場所。

啟動器會先檢查未完成上傳、Drive 下載與來源分析，才停止 ngrok／Cloudflare Tunnel 並切換服務；若還有這些工作，它會拒絕重啟。現行檢查尚未涵蓋 queued/processing compilation，因此切換前也要在頁面確認沒有集錦正在建立。原片、成品、處理紀錄及人工標記都留在 `./data`，切換模式本身不會刪除。

預設仍使用 NVIDIA GPU。只有 NVIDIA Docker runtime 暫時不可用時，才加上 `-CpuOnly`。想確認服務及 GPU：

```powershell
docker compose -f compose.yaml -f compose.localhost.yaml ps
docker compose -f compose.yaml -f compose.localhost.yaml exec pingpong-highlight pingpong-highlight doctor
```

只有用 `-UsePublishedImage` 啟動時，才在上述兩個命令中一併加入 `-f compose.release.yaml`。

`pingpong-highlight` 應顯示 `healthy`，`NVIDIA NVDEC` 與 `NVIDIA NVENC` 都應顯示 `可用`。停止 localhost 服務但保留所有資料：

```powershell
docker compose -f compose.yaml -f compose.localhost.yaml stop pingpong-highlight
```

`127.0.0.1` 只能由這台電腦開啟，手機即使連同一個 Wi-Fi 也不能使用這個網址；需要同一 Wi-Fi 的手機操作時，請用後面的 LAN 模式，需要手機外網時才使用 ngrok。localhost 模式不會產生 tunnel 流量，但第一次 pull Docker image、從 source build dependencies，或讓電腦從 Google Drive 匯入影片，仍會使用一般對外網路。

## 從手機外網使用（ngrok）

只有需要從手機外網操作時才走這條流程。預設使用 ngrok，因為它的主要 tunnel 連線走 TLS 443，不像 Cloudflare Tunnel 需要目前被學校網路封鎖的 7844 連接埠。第一次使用需要免費 ngrok 帳號與 authtoken。

1. 開啟 Docker Desktop，等到左下角顯示 Docker Engine 正在執行。
2. 開啟 Git Bash，進入專案目錄：

   ```bash
   cd /d/projects/pingpong-auto-highlight
   ```

   如果之後移動了專案資料夾，請把路徑換成新的位置。

3. 第一次使用時，先在 [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) 建立帳號並複製 authtoken。它相當於 ngrok 帳號密碼，請只貼進下一步的本機隱藏提示，不要貼到聊天、README 或公開場所。
4. 先確認既有頁面沒有 upload、Drive import、來源分析或 compilation 正在進行，再啟動剪輯服務與手機外網入口；啟動器可能更新並 recreate app container：

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

使用期間請讓電腦保持喚醒，並維持 Docker Desktop 與網路連線。任何 upload、Drive import、來源分析或 compilation 活動時都不要重啟 Docker、tunnel 或電腦；只有上傳／Drive 送入原片完成後，手機頁面可以關閉，電腦會繼續分析與剪輯。

### 用 Google Drive 加入影片（大檔案建議）

這條路徑通常比手機透過公開 tunnel 把大檔案直接傳到電腦更穩定，而且貼完連結後可以立刻關閉手機頁面：

1. 在手機的 Google Drive 上傳原始影片，等 Drive 顯示上傳完成。
2. 開啟該影片的「管理存取權」或「共用」，把一般存取權改成「知道連結的任何人」，角色選「檢視者」。只要允許下載，不需要給編輯權限。
3. 複製的是單一影片連結，不是資料夾連結。
4. 在 HighlightCraft 的 Google Drive 區塊貼上連結，按「開始匯入」。
5. 「Drive 下載中」會顯示電腦端實際收到的進度；完成後會自動消失並變成 GPU 剪輯工作，不需要再按一次開始。

系統端不需要 Google 帳號、OAuth、API key 或額外設定。公開連結等同任何拿到連結的人都能讀取，因此不要用在敏感影片。等 HighlightCraft 狀態已從 Drive 下載切換成「排隊中／分析中」後，就可以把 Google Drive 共用權限改回「受限制」；電腦已下載的本機副本不受影響。

Drive 匯入的狀態與暫存檔都在 `./data`。網路中斷或服務重啟時，會保留已下載部分並在下次啟動續傳；失敗項目也可以直接在頁面重試或刪除。Google 仍可能因下載次數、擁有者禁止下載或組織政策拒絕公開下載，這時頁面會保留進度並顯示權限提示。

### 精彩球素材庫與跨影片集錦（桌面）

桌面版會把所有來源的候選片段放在同一個素材庫。可用檔名搜尋，或依來源影片、拍攝日期、來源內相對分數與片段長度篩選排序。Pixel 類型的 `PXL_YYYYMMDD_HHMMSS...` 檔名會解析成拍攝日期；無法解析時才使用加入系統的日期。

來源篩選可以複選，因此可只勾兩場，再按「加入目前篩選的各來源前 6 球」一次取出 12 球；也可逐場篩選，既有勾選不會消失。右側工作台支援上下調整順序，並顯示估計總長；超過 60 秒仍可照常建立。送出前的勾選與順序只存在目前頁面的記憶體，重新整理會清除；送出後才寫入 SQLite。成品以第一個片段的解析度、比例與 FPS 作 canvas，其他比例會 letterbox，因此第一球與排序也會影響輸出規格。集錦、原片分析與 CLI 重建共用同一個跨程序 GPU media queue，避免同時搶顯存。

舊版工作只能匯入當年實際輸出的片段；被舊版 55 秒預算捨棄的候選並不存在。要用目前的寬鬆素材門檻安全重建某一支舊片，可執行：

```powershell
docker compose exec pingpong-highlight `
  pingpong-highlight rebuild-library <job-id> --data-dir /data
```

`rebuild-library` 只存在目前 source checkout，公開的 1.3.0 image 不支援。`<job-id>` 是 `/api/jobs` 回傳的內部工作 ID；目前 UI 尚未提供複製按鈕。新版輸出會放在該 job 的獨立 `clip-sets/` 子資料夾，全部成功且至少有一球符合門檻後才切換為素材庫的 active 版本。Ctrl+C、失敗或零合格素材都保留原 active 版本，但可能留下未啟用的 timestamp 目錄；反覆重建會增加磁碟用量，目前不會自動清理。舊片段、舊 Reel 與人工標記都不受影響。若網站正在分析或建立集錦，命令會等待共用 GPU 鎖，不會同時執行重型媒體工作。

### 人工標記精彩球（桌面開發工具）

手機使用的上傳、處理進度、成品播放與下載都集中在頁面上方。「人工標記精彩球」是另外一個桌面限定的開發工具區塊，只列出已完成的影片，不會出現在手機版，也不屬於一般成品操作流程。

在電腦版頁面下方選擇影片並按「開啟標記」，就會進入大播放器與右側標記清單；原片到這時才會以 HTTP Range 載入，因此可以直接拖曳長影片，不必先下載整支檔案。

1. 播放或拖曳原片，在該回合實際開始的位置按 `I` 設起點。
2. 到該分結束時按 `O` 設終點，再按 `Enter` 儲存；全程不需要抄寫或輸入時間碼。
3. 選「值得收錄」；如果它是系統自動選到但你不喜歡的球，改選「不該收錄」。精彩標籤可以複選，例如同時選「相持」與「搶攻」；只有不在常用選項裡的內容才需要點「其他…」手動輸入，也可以完全不選。
4. `Space` 控制播放／暫停，方向鍵前後 1 秒，`Shift` 加方向鍵前後 5 秒，`Esc` 關閉工作區。標記會綁定原片與時間碼，存在 `data/state.sqlite3`，重新整理、更新容器或重跑分析都不會消失。

第一輪建議選 2–3 支不同角度、距離或光線的影片，把其中所有你會放進集錦的球標完；另留 1 支完全不參與調整，最後才用來驗證是否真的改善。約 15–30 個「值得收錄」的球已可開始做錯誤分析與版本比較，但正樣本本身不足以訓練可靠分類器；未完整審閱的空白區段也不能自動當作負樣本。正式訓練前仍要補明確的「不該收錄」與每支來源的審閱完成狀態。

即使頁面重新整理或公開網址改變，電腦已收到的分塊也不會遺失。請回到持有原始影片的手機，重新選擇同一支影片；只要伺服器上剛好有一筆檔名與大小相同的未完成紀錄，系統就會從保存的 offset 續傳。其他裝置可同步查看進度，但無法代替來源手機提供原始檔案。若同一影片不小心留下多筆紀錄，頁面會先要求刪除重複項目，避免再建立第四筆；「刪除這筆上傳」只會刪除電腦上的未完成分塊，不會影響手機原片。

ngrok 額度用完或公開網址失效時，請等正在進行的上傳與剪輯完成，再切回有完整本機網址與安全檢查的 localhost-only 模式：

```bash
./scripts/start-localhost.sh
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

上傳原片、Drive 下載暫存、續傳資訊、工作狀態與成品都掛載在電腦的 `./data`，重新 build 或刪除容器不會遺失。`up -d --build` 仍可能 recreate 正在工作的 app；更新前先確認沒有 upload、Drive import、來源分析或 compilation 處於活動狀態，再執行：

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

根目錄的 Compose 檔案是「一個基底加幾個小型開關」，不是六套不同服務。平常只需要 `docker compose up -d --build`；啟動腳本會視需要自動疊加 localhost、CPU、Docker Hub release、ngrok 或 Cloudflare 的小型設定，不需要手動挑選。

常用管理指令：

```powershell
docker compose ps                    # 查看服務與健康狀態
docker compose logs -f pingpong-highlight
docker compose restart               # 重新啟動
docker compose down                  # 停止服務；保留 ./data
```

若希望登入 Windows 後一直可用，請同時開啟 Docker Desktop 的「Start Docker Desktop when you sign in」。

## 影片播放效能

`localhost` 只表示瀏覽器與服務在同一台電腦，不代表影片不需要經過 HTTP、Docker bind mount 與瀏覽器解碼。目前影片端點使用可感知斷線的 HTTP Range 串流與較大的讀取區塊；播放器收合或拖曳造成舊請求中斷時，伺服器會立即停止讀檔，不會在背景繼續把整支原片讀完。完成影片可保留在瀏覽器私有快取，重播與倒退也不必每次重新傳輸。

新產生的 H.264 成品最多使用 30 fps、約兩秒一個 GOP，並限制目標與峰值碼率。播放優化前已完成的 MP4 不會被自動覆寫；它們仍可透過目前的串流端點播放，但若希望把舊的 120 fps／高碼率檔案縮小，需要重新處理原片。

正式提供多人或外網使用時，不應讓免費 tunnel 兼任大量影片配送。建議讓 HighlightCraft 保留驗證、工作狀態與剪輯 API，把完成影片交給支援 Range／sendfile 的反向代理，或使用有權限控制的 object storage／CDN；這樣同時改善並行播放、流量成本與跨地區延遲。

## 在 4090／其他 NVIDIA 電腦使用已發佈 image

目前最後公開的 Docker Hub image 是 `docker.io/momonong/pingpong-auto-highlight:1.3.0`；它不包含這個 checkout 尚未另行發佈的新素材庫／集錦變更，需要這些功能時請從目前 source build。RTX 5090 Laptop 與 RTX 4090 Desktop 都使用同一個 `linux/amd64` image；image 不包含 NVIDIA driver，啟動時由主機的 NVIDIA Container Toolkit 提供 NVDEC／NVENC 所需元件，因此不要建立 `5090` 或 `4090` 專用 tag。

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

目前 checkout 的 package metadata 仍是 `1.3.0`，但 public `1.3.0` 不含新素材庫。**在同步 bump 下一個版本、建立對應 Git tag，並加上防止覆寫既有 version tag 的 publisher 檢查前，不要執行以下發佈器**；否則可能讓同一個 1.3.0 tag 指向不同內容。

完成上述 release hardening 後，也只有在變更已提交、合併到乾淨且與 remote 同步的 `main`，Docker Desktop 已登入 `momonong` 時才執行：

```bash
./scripts/publish-dockerhub.sh
```

發佈器固定產生 `linux/amd64`，使用 `pyproject.toml` 的版本建立 immutable version tag 與 `latest`，附加 SBOM／provenance，確認 repository 是 public，再從 Docker Hub pull 回來檢查版本與 NVIDIA NVDEC／NVENC。Python base image、production dependencies 與 build dependencies 都以 digest 或 hashes 固定；`data`、`.env`、影片及 token 由 `.dockerignore` 排除。

## ngrok Tunnel 細節

Git Bash 的標準啟動指令：

```bash
./scripts/start-ngrok-tunnel.sh
```

若手機第一次開啟時先看到 ngrok 的「Visit Site」頁面，按下後請再開一次終端顯示的完整網址（必須包含 `#token=...`）。第一次只是在該手機建立 ngrok 的瀏覽器 cookie，跳轉時可能沒有保留 fragment；HighlightCraft 收不到 token 時仍能顯示首頁，但會無法讀取伺服器端的處理 session。此時可重新開啟完整網址，或把完整網址／存取碼貼進頁面的解鎖欄位。session 與成品都還保存在電腦的 `data`，不會因手機換頁或重新整理而消失。

PowerShell 也可以使用同一套流程：

```powershell
.\scripts\start-ngrok-tunnel.ps1
```

啟動器會確認 Docker、啟動預設 GPU 服務與 ngrok、等待公開 health check 通過，再產生一條可直接在手機開啟的網址。ngrok 的 HTTP 請求檢視預設關閉，避免在本機 Traffic Inspector 保存影片要求與帶有 HighlightCraft 存取權杖的下載網址；4040 只綁在 `127.0.0.1`，啟動器用它讀取 tunnel 網址，不會對區域網路公開。

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

先確認頁面沒有 upload、Drive import、來源分析或 compilation 正在進行，再執行切換腳本；目前 Cloudflare／ngrok 啟動器不會完整阻止 active work 被 container recreate。腳本會啟動本機服務、保留健康的既有 tunnel、在 tunnel 失效時自動建立新的臨時 HTTPS 網址、確認公開 health check，最後顯示一條可直接在手機開啟、含有存取權杖的專用網址。不要把完整網址轉傳給別人。網址裡的 upload token 放在 `#` 後方，不會隨第一次 HTTP 請求送到 Cloudflare；頁面讀取後也會立刻從網址列移除。

目前 `compose.cloudflare.yaml` 沒有覆寫 base service 的 host binding，所以 Cloudflare 模式同時仍以 `${PINGPONG_PORT}` 開放在整個 LAN；它不是 tunnel-only 模式。ngrok 與 localhost overlay 則會明確綁 `127.0.0.1`。在 Cloudflare overlay 補上 loopback binding 前，請把這個 LAN exposure 納入防火牆與 token 管理。

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

需要 Python 3.11 以上、`uv`、`ffmpeg` 與 `ffprobe`。依 committed lockfile 建立開發環境：

```powershell
uv sync --frozen --extra dev
uv run --frozen pingpong-highlight doctor
uv run --frozen pingpong-highlight serve
```

終端機會顯示手機網址與 QR code。手機和電腦連到同一個區域網路後，用手機開啟網址並選擇影片。上傳採用可續傳分塊；中斷後重新選擇同一檔案即可接續。

上傳百分比以電腦實際保存的分塊為準，受 token 保護的網址可在手機或電腦跨裝置監看。重新整理不會遺失已傳資料，但瀏覽器基於安全限制不會自動恢復手機相簿裡的檔案；請在原來源裝置重新選擇同一支影片。系統會先使用該瀏覽器保存的工作階段；若網址已改變，則以完全相同的檔名與檔案大小接回伺服器上唯一的未完成紀錄。其他裝置能監看，無法代替來源裝置送出它沒有的原始檔案。

完整操作流程：

1. 電腦執行 `uv run --frozen pingpong-highlight serve`，保持終端機與電腦開啟。
2. 手機掃描 QR code，從相簿選擇原始影片，或貼上公開 Google Drive 影片連結。
3. 手機直傳需等上傳完成才能關頁；Drive 連結送出後即可關頁，電腦會在背景下載並處理。
4. 回到同一網址，在桌面素材庫篩選並勾選想要的精彩球。
5. 自由排序後建立集錦，再預覽或下載 MP4。

各來源的分析結果與逐球素材會保留在 `data/outputs/<job-id>/`，最後建立的自訂集錦則在 `data/compilations/<compilation-id>/highlight_compilation.mp4`。LAN 模式不需要雲端帳號或訂閱；Quick Tunnel 模式的傳輸會經過 Cloudflare，但分析、剪輯與持久儲存仍只在這台電腦進行。

也可以直接分析電腦上的影片：

```powershell
uv run --frozen pingpong-highlight analyze "D:\videos\match.mov"
```

這是 standalone 的手動分析：預設輸出到 `data/outputs/manual-*`，不會建立 upload/job，也不會把片段登錄到網頁素材庫。要讓既有網站來源以目前演算法重建可篩選的素材，請使用前述 `rebuild-library <job-id>`。

每次分析會輸出：

- `highlight_###_rank_###.mp4`：達素材門檻的各個候選片段；
- `analysis.json`：所有候選、素材與推薦門檻、切點、排名和媒體資訊；
- 自訂集錦另存於 `data/compilations/<compilation-id>/highlight_compilation.mp4`。

## 流程

```mermaid
flowchart LR
    A["手機影片"] -->|"可續傳分塊上傳"| B["電腦本地儲存"]
    GD["Google Drive 公開影片"] -->|"電腦背景續傳"| B
    B --> C["時間戳式音訊與畫面分析"]
    C --> P["逐分切點"]
    P --> E["70% 素材門檻與來源內排名"]
    E --> F["獨立精彩球素材庫"]
    F --> G["跨影片篩選、勾選與排序"]
    G --> H["任意長度自訂集錦"]
```

音訊瞬變提供擊球節奏，局部畫面動態協助排除缺乏比賽活動的雜音。分析以 FFmpeg 時間戳為準，可處理手機常見的 HEVC、VFR 與 rotation metadata。預設容器使用 NVIDIA NVDEC 解碼、NVENC 編碼；不相容的影片或 GPU runtime 失效時會自動退回 CPU／`libx264`。音訊分析、NumPy 訊號計算與部分 FFmpeg filters 仍會使用 CPU，因此 CPU 用量不會完全歸零。

## 主要設定

| 環境變數 | 預設值 | 用途 |
| --- | ---: | --- |
| `PINGPONG_DATA_DIR` | `./data` | SQLite、原片、素材與集錦共用的持久化根目錄 |
| `PINGPONG_HOST` | `0.0.0.0` | LAN 服務位址 |
| `PINGPONG_PORT` | `8000` | LAN 服務連接埠 |
| `PINGPONG_PUBLIC_URL` | 自動偵測 | QR code 與手機要開啟的公開基底網址；Docker 建議明確設定 |
| `PINGPONG_MAX_UPLOAD_BYTES` | 100 GiB | 單檔上限 |
| `PINGPONG_DOWNLOAD_MIN_FREE_BYTES` | 2 GiB | Drive 匯入時保留的最低可用空間 |
| `PINGPONG_VIDEO_SAMPLE_FPS` | 8 | 畫面分析取樣率 |
| `PINGPONG_LIBRARY_MIN_POINT_SCORE_RATIO` | 0.70 | 候選要實際存入素材庫的來源內相對分數下限 |
| `PINGPONG_MIN_POINT_SCORE_RATIO` | 0.87 | 在素材庫標成「推薦」的來源內相對分數下限；不是機率 |
| `PINGPONG_MAX_POINTS` | 0 | 選用的素材數安全上限；`0` 表示不限制 |
| `PINGPONG_REEL_TARGET_SECONDS` | 55 | 舊版相容設定；新版素材擷取與自訂集錦不套用此上限 |
| `PINGPONG_CLIP_PRE_ROLL_SECONDS` | 1.5 | 每球在實際回合前保留的秒數 |
| `PINGPONG_CLIP_POST_ROLL_SECONDS` | 1.5 | 每球在實際回合後保留的秒數 |

這張表同時包含 local Python 與 container 設定。Docker Compose 目前把容器內 `PINGPONG_DATA_DIR=/data`、`PINGPONG_HOST=0.0.0.0` 固定，主機儲存路徑則固定 bind mount `./data`；在 `.env` 改 `PINGPONG_DATA_DIR` 不會把影片搬到外接硬碟。`PINGPONG_MAX_UPLOAD_BYTES` 與 `PINGPONG_VIDEO_SAMPLE_FPS` 目前也沒有由 `compose.yaml` 轉送，只對直接執行 Python 或自行擴充 Compose environment 生效。其餘列在 `compose.yaml` 的變數可由 `.env` 覆寫。

候選先套 70% 素材保存門檻，再依來源內分數排名並各自輸出；不套 Reel 秒數預算，也不會為了達到最低球數而回填。87% 只影響「推薦」標記。舊的 `PINGPONG_MAX_HIGHLIGHTS` 仍可作為 `PINGPONG_MAX_POINTS` 的備援值。若既有 `.env` 寫了 `PINGPONG_MAX_POINTS=6`，它仍會成為明確安全上限；改成 `0` 才是不限制素材數。

相對門檻可適應不同球館、收音與鏡位造成的分數尺度差異，但來源第一名永遠是 100%，所以跨影片排序仍只是 heuristic。要可靠判斷「整支影片都不夠精彩」或建立真正可比較的全域精彩度，仍需補齊明確負向標記，再把分數換成校準過的模型機率。

## GPU 訓練環境（uv）

訓練套件放在獨立的 `train` dependency group，不會增加一般網站服務的安裝內容。Windows／Linux 使用已鎖定的 PyTorch CUDA 13.0 wheel：

```powershell
uv sync --frozen --group train
uv run --frozen --group train python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

需要同時執行測試時再加上開發套件：`uv sync --frozen --group train --extra dev`。這只準備並驗證 GPU 執行環境；正式訓練仍應等 candidate 資料與正／負標註檢查完成後，再由固定的 training entrypoint 啟動。

## 驗證

```powershell
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev pytest -q
```

延伸文件：

- [系統架構](docs/architecture.md)
- [儲存與資料生命週期](docs/storage.md)
- [評估方式與證據界線](docs/evaluation.md)
- [版本歷史](docs/history.md)
