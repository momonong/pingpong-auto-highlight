# Deployment, backup, and migration

HighlightCraft 不需要在實際剪片電腦上保留 Python、uv、原始碼或編譯工具。建議把「建置」與「執行」拆開：開發電腦或 CI 測試並發佈 image；有 GPU 與大容量磁碟的部署電腦只 pull 已驗證的 image，掛載自己的 `data`，再以 `.env` 注入該主機的設定。

## 三種內容的邊界

| 類型 | 放哪裡 | 能否進 Git / image | 換機時怎麼做 |
| --- | --- | --- | --- |
| Compose 部署 bundle | Git 或 release artifact | 可以 | 複製同一版本的 YAML 與 `.env.example` |
| `.env` 主機設定與 secret | 部署主機或 secret manager | 不可以 | 在目標主機重新建立，只手動帶入必要值 |
| `data` 狀態與媒體 | 部署主機的持久磁碟 | 不可以 | 停機後完整備份／還原，保留隱藏檔 |
| Docker image | container registry | 可以發佈 | 以固定 tag，最好是 digest pull |

最小部署 bundle 是 `compose.yaml`、`compose.deploy.yaml`、`.env.example`；只綁本機另帶 `compose.localhost.yaml`，CPU fallback 另帶 `compose.cpu.yaml`，需要 tunnel 才帶 `compose.ngrok.yaml` 或 `compose.cloudflare.yaml`。啟動腳本是 Windows 方便工具，不是 production 必需品。

## 在開發電腦或 CI 建置

1. checkout 要發佈的 commit，跑完整測試與真實媒體 smoke test。
2. 先推一次性的 candidate tag，取得 immutable digest，再以該 digest pull 回來檢查套件版本與 GPU；不要先公開正式 semantic version tag。
3. 先在 registry 啟用 semantic version tag immutability，再驗證該版本尚不存在，才把已驗證 digest promote 成 semantic version（及選擇性的 `latest`）。已發佈版本不可覆寫。
4. 記下 registry 回傳的 manifest digest，讓部署端鎖定同一份 bytes。

專案維護者可在 Windows PowerShell 使用既有發佈腳本：

```powershell
.\scripts\publish-dockerhub.ps1 -SkipLatest
```

一般 CI 可採同等流程：

```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=mode=max \
  --sbom=true \
  --tag registry.example.com/highlightcraft:candidate-1.4.0-abcdef123456 \
  --push .
docker buildx imagetools inspect registry.example.com/highlightcraft:candidate-1.4.0-abcdef123456
docker pull registry.example.com/highlightcraft@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
docker run --rm registry.example.com/highlightcraft@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  python -c "import pingpong_highlight; print(pingpong_highlight.__version__)"
docker buildx imagetools create \
  --tag registry.example.com/highlightcraft:1.4.0 \
  registry.example.com/highlightcraft@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

上面的 commit 與 digest 都是格式佔位，必須換成實際值；`1.4.0` 也只是命令格式範例，實際版本須與 release metadata 一致。CI 還應在 promote 前做與正式主機相符的 GPU smoke test，並以 registry API fail closed 檢查 version tag 不存在且 immutability 規則確實涵蓋該 semantic version。Docker Hub 可將 Specific tags 設為 immutable，純 `X.Y.Z` 版本使用 RE2 規則 `^[0-9]+\.[0-9]+\.[0-9]+$`；規則不應涵蓋 `latest` 或 candidate tag。部署端的 `PINGPONG_IMAGE` 建議寫成 `registry.example.com/highlightcraft@sha256:...`。不要用 `latest`，否則一次普通重啟就可能在未備份資料庫的情況下換版。

## 在執行電腦首次部署

主機只需 Docker Compose v2.30.0 以上、可寫入的持久磁碟，以及 GPU 模式所需的 NVIDIA driver / Container Toolkit。`gpus` 與 Compose overlay 使用較新的 Compose 語法；舊版 plugin 即使跑 CPU 模式也會在解析設定時失敗。先以 `docker compose version` 確認版本，再把部署 bundle 放進固定目錄：

```bash
cp .env.example .env
```

原生 Linux 不論使用預設 `./data` 或自訂絕對路徑，都必須先把目錄建立成 container 服務帳號 UID/GID `10001` 可寫。預設路徑可執行：

```bash
sudo install -d -m 0750 -o 10001 -g 10001 ./data
```

正式磁碟例如改成 `sudo install -d -m 0750 -o 10001 -g 10001 /srv/highlightcraft/data`，並同步設定 `PINGPONG_DATA_PATH`。Docker Desktop 的共享資料夾權限由 Desktop 管理，Windows/macOS 只需建立 `data` 資料夾，不要套用 Linux ownership 指令。

編輯 `.env`：

- `PINGPONG_IMAGE`：填入 CI 發佈並驗證過的固定 digest；
- `PINGPONG_DATA_PATH`：正式資料磁碟，例如 `/srv/highlightcraft/data`；
- `PINGPONG_PUBLIC_URL`、`PINGPONG_PORT`：依這台主機調整；
- `PINGPONG_BOOTSTRAP_ADMIN_USERNAME`：第一位管理員帳號；
- `PINGPONG_BOOTSTRAP_ADMIN_PASSWORD`：可留空讓系統產生，或只在第一次啟動前以 secret manager 注入至少 8 個字元的密碼；
- `PINGPONG_SESSION_TTL_SECONDS`：預設 `604800`（7 天）；
- `PINGPONG_SESSION_COOKIE_SECURE`：HTTPS 對外入口設 `true`，HTTP localhost/LAN 設 `false`。

GPU 啟動：

```bash
docker compose -f compose.yaml -f compose.deploy.yaml pull
docker compose -f compose.yaml -f compose.deploy.yaml up -d --wait --wait-timeout 180
docker compose -f compose.yaml -f compose.deploy.yaml exec pingpong-highlight pingpong-highlight doctor
```

沒有 NVIDIA runtime 時才疊加 CPU override：

```bash
docker compose -f compose.yaml -f compose.deploy.yaml -f compose.cpu.yaml up -d --wait --wait-timeout 180
```

空資料庫第一次啟動會建立 bootstrap admin。若 `.env` 沒有設定密碼，隨機密碼只會寫在資料目錄的 `.admin-password`；啟動訊息只提示路徑，不會把 secret 印到 log。原生 Linux 依前述 `10001:10001`、`0750` 權限建立資料夾時，以 `sudo cat /srv/highlightcraft/data/.admin-password` 讀取；Docker Desktop 則從主機共享的 data 資料夾讀取。登入後立即改密碼；驗證新密碼可用後，可刪除該一次性密碼檔。若 bootstrap 密碼曾放在 `.env`，清空後以 `docker compose ... up -d --force-recreate` 重建 container，才會一併從 container environment 移除。已存在使用者資料時，bootstrap 變數不會重設帳號或覆寫密碼。

`PINGPONG_UPLOAD_TOKEN` 預設不能登入任何 API。只有明確設定 `PINGPONG_ENABLE_LEGACY_TOKEN_AUTH=true` 時，舊 client 才能暫時存取 bootstrap 管理員名下的舊資料，而且不會取得管理員或其他使用者權限。新網頁使用帳號、密碼與 HttpOnly session cookie，分享網址時不要再加 `#token=...`。本機啟動器檢查工作狀態使用獨立的 `data/.maintenance-token`，它只允許讀取活動數量，不可用來存取影片或管理帳號。

## Tunnel 部署

ngrok 與 Cloudflare override 會把 session cookie 強制設為 `Secure`，因此使用者必須從 tunnel 的 HTTPS 網址登入。以 ngrok 為例：

`compose.ngrok.yaml` 會從 `PINGPONG_DATA_PATH/.ngrok-agent.yml` 掛入憑證。完整 repository 的 `start-ngrok-tunnel` 啟動器會用隱藏輸入建立它；精簡部署 bundle 則應由 secret manager 在目標主機建立該檔，權限限制為服務管理者可讀，不能把 authtoken 放進 bundle 或 image。

```bash
docker compose \
  -f compose.yaml \
  -f compose.deploy.yaml \
  -f compose.ngrok.yaml \
  up -d --wait --wait-timeout 180
```

Quick Tunnel 適合少量受邀測試，不適合長期公開、多使用者的大檔傳輸。正式外網服務應採固定網域、TLS reverse proxy、存取紀錄與有容量監控的儲存；`PINGPONG_PUBLIC_URL` 也應固定為該 HTTPS 網址。

## `.env` 與 secret 規範

- `.env` 已被 Git ignore；不要把它貼進 issue、聊天記錄、release 壓縮檔或 Docker build context。
- 不要把真實密碼寫進 `.env.example`。範例裡的空密碼代表由系統產生，不是允許空密碼登入。
- 若第一次啟動曾在 `.env` 寫入 bootstrap 密碼，建立帳號後就刪除該行的值並重建 container；之後以資料庫內的 password hash 為準。
- `data/.admin-password`、`data/.maintenance-token`、`data/.upload-token`、`data/.ngrok-authtoken` 都視為 secret。它們會隨完整資料備份移動，但不可單獨公開。
- `PINGPONG_SESSION_COOKIE_SECURE=true` 只有在瀏覽器到服務的實際入口是 HTTPS 時使用。純 HTTP 誤設為 `true` 會造成登入後看似立即登出。
- 不要跨環境共用 production `.env`。目標主機應從 `.env.example` 重建，再手動填 image digest、URL、磁碟路徑與 secret。

## 一致性備份

`data` 同時包含 SQLite、上傳續傳狀態、原片、剪輯成品、人工標記及本機產生的 secret。只複製 `state.sqlite3` 或只備份 `outputs` 都不能完整還原。

最可靠的方式是在沒有上傳／Drive 下載／剪輯工作時短暫停機，再完整複製資料目錄（包含點開頭的檔案）。備份含所有影片與登入 secret，請放在 repository 與 Docker build context 外；以下原生 Linux 範例使用只有 root 可讀的專用目錄：

```bash
(
  set -euo pipefail
  sudo install -d -m 0700 -o root -g root /srv/highlightcraft-backups
  sudo test ! -e /srv/highlightcraft-backups/highlightcraft-data-20260907-120000.tar.gz
  docker compose -f compose.yaml -f compose.deploy.yaml stop pingpong-highlight
  sudo sh -c 'umask 077; tar --create --gzip \
    --file /srv/highlightcraft-backups/highlightcraft-data-20260907-120000.tar.gz \
    --directory /srv/highlightcraft data'
  sudo tar --list \
    --file /srv/highlightcraft-backups/highlightcraft-data-20260907-120000.tar.gz \
    >/dev/null
  docker compose -f compose.yaml -f compose.deploy.yaml start pingpong-highlight
)
```

請把範例檔名中的時間改成實際備份時間。括號內的命令以 fail-closed 方式執行：任何停止、封存或驗證步驟失敗都不會繼續執行後續命令；若失敗後服務仍停著，先查明原因再人工重啟，不要把不完整壓縮檔當成可用備份。上例適用 `PINGPONG_DATA_PATH=/srv/highlightcraft/data`；若使用預設 `./data`，請把 `--directory /srv/highlightcraft data` 明確換成部署 bundle 的絕對父路徑與 `data`，不要把空白或未驗證的 shell 變數帶進 root 命令。Windows 可在停止服務後，用檔案總管、`tar.exe` 或既有備份軟體複製 `.env` 指定的整個資料夾，並把備份存到 repository 以外、存取受控的位置。

備份完成後至少檢查：壓縮檔可列出、含 `state.sqlite3`、`uploads/`、`outputs/` 和隱藏 secret；重要版本另做一次離機或異地備份。`.env` 應另外放進受控 secret store，不要與未加密的媒體壓縮檔一起散佈。

## 升級與 rollback

### 素材庫開發分支的資料相容性

`codex/preserve-local-20260907` 保存逐球素材庫、跨影片集錦、pCloud 封存與候選評估的開發成果；這些功能尚未整合到 1.4.0。若資料庫含有 `highlight_clips`、`compilations` 或 `storage_objects`，應先保留完整停機備份，並在另一份完整資料副本驗證升級。1.4.0 的重新處理會置換來源工作整個輸出目錄，刪除操作也沒有維護這些開發版索引；通過相容性驗證前，不要用它操作現役素材庫。

只要先試用帳號、上傳／Drive 匯入與自動 Reel，可將 `PINGPONG_DATA_PATH` 指向獨立空目錄，使用另一個 Compose project name 與未佔用的 localhost port。新目錄的管理員帳號與資料獨立於既有服務。pCloud 封存紀錄或只保留 Git 分支都不能代替完整 `data` 備份。

### 已支援資料版本的升級

1. 確認目前沒有 active upload/import/job，記錄現有 `PINGPONG_IMAGE` digest。
2. 依上一節做完整停機備份。
3. 把 `.env` 的 `PINGPONG_IMAGE` 換成已驗證的新 digest。
4. 執行 `pull` 與 `up -d --wait`，再檢查 `/api/health`、登入、使用者清單、舊原片與舊成品。
5. 若啟動或資料遷移失敗，依下方 rollback 流程同時還原舊 image digest 與升級前的完整 `data`。不要只把 image 降版後繼續使用已升級的資料庫。

原生 Linux rollback 範例（路徑與備份檔名必須先人工確認）：

```bash
(
  set -euo pipefail
  sudo test -f /srv/highlightcraft-backups/highlightcraft-data-20260907-120000.tar.gz
  sudo test -d /srv/highlightcraft/data
  if sudo test -e /srv/highlightcraft/data.failed-20260907-121500; then
    echo '拒絕覆寫既有 data.failed 目錄；請先人工確認。' >&2
    exit 1
  fi
  sudo tar --list \
    --file /srv/highlightcraft-backups/highlightcraft-data-20260907-120000.tar.gz \
    >/dev/null
  docker compose -f compose.yaml -f compose.deploy.yaml stop pingpong-highlight
  sudo mv /srv/highlightcraft/data /srv/highlightcraft/data.failed-20260907-121500
  sudo tar --extract --gzip \
    --file /srv/highlightcraft-backups/highlightcraft-data-20260907-120000.tar.gz \
    --directory /srv/highlightcraft
  sudo test -f /srv/highlightcraft/data/state.sqlite3
  sudo chown -R 10001:10001 /srv/highlightcraft/data
)
```

接著把 `.env` 的 `PINGPONG_IMAGE` 改回升級前記錄的 immutable digest，人工確認無誤後才執行：

```bash
(
  set -euo pipefail
  docker compose -f compose.yaml -f compose.deploy.yaml pull
  docker compose -f compose.yaml -f compose.deploy.yaml up -d --wait --wait-timeout 180
  docker compose -f compose.yaml -f compose.deploy.yaml images
)
```

若 `up` 超時，先保留失敗現場並查看 `docker compose -f compose.yaml -f compose.deploy.yaml logs --tail 200 pingpong-highlight`，不要用 `start`；`start` 只會啟動既有 container，不會套用剛改回的 image reference。驗收舊版與資料後，暫時保留 `data.failed-*`，確認不再需要才由維運者另行清除。

SQLite schema migration 由應用程式啟動時執行，因此升級前備份是必要步驟。production 不應直接對同一份 `data` 跑兩個 app container。

從沒有帳號欄位的舊版第一次升級時，既有未標 owner 的上傳與 Drive 匯入會歸到 bootstrap 管理員，避免升級後變成任何一般使用者都看不到的孤兒資料。先以管理員抽查舊影片完整，再開始建立試用帳號；系統不會僅依檔名猜測舊影片屬於哪一位新使用者。

## 搬到另一台電腦

1. 停止來源服務並建立、驗證完整資料備份。
2. 在目標主機安裝 Docker / GPU runtime，放入與目標 release 相符的部署 bundle。
3. 從 `.env.example` 新建目標 `.env`。手動設定新主機的 data path、port、public URL、HTTPS cookie 和 image digest；不要整包搬開發機 `.env`。
4. 將備份還原到 `PINGPONG_DATA_PATH`，確認 Docker process 可讀寫。使用者、密碼 hash、影片歸屬、處理紀錄與成品都由這份資料保留。
5. `docker compose ... up -d --wait`，以管理員登入並抽查至少一支原片與成品。
6. 驗收完成前保持來源資料與備份不動；確認 DNS/tunnel 已切換後才退役舊服務，避免兩台同時寫同一份狀態。

如此一來，日常開發可以完全在其他電腦或 CI 完成；這台剪片主機只承擔 pull、執行、資料保存與 GPU 處理。
