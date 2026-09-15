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

回合預標註第一版請先使用 `python -m pingpong_highlight.preannotate serve --store <experiment>/review.sqlite3 --port <unused-port>` 的獨立 loopback 服務。它不建立網站正式 state store，也不執行素材庫 migration。主網站的新審核紀錄另存於 `data/rally-review/`；備份時須包含該目錄。這不是現役素材庫升級已通過的宣告。

整片人工審核的「準備整片播放」需要服務程序的 PATH 可找到 FFmpeg/ffprobe；不需要 VLM 套件。播放副本與 manifest 位於對應 review.sqlite3 同目錄的 `review-media/`，只供瀏覽器相容播放，來源檔不變。主網站端點沿用 job owner/admin 權限，實驗服務只綁定 loopback。重新啟動同一 URL 後，使用者重新整理可恢復本機未保存草稿；已保存人工紀錄仍在原審核 DB。

模型是選用依賴，正式網站 image 不必包含。Windows 模型固定存入 `D:\hf\_models`；`model_cache.py` 在 Hugging Face import 前設定本程序 HF_HOME、Hub、Xet、assets、modules、Torch、編譯快取及 TEMP/TMP，offload 目錄也在此根目錄。模型載入只允許已存在的固定 snapshot、CUDA BF16、單一模型／單推論 worker，沒有 D 槽失敗後改存 C 槽的路徑。不要複製權重進 worktree/image，不改全系統環境。權重可跨工作目錄共用，但下載或推論前仍需確認既有工作與 VRAM；不終止其他 GPU 工作。

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

## Linux 隔離驗收

Docker Desktop 的 GPU 支援限定 Windows WSL2（[Docker 官方說明](https://docs.docker.com/desktop/features/gpu/)）；Linux NVIDIA 服務使用原生 Docker Engine 與 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。先唯讀執行 `docker context ls`、`docker --context default info`、`docker --context default ps`、`nvidia-smi`，核對 socket、已登錄的 `nvidia` runtime 與現役服務。不要為了測試執行 `docker context use`、重新設定 daemon 或重啟 Docker。原生 Engine 若已可用，可與 Desktop 分開建立驗收 project；不同 Engine 仍共用主機 port、磁碟及 GPU。

`compose.acceptance.yaml` 限定 loopback，要求明確指定 image、commit 與獨立資料路徑。以下必須在驗收 worktree 執行，且每次新驗收選新目錄與 project；現役 bind mount 不可重用：

```bash
export PINGPONG_ACCEPTANCE_IMAGE=highlightcraft:acceptance-213fc18
export PINGPONG_ACCEPTANCE_REVISION=213fc1895a50ce2bae131e52ec2159b6e2e8a885
export PINGPONG_ACCEPTANCE_PORT=18082
export PINGPONG_ACCEPTANCE_DATA="$PWD/data/host-acceptance"
# 確认 port 未占用、目錄尚不存在，才建立；此步不可對現役資料執行。
sudo install -d -m 0750 -o 10001 -g 10001 "$PINGPONG_ACCEPTANCE_DATA"
docker --context default compose -p highlightcraft-annotation-acceptance \
  -f compose.yaml -f compose.acceptance.yaml config -q
docker --context default compose -p highlightcraft-annotation-acceptance \
  -f compose.yaml -f compose.acceptance.yaml build
docker --context default compose -p highlightcraft-annotation-acceptance \
  -f compose.yaml -f compose.acceptance.yaml up -d --no-build --wait
docker --context default compose -p highlightcraft-annotation-acceptance \
  -f compose.yaml -f compose.acceptance.yaml exec -T pingpong-highlight pingpong-highlight doctor
```

套件版本仍可能顯示 `1.4.0`；必須同時記錄 OCI revision 與本機 image ID。`v1.4.0` tag 的 `321da1e` 不含 PR #5 的整片人工標註，不可用版本字串代替 commit 驗收。只有 FFmpeg 實際 decode/encode 通過才算 GPU 可用，不能僅看 encoder 清單。

本機 dev 環境使用 `uv sync --locked --extra dev --no-default-groups`，不啟用 `vlm` 或 `train`。完整 pytest、`tests/browser/full-review.cjs`、`rally-review.cjs`、`workspace.cjs` 均須在部署主機重跑；Windows 的 pass/skip 為歷史證據。

`tests/browser/deployment-review.cjs` 可對隔離 Docker 入口測試真實 HTTP 上傳／正式 processor、登入、owner 隔離、整片 H.264 Range、I/O、保存續播、coverage、草稿重載與匯出。設定 `HC_ACCEPTANCE_URL`、`HC_ACCEPTANCE_PASSWORD_FILE`（該隔離 instance 的 admin 密碼檔）及全新 `HC_ACCEPTANCE_ARTIFACTS` 目錄，並提供 Playwright/Chromium。測試會新增兩個明示 AUTOMATION FIXTURE 的帳號及生成影片，只可用於獨立驗收資料；它不刪資料，不代表真人真值。測試帳號密碼沿用該隔離測試密碼，這份 data 不可直接升為正式服務。

真實原片需另外核對 SHA-256、時長、首中尾實際影像，再於獨立副本驗證標註操作。自動化寫入一律保留 fixture 標示；正式人工資料不得混入這些紀錄。開發機獨立 `review.sqlite3` 內含 Windows 絕對來源路徑，不能直接當 Linux 已可用的資料：先取得一致性備份，在副本重設路徑並核對 source hash、模型 runs 及 review export；原備份保持不動。`review-media/preview.json` 也可能包含舊絕對路徑，需驗證或重新準備相容播放副本，不能只搬 MP4。

2026-09-12 此主機的實際映像、資料目錄、測試結果與尚缺原片詳見 [部署主機驗收紀錄](evaluation.md#部署主機人工標註驗收2026-09-12linux-rtx-4090)。該輪重用既有依賴層的映像與上面的完整建置命令不同，以 image receipt 為準。

## 正式部署整合候選（2026-09-13）

**本機整合已驗證；尚未切换正式服務或公開入口。** 實際證據見
[evaluation](evaluation.md#部署整合驗收2026-09-13) 與
[evidence](evidence/deployment-integration-20260913/)。以下命令中 `CHANGEME`／`REPLACE` 必須
用已審查的 hostname、來源與目的資料替換；未解決資料來源及入口歸屬前不能執行正式切換。

### 可交付的本機產物

整合 worktree：`/home/ubuntu/projects/pingpong-auto-highlight/.worktrees/deployment-integration-20260913`；
分支 `codex/deployment-integration-20260913`。来源未提交的 44 個檔案完整保存在
`0315009`，逐檔 SHA 與原始 tar hash 見 `source-snapshot.json`；來源 worktree 與 main 未切換。

候選 runtime commit `b367685`，本機映像 `highlightcraft:deployment-20260913`，image ID
`sha256:160214f083a28828d089402b1e7f534009c8f845261cdbc98bd8c57179574e3f`。
使用 `deploy/Dockerfile.candidate` 在 `--network=none` 下重建 wheel，重用已核對的
acceptance builder／runtime 層。45 個 package 檔案與 31 個鎖定依賴一致；
**沒有完整冷建置、SBOM 或 registry manifest digest**，不是正式已發布產物。
最終 bundle receipt 另記 deployment commit、檔案 hash 與 `docker save` tar hash。
任何發布及推送須另行授權；不可拿本機 image ID 冒充 registry digest。

新的 `compose.production.yaml` 為獨立配置，不疊原開發/tunnel 啟動器；
`pull_policy: never` 避免重啟偷偷換版。先明確 `docker load` 本機 tar，或在發布授權後
明確 pull 固定 registry digest。`HC_IMAGE` 可直接使用 image ID；套件字串 1.4.0
不能辨識這次新功能。資料 bind mount 禁止自動建立空目錄。

### 入口選擇與現況

目前 hostname、DNS 管理商、Cloudflare 帳號設定、路由器 WAN IP／CGNAT、TLS 憑證仍待確認。
此主機 Nginx 1.24.0 在 80 port；可讀 active include 只有預設 server，沒有 hostname/TLS。
LAN 為 192.168.68.62，default gateway 為 192.168.68.1；單憑私有 LAN IP 不能判定 CGNAT。
沒有 reload 系統 Nginx，也沒有改 DNS、port forwarding 或建立公開 tunnel。

已確認的入口決策（2026-09-15）：整個 HighlightCraft 由自有 Nginx 提供 HTTPS 與
子路徑分流，保留 `https://你的網域/pingpong-highlight/`；網頁、登入、標註 API、
上傳、播放與下載使用同一 origin。Cloudflare 僅管理該 hostname 的 DNS，設為
DNS-only／灰雲，瀏覽器直接連公開 Nginx，不經 Cloudflare HTTP 代理或 public Tunnel。
不新增跨 origin 媒體授權／CORS 分流。其他獨立 hostname 的服務仍可另用 Cloudflare 代理。

先沿用已有可靠公開 Nginx；若主機可取得公網 IP，核對 NAT／port forwarding 後再配置。
若有 CGNAT 或無法提供直接入口，評估公開 VPS 的 Nginx 經私人通道連回本機，先核對
供應商流量額度、影片使用規範與維運責任。Nginx 本身不會解決 NAT，外網延遲與頻寬
仍須實測。以下 tunnel 比較保留為備案調查，不代表已選用或已驗證正式入口。

DNS-only／代理設定作用於整個 hostname，不能按 URL path 分流。修改 DNS 或關閉代理前，
必須確認同 hostname 的所有既有網站、TLS 及防護需求均已妥善承接；不能僅為新增子路徑
覆蓋整個 hostname。參見 [DNS 代理模式](https://developers.cloudflare.com/dns/proxy-status/)。

Cloudflare 並非只能有一條免費 tunnel：官方目前預設每帳號 1,000 條；
named／managed tunnel 可綁自有 hostname，保留 tunnel UUID／credential 後重啟沿用。
Quick Tunnel 產生隨機 `trycloudflare.com` 網址，僅供測試。
[帳號上限](https://developers.cloudflare.com/cloudflare-one/account-limits/)、
[固定入口設定](https://developers.cloudflare.com/tunnel/get-started/)、
[影片政策](https://developers.cloudflare.com/fundamentals/reference/policies-compliances/delivering-videos-with-cloudflare/)。

2026-09-13 實測 `region1.v2.argotunnel.com:7844`、`region2.v2.argotunnel.com:7844`
TCP 可連，`connect.ngrok-agent.com:443` 亦可連。這只驗證出站 TCP，沒有驗證 UDP/QUIC、
帳號登入、實際 tunnel、行動網路連入、長時間穩定性或公開 TLS。舊 README 所述學校封鎖
7844 是歷史狀態，不適用於此次測量。

Cloudflare 官方文件目前列 Free/Pro 單次 request body 100 MB（zone 可設更小），
本機前端 8 MiB／後端上限 32 MiB 在此大小內，但慢速連線仍可能逾時。
Connection limits 列 Proxy Read 125 秒、Proxy Write 30 秒；本服務準備整片播放的同步
POST 在長片可能超出，因此本機 Nginx 900 秒不能消除上游限制，必須在真實方案重測，
必要時另排非同步播放準備工作，不能僅憑分塊通過就宣布整條鏈路通過。
[413 限制](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/)、
[連線限制](https://developers.cloudflare.com/fundamentals/reference/connection-limits/)。

ngrok Free 目前提供一個固定分配的 dev domain，非自訂網域，且有低流量額度；
你記得「只能一個」可能來自此方案。需要自有 hostname 或影片流量時應先核對帳號方案，
本輪未購買或建立 ngrok 入口。[官方方案](https://ngrok.com/docs/pricing-limits)。

### Nginx 與信任契約

使用 `deploy/nginx/highlightcraft.location.conf`，加入**現有 server**，保留其他 location。
DNS 只能路由 hostname，不能路由 URL path。若把現有 hostname 指向 tunnel，會改變
整個 hostname 的入口，必須先確保 Nginx 已保留所有既有網站；不得只為這個子路徑覆蓋 DNS。

- 直接 TLS 終止於 Nginx：設定 `$hc_external_scheme=$scheme`、`$hc_client_ip=$remote_addr`，
  參照 `tls-server.example.conf`；使用真實有效憑證，不使用本次自簽測試憑證。
- TLS 終止於核准的 Cloudflare edge：專用 `127.0.0.1:18087` listener 只給本機
  cloudflared，scheme 固定 https；只信任該直接 caller 的 `CF-Connecting-IP`。
  參照 `tunnel-server.example.conf` 及 `deploy/tunnel/cloudflared.example.yaml`。
  若 connector 在另一容器/主機，需改為實際私網 IP 並限制防火牆，不能照抄 loopback。
- Nginx 到 app 綁 `127.0.0.1:18084`；容器看到的來源可能是 Docker bridge gateway。
  必須 inspect 對應 network，並實測 redirect 確認 scheme；本次 bridge 驗收看到 `172.17.0.1`，
  **不是所有 production Compose network 都用這個地址**。`HC_PROXY_IPS` 指定實際 IP，禁止 wildcard。
- `HC_ALLOWED_HOSTS=正式hostname,127.0.0.1` 保留 healthcheck；不同 instance 使用獨立 cookie 名稱。
  `HC_ROOT_PATH=/pingpong-highlight`，`HC_PUBLIC_URL=https://正式hostname/pingpong-highlight`，
  `HC_COOKIE_SECURE=true`。根路徑則明確設 `HC_ROOT_PATH=`、對應根 public URL。
- `client_max_body_size 32m` 對應 app 最大 PATCH；前端 8 MiB。request/response buffering off，
  cache off/no-store，write 不做 upstream retry。proxy timeout 900 秒是本機等待上限，
  不是所有外部 tunnel 的端到端保證。Cloudflare 等外層必須對此 path 設 cache bypass，禁止 Cache Everything。

### 資料與備份界線

本輪備份均在整合 worktree 的 `data/`，權限 0700/0600；含帳號 hash、session、token、
原片與成品，**不可加入 Git、公開 bundle 或寄到公開服務**。同磁碟封存不等於離機災難備援。

| 來源 | 本機封存／還原 | 已驗證／缺口 |
| --- | --- | --- |
| Desktop 18080，主 checkout `/data` | `18080-online-20260913.tar` → `restore-18080-20260913/data` | 11 檔，SQLite ok，1 舊工作，6 原片；3 development SHA 與 manifest 相符。排除獨立 e2e 與本輪 snapshot 目錄 |
| native 18082，來源 worktree `data/host-acceptance` | `18082-online-20260913.tar` → `restore-18082-20260913/data` | 31 檔、5 DB ok、7 users、4 jobs；真實 job 已註冊；含既有 fixture，不能直接正式化 |
| Desktop 18081，主 checkout `data/e2e-v140` | 維持原資料，保護狀態已核對 | 非正式資料候選，不合併 |
| container configuration／必要 secret | `secure-config-backup/` | 容器 inspect 含環境值，另存 acceptance env/password 及存在的主 `.env`；不隨 Git 分享 |
| Windows 正式 state／舊 review.sqlite3／runs | 未提供 | 仍 UNKNOWN；不得把原片存在當 DB 遷移完成 |

備份使用 SQLite backup API，不直接複製 WAL 主檔；每份 DB 完整性、所有檔案 SHA、
來源／成品／播放 manifest 關聯都在還原檢查。線上副本會偵測備份期間檔案或 DB 變動並拒絕，
但仍標為 `online-rehearsal`，不是最終切換時間的一致性備份。來源 worktree 的 env/password
不在 app data 內，故另行保護。未知 schema（含 preserve-local 的 highlight_clips、compilations、
compilation_items、storage_objects）或 Windows 絕對路徑會阻擋還原驗收；不直接混用或整支合併。

`data-selection.json` 只列審查提案，未刪任何帳號或資料。正式來源必須先確認是否以 18082
真實 job 的 owner 資料為基礎、如何保留其新增真人標註、是否還有 Windows 正式資料。
不要以 fixture 的 uncertain 紀錄、帳號命名或交接時 0 筆推導目前真人真值。

草稿在瀏覽器 localStorage，**服務端 data 備份不含瀏覽器草稿**。更換 origin／prefix 前，
請在原頁面保存回合或保留草稿匯出；未確認前保留原 18082 入口與原瀏覽器 profile。
不同 origin 的 localStorage、登入 cookie、未完成上傳本機索引不會自動搬移。上傳可由同一原檔
向伺服器重新發現 offset；草稿需要按 job/source 核對後搬入，不能用同名影片猜測。

### 可執行切換流程（須先取得正式停寫與入口變更授權）

先完成來源選擇及 fixture 排除的獨立審查；本輪沒有授權刪除或自動合併正式資料。
下列 `HC_SOURCE_DATA` 是**核准的單一來源**，不可能同時填 18080 與 18082；若需整合兩份資料，
必須先在新副本按 user/upload/source SHA 保留 owner 與 review，再驗收該整合副本。
切換前不得忽略 18082 後續新增真人資料。保留本輪 tar 只是演練依據，不能取代當天最終備份。

1. 準備已審查 bundle、候選 tar、目的磁碟與 secret；記錄當天實際 image ID、container ID、
   context、mount、Nginx 生效設定 hash、目前 DNS／tunnel 設定與原指向。先跑 `nginx -t`
   驗證離線配置，確認其他網站與 location 仍存在。
2. 用 admin activity／maintenance API 確認 queued/processing jobs、uploads、Drive imports 為零；
   另外確認沒人在準備播放副本或標註。通知使用者保存草稿並關閉寫入頁面。
   「沒有 active job」不能防止下一秒有人寫入，因此最終一致性必須停止所有該 data 的 writer。
3. **此點需要使用者明確授權**：只停止核准的 app container（不要停止 Docker Desktop、其他網站或 daemon）。
   例如若正式來源核准為目前 18082，停止對象是
   `docker --context default stop --time 120 highlightcraft-annotation-acceptance-pingpong-highlight-1`。
   若來源另選 18080，必須替換成它的 context/container，不可照抄。
4. 重查 `docker ps`、bind mounts 與主機原生 app process，確定來源無 writer。
   使用以下停寫備份命令，檔名每次新建，避免 shell 覆寫舊備份：

```bash
# 在已解開的候選 bundle 執行。先填入核准值；以下不會自動停機或改 DNS。
set -euo pipefail
HC_SOURCE_DATA=/REPLACE_WITH_APPROVED_SOURCE_DATA
HC_BACKUP=/REPLACE_WITH_BACKUP_DISK/highlightcraft-final-YYYYMMDD-HHMMSS.tar
HC_RESTORE=/REPLACE_WITH_NEW_RELEASE_DIRECTORY
HC_LOCAL_IMAGE=sha256:160214f083a28828d089402b1e7f534009c8f845261cdbc98bd8c57179574e3f
umask 077
set -o noclobber
# helper 不啟動 app；掛載來源唯讀，SQLite 的 WAL/journal 在 /tmp 副本還原。
docker --context default run --rm -i --user 0 --network none --entrypoint python \
  --mount "type=bind,source=$HC_SOURCE_DATA,target=/data,readonly" \
  "$HC_LOCAL_IMAGE" - snapshot --source /data --consistency stopped-writers \
  < scripts/deployment-data.py > "$HC_BACKUP"
# 任一步 exit 非 0 都停止；缺 manifest 或驗證不符的 tar 不可部署。
python3 scripts/deployment-data.py restore --archive "$HC_BACKUP" --destination "$HC_RESTORE"
python3 scripts/deployment-preflight.py --data "$HC_RESTORE/data"
sha256sum "$HC_BACKUP"
```

若來源是主 checkout data，必須依已審查 inventory 加
`--exclude e2e-v140 --exclude deployment-integration-20260913`，避免把其他 instance 及備份本身
納入；其餘新出現目錄先核對用途，不任意排除。將最終 tar 複製到核准的**另一磁碟或離機位置**，
再次核對 SHA；目的地目前未提供，因此本輪未完成離機備份。

5. 還原成功後保留該還原基準不動，複製到新的部署 data（例如 release 的 `app-data`），
   不共用 inode/hardlink；可用 reflink 複製。先 `deployment-preflight.py --data ...`，
   再將副本設為 UID/GID 10001，secret 維持 0600；不得 chown 來源或舊備份。
   編輯 `production.env`：`HC_DATA` 指向新的 `app-data`，其餘值依上節確認。
6. 確认 18084 未占用、候選 image ID 正確，使用固定 project：

```bash
docker --context default image inspect "$HC_LOCAL_IMAGE" --format '{{.Id}}'
docker --context default compose -p highlightcraft-production \
  --env-file production.env -f compose.production.yaml config -q
docker --context default compose -p highlightcraft-production \
  --env-file production.env -f compose.production.yaml up -d --wait --wait-timeout 180
```

7. 使用已準備的獨立 Nginx 入口先 smoke：登入、實際既有 job/source/owner 審核可讀、Range、下載。
   正式 data 不跑會新增 fixture 的 `tests/browser/*`；寫入驗收只能另用指定試用帳號與明確可刪測試資料，
   或由真人在正確 job 接續操作。保留前後 review export 與草稿狀態。
8. **入口修改須已有明確授權**：備份系統 Nginx 設定後，只加入已審查 location，`nginx -t`
   成功才 reload；若 hostname 已有公開入口，DNS 不變。若需 tunnel/DNS，逐項套用已核准差異，
   不能用範例 catch-all 404 取代既有網站。
9. 從不同外網（手機行動網路）驗證有效 TLS、子路徑登入、長片 TUS 續傳、播放拖曳、下載、
   慢速連線、tunnel 斷線重連及其他網站正常；記錄實際使用方案與流量。
   未完成這一步只能報告本機通過。確認完成後才讓真人接續標註；舊 data/image/設定保留，
   是否退役 18080／18081 另行决定，不因此次新入口自動停止。

### Rollback

任一資料 hash／關聯、owner 權限、登入、Range 或現有網站回歸失敗，先停止開放新寫入。
不要重複覆寫同一 data 來試錯。

- **尚未開放正式寫入**：停止 `highlightcraft-production` 的 app；回復原 Nginx location／
  tunnel route／DNS 記錄，`nginx -t` 成功才 reload；重新啟動先前被核准停止的單一來源 app。
  原 image 與來源 data 未改動，可直接回復。DNS 若有變更須考慮 cache/TTL，保留回復觀察窗口。
- **已產生新真人標註／uploads**：先保留失敗部署完整資料與其設定，停寫後再取得一份新 tar；
  不把新資料丟掉，也不直接用舊 image 開啟已 migration 的 DB。
  從 pre-cutover tar 還原到**另一全新目錄**，重跑 integrity/hash/關聯檢查，
  用原 image ID 啟動一個唯一 writer，再回復入口。新舊期間增加的標註與原片保留待核對合併，
  對使用者明說哪些新寫入暫時不在回復後網站內。
- 回復後驗證登入、原有 job 數量、owner review export、播放、下載及其他網站；
  同一份 data 絕不可同時讓新舊 app 寫入。保留失敗副本、tar、image 和紀錄，不做自動清理。

```bash
# 只有在 rollback 已授權時執行；不停止整個 Docker context。
docker --context default compose -p highlightcraft-production \
  --env-file production.env -f compose.production.yaml stop app
# 還原至新的目錄；之後建立 rollback.env，使用原 image ID 與新的回復 data。
python3 scripts/deployment-data.py restore --archive "$HC_BACKUP" \
  --destination /REPLACE_WITH_NEW_ROLLBACK_DIRECTORY
python3 scripts/deployment-preflight.py --data /REPLACE_WITH_NEW_ROLLBACK_DIRECTORY/data
# 原來源可重啟的前提：其 data 始終保留且未被新版 app 寫入。
```

### 本機隔離驗證的重跑入口

本輪使用 18084 app → 18085 HTTP／18443 HTTPS Nginx；根模式 18086，舊 DB migration 18088。
均為原生 Docker `default`、獨立資料與 `hc-integration-*` project；TLS 為自簽測試憑證。
系統 Nginx 的 80、正式／真人 18080–18082 及停止中的 18083 均未切換。
每次重跑先重新查 port/process，使用新副本和新的 artifacts 目錄，不對真人入口執行。
`proxy-deployment.cjs` 的入口 guard 只接受 18443 的指定子路徑或 18086 根入口；
`proxy-real-playback.cjs` 只讀副本，`proxy-upload-limits.cjs` 只清自己的具名上限測試 fixture。
Playwright 必須已安裝；本輪使用主機既有套件，沒有在 repo 加前端依賴。

驗證結束時已停止本輪三個 `hc-integration-*` 容器與自訂 Nginx，保留映像、資料、副本及所有證據；
18080／18081／18082 仍運行且健康。`data/nginx/nginx.conf`、`data/*.env` 為本機演練設定，
不套用至現役系統 Nginx。部署 bundle 的產生方式是
`python3 scripts/package-deployment.py --context default --image highlightcraft:deployment-20260913 --output data/deployment-bundle-20260913`。
