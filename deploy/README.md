# HighlightCraft 本機部署候選

這是可審查的本機候選，沒有正式發布、DNS 或服務切換授權。
先閱讀 `docs/deployment.md` 的「正式部署整合候選（2026-09-13）」及 `bundle.json`。

解開 bundle 後，在其根目錄驗證所有 bytes：

```bash
sha256sum -c SHA256SUMS
# 僅匯入本機映像，不會啟動容器：
docker --context default image load -i image.tar
```

`bundle.json` 分別列出 runtime commit、部署 commit、image ID 與檔案 SHA；
registry digest 為 null，不能把 image ID 當 registry manifest digest。
依 `production.env.example` 建立私有 `production.env`，依部署手冊準備**新的核准 data 副本**。
不得直接使用含 fixture 的驗收 data，不得重建空 DB，更不可讓兩個 app 寫同一份 data。

尚待 hostname／現有網站設定、正式資料來源、離機備份位置與公開影片流量方案確認。
正式停寫、切換、Nginx reload、DNS／tunnel 修改、推送及發布必須另有明確授權。
本 bundle 不含任何影片、正式 DB、secret、個人草稿或 TLS 私鑰。
