# E0 研究材料：公開去識別模板

此包隨 1.4.1 Git/source 研究材料 patch 提供，包含完整抽樣規格、兩份人工空表與匯入契約，以及 controller／tool／budget／MCP 設計和恰三條合成走讀。
這是去識別模板，**不是可執行資料集、實際抽樣結果或已完成模型驗證**。公開內容無法重建原本16個真實觀察窗，也不提供原片、候選、標記、真人答案、來源對照或執行收據。

## 使用入口

- [抽樣規格](sampling-contract.md)：固定16窗×Q1–Q3＝48槽、四場次配額、Decimal時間轉換、SHA-256選樣與觀察窗去重。
- [reviewer A](reviewer-a/README.md)／[reviewer B](reviewer-b/README.md)：各48個空白答案、assignment與rubric；CASE-DEMO-01..16全部是無來源映射的示例身分，兩份順序不同。
- [匯入契約](IMPORT-CONTRACT.md)：CSV／JSON複合欄位、relative offsets、原子拒絕、unknown／invalid／pending及雙人覆核。
- [controller契約](controller-contract.md)、[工具契約](tool-contracts.md)、[機讀budget/status](budget-status-contract.json)：host執行預算與身分／PTS約束，runtime未准入。
- [三條合成走讀](synthetic-traces.json)：正常A、證據不足B、工具結果未知；E0-W01..16只是抽象槽名，沒有來源對照。SYN操作／證據／PTS引用均為合成符號，並非媒體或工具證據。
- [MCP決策](MCP-decision.md)、[未決事項](unresolved-decisions.md)、[來源與版本](sources-and-versions.md)。
- [合成格式案例](synthetic-examples/README.md)：獨立SYN示例，不能填入正式答卷或當語義真值。

## 尚未完成的事項

既有第一候選Qwen2.5-VL-7B-Instruct及固定weight revision維持提案身分；code／processor／decoder／資源、媒體registry與native PTS仍待准入。
人工reference、語義correctness／unsupported、偏好與新場次驗收仍pending；有格式或support引用不代表claim語義受到支持。
未實作controller、匯入器、MCP服務、SDK或正式schema validator；未做影片解碼／渲染、模型推論／訓練或部署。
日後實際研究必須另行凍結合法資料與私有身分映射、取得執行授權及全部runtime准入。公開DEMO IDs不可對回原始來源。
若分發真正盲審材料，只提供各reviewer所需的獨立資料夾與准入媒體，不分發steward對照；本模板本身不證明已實施人員ACL或真人盲測。

公開[產物hash manifest](artifacts.sha256.json)僅涵蓋本目錄公開檔案，不包含私人輸入、原始研究收據或來源hash。
既有[2026-10-26研究方向](../2026-10-26-direction/README.md)保留原內容；這一包細化資料／控制契約，不啟動E0或E1。
