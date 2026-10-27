# 來源、版本與公開範圍

本去識別研究模板隨1.4.1 Git/source patch交付。它不是資料集、模型結果、runtime准入或Docker image發布。
固定設計沿用[2026-10-26實驗提案](../2026-10-26-direction/experiment.md)與[架構提案](../2026-10-26-direction/architecture.md)，原文件保持不變。
原設計的來源身份與研究資料不隨本模板提供；新sampling-contract以抽象四場次／五來源描述相同選樣規則，不揭露實際選中窗。

既有第一候選是Qwen/Qwen2.5-VL-7B-Instruct，weight revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`；保留existing candidate proposal，沒有重新選型或宣稱已完成runtime准入。
code／processor／decoder／環境／觀察recipe／資源caps仍pending。weight身份已知不能代替實裝、影片PTS或品質證據。
MCP的三份固定官方來源與版本界線只在[MCP決策](MCP-decision.md)摘要；沒有把舊研究文件回寫成新版SDK已接通。

CASE-DEMO-01..16不映射私人case／來源；controller的E0-W抽象槽名沒有對照。SYN IDs、答案及工具事件完全合成，沒有真人／模型輸出或實測成本。
JSON中document hashes僅指本公開目錄或既有公開研究文件；公開artifact manifest只封存公開檔案。沒有私人母體／影片hash、機器／人員／任務識別、執行收據或私有對照。
公開檔案可核對格式、48槽與設計一致性；不能據此重建原16真窗、計模型正確率、證明人員盲化或完成新場次接受。
