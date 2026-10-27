> 公開去識別研究模板；不是可執行資料集或已准入runtime。E0-W槽名是無來源映射的抽象標籤。

# 未決與停止入口

本包所有 runtime 必要准入仍 pending；其中任一必要值未定就停在 `BLOCKED_PENDING_ADMISSION`。人工reference只阻擋語義評分，optional MCP只影響轉用MCP，不阻擋日後已准入的local函式E0。
call/frame政策已有數值，不代表其餘runtime資源可略過。

既有第一候選提案為 `Qwen/Qwen2.5-VL-7B-Instruct`，weight revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`，沿用[固定架構文件](../2026-10-26-direction/architecture.md)的已知身分；不重新選型或查HF。
這是 existing candidate proposal，並非已完成runtime准入。code／processor／decoder、實際環境及資源等尚未核對，仍null/pending；不能把已知候選抹成未知，也不能由weight revision推定runtime可執行。

| 待決內容 | 為何阻擋 | 本包保留值／接續方式 |
| --- | --- | --- |
| 既有候選的runtime／code／processor／decoder准入、資料與實裝使用條件 | 第一候選及weight revision已知，但不足以證明實際環境／條件 | 保留上列既有proposal；尚未核的runtime欄位null/pending，不下載或推論補證明 |
| A shared evidence生成方法與成本；B初始／fixed後續grid、ROI、pixels與觀察recipe | 目前只有原16＋24＋24提案，沒有可執行時間排程 | null/pending；先封存recipe，再執行；不能依標記或結果調整 |
| pixel／input-output tokens／GPU seconds／wall／費用的單call、單窗與全實驗上限及成本分攤 | 無法完整reserve／硬限／比較實際成本 | null/pending；這裡不新增門檻，不以準備包10分鐘額度充當runtime准入 |
| 真媒體registry、nativePTS/time_base/stream start／來源映射與版本lineage | 相對offset或metadata不足以證明實際影格 | pending_media；需另准入的真實媒體取得及核對，本輪不讀影片 |
| Request canonical encoding、實裝持久化／原子reserve與receipt對帳 | 不明結果不能視為0成本或新ID重試 | null/pending；後續最小實作另授權，這裡只定契約 |
| 整體E0執行授權與新資源／失敗後接續 | 文件完成不是runtime或重試授權 | 維持未執行；未知operation先查原receipt，重大缺口交原整合入口 |

下列是分開的gate，不列入local runtime的必要准入：

| Gate | 影響範圍 | 保留狀態 |
| --- | --- | --- |
| 人工reference、人員獨立性、盲化分發與覆核 | 僅semantic evaluation gate：阻擋correctness／unsupported／偏好評分；不把人工資料送planner | 48 pending；沿用盲化裁決模板；即使日後runtime有效，本輪不打品質分 |
| 真正跨host／多client需求與MCP adapter版本／認證權限 | optional MCP switch gate；不因尚無MCP需求／准入而阻擋local functions | 預設local functions；需求與准入皆明確後才重議 |

原E0篩選政策與E1品質／成本／退步／新場次門檻維持既有提案與待決狀態；這包不增訂新性能或品質數值，也不啟動E1。
三條合成走讀只檢查邏輯分支：符號證據／PTS refs不是已核媒體，null成本不是免費，模擬完成不代表完整16窗比較。
此公開包不含私人輸入、來源對照、人工答案或執行收據；沒有新增服務、模型下載／執行或媒體操作。
