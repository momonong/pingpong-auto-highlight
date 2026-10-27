> 公開去識別研究模板；不是可執行資料集或已准入runtime。E0-W槽名是無來源映射的抽象標籤。

# MCP 採用決策（研究提案）

**先採 typed local functions；MCP 保留為可選 adapter。** E0 目前是單機、固定 evidence／budget／receipt 的有界研究，沒有已確認跨 host 或多 client 相容需求；新增服務與協定層尚無必要收益。
這不授權實作或重建系統；本包只有工具契約。Host 的 caps、身分、PTS、idempotency、原 operation 對帳與 truth 審查不交給協定保證。

只有多 client／host 的具體相容需求成立，並另完成版本、SDK／adapter、認證／權限、timeout／receipt與錯誤映射准入，才重新考慮 MCP。
新 adapter 必須維持相同 typed I/O、48 槽、預算 reservation、不可覆寫 raw、未知 outcome hold 及整 run 停止行為；協定 error feedback 不能增加重試權限。
unexpected 額外互動先停等；capabilities、工具 annotations 或 statehandle 字串不構成 E0 授權。

## 官方來源 Ledger（恰三份）

固定參考以下三份2026-07-28官方頁面；這是文件依據，不是已安裝SDK／runtime版本證據。每頁只保留短摘要，不複製規格正文。

| 官方頁 | 精簡核對事實與本案界線 |
| --- | --- |
| [Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) | tools capability、tools/list/call 提供介面；schema／structuredContent 仍須 client 核對，非 LLM constrained generation 保證。isError 與 protocol error 分開。annotations不可信、statehandle只是字串；input_required 的非預期互動在E0停等。 |
| [Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources) | application-driven 的 URI context，可 list/read text或base64 blob。**本案推論**：URI／mimeType不保證影片理解或正確PTS，不能取代registry。 |
| [Versioning and Compatibility](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning) | 此核對版採per-request version／identity／caps、server/discover，無initialize negotiation；unsupported version為-32022。舊2025-11-25架構文件保留，不回寫為已升級。 |

本輪沒有核對 current tasks／cancellation；不沿用舊 methods 為現行事實，也不承諾 MCP 能停止 compute。
來源頁描述協定；本案選用 local functions、pending gate與整 run 停止政策是研究設計決策，不假稱官方替我們保證 caps、真值或exactly-once。
