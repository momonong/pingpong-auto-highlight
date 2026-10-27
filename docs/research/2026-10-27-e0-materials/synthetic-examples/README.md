> 公開合成格式示例，不映射任何私人來源。

# 合成格式案例

所有 SYN IDs、support 與答案均為格式示範，未觀察影片、不構成語義真值、不得匯入正式 16-case 答卷。
此小案例 allowlist 僅 SYN-PUBLIC-CASE-01×Q1–Q3＝3 槽，方便展示；正式分母維持 48，不能用本案例 allowlist 取代 assignment。
format-valid.csv 示範標準 CSV quoting、yes/no 配對證據欄、unknown 原因及 0／8000000 時間端點。
rejected-duplicate.csv 額外重複 Q1，預期整批拒絕，診斷投影仍保留合成期待 3 槽（正式為 48）。
format-cases.json 列出接受／拒絕預期，涵蓋缺槽、未知 ID、缺答、JSON 型別、bool/float、負值、零長與越界。
format-valid 僅指欄位格式：沒有媒體 registry，狀態仍 pending_media，不能發布 reference；本輪未實作正式匯入器。
