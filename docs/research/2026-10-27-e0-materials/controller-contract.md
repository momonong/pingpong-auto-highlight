> 公開去識別研究模板；不是可執行資料集或已准入runtime。E0-W槽名是無來源映射的抽象標籤。

# E0 Controller 研究契約（提案，未准入）

唯一執行權威是 host。模型只能提出 allowlisted typed request 或 stop；不能自行取得媒體、呼叫任意工具或放寬規格。
每個 arm 先由另行准入的凍結manifest初始化固定 16 窗×Q1/Q2/Q3＝48 槽與原 slot IDs，預設 `not_executed`、answer null；不按回答數縮分母。
planner 只取凍結窗的必要 metadata、已准入 evidence、合法工具 receipt、當前 caps／狀態與自己的歷史 raw 回應。
56 舊標記、排名、GT、偏好、reference、盲審人員及對照資料不進 planner；不讓 planner 換來源、窗、邊界或選樣順序。

## A／B 的執行邊界

| 比較 | 相同的部分 | 可變的部分 | 必須由 host 禁止 |
| --- | --- | --- | --- |
| A fixed／controller | 同封存 frame IDs、native PTS、ROI、pixels、metadata、同一次 perception 原文與取得成本 | 後續文字 planner 的有界步驟／停止 | 新 decode／frame／ROI／feature／其他 perception，或把封存影像再次送入模型 |
| B fixed／adaptive | 同初始觀察 recipe、初始 evidence、同 caps | fixed 用預先封存後續排程；adaptive 說明欲消除的 unknown 後提出允許的重看 | 換窗、插幀、同 frame 充新證據、免費 planner、看結果才修改 fixed 排程 |

A 的 shared perception 若用了 1 call，每 arm 僅餘 2 個文字 planner calls；總 model＋planner≤3，不能再加免費解說／重試。
B 的 16＋24＋24 新 frame 是原設計初始提案；每 call 實送≤32，每窗合計≤64，重複輸入也計總量。全部來自既定 8 秒窗。
初始／後續 grid、ROI、pixels、tokens、版本與總實驗資源仍未定，不能因 call/frame 上限已知就執行。
shared perception 的實體成本只記一次，另向每 arm 歸入相同取得成本及 call/frame 帳；各 arm 增量分列，避免隱藏共用成本或重複計算實付費用。
重用任何既存模型輸出的 cache，都必須連結原 operation／evidence／取得成本，並計入該 arm 可取得的 logical call/frame 額度；不能靠 cache 偷取額外證據或呼叫。

## 狀態表

表中 observation 是該狀態可以讀的資料，不表示已取得影片觀察。所有 transition 均由 host 驗證及寫 receipt；以下是契約，未執行。

| 狀態 | Input | 可讀 observation | Allowlisted action | Stop／下一步 |
| --- | --- | --- | --- | --- |
| `INIT_FIXED_SLOTS` | 固定manifest身分及 48 IDs | 窗 metadata、規格 hash | `load_frozen_window`；初始化完整槽位帳 | 身分／hash 不符→`STOP_INVALID_RUN`；否則→`CHECK_ADMISSION` |
| `CHECK_ADMISSION` | runtime必要的版本、recipe、caps、registry／資料條款准入 | 簽定且可查的准入記錄 | host 檢查runtime必要項；reference與optional MCP分開 | runtime必要項任一 null/pending→`BLOCKED_PENDING_ADMISSION`；全數另獲准入才→`READY_WINDOW` |
| `BLOCKED_PENDING_ADMISSION` | 未決清單 | 既有提案／receipt | 保存阻礙，交回決策 | 不 dispatch；這是本包真實狀態 |
| `READY_WINDOW` | host 排定的下一固定 window／arm | 當前槽位帳、caps、既有 valid 快照 | 準備 A sealed read 或 B 准入 recipe request | 全窗完成→`SEAL_RUN`；否則→`RESERVE_OPERATION` |
| `RESERVE_OPERATION` | typed request、原 operation ID、request hash | 已用＋held budgets、原 receipt | 原子保存 reservation；dispatch 前檢查 caps | 正常窗額度用盡且有合法快照／相同request無進展→`FINALIZE_WINDOW`；實驗總額度不足續走→`STOP_PARTIAL_BUDGET`；真正違規→`STOP_INVALID_RUN`；合格才dispatch |
| `ACQUIRE_EVIDENCE` | 已 reserve 的 observation request | A 僅 sealed registry；B 僅准入 recipe／registry | `acquire_admitted_observation` | 來源／PTS／schema／環境錯→全 run 停；未知 outcome/cost→`STOP_RECONCILE`；成功→`MODEL_STEP` 前 reserve 模型 call |
| `MODEL_STEP` | 已 reserve call、允許的 evidence／文字 context | 同一次 raw response，含三題、事實、support、next request／stop | `model.observe_or_decide` 一個 model/planner call | 成功→`VALIDATE_RESPONSE`；沒有成功回應不能補 unknown；失敗按下述規則停 |
| `VALIDATE_RESPONSE` | 原 raw、typed 回應、registry refs | host 身分／PTS／schema 檢查結果 | 保存 raw；結構有效才建立衍生快照 | fatal violation→`STOP_INVALID_RUN`；未知狀態→`STOP_RECONCILE`；合法 request→reserve；stop／上限／無進展→`FINALIZE_WINDOW` |
| `FINALIZE_WINDOW` | 最新合法回應快照或未成功原因 | 三題固定槽及成本帳 | `persist_answer_batch`，保持原 raw、版本及 48 槽投影 | 成功才回 `READY_WINDOW`；提交未知查原 ID，不宣稱已 commit |
| `SEAL_RUN` | 全部 16 窗、48 槽及 reconciled receipts | 身分／成本／停止狀態與歷史快照 | 保存不可覆寫的 run manifest | 工程完整仍待 reference；partial/invalid 不作完整 paired 比較 |
| `STOP_RECONCILE` | 原 operation ID／request hash | `get_operation_receipt` 的原帳 | 只查原 ID、保存 unknown 與 held reservation | 不自動 retry／resume、不換 ID；無可核 outcome/cost 保持停止 |
| `STOP_PARTIAL_BUDGET` | 已知總額度不足以完成後續窗 | 既有valid快照、未執行槽、成本帳 | 保存合法已完成內容及全48槽partial狀態 | 不超額dispatch、不宣稱完整比較；未執行維持not_executed與原因 |
| `STOP_INVALID_RUN` | violation／全 run stop 原因 | 已成功快照、原 raw、錯誤 receipt | 保存 whole-run invalid/partial 及完整 48 槽 | 未執行槽留 invalid/not_executed；不另起窗或報勝負 |

`FINALIZE_WINDOW` 不讓 host 擅自改寫答案：有合法最新回應才可封存其中 yes/no/unknown；沒有回應則記未執行／invalid 與原因，不能製造合法 unknown。
相同 typed request 以 host canonical request hash／evidence identity 判重；若不能指出新的可准入觀察或文字決策進展即停，不增設可循環重試的 counter 額度。
用罄既有 calls 後只能保存最新合法答案及 unknown 原因，不能再呼叫一次「總結模型」。
正常用盡窗的calls/frames本身不是invalid；有合法快照就封存，無合法快照則保留缺答／invalid原因，不能補unknown。實驗總預算不足續走屬partial停止；超額dispatch、違規request或無法硬控才是整run違規。

## 回應有效性與整體停止

legal unknown 必須來自成功且 schema-valid 的回應，附本題證據不足原因；工具失敗、沒看、未答或缺題不是 unknown。
yes/no 必須列已取得的 support IDs 與 registry actual PTS 引用；no 另需足夠範圍支持不存在。缺欄／錯 ID／錯 PTS 不把 raw 改寫為成功。
host 檢查來源／hash／PTS／schema／caps，但 syntactic support 存在不代表 claim 語義受到支持；unsupported 由獨立人工依實際 evidence 審查，當前仍 pending。

來源／hash／PTS／schema／budget／環境、OOM、CPU fallback 或 backend 偷換均停止**整個實驗**；不能只跳過一窗繼續湊完整結果。
外部暫時失敗或 outcome/cost 未知先停並查原 operation；MCP error feedback、timeout、取消標記均不構成重試授權或 compute 已停止的證據。
已成功的 valid 快照仍原樣保留，run 同時標 partial/invalid；目前失敗與未執行槽各記原因，原 48 槽完整保留，不刪成功、不補零、不宣稱完整可比。

## 人工 reference 與評分分離

當前 human reference 48 槽全 pending；truth correctness、wrong／semantic unsupported 與模型 48 題評分均不產生。
unknown 不是錯誤或零分；合法 model unknown、執行 invalid、人工檔匯入 invalid 是不同欄位，後者不能直接算作 model invalid。
reference 完備後才按原 rubric：human-known 的四互斥類（correct且supported、wrong或unsupported、合法model unknown、model invalid）加 human-unknown＝48。
reference 未完備另保留 pending-reference 槽位帳；unsupported 是 human-known／human-unknown 中確定 claim 的旁標，不與主類加總，也不能因 reference 未知而免審。
此包不改原研究篩選／品質門檻，不把合成走讀或工程有效性當品質勝出。
