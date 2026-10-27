> 公開去識別研究模板；不是可執行資料集或已准入runtime。E0-W槽名是無來源映射的抽象標籤。

# 最小 Typed Local Tool 契約（未實作）

這些是單機函式的概念 I/O，不是可呼叫 API、SDK 或正式 JSON Schema。Host 唯一執行 cap、身分、source/hash、PTS、schema 與環境約束；模型與任何 adapter 均不能覆蓋。
host 依另行准入manifest凍結的 original slot/window 身分持有資料；planner 不讀 reference、56 標記、排名、偏好或盲審人員資料。
不提供選片、render、訓練、任意 URL、shell、下載、換來源或改邊界工具。

## 共用型別與 receipt

`RunId`、`ArmId`、`WindowId`、`OperationId`、`EvidenceHandle` 是 host allowlist 的原值；字串本身不是授權。
`RequestHash` 為 host 以已固定 canonical encoding 計算的 request 身分；encoding 尚需准入時固定，不由模型提供可信 hash。
`NativePtsRef` 指向 registry 的實際 native PTS 記錄，不接受由相對 offset／FPS 推算而宣稱為實際影格的值。

每個 dispatch 之前 host 原子保存：operation ID、run/arm/window、tool及request hash、input artifact/evidence refs、requested budget、held reservation、版本／recipe身分。
同一 ID＋相同 hash 查原 operation；同 ID 不同 hash 為衝突，拒絕。未知結果不能改 ID 重送。
receipt 至少保存：原 ID/hash、狀態與原錯誤、開始／結束／最後查核時間（未知可 null）、輸入／輸出 artifact hashes、actual evidence registry refs、budget reservation、實際可核用量／未知欄、停止原因及 commit snapshot 引用。
receipt 的 operation outcome 與 cost completeness 分開：`succeeded/failed/in_flight/unknown` 不自動決定成本是否已知；沒有量測的值是 null/unknown，不是 0。
prepare、decode／cache 取得、歷史重送、重試、失敗、tokens、planner、實送／unique frames、pixels、input seconds、GPU/wall／費用均入帳；cache hit 仍揭露最初取得成本。
logical per-arm budget 與 physical execution／實付成本分開。A shared感知實做一次，向每arm計1 call及相同frames；共享實付只報一次並列分攤。任何cache模型輸出重用都綁原operation/cost，不得突破arm的證據或call額度。

## 最少函式

| Typed function | 最小 input | 最小 output／receipt | Host 必查 |
| --- | --- | --- | --- |
| `load_frozen_window` | run、host 排定 window ID、凍結manifest identity | 只讀 `WindowHandle`、原核心／requested/actual範圍的 host metadata、Q1–Q3原槽位 refs、input hash receipt | 固定16窗／48槽、source/hash/session／端點一致；不選新窗，不讀標記 |
| `acquire_admitted_observation` | 原 operation ID、reservation、window/arm、mode、准入 recipe／sealed handle；B 重看須 unknown-to-resolve | `ObservationReceipt`、immutable evidence handle、實際 registry entry refs、requested/actual取得用量 | A 僅讀同 sealed set，不能新增 decode／frame／ROI；B 僅准入 grid/ROI、來源窗內、frame novelty／caps；outcome/cost未知則停對帳 |
| `model.observe_or_decide` | 已 reserve 的原operation、固定三題/rubric、合法 evidence handle或A文字context、剩餘caps | 原 raw 回應 artifact；typed `{observed_facts,answers,next_request_or_stop}`；model/planner用量receipt | 一次 model＋planner call；A shared後只文字，B送影像也含決策；schema／support／PTS／request allowlist，不添加免費planner |
| `persist_answer_batch` | host 已核的原 operation、immutable raw refs、衍生槽位快照、完整48槽投影、run停止／成本狀態 | 原子commit receipt、snapshot ID/hash或unknown；不覆寫原raw／舊snapshot | 固定IDs與48分母、raw/derived分離、reference/scoring pending；缺／重複槽拒絕，不first/last wins |
| `get_operation_receipt` | 原 operation ID及request hash | 原 operation 的 outcome、cost completeness、commit／compute可核狀態 | 唯讀對帳；不觸發新工作、不視查詢成功為原執行成功／已停止 |

`answers` 的 final／provisional typed 回應均須精確列 Q1–Q3；每題 `answer∈yes/no/unknown`，確定答案含非空 support IDs 與 NativePtsRefs，unknown 含具體證據不足原因。
回應只提出一個 allowlisted next action 或 stop；A 可要求下一次有界文字決策，不能要求再送影像；B 可要求固定窗內准入 recipe 及待消除 unknown，或 stop。
未成功呼叫／未答／schema錯保留 raw/error 與 invalid 原因，不由 host 代補成合法 unknown。
`persist_answer_batch` 是模型執行記錄的概念儲存接口；不是盲化裁決模板的人類 CSV 匯入器，也不實作人類 reference 發布。

## Runtime Evidence Registry（尚未建立）

Registry 由 host 保存 immutable entries；每 entry 至少需要：

- evidence/support/frame ID、run/arm/window及source/hash身分、媒體／frame內容 hash、stream identity。
- 真正 native PTS、time_base 的分子／分母、stream start 與來源時間映射；保留 requested／actual位置、觀察／裁切 recipe 與 decoder/processor/code 固定版本。
- 實際 ROI／pixel dimensions、實送／unique frame identity、取得 operation receipt、raw perception artifact（A 同一次封存原文）。
- 封存 set hash及可核 lineage：A兩arm對應完全同set／原文；B共同初始set與後續各arm分支分開記錄。

缺少來源映射、PTS、版本或真正媒體 evidence 時保持 pending_media，不生成可執行的 ObservationReceipt。
相對 0–8000000µs 只用於定位；不能直接加窗起點或乘 FPS 冒充 native actual PTS。Registry 的具體表示與轉換 recipe 尚待准入，不在此假造。
形式可解析／URI或mimeType存在不證明已看懂影片或取得正確 PTS；registry 可核身分也不能自動證明模型語義 claim。

## 預算交易及錯誤

每次 call 的 `used + held reservations + requested` 均須在已准入 caps 內，先 reserve、保存 ID/hash，才 dispatch。未決 cap 不可用無限大、0或CPU fallback代替。
call 失敗仍計 dispatch／嘗試成本；outcome或cost未知保留 held reservation，不釋放為0。只有原receipt證明完整結果及用量才可對帳，不因transport error自動退款或重試。
`not_dispatched` 可記0次實際執行；已dispatch但outcome_unknown的請求，其實際執行／成本是unknown，不能記0或放回預留額度。
同 request／同 evidence 沒有新進展時 stop；上限用罄只保存最新合法快照，不額外呼叫模型整理。
正常窗額度用盡不自動invalid；有合法快照即finalize。已知全實驗額度不足以繼續未走窗時保存partial／not_executed並停；超額dispatch或違規request、無法硬控才整run invalid。
環境／OOM／backend/fallback、source/hash/PTS/schema/budget違規停止整個實驗；暫時／未知外部失敗也停並先查原 ID，不自行 retry／resume。
沒有核對 compute／commit 結果就維持 unknown；取消或 timeout 不自動代表硬體已停或資料已提交。
