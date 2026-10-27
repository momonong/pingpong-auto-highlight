> 公開去識別研究模板；不是可執行資料集或已准入runtime。E0-W槽名是無來源映射的抽象標籤。

# E0 盲化答卷匯入契約 v1

狀態：可實作的資料契約；本輪未建立正式匯入器或應用功能，未匯入真人答案。示例reference為48個抽象槽位且全部pending；CASE-DEMO IDs無私人來源映射。
本契約不改 Q1/Q2/Q3、16 固定窗、觀察範圍或既有裁決 rubric。實際資料若另建私有case→原槽位對照，僅管理者持有；本公開模板沒有對照，不按CSV列位置配對。

## CSV 與批次封套

答卷是 UTF-8、無 BOM 的 CSV；可用 LF 或 CRLF，欄名、順序固定如下，不增刪欄位，不以空行當資料列：

```text
case_id,question_id,answer,support_ids_json,time_ranges_us_json,unknown_reason,notes
```

使用標準 CSV 引號規則：含逗號、雙引號或換行的欄位須以雙引號包住，欄內雙引號寫兩次。
證據欄整格必須是 JSON；用 JSON parser 解碼，不能以逗號拆字串。範例欄值 `[[0,8000000]]` 在 CSV 中須寫成 `"[[0,8000000]]"`。
未閉合引號、錯誤 UTF-8、重複／多餘／缺少欄名、多餘／缺少儲存格與空白資料列均須明示拒絕；不得靜默跳過。

匯入呼叫另附批次封套（管理者保存，不在答卷增加姓名欄）：

```json
{"schema_version":"e0-blind-answer-v1","packet_id":"指定 assignment 中的原值","reviewer_code":"指定 assignment 中的原值","batch_id":"本次不透明唯一 ID","mode":"draft 或 submitted"}
```

packet/reviewer 必須精確符合指定 assignment；schema 固定。batch ID 相同且原始 bytes hash 相同可視為同一提交查詢；相同 ID、不同 bytes 為衝突，保留兩份並拒絕覆寫。結果不明先查原 batch ID，不換 ID 冒充新提交。
CSV 及封套原始 bytes、SHA-256、接收時間與版本必須先保留，再解析；同範圍修正另存新批次並連結前批次，原答不可覆寫。

## 身分與固定分母

- case allowlist 精確等於指定 `assignment.json` 的 16 個 case IDs；question allowlist 精確為 `Q1`、`Q2`、`Q3`。
- 複合鍵 `(case_id,question_id)` 必須唯一；期待鍵集合為 16×3，共 48 槽，與列順序無關。
- case/question/answer 不 trim、不改大小寫、不補零、不模糊配對。未知 ID、重複、缺槽、多槽一律診斷，禁止 first/last wins、合併或跨 reviewer 覆蓋。
- submitted 任一結構錯誤使**整批原子性拒絕／隔離**，不部分發布。draft 也須保留完整 48 鍵；只有答案欄可尚未完成。
- 診斷投影始終建立期待 48 槽；缺槽為 missing/invalid，重複槽保留所有衝突原列及行號，額外／未知列放批次診斷旁表。不得刪錯誤槽、縮小分母或用 unknown 代補缺題。

## 欄位與空白規則

| 欄位 | 精確規則 |
| --- | --- |
| `case_id` | assignment 的不透明原值，非空，不含來源或時間。 |
| `question_id` | `Q1`／`Q2`／`Q3`，不可省題。 |
| `answer` | draft 可為真正空字串；submitted 僅 `yes`／`no`／`unknown`，全小寫。`null`、空白字元或其他字串不是空值。 |
| `support_ids_json` | JSON array of unique nonempty strings；ID 不帶前後空白，精確查 registry。初始 `[]`，不接受只含空白的欄值、空字串、逗號清單、object 或字串化 array。 |
| `time_ranges_us_json` | JSON array of two-item arrays；每端是 JSON integer，**不得是 bool、float、字串或 null**。初始 `[]`。 |
| `unknown_reason` | unknown 必填非空白原因；yes/no 必須空字串。空 draft 亦為空字串。原文字保留，不以 trim 改寫。 |
| `notes` | 選填文字，初始空字串；不得在此補變更身分、範圍或替代證據欄。 |

兩個證據 array 長度必須相同，按 index 一對一表示「此 support ID 支持此時間範圍」。不可猜測排列或讓一格多個 ID 對不明範圍。
相對時間一律是固定 8 秒窗內的整數微秒，半開區間 `[start,end)`：`0 <= start < end <= 8000000`；零長、負值、越界皆 invalid。
draft 空白答案必須配 `[]`／`[]` 及空 unknown_reason；不替未答預填 unknown。draft 已填答案也須符合相同欄位規則。

`yes`、`no` 都需非空且可核對的 support／範圍。`no` 還需足夠範圍支持不存在，稀疏取樣未看到不能直接判 no；範圍長度本身也不能自動證明語義。
`unknown` 是人工實際觀察後依題目邊界作出的合法答案，必須有具體原因；可不引用確定事件，此時兩個證據 array 都為 `[]`。如提供引用，仍須配對且由 registry 核對。
媒體尚未提供、根本未閱覽或未答不等於 human unknown；保留 pending_media／pending_adjudication。

## 相對定位與 actual PTS 的界線

reviewer 填的微秒偏移只供定位，不是 native actual PTS。不得直接加上原窗起點、乘 FPS、插幀或用假 frame ID 宣稱已取得實際影格。
日後准入的媒體 evidence registry 由管理者保存，至少包含：

- 不透明 support ID、所屬 case ID、實際媒體／frame identity 與內容 hash；與固定原窗的身分關係。
- 實際 native PTS、`time_base` 有理數、stream start、stream identity 及來源映射；缺值／不確定映射保持 pending。
- support 實際涵蓋的影格／區間及其與 reviewer 相對定位的可核對關係；轉換 recipe 與媒體處理版本另行固定。

管理者必須確認引用屬於該 case、固定窗、實際取得 evidence 及可核的 native PTS，並保存對應證據，才可通過媒體核對；不能僅因數字在 0–8000000 間而視為真實 PTS。
registry 不分發來源對照給 reviewer；日後 reviewer 僅取得 case 命名的准入媒體及不透明 support IDs。
本輪 registry 與媒體都不存在：**格式有效僅為 `format_valid_pending_media`，不得發布 reference**。語義正確及 support 是否足夠仍待人工，形式有效不證明真實。

## 狀態與拒絕

| 情況 | 狀態及行為 |
| --- | --- |
| 完整 draft、answer 空白 | `draft_pending`；不計為 human unknown。 |
| submitted 缺答／錯格式／錯鍵／重複／缺槽／越界 | 批次 `rejected_quarantined`；預期 48 槽診斷仍完整，保留 invalid／原因／原列。 |
| 欄位格式有效但尚無准入 registry | `format_valid_pending_media`；原答保留，reference 不變。 |
| 人工 unknown，原因合法且媒體與獨立觀察經核對 | 合法 human-unknown 候選答案；仍須後續封存。 |
| support 不存在、錯 case、PTS／映射不符 | 明列失敗，隔離；禁止回填看似合理數字或暗改為 unknown。 |

unknown 是答案類型，invalid 是提交／欄位錯誤，pending 是未完成程序；三者不能互相代替。
診斷也應分開 batch structural errors、各槽 field errors、media pending／mismatch 與人工 semantic review，不把其中任一通過當全流程通過。
本契約的「匯入 invalid」指人工裁決檔／證據資料未合格，**不能直接計入原 E0 評分的 model invalid 類別**。human reference 尚 pending 時，不產生 truth correctness 或 48 題模型評分；兩個評估階段必須分開保存。

## 兩位獨立答卷與覆核

R-A、R-B 是占位代號，尚未指定真人或建立人員分離。兩人應獨立閱覽與作答，不看對方答案、arm 名稱、模型答案或建議；先封存原答，再比對。
每份原答及後續版本各自保存。每位 48 槽，最終對照仍是一份 48 槽；不能合併成 96 個 reference 或用重複提交增加票數。

1. 兩份均完整、欄位／媒體核對有效且答案相同（含 unknown）時，只標 `agreement_pending_seal`，不自動發布 reference；保留兩份 support、範圍及原因。
2. 不同答案，包括 unknown vs yes/no，一律 `needs_adjudication`；不多數決、不偏選 known、不讓同模型自評補真值。
3. 任一份缺失／invalid 先補正並保存版本；另一份不能自動晉升 reference。媒體未准入則仍 pending_media。
4. 第三位或指定人工只對同一固定窗、同一題覆核；保留雙方原理由、覆核者自己的 support／時間／registry 核對及最終判斷。不得延長窗、換樣或改題。
5. 無法裁決就保持 `pending_adjudication`；若人工已實際覆核並確認題目不可判定，可明示 human unknown 並記原因。未回覆不能視為接受。

指定覆核者、實際人員獨立性、媒體准入、資料分發及 reference 發布均待人決策，這份契約不構成已執行證據。
最終每槽保留原 slot ID、case ID、雙方提交引用、agreement／disagreement／invalid／pending 狀態及封存者證據；reference 仍可 null/pending，不強迫湊成 48 個已知 yes/no。
封存後既有 E0 human-known／human-unknown、四個互斥評估主類及固定 48 分母規則繼續適用；本包不產生 arm 得分或品質結論。
