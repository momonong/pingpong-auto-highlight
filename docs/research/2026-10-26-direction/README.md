# 方向研究：從可追溯證據到個人化桌球集錦

查閱日：2026-10-26。狀態：研究整理與實驗提案；本輪沒有下載資料／權重、模型推論、訓練、影片解碼／渲染或服務變更。

**建議先研究單一有界 controller 配合短窗視覺證據，保留既有訊號、VP／VJ 與弱監督方法作對照。** MCP 先作可替換的接口，球偵測、事件辨識與偏好資料各自處理；先用小型可反證比較決定是否值得進入整片排名，尚未選定產品新架構或啟動實驗。

## 長期目標與目前差距

從完整自有桌球影片找出完整、可看的球，依使用者偏好選出每片不超過 55 秒的集錦；每段都有來源時間、可追溯證據及明示未知。成品在**新場次盲看片被使用者接受**，才是產品品質證據。

既有 main 有上傳、來源綁定、人工審核／coverage、模型預標註與剪輯流程；本研究沿用其資料保護與人工紀錄，不把工程功能當作精彩品質。固定來源 hash／PTS、候選母體、deterministic selector、render／probe 及回歸記錄是推薦保留的研究骨架；更嚴格的工具預算與原始 PTS 契約仍是提案，見[架構](architecture.md)。

[已核歷史摘要](existing-evidence.json)涵蓋五片、四場次、56 個人工精彩正例及 2,145 候選。signal 選中 5/56，weak MLP 為 6/56，但新增 5 個同時失去 4 個；候選 pool 涵蓋 53/56。這些是同一開發母體的診斷，不能稱為全片召回、一般化或偏好改善。40 個 AI nonplaying 標記不是普通球的人工偏好負例；未標候選維持 unknown。歷史程式與 WIP 尚未合併者，也不因文件收錄而成為 main 能力。

## 三條路的比較

| 方向 | 可解決的問題 | 主要限制與判斷方式 |
| --- | --- | --- |
| 保留訊號／VP／VJ／弱監督 baseline | 成本可控、可重播，提供固定候選與排序對照 | 現有 +5／−4 置換不足以證明偏好改善；逐片列收益、退步及成本 |
| 單一 VLM＋有界 agent | 尋找缺失證據、決定重看時間或 ROI，描述可見事件與未知 | 稀疏影格看不到的接觸不能補成事實；先區分同證據控制與同預算重看 |
| 專項模型＋事件／偏好資料 | 球位置、回合事件、個人歷史偏好分別補強 | 授權、取用成本與域差先准入；球追蹤或群體 saliency 不直接等於精彩 |

推薦第二條作下一個最小研究問題，第三條作分開的能力與資料准入工作；不先建立多代理投票平台。第一候選為固定卡片版本的 Qwen2.5-VL-7B，Qwen3-VL-8B 留作備案；兩者在本來源的效果、硬體容納量及吞吐尚未證明，不同時大跑。

## 可判別的下一步

[實驗提案](experiment.md)先做 E0：四場次各四窗，共 16 個固定候選短窗。比較 A 給所有方法相同封存證據，禁止新增感知；比較 B 給相同初始觀察及資源上限，比固定排程與自適應重看。所有模型與 planner 成本都計入；小窗完整回合不可見就記 unknown，不以語言合理性當真值。

E0 有有效、忠實的證據使用結果後，才考慮 E1：對完整 2,145 候選作 paired full-film ranking，固定每片 55 秒、剪輯與 matching 規則，先封存 prediction 再讀 56 標記。這仍是 development；新場次、人工盲偏好與可接受回歸門檻另行決策。失敗不自動換模型、改 rubric 或追加資源。

## 本輪完成與待決策

本輪完成一手來源查證整理、既有 JSON／hash 摘要核對、資料與模型准入缺口、工具契約和兩階段反證方案。未生成 E0 的實際 16 窗 manifest，未取得獨立人工裁決，也沒有新增 GPU 效果證據。

E0 待確認的是可觀測事件 rubric、獨立人工裁決安排，以及單一 checkpoint 的 code／processor／decoder、合法條款和 runtime／GPU／費用／時間上限。精彩偏好 rubric、新場次與回歸／值得成本門檻屬 E1／產品決策，不阻擋 E0 的工程準備。這份文件仍不構成下載、推論或訓練授權。

## 來源與閱讀入口

- [資料集比較與使用條件](datasets.md)：OpenTTGames／Extended、TTStroke-21、BlurBall、QVHighlights、PHD²。
- [架構、模型與五工具契約](architecture.md)：來源事實、推論與提案分列。
- 一手研究依據：[VideoAgent 正式論文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10325.pdf)、[MotionBench v1](https://arxiv.org/html/2501.02955v1)。前者是影片問答，後者提醒細動作限制，均未證明本產品品質。
- [MCP Tools 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)提供結構接口；[Tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks)仍為 experimental。schema 有效、task cancelled、影片可播放各自不等於事實正確、GPU 已停或人工接受。
