# 實驗提案：先證據有效，再整片排名

查閱日：2026-10-26。狀態：**未執行**；未產生E0實際16窗manifest、模型結果或新GPU證據。
本文件不是下載、推論、訓練、渲染或服務變更授權。

**先用E0區分控制流程與自適應取樣的作用，通過後才提案E1整片排名。**
16窗只能評估局部證據使用／工具契約，不能報56例或全片選球勝出。
單一候選模型、單一預先固定rubric、一次比較；失敗不自動換模型或搜參數。

## 共同凍結與准入

凍結五來源／四場次、source hash／時間映射、2,145 candidate IDs／核心區間及feature revision。
56歷史精彩正例只給evaluator；40個AI nonplaying不是普通球偏好負例，未標一律unknown。
56例是編輯區間，不是完整回合或逐拍GT；所有場次皆development。
現有5/56、6/56、+5／−4、pool53/56只作[歷史診斷](existing-evidence.json)。

執行前須另確認：模型／code／processor／decoder固定版本、合法資料條款、runtime／GPU可用性及總資源上限。
第一候選與備案見[架構](architecture.md)；本輪只有weight card identity，尚未完成runtime准入。
預先固定rubric、schema、選擇規則、觀察／裁切recipe、隨機seed與每項成本計法。
來源標記不得進入planner、視覺prompt或抽樣分層；零shot rubric不得事後依56例調整。

## E0：四場次均衡的16個短窗

以下是**待執行的可重現生成規則**；不是已封存的實際抽樣結果。
不按正例、方法命中或top-k挑窗，也不以人工精彩標記補樣。

1. 對固定母體按session分組，每場4個名額，共16；五片中的Apr25場次含兩片，先分配每片2窗。
2. 其他三場各一片，各4窗；source ID以固定manifest的穩定ID為準，不能用本機路徑排序。
3. 各source內按candidate核心中心時間排序，tie用candidate ID；N筆排序列的Qk取索引`floor((k−1)N/4)`至`floor(kN/4)`（不含尾），k=1..4。
4. 單片場次在每個四分位取1窗。Apr25每片先分早半場（Q1＋Q2）與晚半場（Q3＋Q4），各取1窗。
5. 每個stratum依SHA-256最小值選；輸入為UTF-8 JSON array `[0, source_id, start_ms, end_ms, candidate_id]`，無空白、非ASCII原字元不跳脫、毫秒整數。
6. hash tie以`source_id,start_ms,end_ms,candidate_id`字典序打破；`start_ms/end_ms`是原candidate核心，不因上下文裁切改identity。
7. `center_ms=floor((start_ms+end_ms)/2)`；觀察窗固定中心±4秒，clip到source端點，保留原核心、requested／actual窗及不足8秒原因。
8. 同一source已選窗若有正長度重疊，在該stratum依hash順序取下一個；端點相接不算重疊。
9. stratum按source ID、Q1至Q4／早晚順序處理；source hash相同的重複檔先視為同來源，禁止重複計樣。
10. 無可用候選、來源／session映射矛盾或不足16窗，明示不足並停止生成；不跨場次、跨stratum補樣。

原始核心以manifest整數毫秒為準；若只有小數秒，以十進位原字串×1000、ROUND_HALF_UP轉整數，保留原值與轉換revision。
四分位以candidate時間排序分布定義，不保證等長的來源秒數；manifest須記此限制與候選密度。
固定序列、hash encoding、母體hash、選中／跳過原因與重疊檢查一併封存，讓別人可重建同16窗。
若日期分組與來源manifest不符，先修正資料契約並交回核對，不能默默換樣。
短窗未拍到完整一球、未看見接觸或結果，完整性／該事件即unknown，不延長窗來湊真值。

## E0-A：完全相同證據，只比較控制方式

兩arm：預定步驟的fixed流程，與可在有界工具內選步驟／停止的controller。
所有arm取得**同一份已封存evidence**：同frame IDs、actual PTS、ROI／像素、metadata及同一次感知原文。
禁止額外decode、新frame、新ROI、其他模型perception或新特徵；不能把新觀察混入此比較。

shared evidence的生成方法與成本預先固定；若包含一次模型感知，該call計入每arm單窗上限。
後續planner僅使用同一封存內容，不能再次送影像作額外perception；規則如何使用evidence另記。
例如共用1次感知後至多2次文字planner，全部合計仍≤3，不把planner當免費工作。
報告共享準備成本與各arm增量成本；cache重用不隱藏最初取得證據的費用。

此比較回答「相同證據下控制流程是否有用」，不回答「agent是否更會找新證據」。
若只整理理由、改輸出格式或更會引用，不能直接稱精彩品質勝出。
A沒有可重現證據使用收益且增加unsupported claims時，不保留額外agent推理層。

## E0-B：相同初始觀察與上限，固定取樣對自適應重看

兩arm共用初始觀察recipe；fixed使用預先固定時間／ROI排程，adaptive依unknown要求下一次觀察。
兩者後續證據會不同；這是**同預算**比較，不能稱為A的同證據比較。
fixed的密集重看recipe在看結果前封存，adaptive每次列出希望消除的unknown。

初始可行提案：16個實際來源frames，後續至多24＋24個新frames；全部來自已固定8秒窗。
採原始source timestamps取樣，記錄requested與actual PTS；不插幀補事件，不用重複frame充當新證據。
具體時間grid、ROI候選與pixel上限在准入時一次凍結，不能看56標記後調整。
若來源可用frame不足，保留不足／unknown，不默默變更FPS、放大或換backend。

每次模型回應同時輸出觀察事實、support及下一個typed request／stop，讓planner與描述共用call。
controller驗證request後執行，fixed arm則使用封存排程；不另加免費planner呼叫。
若runtime無法在總≤3 calls內滿足schema與規劃，縮小schema後重新決策，或停止；不能放寬為3次觀察加額外planner。
B無可重現收益，或固定密集觀察相同／更好，就不擴為多代理或更多回合。

## E0共同預算與記錄（上限仍待准入）

| 項目 | 初始政策提案 | 目前狀態 |
| --- | --- | --- |
| 單窗model＋planner calls | 總計≤3，包含shared感知、重試與planner | 未執行；不可分開加額度 |
| 實際送入frames | 每call≤32，單窗合計≤64；重複輸入也計入總量 | B的16＋24＋24是初始recipe；A不得新增perception |
| pixels、input／output tokens | 所有arm同最大值，含歷史重複輸入及規劃 | 具體數值待runtime准入，未確認前不能跑 |
| GPU seconds、wall time、費用與全實驗上限 | 每窗及總量都要硬限，含準備／失敗 | 待決策；工作時段不等於任意資源額度 |
| 外部job／中止 | 固定operation ID，查receipt、核對compute停止 | 不明先查原operation，不能新ID重送 |

同最大額度不等於實際等成本；兩者都報，無法相等就明示差異，不硬稱等成本勝出。
記錄decoded frames、實送／unique frames、pixels、unique source seconds／總input seconds、cache hit。
另列tokens、模型／planner calls、GPU seconds、wall time、tool errors、abstention、終止原因及產物hash。
重試、失敗、cache準備與planner歷史輸入都計費，不只報成功工具回合數。

## E0：固定48個題目槽位與獨立裁決

每個arm固定16窗×下列3題＝**48 slots**；三題均須回答`yes / no / unknown`，不允許靠少說縮小分母。

| 固定predicate | yes的必要證據 | 不可推定與unknown邊界 |
| --- | --- | --- |
| Q1：窗中是否可支持正在對打？ | 引用可見動作序列及其actual PTS／範圍 | 單張揮拍不等於擊球；序列不足即unknown |
| Q2：是否至少一次直接可見球桌接觸（bounce）？ | 直接可見的接觸證據、actual PTS與support IDs | 不從姿勢／常識猜bounce；事件不可見即unknown |
| Q3：窗內是否有可見開始與結束，足以支持完整一球？ | 同一球可見開始／結束及窗內時間序列 | 缺任一端、回合關係不清即unknown，不延長窗湊答案 |

所有確定答案（含`no`）都要有support IDs及時間範圍；`no`需足夠範圍證據支持不存在，稀疏取樣未看到不能直接答no。
未答、schema錯或缺題仍占原48 slots，保留invalid／錯誤原因，不刪窗、不丟分母。

獨立人工先對原固定窗的同三題封存reference（可unknown），不看arm名稱、模型答案或建議。
再按各arm**實際取得的evidence**盲審support；原窗reference不能替模型補上未觀測影格。
固定判定rubric與分歧處理先封存；AI代標僅為弱證據，同模型自評不是獨立GT。

human-known slots（reference為yes／no）分成四個互斥主類：correct且supported、wrong或unsupported、合法model unknown、invalid（未答／schema錯／缺題）。
四類合計＝human-reference-known slots；invalid獨立保留，不當作事實錯誤，也不與前三類重複計數。
human-unknown slots另列，不納入truth correctness；模型確定claim仍可另審evidence是否支持，不能因reference未知就免審。
若人工無法裁決，保留pending；不能由同模型自審補成已知reference。

每個arm須報四個互斥主類及human-unknown完整計數；四類合計加human-unknown＝48。
同窗同題配對列出correct gain／loss及slot IDs；unsupported gain／loss明列新增／消失的unsupported slots，避免方向混淆。
另報answered coverage＝合法yes／no slots／48，human-reference-known coverage＝人工yes／no slots／48；兩者分母固定48。
unsupported count是旁標，涵蓋human-known及human-unknown slots中確定claims的support裁決；不與四個主類加總48，也不能只報低unsupported率。
沒有獨立人工時，只報schema、來源／時間、工具／預算與成本；語義正確及偏好維持pending。

**未執行的研究篩選政策提案**：0來源／PTS／schema／budget違規，且在人工可裁決slots的paired net correct gain≥1、unsupported count不增加，才值得考慮保留該agent元件。
這不是統計顯著性、產品驗收或使用者已接受品質；小窗結果不外推全片。
全unknown、沒有人工可裁決slots或support仍無法裁決時為inconclusive，不算通過。
若語意結果完全相同但成本更省，只稱效率改善，保留固定流程作品質基線。

## E1：E0通過後才提案完整候選paired ranking

E1保留五片全部2,145候選列，不只重排先前選中或top-k；固定candidate identity、source範圍、單一模型／rubric及selector設定。
每列保留`status`（scored／unscorable／error），`score`須為finite數值，或`null`並附明確reason；缺結果也保留該列。
event某欄unknown不必使整列unscorable：只有其他可支撐證據足以依固定rubric評分時才給score。
無可支撐score不能補0、默認低分或偷偷回填baseline；unknown不是低分的同義詞。

任一candidate unscorable／缺結果時，保留完整2,145列與partial狀態，停止進入正式paired selection報告，不宣稱完整五片品質勝負。
可報資料可觀測性／失敗診斷，不能縮母體再比較；hybrid baseline fallback必須另定新arm、校準及排序契約，本輪不默認。
即使全列可評分，也不能聲稱救回候選外漏球；pool53/56仍只代表已知正例的開發覆蓋。

比較沿用同一deterministic selector：每片55秒、前後1.5秒、完整球不truncate、相同去重／核心不重疊及tie規則。
在執行前固定selector程式版本；排序total order為`(-score, start, end, candidateID)`，時間沿用同一canonical來源時間單位，四欄依序升冪。
one-to-one matching在同source進行，交集至少人工區間50%且IoU≥0.3；相同matching實作與邊界規則。
先封存全母體prediction、selector計畫、hash及成本，再讀56標記作評估；不事後改分數／rubric。

保留四session分組；凡train／fit／校準／閾值選取，都排除整個留出場次，Apr25兩片不可拆兩邊。
既有四場全是development，即使做grouped comparison也不能改稱新場次held-out泛化。
零shot新方向不以56例調整；若必須調，另記新研究決策，不沿用一次比較名稱。

| E1必報項 | 解讀界線 |
| --- | --- |
| 每片／每場hits、新增與失去ID、對稱差集及淨差 | 淨增不能掩蓋回歸；未標片段不是FP |
| 全母體unknown／錯誤、重複、超時長、來源／PTS與render／probe結果 | 工程可播放不等於回合完整或精彩 |
| shared及arm成本、wall／GPU／token、終止原因 | 不只比命中，報實際品質／成本取捨 |
| 人工盲偏好、完整性及新場次接受 | 未做就pending；不能由56開發正例代替 |

E1 promotion的net wins、每片退步容忍度、值得成本、盲偏好接受與新場次門檻均待人決策。
政策提案是先看E0 validity／faithfulness，再看E1收益／回歸／成本；最終品質仍須新場次人工接受。

## 全程停止條件

- source／hash／PTS／schema錯誤，或來源與候選identity無法對上：停止該實驗，不修補成成功。
- budget無法硬控、到上限、OOM、需要CPU fallback或換backend：停止，保留receipt，不偷偷繼續。
- claim無support，或高速接觸不可見：該claim回unknown；若核心研究問題因此不可判別，停止該實驗。
- 需改權重、看標記、擴母體、換rubric、換供應商或追加資源才能推進：交回決策，不自動搜尋。
- 外部job結果不明或cancelled：先查原operation並確認compute狀態，不把取消標記當硬體已停。
- data／code／weights使用條件或runtime仍有缺口：保持未准入，不下載／推論來補證明。

停止後保留原manifest、raw/error、預算、未知與失敗原因；不丟掉不利窗或重挑樣本。
本輪交付到研究文件與靜態核對為止；E0、E1及新場次人工驗收都尚未啟動。

回到[方向結論](README.md)；[工具／模型來源](architecture.md)與[資料集條款](datasets.md)分別核對。
