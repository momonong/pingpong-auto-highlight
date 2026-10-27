# 固定16窗／48槽抽樣規格

公開去識別規格模板；以下是生成規則，沒有公開實際候選、選中窗、來源身分、絕對時間或母體hash。CASE-DEMO及E0-W抽象槽名不能重建原16真窗。

## 凍結與隔離

實際執行前另行凍結來源／場次映射、source hash、精確duration_us、候選row/candidate IDs及核心時間、feature revision、Decimal轉換revision與candidate-only母體hash。
來源metadata與候選identity只供必要的host映射與抽樣。人工正例、review、偏好、方法命中、模型分數及GT均不得用於選樣、分層或planner輸入。
候選投影採白名單，不能把含標記／歷史score的原始資料直接交給sampler；與來源候選核心及handoff的ID／session／sourcehash／核心對應矛盾就診斷停止。
原有開發資料不能因本規格改稱held-out；短窗結果不外推全片或精彩偏好。

## 配額與選樣

1. 固定四場次，每場4窗，共16。五來源中一場含兩來源，各2窗；其餘三場各一來源，各4窗。實際場次／來源ID另存私有manifest，本模板不指定真實日期。
2. source依穩定source ID排序，不按本機路徑。相同source hash視為同來源，禁止重複計樣；若去重後與固定配額矛盾，先診斷，不擅自補樣。
3. 每source按candidate核心中心排序；整數ms下用`(start_ms+end_ms,candidate_id)`，保留可能0.5ms的中心差，不能先floor再作排序。
4. N筆的Qk索引為`floor((k-1)N/4)`至`floor(kN/4)`，尾不含。單來源場次每四分位取1；雙來源場次每片在早Q1+Q2、晚Q3+Q4各取1。
5. 固定seed=0；每stratum取SHA-256最小者。hash輸入為UTF-8 JSON array `[0,source_id,start_ms,end_ms,candidate_id]`，`ensure_ascii=False,separators=(',',':')`；hash tie按`source_id,start_ms,end_ms,candidate_id`字典序。
6. 按source ID、Q1至Q4／早晚處理。若與相同source hash已選觀察窗有正長重疊，依hash rank取下一候選；端點相接允許。保存每次rank、hash、跳過原因與衝突窗引用。
7. 任一stratum不足、無合法候選、映射不符或不足16窗就停止；不跨來源／場次／stratum補樣，不按精彩標記重選。

## 時間與證據

優先沿用manifest整數毫秒；若只有小數秒，從JSON原數字token直接讀Decimal，乘1000後ROUND_HALF_UP轉整數ms，保留原token及轉換revision，禁止先經binary float。
原candidate核心identity不因觀察窗裁切而變更。核心須非負、start<end；其end_ms不得超過duration_us按ROUND_HALF_UP轉成的毫秒端點；保留可能最多500µs的量化差診斷，不改寫原核心。
觀察中心才取`center_ms=floor((start_ms+end_ms)/2)`，requested為中心±4000ms，actual目標範圍嚴格clip至`[0,duration_us]`。
保留原核心、requested／actual範圍、精確微秒端點、不足8秒原因及requested/actual取樣對照。actual目標窗不等於已解碼影格或native PTS。
日後只有准入registry中的實際frame/support ID、native PTS、time_base、stream start及來源映射可當證據；不得用FPS或相對offset假造PTS。
四分位代表候選中心排序分布，不是等秒數切片；封存候選密度及此限制。短窗看不到開始／結束或接觸就保留unknown，不延長窗湊真值。

## 48槽、封存與重建界線

每窗固定Q1對打、Q2直接可見bounce、Q3完整一球，共48槽；詳見[裁決rubric](reviewer-a/rubric.json)與[匯入契約](IMPORT-CONTRACT.md)。未答／缺題／schema錯仍占原槽位，不縮分母。
真實抽樣包應保存candidate-only母體、來源身分、原row/candidate IDs、選中窗與跳過audit、spec/script/產物hash、空白reference及成本收據，保持私人。
重建必須輸出到新目錄，不覆寫原成果或人工答案；先核凍結母體／spec／script身份，再驗配額、hash minimum、映射、無重疊、端點與48空槽。
這份公開模板沒有上述私有母體、mapping或實際manifest，因此只能重用規則，不能重建原16真窗，也不能把格式檢查當模型／人工品質驗證。
