# 公開DEMO人工空表

本資料夾是去識別示例：CASE-DEMO-01..16完全沒有來源／私有窗映射，沒有媒體，不能用來重建或評分原研究。每case三題，共48槽；兩人答卷最終仍是48個reference，不能相加為96。
answers.csv保持原空表，試填另存新檔。assignment.json列精確allowlist／順序；rubric.json列Q1–Q3固定規則。
answer可為yes/no/unknown，未答保持真正空白；兩個證據欄初始[]。unknown須人工實際觀察後記具體原因，不能代替沒看或未答。
yes/no均需真實可核support及範圍；no需足夠範圍支持不存在。相對微秒只作定位，不能當native actual PTS；沒有准入媒體registry就維持pending_media，不發布reference。
兩證據欄用標準CSV quoting保存JSON arrays，按index成對，時間為0<=start<end<=8000000的integer，不接受bool或float。
真正研究須另行准入媒體並分發獨立材料；reviewer不看對方、模型答案或來源對照，原答另存不可覆寫。這份公開空表不證明已安排真人、人員ACL或盲測。
完整規則見[匯入契約](../IMPORT-CONTRACT.md)。
