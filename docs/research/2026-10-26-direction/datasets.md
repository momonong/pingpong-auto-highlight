# 資料集：球、事件與偏好分開准入

查閱日：2026-10-26。下列是官方來源事實與本研究適配判斷；未下載、申請帳號、接受條款或執行模型。

**尚未找到已確認授權、同時具備業餘固定單鏡完整回合、精彩度與本使用者偏好的現成資料。**
推薦分工：BlurBall 核對球偵測；OpenTTGames／Extended 核對事件；QVHighlights／PHD² 借用偏好標註模式。
這是提案，不是已驗證的轉移效果，也不是資料取用批准。

## 六個核心候選

下表的「未知」表示官方來源未明示或本輪未核，不表示允許。code、data、weights 必須各自核對。

| 候選／官方來源 | 規模與監督 | data／code／weights 授權 | 取得成本與未核缺口 | 域差與用途判斷（推論） |
| --- | --- | --- | --- | --- |
| [OpenTTGames／TTNet](https://lab.osai.ai/) | 5 訓練影片＋7 短測試片；4,271 bounce／net／empty events；事件前4／後12影格球座標與 mask | data：CC BY-NC-SA 4.0；原作者官方 code、weights 發布及其授權未定位 | 公開直鏈；影片頁列大小加總約35.397 GB、markup約61.7 MB；下載 bytes／hash 未核 | 1080p／120 fps、相近側面視角；事件稀疏標註不等於全片追蹤，更不等於精彩度 |
| [Extended OpenTTGames](https://github.com/moamal01/table_tennis_data) | 同一批12片，1,457次擊球；球種、lean、feet、結束點／原因；回合數有矛盾 | data及新annotation：CC BY-NC-SA 4.0；LICENSE只指dataset，code未知；獨立weights未定位 | 12個JSON可公開讀；影片沿用上游，不能當另一獨立影片集；annotation體積及全量內容未核 | 最接近回合事件結構；120 fps、無音軌及乾淨場景仍有域差；不是已驗證完整rally intervals |
| [TTStroke-21／ME22](https://multimediaeval.github.io/editions/2022/tasks/sportsvideo/) | ME22：1,155 trimmed clips，另28 untrimmed片約100分鐘；20種擊球＋非擊球及起迄 | data：自訂協議，ME22要求2023-01-30前銷毀；ME22／23 baseline code未找到LICENSE；weights獨立授權未知 | ME22為46.1 GB；ME23說須免費註冊、機構email與兩項協議；現行申請及更新條款未驗證 | player-centered單人動作；不能取代雙人完整回合或普通球偏好負例；到期版本不列現行可自由取用 |
| [BlurBall](https://github.com/cogsys-tuebingen/blurball) | 26來源場景、64,119影格；球中心、可見性、拖影角度／長度與相機校正 | data：paper僅稱來源影片Creative Commons，版本／逐片及annotation完整條款未知；code MIT；weights獨立條款未知 | 官方Nextcloud資料／weights分享頁可開，內容、大小及帳號要求未核；網站CC BY-SA不等於dataset授權 | 固定相機、業餘與專業較貼近；無orange balls，重複影格會影響多影格模型；只支持球／拖影 |
| [QVHighlights／Moment-DETR](https://github.com/jayleicn/moment_detr) | 10,148影片、10,310 queries、18,367 moments；相關2秒clip由3人給5級saliency | annotation：CC BY-NC-SA 4.0；code MIT；YouTube原片另核；checkpoint獨立權重授權未知 | 公開annotation與8 GB feature archive；官方checkpoint頁顯示16.2 MB；raw video大小、下載可用性未核 | Vlog／news、query群體評分；可借moment／saliency契約，不能當桌球或個人偏好真值 |
| [PHD²／PHD-GIFs](https://github.com/gifs/personalized-highlights-dataset) | 13,822 users、222,015 annotations、119,938 YouTube影片；選成GIF的起點／長度及user history | metadata／annotation repo未見LICENSE；官方可重用model code及weights未定位；原片權利另核 | train／test CSV公開，原片依YouTube ID另取；存活率、體積及帳號需求未驗證 | 可借個人歷史條件式ranking模式；不是桌球benchmark，未選區間不能直接當不喜歡 |

## 監督不能互換

| 監督 | 直接支持的問題 | 本專案仍缺的判斷 |
| --- | --- | --- |
| ball detection／tracking | 球是否可見與2D位置；BlurBall另有拖影 | 完整回合、好不好看、使用者是否收錄 |
| event／rally structure | bounce、net、擊球與結束事件 | 可從事件提議區間，但仍需驗證一球是否完整 |
| stroke／action | 前反手、球種與局部姿態 | 某球種或長回合是否精彩 |
| highlight／saliency | query相關性、群體評分或剪輯行為 | 本使用者對業餘桌球的喜好 |
| personal history | 過去選取如何條件化下一次ranking | 未檢視區間的負例與跨影片獨立評估 |

上述區分是依官方標註定義作的研究推論，不是資料集已替本產品完成驗收。

## OpenTTGames／Extended 的矛盾與邊界

[原版 landing](https://lab.osai.ai/)與[TTNet論文](https://arxiv.org/pdf/2004.09927)給的是事件附近標註。
`(-1,-1)` 的明示缺球、缺少一筆frame annotation、來源影格中真的無球，是三個不同狀態。
因此未標影格不可自動當負例，也不能以標註密度宣稱全片球追蹤 coverage。

[Extended論文v1](https://arxiv.org/html/2512.19327v1)與[官方repo](https://github.com/moamal01/table_tennis_data)須保留以下待解項：

- Table 1 rally endings為223 train＋58 test，逐項合計281，但total寫282。
- 正文稱3位left-handed players；Table 1 test寫4，repo的test_3／test_7各1位、test_5兩位；唯一人物或人次尚未釐清。
- 原TTNet的`empty_event`是避免過擬合的空事件目標；Extended部分把它用為擊球接觸時刻，語義不能直接等同。
- Extended建議重新考慮split，原train／test有球種缺席及不平衡；本研究推薦按match／session分組，未實作。

本輪沒有讀完全部JSON、重新計數或自行消除矛盾。
新annotation的[LICENSE](https://raw.githubusercontent.com/moamal01/table_tennis_data/main/LICENSE)明指dataset，不能推定附帶scripts取得軟體授權。
[下載script](https://raw.githubusercontent.com/moamal01/table_tennis_data/main/download_videos.sh)只用來核對上游來源，本輪未執行。

## BlurBall 的優先准入項

[原論文v1](https://arxiv.org/html/2509.18387v1)列22 games／363 clips／51,423 frames訓練，4 games／80 clips／12,696 frames測試。
固定相機且有業餘場景，使它成為球偵測的優先查核對象；此適配仍未在自有影片量測。

其多影格模型對重編碼造成的duplicate frames敏感；使用前須保留原始fps／PTS及標註座標約定。
可見性與拖影標註能回答「看不看得到球」，不提供精彩、球種或個人喜好真值。

- [官方code LICENSE](https://raw.githubusercontent.com/cogsys-tuebingen/blurball/main/LICENSE.md)：MIT，含WASB的NTT copyright。
- [官方資料分享](https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7)：入口可達，未核檔案內容與bytes。
- [官方weights分享](https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC)：未核獨立weights條款與大小。
- [project page](https://cogsys-tuebingen.github.io/blurball/)的CC BY-SA 4.0明指網站，不能代替資料授權。

在data及weights條款未清楚前，保持未准入，不以可點下載或MIT code作替代。

## TTStroke-21 的版本與到期問題

[ME22作者overview](https://arxiv.org/pdf/2301.13576)的trimmed split為807／230／118，untrimmed為16／6／6。
不同release規模不可拼在一起。單人XML的球種、handedness與動作起迄也可能相鄰重疊，不能直接當兩人回合。

[ME22使用協議](https://multimediaeval.github.io/editions/2022/docs/MediaEval2022_UsageAgreement.pdf)之外另有Bordeaux特別條件。
ME22已到期；[ME23官方說明](https://raw.githubusercontent.com/ccp-eva/SportTaskME23/master/README.md)提到後續申請流程，今日能否取得及條款是否改變仍未知。
[ME22 baseline](https://github.com/ccp-eva/SportTaskME22)與較早[CRISP](https://p-emartin.github.io/CRISP/)的授權不能互相套用。
本研究不把歷史賽事註冊流程視為目前已取得使用權。

## 偏好模式：QVHighlights 與 PHD²

[QVHighlights論文](https://proceedings.neurips.cc/paper/2021/file/62e0973455fd26eb03e91d5741a4a3bb-Paper.pdf)區分moment retrieval與saliency。
[官方data README](https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/README.md)定義query相關2秒片段、3人與0–4評分。
可借其分離「有關」與「值得收錄」的契約，不能把群體評分直接當個人桌球偏好。

[annotation LICENSE](https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/LICENSE)與[code LICENSE](https://raw.githubusercontent.com/jayleicn/moment_detr/main/LICENSE)分開。
[原補充材料](https://proceedings.neurips.cc/paper/2021/file/62e0973455fd26eb03e91d5741a4a3bb-Supplemental.pdf)對YouTube原片的研究用途論述，不等於影片已同受CC授權。
[checkpoint metadata](https://github.com/jayleicn/moment_detr/blob/main/run_on_video/moment_detr_ckpt/model_best.ckpt)不是獨立weights准入證明。

[PHD²原論文](https://arxiv.org/pdf/1804.06604)與[官方CSV schema](https://github.com/gifs/personalized-highlights-dataset)提供user_id、歷史與最後影片的模式。
只看見被選成GIF的片段，沒有證明該使用者看過並不喜歡所有其他區間。
因此可借偏好資料結構，不能由缺標建立普通球負例；metadata授權缺口也仍保留。

## 本專案資料契約（提案）

自有影片繼續固定source hash、來源時間、session group與人工紀錄來源。
56個精彩正例是歷史編輯區間，不是所有回合的精確邊界；40個AI nonplaying也不是普通球的人工偏好負例。
新增標註至少分開回合有效性／完整性、精彩偏好、coverage、unknown及標註者身份。
AI代標只作弱證據；同一模型自審不能變成獨立GT。

任何公開資料的後續取用，須先記錄具體release、逐資產條款、來源清單、體積／hash、split與預算。
取得合法資料不代表已在自有來源泛化；小樣本域差檢查、訓練與新場次驗收是另外的授權階段。
來源頁可能更新，實驗需再固定可重現版本；以上只代表2026-10-26查閱結果。

回到[方向結論](README.md)；工具與模型見[架構](architecture.md)，比較方式見[實驗](experiment.md)。
