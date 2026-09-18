# 固定回合剪輯實驗

本文件只涵蓋固定候選的剪輯、試看與回饋套用，不包含全片搜尋或精彩排名。

## 固定回合的事件邊界、節奏與合輯比較（2026-09-18，任務 B）

**完成可重跑試剪與實際媒體核對，尚未取得新真人驗收。** 本輪不研究搜尋或排名，
固定 `candidate-family-v3` 第一來源 X 的六個 ID 與原順序。來源完整 SHA 為
`414c278b38ce4de7bce5a9df3ab65ee7f335aec198e5d5443620e616fc5251ad`。
主目錄與另一任務的工作皆保留，修改位於 `.worktrees/rally-edit-comparison`，分支
`codex/rally-edit-comparison`；HEAD／起始版本 `58ae3a3`，另帶入 88 個既有未提交依賴檔案的
逐檔 SHA 快照（`data/dependency-snapshot.json`），沒有提交他人的原型。

### 來源、人工標記與版本辨識

重新計算五支原片完整 hash，全部吻合復原 dataset；主 DB 的 56 筆標記逐欄吻合
`windows-import-20260917/dataset.json`，沒有新增、變更或刪除。原片與 DB 只讀。
人工契約只有 `human-selected point interval`，無法區分精確事件、偏好剪輯或粗略精彩
區間，因此 **annotation_time_semantics = unknown**；不把原始標記改造成影格真值。
本批只有第六段重疊已知人工正例 `cd934039319e4f218cb3ca8334f290ba`，
原標記 **140.500–152.900、相持** 原樣保存。其餘五段來自已選 X，沒有補成五筆人工正例。
未擴大到其餘 55 筆人工正例，先等待這批代表案例的完整性與節奏核對。

實際 manifest／媒體 hash 核對見 [prior-media-audit](evidence/rally-edit-v1/prior-media-audit.json)：

| 試看包 | 來源 | 字母的實際內容 |
|---|---|---|
| quality-v1 blind-packet | `636c4bb3…` | X = multi-signal；Y = baseline |
| candidate-family-v3 source-1 | `414c278b…` | X = union-family 六段；Y = union-raw 九段 |
| agentic v2 P | `414c278b…`，候選 b28 | 原片 137.492–151.492，14 秒 |
| agentic v2 P+ | 同 P，核心不變 | 原片 137.492–152.492，15 秒 |

P 的原文確認「看得出最後一球結束，但突然」，不等於發球完整；P+ 無已綁定的真人
評分。「好一點，但略急」保留為本次使用者提供的觀看偏好，未擅自綁定 P+。X 首段
太拖、另有一段太急仍保留原回饋，未將未指明的段號擅自填成第四段。

### 原片重看與分離欄位

先看原片 2 fps 前後文，再對發球／結束提高至 8 fps。表內時間是**助手視覺觀察的
不確定區間**，不是人類標註，也不是精確球拍接觸、落桌或勝負判定。8 fps 只代表取樣
間隔 0.125 秒，不等於邊界精度；看不到的精確終拍與勝者保持 null。

| ID | 舊播放範圍 | 開始不確定範圍 | 結束不確定範圍 | 試剪採用範圍 | 原片觀察 |
|---|---|---|---|---|---|
| b2 | 28.250–35.000 | 27.625–28.125 | 31.125–31.500 | 26.750–32.500 | 原入點在發球動作後，原尾端延續走離球桌 |
| v4-3 | 44.375–50.692 | 44.875–45.375 | 48.625–49.125 | 維持原範圍 | 有發球準備與結束，先保留為未改的對照 |
| v4-6 | 78.908–88.375 | 77.250–77.875 | 81.875–82.500 | 76.250–83.500 | 原入點漏發球，原尾段進入下一分準備 |
| b15 | 89.892–93.644 | 90.875–91.375 | 93.625–94.125 | 89.892–95.000 | 原出點在最後攻防轉為停止的過渡中 |
| v4-9 | 120.828–133.875 | 119.625–120.125 | 124.125–124.750 | 118.500–126.000 | 漏發球，且約133秒另一分已發球；需排除第二分 |
| v4-11 | 141.500–156.625 | 141.875–142.375 | 150.625–151.250 | 140.500–152.900 | 相持後停止與走動可見；收尾長度仍需偏好判斷 |

v4-9 以原核心開始所屬回合為剪輯對象，原候選可能跨分的缺陷明列，不更動其原始 core、
ID、分數或候選家族。未將第二分新增為收錄項。第三、四段在原片有重新準備／發球，
本輪觀察支持它們是分開的事件，不自動合併。

[manifest](../examples/rally-edit-v1.json) 分開保存 original_candidate_range、原播放範圍、
event 的不確定範圍、提出待比較的可接受入出點範圍、human_acceptable_render（目前 null）、
實際 selected_render、理由、原始人工標記、回饋與版本。事件範圍和呈現範圍不互相覆寫。
第二段維持舊剪法，雖另記較短尾段假說，尚未自動套用。

### 少量單因素實際比較

`data/experiments/rally-edit-v1/review-01/index.html` 沿用本機 review pack 形式，不接正式
網站或 DB。六組共十二支比較片，方法名稱隱藏，數字代號使用固定 seed，對照先播／
變體先播各三組，提供反向順序。這是降低期待偏差的平衡呈現，**不是已完成的真人盲測**。
原片、已見過的動作及片長仍可能洩漏版本；展開「來源時間與差異」會解除部分遮蔽。

1. b2：出點 35 固定，只比較 28.25／26.75 入點。
2. b15：入點 89.892 固定，只比較 93.644／95 出點。
3. v4-9：出點 133.875 固定，只比較 120.828／118.5 入點；兩片故意保留舊污染尾段供核對。
4. v4-9：入點 118.5 固定，只比較 133.875／126 出點，隔離下一分污染。
5. b2：修正入點 26.75 固定，只比較 35／32.5 尾段。這是移除已觀察空檔，非依時長上限截球。
6. v4-11：入點 140.5 固定，只比較 152.1／152.9 收尾，差 0.8 秒。

完整性選項為「完整／不完整／無法判定」，節奏為「太急／合適／太拖」，各自可留待確認。
六支採用切點片另有完整性欄位，皆確認後才展開合輯偏好，避免用好看的轉場掩蓋事件問題。
回饋按瀏覽器輸入追加時間戳，匯出綁定 plan SHA、媒體 SHA、播放順序與版本；沒有自動
匯入原標記。此次沒有填入任何新真人答案。

### 固定切點後的合輯呈現

- K：原六段順序、直接切換，**44.400 秒**。
- M：相同六段、順序、切點，**0.2 秒 dissolve + 同長度線性音訊 crossfade**，**43.400 秒**。
- N：相同六段與切點、直接切換，只把長相持 v4-11 移到最前，其餘保持原序，**44.400 秒**。

K/M 只比較轉場；K/N 只比較順序，沒有改精彩排名。試剪全批的節奏尚未通過真人，
所以三份合輯是供下一階段核對的預覽，不是已接受成品。

M 五處重疊皆逐筆核對，落在事件不確定包絡之外：前片結束後／後片發球前。
實際中心影格已檢视，短 dissolve 有明顯人物重影，這是風格取捨，不能預設較流暢。
聲音在重疊處同時混合兩段背景音；訊號連續不等於聽覺舒適，轉場偏好仍待真人。
時長減少1秒完全來自五處0.2秒重疊。保留全部六段；55秒仅作超出回報，不截必要內容。
未提出「單回合越短越好」或一律增加固定尾秒的規則。

### 實際媒體與工程驗證

- 27支MP4（12比較、6上下文、6採用切點、3合輯），總輸出393.000秒。
  30fps H.264/yuv420p、AAC、原比例640×360；原片不變。
- 全27支完整影音解碼通過、A/V起點及時長核對。輸出切點向外量化到30fps格點，
  兩端各最多1影格；採用六段理論44.325秒，實際44.400秒，未悄悄把量化差當原片事件時間。
- 18份比較／採用切點影片的首末實際解碼影格對照原片，容許搜尋±1影格，灰階RMSE最大2.262/255。
  原音與成品8kHz mono解碼相關係數最低0.99428，無靜音缺軌；這是訊號核對，不是真人聽感評分。
- 六段成品首尾與五個dissolve中心影格已直接檢視。解碼成功、聲音相關與影格對應均不當成真人完整率。
- **187項測試通過，19.30秒**，含11個新增測試（事件包絡、下一分、固定集合、單因素、
  不因預算截球、dissolve保護及真正FFmpeg影音時鐘整合測試）。Ruff、diff check通過。
  沙箱完整回歸曾停滯，僅中止自己的測試後在主機以120秒上限重跑通過。
- Codex內建瀏覽器可顯示表單，但在啟動MP4播放時崩潰；按需載入後再試仍重現。
  因此**瀏覽器影片播放、回饋暫存／匯出端到端仍未驗收**，沒有以DOM出現冒充播放成功。
  可用本機播放器直接打開MP4；完整解碼核對與瀏覽器問題分開記錄。暫時loopback測試server收尾停止。
- 使用CPU FFmpeg；零模型呼叫、零第三方影片上傳。

可支持的規則：事件邊界與剪輯邊界必須分開；起點缺失不能靠收尾修好；低motion不保證
自然收尾；下一分需明確排除；轉場不得蓋住事件包絡。**各片可接受入出點、最佳尾長、
直接切換／dissolve與順序偏好仍需逐片真人判斷**。只在一個development來源觀察，未計
整批真人接受率、held-out品質或其餘55筆正例的改善。

### 重跑與交接

在本worktree執行，原片唯讀共用、每次使用新的輸出目錄：

```bash
uv sync --locked --extra dev --no-default-groups
uv run --locked --extra dev --no-default-groups python -m pingpong_highlight.edit_comparison \
  --plan examples/rally-edit-v1.json \
  --source ../../data/uploads/a6ef1ae5b64d4f62b19bf2d850b9282b.mp4 \
  --output data/experiments/rally-edit-v1/repeat-01
uv run --locked --extra dev --no-default-groups python scripts/verify-edit-media.py \
  --packet data/experiments/rally-edit-v1/repeat-01/packet.json \
  --source ../../data/uploads/a6ef1ae5b64d4f62b19bf2d850b9282b.mp4 \
  --output data/experiments/rally-edit-v1/repeat-validation-01
```

輸入plan是本輪可核對的人工策展／助手觀察試剪設定，不是從任意新片自動找切點的算法。
只要變更事件或剪輯假說，另存新版本；人類評分綁定舊plan hash，不遷移成新版本接受。
原型不依賴另個任務A的新候選，未將其較新搜尋結果帶入本輪固定比較。

初輪新增檔案：`edit_comparison.py`、`scripts/verify-edit-media.py`、對應測試與
`examples/rally-edit-v1.json`、[精簡證據](evidence/rally-edit-v1/)。更新這份既有報告、README、
architecture、evaluation。其餘未提交檔是起始快照依賴，不是本輪新增成果。
所有成果未提交，未合main／推送／發布／部署。新worktree保留作看片與後續整合，
占用自己的工作分支，不占用main；既有worktree未清理。整合時先保存既有原型，再只取B的
增量與文件補記，不直接把整份包含88個依賴的工作目錄當B提交。

### 試看流程修復（review UI v2，2026-09-18）

沿用同一份 `review-01`，27支MP4、plan、切點與順序均未重製或改寫。
新增 `pingpong_highlight.edit_review`，直接讀取既有 pack，在啟動時核對 plan 與全部媒體 SHA。
共用一個播放器、按版本按鈕才載入；切換試看項目清除上一段畫面，避免誤評。
提供重播、暫停、前後1秒與標準影片控制列。HTTP媒體支援Range 206。
單播放器加Range後，同一份H.264/AAC已可在Codex內建瀏覽器播放；
**舊頁崩潰的確切根因未單獨證實**，不把改善歸因為影片損毀或GPU問題。

正式回饋另存在ignored `data/experiments/rally-edit-v1/review-ui-v2/human-feedback.json`。
不修改舊pack的 `human-feedback.json`、56笔原始人工標記、任務A的候選或正式DB。
評分採追加歷史，完整性／節奏／轉場／順序分開，含原片與plan/media hash、播放順序、時間與actor。
自動保存成功才顯示「已保存」；失敗保留待存佇列並可重試，以相同entry ID防止重複寫入。
暫存佇列另存瀏覽器localStorage，成功後清除；後端以fsync及原子替換保存。
反向順序亦保存並於重載恢復。匯出提供完整已保存JSON；未保存時隱藏匯出入口。
六段採用切點全部明確評為完整才開放合輯偏好；若撤回某段，重新關閉，保留舊偏好歷史但
`compilation_eligible=false`，不能把舊轉場偏好解讀為當前已通過。

工程測試使用獨立8877與 `automation-fixture.json`，所有操作強制標記為fixture。
正式入口8876使用獨立cookie與回饋檔；只bind loopback，不是部署／公開服務。
每份回饋檔限一個服務程序使用。本輪未實作多人合併、匯入旧版localStorage或跨裝置同步。

實際驗證：

- 同一內建瀏覽器，306播至6.7667秒結尾、778可播放／重播／暫停，畫面可辨識；
  暫停後向前定位由2.7到3.7秒。合輯M播至43.4秒結尾。
- 分開評分、備註、反向順序、重載還原、完整性gate開啟與撤回、匯出下載事件均確認。
  暫停**自己的fixture服務**後輸入備註，頁面明確報保存失敗；恢復服務再重試成功，
  磁碟保持14筆fixture紀錄，未重複。正式檔於交付核對仍0筆、全待確認。
- 18項受影響測試通過（既有剪輯11、新試看服務7），2.07秒；Ruff與diff check通過。
  新測試涵蓋Range／來源限制、actor隔離、保存續讀、重試冪等、gate撤回與磁碟失敗。
  首次磁碟失敗測試的TestClient例外設定方式錯誤，修正測試建構參數後通過；未掩蓋服務錯誤。
  本次不把先前187項全回歸結果重算為新全回歸。
- 主目錄88個依賴快照hash仍一致。本輪0GPU、0模型呼叫、0第三方媒體上傳。

瀏覽器工程流程已可用，不等於真人完整率或聽感驗收；未逐支做瀏覽器播放，
其餘素材沿用前輪全27支FFmpeg解碼、來源影格／音訊核對。真人完整性、節奏與轉場偏好仍待確認。
第一輪先看「比較1、2」，再依回饋處理代表片段，不先把六段擴大到全部正例。

在本worktree啟動（環境已依uv.lock建立）：

```bash
UV_CACHE_DIR=/tmp/rally-edit-uv-cache uv run --locked --no-sync \
  -m pingpong_highlight.edit_review \
  --pack data/experiments/rally-edit-v1/review-01 \
  --feedback data/experiments/rally-edit-v1/review-ui-v2/human-feedback.json \
  --port 8876
```

瀏覽 `http://127.0.0.1:8876/`。原render產生的舊靜態index.html保留作歷史版本，
後續看片使用上述入口。精簡工程證據見 `docs/evidence/rally-edit-v1/review-ui-v2.json`。
本輪仍在 `.worktrees/rally-edit-comparison`、`codex/rally-edit-comparison`，HEAD維持58ae3a3；
新增server、靜態資產、7項測試及文件均未提交。8876保留供使用者試看，8877fixture服務已停，
worktree保留；未合main、推送、發布、部署或清理任何既有worktree。

### 第一筆剪輯真人回饋：b2 入點與尾長（2026-09-18）

使用者依序看306、778後表示：

> 306 開頭有點太急 開頭的時候就已經在拋球了
> 778 我覺得開頭的節奏剛剛好 不過結尾有點太拖 可以早點結束

原話、候選／来源／plan／媒體hash與時間碼保存於
[evidence/rally-edit-v1/human-chat-feedback-01.json](evidence/rally-edit-v1/human-chat-feedback-01.json)。
這是使用者聊天回饋，由助手轉錄；不冒充browser-input或自動fixture。
306要求入點28.25秒（實際28.2333）被評為略急；778要求入點26.75秒（實際26.7333）
被評為合適，35秒出點被評為太拖。只能支持本段這個入點樣本，不能推成可接受時間區間，
也不能推成固定多留1.5秒的全域規則。原事件時間、原始人工標記與凍結manifest未修改。
兩片整體完整性與整體節奏沒有被明確評定，維持null；真人沒有指定精確新出點。

下一輪沿用既有740（原比較5）：入點與778完全相同，只將出點改為32.5秒，
實際全長5.766667秒。這是既有助手試剪假說，不是使用者已核准的終點。
本輪重新核對306／778／972／740媒體hash，確認972與778相同、740與point-1相同，
740全影音解碼通過，重看已驗證首末影格；前輪原音相關係數0.99727。
交付740供判斷終拍／結果是否清楚及收尾節奏，仍未擴大到其他候選或合輯。
聊天紀錄另存，不改寫運行中的browser回饋檔；該表單的整體評分仍待確認。

### b2 尾段微調：740 → 582（2026-09-18）

使用者對740回覆「可以 都差不多 結尾可以稍稍慢一點點點」，原話與媒體綁定保存於
`evidence/rally-edit-v1/human-chat-feedback-02.json`。這是大致可接受、仍想多留少許尾段的回饋；
並未指定精確出點，也未逐項確認完整性。保留原話，不換算成整批完整率。

固定同一b2與入點26.75秒，由助手提出將出點32.5微增至32.7秒（增加0.2秒、6影格），
播放速度與原音不變。新token582，实际26.733333–32.7秒，長5.966667秒，audio5.966秒。
事件不確定區間保持原紀錄，入點只保留已獲偏好支持的樣本，不自動擴成可接受時間區間。
新出點仍待真人；未將單片回饋推成全部加0.2秒的規則，未改舊plan／六段合輯或其他候選。

新影片全影音解碼、A/V起訖、來源首末影格與原音核對通過（audio相關0.99719）；
首末拼圖已檢視。精簡收據為`evidence/rally-edit-v1/tail-refinement-01.json`。
ignored本機`data/experiments/rally-edit-v1/tail-refinement-01/`保存影片、packet、完整manifest
及validation；manifest保留實際FFmpeg command，重跑時只需將最後輸出路徑改成新檔，
或使用既有uv管理的`edit_comparison.encode(source, new_target, [26.75, 32.7])`。
本輪未修改程式碼，僅增量回饋／試剪／文件，不重跑無關全套測試。0 GPU／模型，尚未提交。

### 接受582後的決策套用與串接修正（2026-09-18）

使用者對582明確表示「有 我覺得這樣很完美 我們應該可以直接繼續開發了嗎」。
原話與exact media hash保存於`evidence/rally-edit-v1/human-chat-feedback-03.json`，
記為**此支影片的剪輯呈現已接受**。它不是新的精彩收錄標記、精確事件邊界真值，
也不推為所有回合的接受率。此前「0筆新真人回饋」是當時快照，現在已有3筆聊天回饋。

新增 `edit_decisions.py` 與 `examples/rally-edit-decisions-v2.json`：

- 決策綁定parent plan SHA、來源SHA、candidate ID、媒體SHA、要求／實際切點與原話紀錄。
  舊版本、不同來源、重複ID、fixture actor、回饋與媒體不符都拒絕套用。
- 只更新指定candidate的採用切點與呈現接受紀錄；保留原事件不確定區間、原候選、
  人工原始標記與其餘五段。接受一個切點樣本不擴張成可接受時間區間。
- 實際影片以copy保留已接受單段的bytes；另產固定六段、固定原序、直接切換的合輯預覽。
  原有27支媒體與v1 plan不改寫。新預覽不是原`edit_review`的完整比較pack，勿以該UI CLI載入。
- 預覽紀錄有`presentation_accepted_count=1`、逐段待確認状态及獨立event completeness欄位。
  五段未評分不會因第一段獲肯定而通過。55秒不足仍報超出，不剪必要內容。

實際串接驗證發現並修復AAC padding問題：舊concat讓解碼音訊尾端補樣影響segment長度，
後半接點發生約一影格偏移。第一份`accepted-preview-01`雖可完整影音解碼，
逐影格比對發現b15、v4-9、v4-11接點不一致（RMSE最高49.447／255），
已另存`superseded.json`標為失敗、保留證據，不能交作已通過成品。

`edit_comparison.reel` 現以每段decoded frame count建立精確30fps時計：影像trim到影格數，
音訊resample48kHz後apad／atrim到對應樣本數，再重設timestamp並串接。
`verify-edit-media.py`增加直接切換合輯的逐影格首末驗證，依decoded frame index取樣，
避免用浮點秒數seek跨到相鄰影格。新AAC補樣回歸測試使用6段非AAC block整數長度的合成片，
逐段檢查首末影格及總影格數；fixture不當真人評分。

修正版`accepted-preview-02`：

- 六段／原序／直接切換，**44.600秒、1338影格、audio44.600秒**；無55秒超出。
- 第一段與582 bytes相同，其餘五段與v1 selected clips bytes相同。
- 全部輸出完整影音解碼；六段首末原片對應／音訊核對通過，audio相關最低0.99454。
- 合輯12個segment端點在精確frame index對照單段，RMSE最高1.861／255；首尾與第一接點拼圖已檢視。
- **18項受影響測試通過，4.52秒**，包括6個決策套用案例與12個剪輯案例；Ruff、diff check通過。
  沒有重跑不受影響的網站全套。這是工程時鐘／內容核對，不是新合輯的真人節奏或音訊聽感驗收。

可重跑（使用新的output目錄）：

```bash
UV_CACHE_DIR=/tmp/rally-edit-uv-cache uv run --locked --no-sync \
  -m pingpong_highlight.edit_decisions \
  --pack data/experiments/rally-edit-v1/review-01 \
  --decisions examples/rally-edit-decisions-v2.json \
  --output data/experiments/rally-edit-v1/accepted-preview-repeat
UV_CACHE_DIR=/tmp/rally-edit-uv-cache uv run --locked --no-sync \
  python scripts/verify-edit-media.py \
  --packet data/experiments/rally-edit-v1/accepted-preview-repeat/packet.json \
  --source ../../data/uploads/a6ef1ae5b64d4f62b19bf2d850b9282b.mp4 \
  --output data/experiments/rally-edit-v1/accepted-preview-repeat/validation
```

精簡收據見`evidence/rally-edit-v1/accepted-preview-v2.json`與`decision-pytest.xml`。
下一輪可把其他逐片接受紀錄加入獨立決策版本，而不重新定義原候選或覆寫前輪影片。
本輪不要求補標56筆原資料；先沿既有資料與固定六段往下開發，不從單球歸納固定尾長。

交接仍為`.worktrees/rally-edit-comparison`／`codex/rally-edit-comparison`，HEAD58ae3a3；
新增／修改程式、測試、決策與文件均未提交。0 GPU／模型、無新port；8876原試看頁保持原pack。
不合main、不推送、不部署、不清理worktree。主目錄與任務A資料均未修改。


## 本機提交與整合

本文件從共享未提交研究報告中拆出任務B章節，以便獨立提交；其他任务的研究報告與程式不在此提交。
各節中的「未提交」是當時的實驗快照。最終提交與合併狀態以Git及交接為準。
大型影片、原片、DB、完整命令與本機驗證目錄均留ignored data；Git包含可重跑程式、固定設定與精簡證據。
乾淨checkout可執行工程測試；重播真实媒體需要本機唯讀原片和既有review pack，
`rally-edit-decisions-v2.json`也需要它引用的本機render manifest，不假裝Git已包含媒體。

使用者後續對44.6秒預覽回覆「這個感覺不錯 可以先合併和提交」，記於`human-chat-feedback-04.json`。
這是合輯呈現的正面回饋與本機提交／合併授權，不自動轉成六段完整性逐項通過。

合併前在main 58ae3a3＋任務B精確暫存內容的乾淨worktree驗證：154項完整測試通過（21.62秒）、Ruff與diff check通過。
此數量不同於早期含未提交搜尋原型的187項；其他任務程式與測試未被納入B提交。詳見`merge-validation.json`。
