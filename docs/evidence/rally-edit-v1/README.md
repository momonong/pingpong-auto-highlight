# 固定回合剪輯比較 v1

詳見[既有實驗報告](../../rally-edit-experiment.md)。

- `prior-media-audit.json`：兩套X/Y與P/P+實際manifest、媒體hash核對。
- `protected-inputs.json`：五支原片與兩份DB前後完整hash；56筆人工標記未變。
- `comparison-manifest.json`：凍結候選、事件不確定範圍、可接受切點假說、採用切點、原始人工註記與真人待確認欄位。
- `summary.json`：實際比較媒體hash、來源／合輯時間映射、轉場保護、成本範圍與限制。
- `media-signal-validation.json`：18支單段輸出首末影格與原音核對、3支合輯音訊接點。不是人類觀看或聽感評分。
- `code-receipt.json`、`pytest.xml`：本輪執行／交付版本、187項工程測試。渲染後僅更新驗證上限與播放器表單，切點／編碼不變。

本輪仍未取得新真人評分。原片2fps上下文、8fps邊界觀察、27支MP4與review pack留在
本worktree ignored `data/experiments/rally-edit-v1/`。事件觀察來自助手，不是使用者標註。
後續review UI v2已完成單播放器播放、定位、保存／重載／失敗重試／匯出與gate撤回驗證，見`review-ui-v2.json`；未聲稱舊頁崩潰根因已確定，fixture仍不等於真人驗收。
不發布原片、影格或本機檔案路徑；没有第三方模型上傳。

後續已收到第一筆真人聊天回饋，見`human-chat-feedback-01.json`：306入點略急；
778入點合適、尾段太拖。這不等於整段完整性通過。下一支740固定778入點，只比較出點，待真人判定。

`human-chat-feedback-02.json`記錄740「可以，都差不多，但尾段想再慢一點」；
`tail-refinement-01.json`為助手試剪582的收據：同入點、正常速度多留0.2秒，影音驗證通過，真人仍待判斷。

`human-chat-feedback-03.json`保存582「很完美」的逐片呈現接受。
`accepted-preview-v2.json`及`decision-pytest.xml`記錄決策套用、AAC補樣接點修復與44.6秒新預覽；
1段呈現接受、5段待確認，沒有新增精彩收錄或整批完整性宣告。舊收據保留當時狀態，不改寫歷史。

`merge-validation.json`與`merge-pytest.xml`是僅main＋任務B增量的乾淨驗證：154項全測試通過；不包含其他任務未提交原型。
