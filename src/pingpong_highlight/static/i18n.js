(() => {
  "use strict";

  const storageKey = "highlightcraft-language";
  const catalog = Object.freeze({
    "meta.title": ["HighlightCraft — 桌球精彩集錦", "HighlightCraft — Table Tennis Highlights"],
    "meta.description": ["把桌球比賽影片自動剪成逐分精彩集錦的本機工具。", "A local tool that turns table tennis match footage into point-by-point highlight reels."],
    "brand.home": ["HighlightCraft 首頁", "HighlightCraft home"],
    "service.online": ["服務運作中", "Service online"],
    "language.currentChinese": ["中文介面", "Chinese interface"],
    "language.currentEnglish": ["英文介面", "English interface"],
    "language.switchChinese": ["切換至中文", "Switch to Chinese"],
    "language.switchEnglish": ["切換至英文", "Switch to English"],
    "nav.guide": ["使用教學", "How to use"],
    "nav.logout": ["登出", "Log out"],
    "account.defaultName": ["使用者", "User"],
    "login.titleLead": ["你的每場球，", "Every match has"],
    "login.titleAccent": ["都有自己的位置。", "a place of its own."],
    "login.lead": ["登入後上傳的原片、處理進度與剪輯成品，都只會收進你的個人影片庫。", "Your uploads, processing progress, and finished reels stay together in your private video library."],
    "login.kicker": ["SIGN IN / 登入", "SIGN IN / MEMBER ACCESS"],
    "common.username": ["帳號", "Username"],
    "common.password": ["密碼", "Password"],
    "login.submit": ["進入我的剪輯室", "Enter my studio"],
    "login.help": ["帳號由管理員建立；若忘記密碼，請聯絡提供這套系統的人。", "Accounts are created by an administrator. Contact the person who provided this system if you forget your password."],
    "hero.titleLead": ["把整場球，", "Turn a full match into"],
    "hero.titleAccent": ["剪成值得重播的幾分。", "points worth replaying."],
    "hero.lead": ["手機直接傳原片，電腦自動找出每一分、挑選精彩球，再用原始比例剪成一支流暢集錦。", "Upload the original video from your phone. The computer finds each point, selects the best rallies, and edits a smooth reel in the original aspect ratio."],
    "hero.featuresLabel": ["產品特色", "Product features"],
    "hero.featurePoints": ["逐分切點", "Point detection"],
    "hero.featureAspect": ["原尺寸輸出", "Original aspect ratio"],
    "hero.featureLocal": ["本機處理", "Local processing"],
    "upload.title": ["加入一段比賽影片", "Add match footage"],
    "upload.lead": ["從手機直傳，或貼上 Google Drive 公開連結。", "Upload from your phone or paste a public Google Drive link."],
    "upload.privacy": ["本機剪輯", "Edited locally"],
    "upload.filePrompt": ["點一下，從相簿選擇原片", "Tap to choose the original video"],
    "upload.fileMeta": ["也可以把 MOV、MP4、HEVC 影片拖到這裡", "You can also drop a MOV, MP4, or HEVC video here"],
    "upload.start": ["開始上傳與製作", "Upload and create highlights"],
    "transfer.preparing": ["準備中", "Preparing"],
    "transfer.creatingSession": ["正在建立續傳工作階段", "Creating a resumable upload session"],
    "transfer.pause": ["暫停", "Pause"],
    "transfer.resume": ["繼續", "Resume"],
    "drive.divider": ["或交給電腦從雲端下載", "Or let this computer download from the cloud"],
    "drive.background": ["背景下載，關閉此頁也會繼續", "Downloads in the background, even after you close this page"],
    "drive.urlLabel": ["Google Drive 影片共用連結", "Shared Google Drive video link"],
    "drive.placeholder": ["貼上 drive.google.com 影片連結", "Paste a drive.google.com video link"],
    "drive.start": ["開始匯入", "Start import"],
    "drive.hint": ["共用設定請選「知道連結的任何人」且權限為「檢視者」", "Set sharing to “Anyone with the link” and the role to “Viewer”."],
    "upload.onlineTitle": ["上傳時請保持這台電腦在線", "Keep this computer online while uploading"],
    "upload.onlineBody": ["開始處理後可以關閉此頁，電腦會繼續工作。", "Once processing starts, you may close this page and the computer will keep working."],
    "library.title": ["我的影片庫", "My video library"],
    "library.waiting": ["等待第一支影片", "Waiting for your first video"],
    "common.refresh": ["重新整理", "Refresh"],
    "library.emptyTitle": ["上傳進度與第一支集錦會出現在這裡", "Upload progress and your first reel will appear here"],
    "library.emptyBody": ["傳送中可跨裝置監看；完成後可直接預覽、下載，或用手機分享。", "Follow progress from another device, then preview, download, or share the finished reel from your phone."],
    "guide.kicker": ["QUICK START / 使用教學", "QUICK START / HOW TO USE"],
    "guide.title": ["第一次使用？四步完成你的精彩集錦", "New here? Make your highlight reel in four steps"],
    "guide.expand": ["展開教學", "Open guide"],
    "guide.step1Title": ["選擇原始影片", "Choose the original video"],
    "guide.step1Body": ["點選上傳區從手機相簿挑選 MOV、MP4 或 HEVC；也可以貼上已開啟「知道連結的任何人」檢視權限的 Google Drive 連結。", "Choose a MOV, MP4, or HEVC video from your phone, or paste a Google Drive link shared with “Anyone with the link” as a viewer."],
    "guide.step2Title": ["等待上傳完成", "Wait for the upload"],
    "guide.step2Body": ["直傳期間請保持頁面與這台電腦在線；中斷後重新選同一個檔案，就會從已收到的位置續傳。", "Keep this page and computer online during a direct upload. If interrupted, choose the same file again to resume where it stopped."],
    "guide.step3Title": ["讓系統背景剪輯", "Let the system edit in the background"],
    "guide.step3Body": ["看到「已排入佇列」後就能關閉頁面。系統會分析每一分、挑選精彩回合，再輸出一支 Reel。", "You may close the page after the video is queued. The system analyzes every point, selects the best rallies, and creates a reel."],
    "guide.step4Title": ["回到影片庫取件", "Return to your library"],
    "guide.step4Body": ["登入同一帳號即可查看自己的所有影片。完成後可預覽 Reel、下載 MP4，或展開單分片段與分析檔。", "Sign in with the same account to see all your videos. Preview the reel, download the MP4, or open individual clips and analysis files."],
    "guide.privacyLabel": ["隱私提醒：", "Privacy:"],
    "guide.privacyBody": ["每個帳號只看得到自己的內容；請勿共用帳號。管理員為維護儲存空間，可查看與移除所有使用者的影片。", "Each account can see only its own content, so do not share accounts. Administrators can view and remove all users’ videos to manage storage."],
    "guide.languageLabel": ["語言切換：", "Language:"],
    "guide.languageBody": ["右上角的「中 / EN」可隨時切換繁體中文與英文；重新整理或下次開啟時會沿用你的選擇。", "Use the language switch in the top-right corner at any time. Your choice is kept after a reload or on your next visit."],
    "account.kicker": ["ACCOUNT / 帳號安全", "ACCOUNT / SECURITY"],
    "account.changePassword": ["更改我的登入密碼", "Change my password"],
    "account.openSettings": ["開啟設定", "Open settings"],
    "account.currentPassword": ["目前密碼", "Current password"],
    "account.newPassword": ["新密碼", "New password"],
    "account.confirmPassword": ["再次輸入新密碼", "Confirm new password"],
    "account.updatePassword": ["更新密碼", "Update password"],
    "admin.title": ["使用者與資料管理", "Users and data"],
    "admin.lead": ["管理試用帳號、所有人上傳的原片、處理狀態、剪輯成品與主機容量。", "Manage trial accounts, uploaded source videos, processing statuses, finished reels, and host storage."],
    "admin.refresh": ["重新整理管理資料", "Refresh administration data"],
    "storage.used": ["已使用空間", "Storage used"],
    "common.loading": ["讀取中…", "Loading…"],
    "storage.sourceAndOutput": ["來源與成品合計", "Sources and outputs combined"],
    "storage.sources": ["原始影片", "Original videos"],
    "storage.waiting": ["等待容量資料", "Waiting for storage data"],
    "storage.outputs": ["剪輯成品", "Edited outputs"],
    "storage.available": ["可用空間", "Available space"],
    "storage.hostDisk": ["主機磁碟", "Host disk"],
    "admin.accounts": ["試用帳號", "Trial accounts"],
    "common.displayName": ["顯示名稱", "Display name"],
    "admin.usernamePlaceholder": ["例如 wang_chen", "For example, alex_chen"],
    "admin.displayNamePlaceholder": ["例如 王小明", "For example, Alex Chen"],
    "admin.initialPassword": ["初始密碼", "Initial password"],
    "admin.passwordPlaceholder": ["至少 8 個字元", "At least 8 characters"],
    "common.role": ["角色", "Role"],
    "common.user": ["一般使用者", "User"],
    "common.admin": ["管理員", "Administrator"],
    "admin.createAccount": ["建立帳號", "Create account"],
    "admin.loadingAccounts": ["正在讀取帳號…", "Loading accounts…"],
    "admin.allVideos": ["所有影片", "All videos"],
    "admin.pendingSources": ["未完成來源", "Incomplete sources"],
    "admin.loadingPending": ["正在讀取未完成來源…", "Loading incomplete sources…"],
    "admin.jobsAndOutputs": ["處理工作與成品", "Processing jobs and outputs"],
    "admin.perPage": ["每頁 20 支", "20 per page"],
    "admin.owner": ["擁有者", "Owner"],
    "admin.sourceAndStatus": ["來源與狀態", "Source and status"],
    "admin.sourceAndOutput": ["原片與成品", "Source and outputs"],
    "admin.loadingVideos": ["正在讀取影片…", "Loading videos…"],
    "admin.previous": ["← 上一頁", "← Previous"],
    "admin.pageOne": ["第 1 頁", "Page 1"],
    "admin.next": ["下一頁 →", "Next →"],
    "annotation.devTitle": ["人工標記精彩球", "Manually label great rallies"],
    "annotation.devLead": ["短期模型優化工具，與上方的手機上傳、處理及成品下載流程分開。", "A model-improvement tool kept separate from the upload, processing, and download workflow above."],
    "annotation.desktopOnly": ["僅限電腦", "Desktop only"],
    "annotation.summary": ["選擇一支已完成影片，用大播放器與快捷鍵直接標記每一球。", "Choose a completed video and label each rally with the large player and keyboard shortcuts."],
    "annotation.waiting": ["等待可標記影片", "Waiting for videos to label"],
    "annotation.emptyTitle": ["目前還沒有可標記的影片", "No videos are ready for labeling"],
    "annotation.emptyBody": ["影片完成處理後，會自動出現在這個開發工具區。", "Completed videos will automatically appear in this development area."],
    "process.label": ["集錦製作流程", "Highlight creation process"],
    "process.step1Title": ["讀懂每一分", "Analyze every point"],
    "process.step1Body": ["從發球到得分各自成段", "Split the match into serve-to-score sequences"],
    "process.step2Title": ["挑出精彩球", "Select the best rallies"],
    "process.step2Body": ["依節奏、動態與回合排序", "Rank by pace, motion, and rally quality"],
    "process.step3Title": ["串成一支 Reel", "Build one reel"],
    "process.step3Body": ["保留原比例，俐落直接剪接", "Keep the original aspect ratio with clean cuts"],
    "adminPassword.title": ["重設使用者密碼", "Reset user password"],
    "adminPassword.body": ["輸入至少 8 個字元的新密碼。儲存後，該使用者在其他裝置上的登入會全部失效。", "Enter a new password of at least 8 characters. Saving it will sign this user out on every other device."],
    "adminPassword.new": ["新密碼", "New password"],
    "common.cancel": ["取消", "Cancel"],
    "adminPassword.save": ["儲存新密碼", "Save new password"],
    "annotation.workspaceTitle": ["精彩球標記工作區", "Rally labeling workspace"],
    "annotation.noVideo": ["尚未選擇影片", "No video selected"],
    "annotation.closeHtml": ["<kbd>Esc</kbd> 關閉", "<kbd>Esc</kbd> Close"],
    "annotation.closeLabel": ["關閉標記工作區", "Close labeling workspace"],
    "annotation.videoLabel": ["桌面標記原始影片播放器", "Source video player for desktop labeling"],
    "annotation.backFive": ["−5 秒", "−5 sec"],
    "annotation.backOne": ["−1 秒", "−1 sec"],
    "annotation.forwardOne": ["+1 秒", "+1 sec"],
    "annotation.forwardFive": ["+5 秒", "+5 sec"],
    "annotation.shortcutsLabel": ["鍵盤快捷鍵", "Keyboard shortcuts"],
    "annotation.shortcutPlayHtml": ["<kbd>Space</kbd> 播放／暫停", "<kbd>Space</kbd> Play / pause"],
    "annotation.shortcutOneHtml": ["<kbd>←</kbd><kbd>→</kbd> 前後 1 秒", "<kbd>←</kbd><kbd>→</kbd> Back / forward 1 sec"],
    "annotation.shortcutFiveHtml": ["<kbd>Shift</kbd> + <kbd>←</kbd><kbd>→</kbd> 前後 5 秒", "<kbd>Shift</kbd> + <kbd>←</kbd><kbd>→</kbd> Back / forward 5 sec"],
    "annotation.shortcutStartHtml": ["<kbd>I</kbd> 起點", "<kbd>I</kbd> Start"],
    "annotation.shortcutEndHtml": ["<kbd>O</kbd> 終點", "<kbd>O</kbd> End"],
    "annotation.shortcutSaveHtml": ["<kbd>Enter</kbd> 儲存", "<kbd>Enter</kbd> Save"],
    "annotation.boundaryTitle": ["標出一個回合", "Mark a rally"],
    "annotation.boundaryBody": ["直接用播放器目前時間，不必抄秒數", "Use the player's current time—no timestamps to copy"],
    "annotation.start": ["回合起點", "Rally start"],
    "annotation.end": ["回合終點", "Rally end"],
    "annotation.notSet": ["尚未設定", "Not set"],
    "annotation.markStartHtml": ["<kbd>I</kbd> 設為起點", "<kbd>I</kbd> Set start"],
    "annotation.markEndHtml": ["<kbd>O</kbd> 設為終點", "<kbd>O</kbd> Set end"],
    "annotation.thisRally": ["這一球", "This rally"],
    "annotation.include": ["值得收錄", "Include"],
    "annotation.exclude": ["不該收錄", "Exclude"],
    "annotation.tagsLegend": ["精彩標籤（可複選）", "Highlight tags (select any)"],
    "annotation.tagRally": ["相持", "Rally"],
    "annotation.tagAttack": ["搶攻", "Attack"],
    "annotation.tagCounterloop": ["反拉", "Counter-loop"],
    "annotation.tagCounter": ["反壓", "Counter"],
    "annotation.tagPlacement": ["落點", "Placement"],
    "annotation.tagPlacementControl": ["落點控制", "Placement control"],
    "annotation.tagBlock": ["擋球", "Block"],
    "annotation.tagDefense": ["防守", "Defense"],
    "annotation.tagOther": ["其他…", "Other…"],
    "annotation.otherLabel": ["其他標籤", "Other tag"],
    "annotation.otherPlaceholder": ["輸入這一球的特別之處", "Describe what makes this rally special"],
    "annotation.saveHtml": ["<kbd>Enter</kbd> 儲存這一球", "<kbd>Enter</kbd> Save this rally"],
    "annotation.savedTitle": ["已儲存標記", "Saved labels"],
    "annotation.zeroCount": ["0 個回合", "0 rallies"],
    "annotation.openToLoad": ["開啟影片後讀取標記…", "Open a video to load labels…"],
    "stage.queued": ["等待電腦處理", "Waiting for processing"],
    "stage.queued-after-restart": ["重新排入處理", "Requeued after restart"],
    "stage.starting": ["準備分析", "Preparing analysis"],
    "stage.probing": ["讀取影片時間軸", "Reading the video timeline"],
    "stage.audio-analysis": ["分析擊球聲", "Analyzing impact audio"],
    "stage.motion-analysis": ["分析畫面動態", "Analyzing motion"],
    "stage.detecting-points": ["切分每一個得分", "Detecting points"],
    "stage.editing-point-reel": ["剪接得分集錦", "Editing the point reel"],
    "stage.completed": ["完成", "Completed"],
    "stage.failed": ["處理失敗", "Processing failed"],
    "stage.exportingPoint": ["輸出第 {number} 個得分", "Exporting point {number}"],
    "stage.waiting": ["等待處理", "Waiting to process"],
    "error.sessionExpired": ["登入已過期，請重新登入。", "Your session has expired. Please sign in again."],
    "error.loginPayload": ["登入回應缺少使用者資料，請重新登入。", "The sign-in response did not include user data. Please sign in again."],
    "error.connection": ["目前無法連線到服務，請稍後再試。", "The service is unavailable right now. Please try again later."],
    "error.loginInvalid": ["帳號或密碼不正確，請再試一次。", "The username or password is incorrect. Please try again."],
    "error.loginFailed": ["目前無法登入：{error}", "Unable to sign in: {error}"],
    "login.loading": ["登入中…", "Signing in…"],
    "logout.success": ["已安全登出。", "You have been signed out safely."],
    "logout.uncertain": ["本頁資料已清除，但伺服器登入狀態可能仍有效。連線恢復後請再登出，或先關閉瀏覽器。", "This page was cleared, but the server session may still be active. Sign out again when the connection returns, or close the browser for now."],
    "upload.errorLocation": ["伺服器沒有回傳續傳網址", "The server did not return a resumable upload URL"],
    "upload.errorSize": ["已儲存的續傳工作階段與影片大小不同", "The saved upload session has a different file size"],
    "upload.errorDuplicates": ["找到 {count} 筆相同影片的上傳紀錄，請先刪除重複項目", "Found {count} upload records for the same video. Remove the duplicates first."],
    "upload.errorChunks": ["分塊傳送失敗", "Chunk upload failed"],
    "upload.creatingConnection": ["建立可續傳連線", "Creating a resumable connection"],
    "upload.continuing": ["繼續傳送影片", "Resuming video upload"],
    "upload.sending": ["正在傳送影片", "Uploading video"],
    "upload.errorOffset": ["伺服器沒有推進上傳 offset", "The server did not advance the upload offset"],
    "upload.delivered": ["影片已送達電腦", "Video delivered to the computer"],
    "upload.queuedDetail": ["分析已排入佇列；現在可以關閉這個頁面", "Analysis is queued. You may close this page now."],
    "upload.paused": ["傳送暫停", "Upload paused"],
    "upload.pausedDetail": ["{error}。重新按開始會從已完成的位置繼續。", "{error}. Press start again to resume from the completed position."],
    "upload.progress": ["{offset} / {total} · {speed}/s", "{offset} / {total} · {speed}/s"],
    "result.preparingShare": ["準備影片…", "Preparing video…"],
    "result.shareTitle": ["桌球得分集錦", "Table tennis point reel"],
    "result.shareSave": ["分享／存到相簿", "Share / save to Photos"],
    "result.mobileSaveHint": ["可從手機分享選單選擇「儲存影片」。", "Choose “Save Video” from your phone's share menu."],
    "result.downloadSaveHint": ["下載後若要放進相簿，請開啟 MP4，再使用手機的分享或儲存影片功能。", "To add the downloaded MP4 to Photos, open it and use your phone's share or save option."],
    "result.pointDownload": ["得分 {number}", "Point {number}"],
    "result.analysis": ["分析報告", "Analysis report"],
    "result.srLabel": ["{source} 的剪輯結果：", "Edited results for {source}:"],
    "result.playDownload": ["播放與下載", "Play and download"],
    "result.collapse": ["收合", "Collapse"],
    "result.previewLabel": ["{source} 的得分集錦預覽", "Point reel preview for {source}"],
    "result.downloadMp4": ["下載 MP4", "Download MP4"],
    "result.moreFiles": ["單分片段與分析檔", "Individual clips and analysis"],
    "annotation.count.one": ["{count} 個回合", "{count} rally"],
    "annotation.count.other": ["{count} 個回合", "{count} rallies"],
    "annotation.empty": ["還沒有標記。播放原片後按 I、O、Enter 就能存下第一球。", "No labels yet. Play the source video, then press I, O, and Enter to save the first rally."],
    "annotation.playLabel": ["播放第 {number} 個標記", "Play label {number}"],
    "annotation.duration": ["{seconds} 秒", "{seconds} sec"],
    "annotation.deleteLabel": ["刪除第 {number} 個標記", "Delete label {number}"],
    "annotation.loading": ["正在讀取標記…", "Loading labels…"],
    "annotation.sourceVideo": ["原始影片", "Source video"],
    "annotation.playbackError": ["瀏覽器無法播放這個原片編碼。", "This browser cannot play the source video's codec."],
    "annotation.rangeError": ["請先按 I 設起點，再按 O 設終點。", "Press I to set the start, then O to set the end."],
    "annotation.noteLengthError": ["精彩標籤合計不能超過 300 個字。", "Highlight tags cannot exceed 300 characters in total."],
    "annotation.otherError": ["請填寫其他標籤，或取消選取「其他…」。", "Enter an Other tag or clear the “Other…” option."],
    "annotation.saving": ["儲存中…", "Saving…"],
    "annotation.saved": ["已儲存。可以直接繼續播放並標下一球。", "Saved. Keep playing and label the next rally."],
    "annotation.confirmDelete": ["確定要刪除這個人工標記嗎？", "Delete this manual label?"],
    "annotation.deleted": ["標記已刪除。", "Label deleted."],
    "common.recentlyUpdated": ["最近更新", "Updated recently"],
    "common.lastUpdated": ["最後更新 {time}", "Last updated {time}"],
    "drive.statusQueued": ["等待下載", "Waiting to download"],
    "drive.statusResolving": ["檢查連結", "Checking link"],
    "drive.statusDownloading": ["Drive 下載中", "Downloading from Drive"],
    "drive.statusFailed": ["匯入失敗", "Import failed"],
    "drive.detailQueued": ["已交給電腦，輪到這支影片時會自動開始。", "The import is queued and will start automatically."],
    "drive.detailResolving": ["正在確認公開權限、檔名與影片格式。", "Checking public access, filename, and video format."],
    "drive.detailDownloading": ["影片會直接下載到這台電腦，完成後自動排入 GPU 剪輯。", "The video downloads directly to this computer and enters the GPU editing queue when complete."],
    "drive.downloaded": ["已下載 {bytes}", "Downloaded {bytes}"],
    "drive.connecting": ["準備連線至 Google Drive", "Connecting to Google Drive"],
    "drive.downloading": ["下載中", "Downloading"],
    "common.delete": ["刪除", "Delete"],
    "drive.retry": ["從目前進度重試", "Resume import"],
    "drive.cancel": ["取消這筆匯入", "Cancel this import"],
    "drive.defaultFilename": ["Google Drive 影片", "Google Drive video"],
    "drive.requeuing": ["重新排入中…", "Requeuing…"],
    "drive.retryError": ["無法重試：{error}", "Unable to retry: {error}"],
    "drive.confirmDelete": ["確定移除這筆 Google Drive 匯入與電腦上的暫存進度？\n\nGoogle Drive 裡的原始影片不受影響。", "Remove this Google Drive import and its temporary progress from this computer?\n\nThe original video in Google Drive will not be affected."],
    "drive.removing": ["移除中…", "Removing…"],
    "drive.removeError": ["無法移除：{error}", "Unable to remove: {error}"],
    "drive.submitting": ["正在把連結交給這台電腦…", "Sending the link to this computer…"],
    "drive.submitted": ["已開始背景匯入。現在可以關閉此頁，稍後再回來看進度。", "The background import has started. You may close this page and return later to check progress."],
    "upload.statusActive": ["上傳中", "Uploading"],
    "upload.statusWaiting": ["等待續傳", "Waiting to resume"],
    "upload.resumeLocal": ["這台裝置保留了續傳位置；若曾重新整理，請在上方重新選擇同一支影片。", "This device saved the resume position. If you refreshed the page, choose the same file above to continue."],
    "upload.resumeActive": ["來源裝置正在傳送；此頁會自動更新電腦已收到的進度。", "The source device is uploading. This page will update as the computer receives new data."],
    "upload.resumeIdle": ["目前沒有收到新分塊；請回來源裝置重新選擇同一支影片續傳。", "No new chunks are arriving. On the source device, choose the same video again to resume."],
    "upload.deleteRecord": ["刪除這筆上傳", "Delete this upload"],
    "upload.thisVideo": ["這支影片", "this video"],
    "upload.transferredData": ["已上傳的資料", "uploaded data"],
    "upload.confirmDelete": ["確定刪除「{filename}」這筆未完成上傳？\n\n電腦上的 {transferred} 會永久刪除，手機裡的原始影片不受影響。", "Delete the incomplete upload “{filename}”?\n\nThe {transferred} on this computer will be permanently deleted. The original video on the phone is unaffected."],
    "upload.deleting": ["刪除中…", "Deleting…"],
    "upload.deleteError": ["無法刪除這筆上傳：{error}", "Unable to delete this upload: {error}"],
    "video.fallbackName": ["影片 {id}", "Video {id}"],
    "video.sourceDuration": ["{duration} 原片", "Source · {duration}"],
    "video.completed": ["處理完成", "Processing complete"],
    "annotation.loadOnOpen": ["{duration} · 開啟工作區後才會載入原始影片", "{duration} · The source video loads after you open the workspace"],
    "annotation.openLabel": ["開啟 {filename} 的標記工作區", "Open the labeling workspace for {filename}"],
    "annotation.open": ["開啟標記", "Open labeling"],
    "annotation.availableCount.one": ["{count} 支可標記影片", "{count} video available to label"],
    "annotation.availableCount.other": ["{count} 支可標記影片", "{count} videos available to label"],
    "common.unknownUser": ["未知使用者", "Unknown user"],
    "common.deviceUpload": ["裝置上傳", "Device upload"],
    "job.downloadSource": ["下載原片", "Download source"],
    "job.retry": ["重試", "Retry"],
    "job.reprocess": ["重新分析", "Reanalyze"],
    "job.delete": ["刪除", "Delete"],
    "job.statusComplete": ["完成", "Completed"],
    "job.statusFailed": ["失敗", "Failed"],
    "job.statusProcessing": ["分析中", "Analyzing"],
    "job.statusQueued": ["排隊中", "Queued"],
    "job.summaryPoints.one": ["選出 {count} 個精彩得分，已剪成得分集錦。", "Selected {count} great point and edited it into a reel."],
    "job.summaryPoints.other": ["選出 {count} 個精彩得分，已剪成得分集錦。", "Selected {count} great points and edited them into a reel."],
    "job.summaryNoPoints": ["這次沒有足夠可靠的得分回合；可下載分析報告檢查訊號。", "No sufficiently reliable points were found. Download the analysis report to inspect the signals."],
    "job.statPoints.one": ["{count} 個得分", "{count} point"],
    "job.statPoints.other": ["{count} 個得分", "{count} points"],
    "job.statReel": ["{duration} 集錦", "Reel · {duration}"],
    "job.statSource": ["{duration} 原片", "Source · {duration}"],
    "storage.capacityPercent": ["{percent}% 的主機容量", "{percent}% of host capacity"],
    "storage.userSources": ["使用者上傳來源", "User-uploaded sources"],
    "storage.sourceCount.one": ["{count} 個來源檔", "{count} source file"],
    "storage.sourceCount.other": ["{count} 個來源檔", "{count} source files"],
    "storage.outputDescription": ["Reel、單分與分析檔", "Reels, point clips, and analysis files"],
    "storage.outputCount.one": ["{count} 個輸出檔", "{count} output file"],
    "storage.outputCount.other": ["{count} 個輸出檔", "{count} output files"],
    "storage.totalCapacity": ["總容量 {bytes}", "Total capacity {bytes}"],
    "admin.accountCount.one": ["{count} 個帳號", "{count} account"],
    "admin.accountCount.other": ["{count} 個帳號", "{count} accounts"],
    "admin.noAccounts": ["尚未建立其他帳號。", "No other accounts have been created."],
    "admin.currentAccount": [" · 目前帳號", " · Current account"],
    "admin.active": ["啟用中", "Active"],
    "admin.inactive": ["已停用", "Inactive"],
    "admin.nameAction": ["名稱", "Edit name"],
    "admin.passwordAction": ["密碼", "Reset password"],
    "admin.selfPasswordTitle": ["請從帳號安全區更改目前帳號的密碼", "Change your own password in Account Security"],
    "admin.selfRoleTitle": ["不可變更目前帳號的角色", "You cannot change the role of your current account"],
    "admin.promote": ["設為管理員", "Promote to administrator"],
    "admin.demote": ["改為使用者", "Change to user"],
    "admin.selfActiveTitle": ["不可停用目前帳號", "You cannot deactivate your current account"],
    "admin.deactivate": ["停用", "Deactivate"],
    "admin.activate": ["啟用", "Activate"],
    "admin.pendingCount.one": ["{count} 個未完成來源", "{count} incomplete source"],
    "admin.pendingCount.other": ["{count} 個未完成來源", "{count} incomplete sources"],
    "admin.noPending": ["目前沒有未完成的上傳或 Drive 匯入。", "There are no incomplete uploads or Drive imports."],
    "admin.itemCount.one": ["{count} 個項目", "{count} item"],
    "admin.itemCount.other": ["{count} 個項目", "{count} items"],
    "admin.noVideos": ["目前沒有任何影片。", "There are no videos yet."],
    "admin.page": ["第 {current} / {total} 頁", "Page {current} / {total}"],
    "account.passwordMismatch": ["兩次輸入的新密碼不一致。", "The new passwords do not match."],
    "account.updatingPassword": ["正在更新密碼…", "Updating password…"],
    "account.passwordUpdated": ["密碼已更新，其他裝置的登入也已登出。", "Password updated. All other devices have been signed out."],
    "account.currentPasswordWrong": ["目前密碼不正確。", "The current password is incorrect."],
    "account.passwordUpdateError": ["無法更新密碼：{error}", "Unable to update password: {error}"],
    "admin.pendingLoadError": ["無法讀取未完成來源：{error}", "Unable to load incomplete sources: {error}"],
    "admin.dataLoadError": ["無法讀取管理資料：{error}", "Unable to load administration data: {error}"],
    "admin.creatingAccount": ["正在建立帳號…", "Creating account…"],
    "admin.accountCreated": ["已建立 @{username}，可以把帳號與初始密碼交給試用者。", "Created @{username}. You can now give the trial user their username and initial password."],
    "admin.displayNamePrompt": ["輸入新的顯示名稱（留空會顯示帳號）：", "Enter a new display name (leave blank to show the username):"],
    "admin.confirmPromote": ["確定要把這個帳號改為管理員？", "Promote this account to administrator?"],
    "admin.confirmDemote": ["確定要把這個帳號改為一般使用者？", "Change this account to a standard user?"],
    "admin.confirmActive": ["確定要{action}這個帳號？", "{action} this account?"],
    "admin.updateError": ["無法更新帳號：{error}", "Unable to update account: {error}"],
    "adminPassword.lengthError": ["密碼至少需要 8 個字元。", "The password must be at least 8 characters."],
    "job.actionRetry": ["重試處理", "retry processing"],
    "job.actionReprocess": ["重新分析", "reanalyze"],
    "job.actionDelete": ["永久刪除", "permanently delete"],
    "job.deleteDetail": ["原始影片、剪輯成品與分析資料都會一併刪除，且無法復原。", "The source video, edited outputs, and analysis data will all be deleted and cannot be recovered."],
    "job.queueDetail": ["系統會重新排入背景處理佇列。", "The system will add the video back to the background processing queue."],
    "job.confirmAction": ["確定要{action}這支影片？\n\n{detail}", "Are you sure you want to {action} this video?\n\n{detail}"],
    "job.processingAction": ["處理中…", "Working…"],
    "job.actionError": ["無法{action}：{error}", "Unable to {action}: {error}"],
    "library.itemCount": ["{count} 個項目", "{count} items"],
    "library.videoCount": ["{count} 支影片", "{count} video"],
    "file.notVideo": ["這個檔案看起來不是影片", "This file does not appear to be a video"],
    "file.chooseVideo": ["請選擇 MOV、MP4、M4V 或 MKV 檔案", "Choose a MOV, MP4, M4V, or MKV file"],
    "file.ready": ["{size} · 選好後可直接開始", "{size} · Ready to start"],
    "file.uploadThis": ["上傳這支影片", "Upload this video"],
    "upload.pausedRetained": ["已暫停（已傳部分會保留）", "Paused (uploaded data is retained)"],
  });

  const dynamicParameters = new WeakMap();
  const browserLanguages = Array.isArray(navigator.languages)
    ? navigator.languages
    : [navigator.language || ""];

  function readStoredLanguage() {
    try {
      const stored = localStorage.getItem(storageKey);
      return stored === "en" || stored === "zh-Hant" ? stored : null;
    } catch (_) {
      return null;
    }
  }

  function preferredBrowserLanguage() {
    for (const value of browserLanguages) {
      const normalized = String(value).toLowerCase();
      if (normalized.startsWith("zh")) return "zh-Hant";
      if (normalized.startsWith("en")) return "en";
    }
    return "en";
  }

  let language = readStoredLanguage() || preferredBrowserLanguage();
  document.documentElement.dataset.i18nPending = "true";

  function t(key, parameters = {}) {
    const entry = catalog[key];
    const template = entry ? entry[language === "en" ? 1 : 0] : key;
    return String(template).replace(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g, (match, name) =>
      Object.hasOwn(parameters, name) ? String(parameters[name]) : match,
    );
  }

  function setText(element, key, parameters = {}) {
    if (!element) return;
    element.dataset.i18nDynamic = key;
    delete element.dataset.i18nDynamicHtml;
    dynamicParameters.set(element, parameters);
    element.textContent = t(key, parameters);
  }

  function setHtml(element, key, parameters = {}) {
    if (!element) return;
    element.dataset.i18nDynamicHtml = key;
    delete element.dataset.i18nDynamic;
    dynamicParameters.set(element, parameters);
    element.innerHTML = t(key, parameters);
  }

  function clear(element) {
    if (!element) return;
    delete element.dataset.i18n;
    delete element.dataset.i18nHtml;
    delete element.dataset.i18nDynamic;
    delete element.dataset.i18nDynamicHtml;
    dynamicParameters.delete(element);
  }

  function apply() {
    document.documentElement.lang = language;
    for (const element of document.querySelectorAll("[data-i18n]")) {
      element.textContent = t(element.dataset.i18n);
    }
    for (const element of document.querySelectorAll("[data-i18n-html]")) {
      element.innerHTML = t(element.dataset.i18nHtml);
    }
    for (const [attribute, datasetKey] of [
      ["aria-label", "i18nAriaLabel"],
      ["placeholder", "i18nPlaceholder"],
      ["content", "i18nContent"],
      ["title", "i18nTitle"],
    ]) {
      for (const element of document.querySelectorAll(`[data-${datasetKey.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}]`)) {
        element.setAttribute(attribute, t(element.dataset[datasetKey]));
      }
    }
    for (const element of document.querySelectorAll("[data-i18n-dynamic]")) {
      element.textContent = t(element.dataset.i18nDynamic, dynamicParameters.get(element));
    }
    for (const element of document.querySelectorAll("[data-i18n-dynamic-html]")) {
      element.innerHTML = t(element.dataset.i18nDynamicHtml, dynamicParameters.get(element));
    }
    const toggle = document.querySelector("#languageToggle");
    if (toggle) {
      const english = language === "en";
      toggle.setAttribute("aria-checked", String(english));
      toggle.setAttribute("aria-label", t(english ? "language.currentEnglish" : "language.currentChinese"));
      toggle.title = t(english ? "language.switchChinese" : "language.switchEnglish");
    }
    delete document.documentElement.dataset.i18nPending;
  }

  function setLanguage(nextLanguage) {
    language = nextLanguage === "en" ? "en" : "zh-Hant";
    try {
      localStorage.setItem(storageKey, language);
    } catch (_) {
      // The language still applies to this page when storage is unavailable.
    }
    apply();
    return language;
  }

  window.HighlightCraftI18n = Object.freeze({
    apply,
    catalog,
    clear,
    locale: () => (language === "en" ? "en-US" : "zh-TW"),
    setHtml,
    setLanguage,
    setText,
    t,
    toggle: () => setLanguage(language === "en" ? "zh-Hant" : "en"),
    get language() {
      return language;
    },
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply, { once: true });
  } else {
    apply();
  }
})();
