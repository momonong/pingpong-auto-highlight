const elements = {
  tokenWarning: document.querySelector("#tokenWarning"),
  accessForm: document.querySelector("#accessForm"),
  accessValue: document.querySelector("#accessValue"),
  accessMessage: document.querySelector("#accessMessage"),
  dropZone: document.querySelector("#dropZone"),
  videoInput: document.querySelector("#videoInput"),
  filePrompt: document.querySelector("#filePrompt"),
  fileMeta: document.querySelector("#fileMeta"),
  uploadButton: document.querySelector("#uploadButton"),
  transferPanel: document.querySelector("#transferPanel"),
  transferLabel: document.querySelector("#transferLabel"),
  transferPercent: document.querySelector("#transferPercent"),
  transferBar: document.querySelector("#transferBar"),
  transferDetail: document.querySelector("#transferDetail"),
  pauseButton: document.querySelector("#pauseButton"),
  driveForm: document.querySelector("#driveForm"),
  driveUrl: document.querySelector("#driveUrl"),
  driveButton: document.querySelector("#driveButton"),
  driveMessage: document.querySelector("#driveMessage"),
  refreshButton: document.querySelector("#refreshButton"),
  emptyJobs: document.querySelector("#emptyJobs"),
  jobCount: document.querySelector("#jobCount"),
  importList: document.querySelector("#importList"),
  uploadList: document.querySelector("#uploadList"),
  jobList: document.querySelector("#jobList"),
};

const searchParams = new URLSearchParams(window.location.search);
const queryToken = searchParams.get("token");
const fragmentParams = new URLSearchParams(window.location.hash.replace(/^#/, ""));
const fragmentToken = fragmentParams.get("token");
const urlToken = fragmentToken || queryToken;
if (urlToken) {
  localStorage.setItem("pingpong-upload-token", urlToken);
  searchParams.delete("token");
  const cleanSearch = searchParams.toString();
  history.replaceState({}, "", `${window.location.pathname}${cleanSearch ? `?${cleanSearch}` : ""}`);
}
const token = urlToken || localStorage.getItem("pingpong-upload-token") || "";

let selectedFile = null;
let chunkSize = 8 * 1024 * 1024;
let paused = false;
let uploadRunning = false;
let wakeLock = null;
let activityLoading = false;
let authReady = false;
let driveSubmitting = false;
let lastImportsSignature = "";
let lastUploadsSignature = "";
let lastJobsSignature = "";

const uploadActiveWindowMs = 60 * 1000;

function accessTokenFrom(value) {
  const input = String(value || "").trim();
  if (!input) return "";
  try {
    const url = new URL(input, window.location.origin);
    const fragment = new URLSearchParams(url.hash.replace(/^#/, "")).get("token");
    const query = url.searchParams.get("token");
    if (fragment || query) return fragment || query;
  } catch (_) {
    // Treat non-URL input as a raw token below.
  }
  const raw = input.replace(/^#?token=/, "");
  return raw && !/\s/.test(raw) ? raw : "";
}

function showAccessMessage(message) {
  elements.accessMessage.textContent = message;
  elements.accessMessage.hidden = !message;
}

const stageNames = {
  queued: "等待電腦處理",
  "queued-after-restart": "重新排入處理",
  starting: "準備分析",
  probing: "讀取影片時間軸",
  "audio-analysis": "分析擊球聲",
  "motion-analysis": "分析畫面動態",
  "detecting-points": "切分每一個得分",
  "editing-point-reel": "剪接得分集錦與轉場",
  completed: "完成",
  failed: "處理失敗",
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const unit = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** unit).toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(seconds || 0));
  const minutes = Math.floor(value / 60);
  const remainder = value % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatTimestamp(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = (value % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remainder}`;
}

function authHeaders(extra = {}) {
  return { "X-Upload-Token": token, ...extra };
}

function fileAccessUrl(path, { download = false } = {}) {
  const url = new URL(path, window.location.origin);
  url.searchParams.set("token", token);
  if (download) url.searchParams.set("download", "true");
  return `${url.pathname}${url.search}`;
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      message = (await response.json()).detail || message;
    } catch (_) {
      // Keep the HTTP status as the useful fallback.
    }
    throw new Error(message);
  }
  return response;
}

function encodeMetadata(value) {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function fingerprint(file) {
  return `pingpong-upload:${file.name}:${file.size}:${file.lastModified}`;
}

async function createSession(file) {
  const metadata = [
    `filename ${encodeMetadata(file.name)}`,
    `filetype ${encodeMetadata(file.type || "application/octet-stream")}`,
  ].join(",");
  const response = await apiFetch("/api/uploads", {
    method: "POST",
    headers: {
      "Tus-Resumable": "1.0.0",
      "Upload-Length": String(file.size),
      "Upload-Metadata": metadata,
    },
  });
  const location = response.headers.get("Location");
  if (!location) throw new Error("伺服器沒有回傳續傳網址");
  localStorage.setItem(fingerprint(file), location);
  return { location, offset: 0, jobId: null };
}

async function inspectSession(location, file) {
  const response = await apiFetch(location, {
    method: "HEAD",
    headers: { "Tus-Resumable": "1.0.0" },
  });
  const length = Number(response.headers.get("Upload-Length"));
  if (length !== file.size) throw new Error("已儲存的續傳工作階段與影片大小不同");
  return {
    location,
    offset: Number(response.headers.get("Upload-Offset")) || 0,
    jobId: null,
  };
}

async function findOrCreateSession(file) {
  const fileFingerprint = fingerprint(file);
  const saved = localStorage.getItem(fileFingerprint);
  if (saved) {
    try {
      return await inspectSession(saved, file);
    } catch (error) {
      if (!String(error.message).includes("404")) console.info("Starting a new upload:", error);
      localStorage.removeItem(fileFingerprint);
    }
  }

  const response = await apiFetch("/api/uploads");
  const { uploads } = await response.json();
  const matches = uploads.filter(
    (upload) => upload.filename === file.name && upload.size === file.size,
  );
  if (matches.length > 1) {
    throw new Error(`找到 ${matches.length} 筆相同影片的上傳紀錄，請先刪除重複項目`);
  }
  if (matches.length === 1) {
    const location = `/api/uploads/${matches[0].id}`;
    const session = await inspectSession(location, file);
    localStorage.setItem(fileFingerprint, location);
    return session;
  }
  return createSession(file);
}

async function checksumHeader(blob) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  const bytes = new Uint8Array(digest);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return `sha256 ${btoa(binary)}`;
}

async function serverOffset(location) {
  const response = await apiFetch(location, {
    method: "HEAD",
    headers: { "Tus-Resumable": "1.0.0" },
  });
  return Number(response.headers.get("Upload-Offset")) || 0;
}

async function sendChunk(location, offset, blob) {
  const checksum = await checksumHeader(blob);
  let lastError = null;
  for (const delay of [0, 700, 1800, 4000]) {
    if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
    try {
      const headers = {
        "Tus-Resumable": "1.0.0",
        "Upload-Offset": String(offset),
        "Content-Type": "application/offset+octet-stream",
      };
      if (checksum) headers["Upload-Checksum"] = checksum;
      const response = await apiFetch(location, { method: "PATCH", headers, body: blob });
      return {
        offset: Number(response.headers.get("Upload-Offset")),
        jobId: response.headers.get("Upload-Job-Id"),
      };
    } catch (error) {
      lastError = error;
      try {
        const recovered = await serverOffset(location);
        if (recovered > offset) return { offset: recovered, jobId: null };
      } catch (_) {
        // The retry loop will surface the original transfer error.
      }
    }
  }
  throw lastError || new Error("分塊傳送失敗");
}

function setTransferProgress(offset, total, startedAt) {
  const fraction = total ? Math.min(1, offset / total) : 0;
  const percent = Math.round(fraction * 100);
  const elapsed = Math.max(0.25, (performance.now() - startedAt) / 1000);
  const speed = offset / elapsed;
  elements.transferPercent.textContent = `${percent}%`;
  elements.transferBar.style.width = `${percent}%`;
  elements.transferDetail.textContent = `${formatBytes(offset)} / ${formatBytes(total)} · ${formatBytes(speed)}/s`;
}

async function acquireWakeLock() {
  try {
    wakeLock = await navigator.wakeLock?.request("screen");
  } catch (_) {
    wakeLock = null;
  }
}

async function releaseWakeLock() {
  try {
    await wakeLock?.release();
  } catch (_) {
    // The browser may already have released it when the tab lost focus.
  }
  wakeLock = null;
}

async function startUpload() {
  if (!selectedFile || uploadRunning || !authReady) return;
  uploadRunning = true;
  paused = false;
  elements.uploadButton.disabled = true;
  elements.videoInput.disabled = true;
  elements.transferPanel.hidden = false;
  elements.pauseButton.hidden = false;
  elements.pauseButton.textContent = "暫停";
  elements.transferLabel.textContent = "建立可續傳連線";
  const startedAt = performance.now();
  await acquireWakeLock();

  try {
    const session = await findOrCreateSession(selectedFile);
    let offset = session.offset;
    let jobId = session.jobId;
    setTransferProgress(offset, selectedFile.size, startedAt);
    elements.transferLabel.textContent = offset ? "繼續傳送影片" : "正在傳送影片";

    while (offset < selectedFile.size) {
      while (paused) await new Promise((resolve) => setTimeout(resolve, 250));
      const end = Math.min(offset + chunkSize, selectedFile.size);
      const result = await sendChunk(session.location, offset, selectedFile.slice(offset, end));
      if (!Number.isFinite(result.offset) || result.offset <= offset) {
        throw new Error("伺服器沒有推進上傳 offset");
      }
      offset = result.offset;
      jobId = result.jobId || jobId;
      setTransferProgress(offset, selectedFile.size, startedAt);
    }

    if (!jobId) {
      const response = await apiFetch(session.location);
      jobId = (await response.json()).job_id;
    }
    elements.transferLabel.textContent = "影片已送達電腦";
    elements.transferDetail.textContent = "分析已排入佇列；現在可以關閉這個頁面";
    elements.pauseButton.hidden = true;
    await loadActivity();
  } catch (error) {
    elements.transferLabel.textContent = "傳送暫停";
    elements.transferDetail.textContent = `${error.message}。重新按開始會從已完成的位置繼續。`;
    elements.uploadButton.disabled = false;
  } finally {
    uploadRunning = false;
    elements.videoInput.disabled = false;
    await releaseWakeLock();
  }
}

function jobStage(job) {
  if (job.stage.startsWith("exporting-point-")) {
    return `輸出第 ${job.stage.split("-").at(-1)} 個得分`;
  }
  return stageNames[job.stage] || job.stage;
}

function triggerDownload(url, filename) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}

async function shareOrSave(button) {
  const path = button.dataset.url;
  const filename = button.dataset.filename || "best_points_reel.mp4";
  const fallbackUrl = fileAccessUrl(path, { download: true });
  if (!navigator.share) {
    triggerDownload(fallbackUrl, filename);
    return;
  }

  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = "準備影片…";
  try {
    const response = await apiFetch(path);
    const blob = await response.blob();
    const file = new File([blob], filename, { type: blob.type || "video/mp4" });
    if (navigator.canShare && !navigator.canShare({ files: [file] })) {
      triggerDownload(fallbackUrl, filename);
      return;
    }
    await navigator.share({
      files: [file],
      title: "桌球得分集錦",
    });
  } catch (error) {
    if (error?.name !== "AbortError") triggerDownload(fallbackUrl, filename);
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

function renderAnnotationPanel(jobId) {
  return `<details class="annotation-panel" data-job-id="${escapeHtml(jobId)}" data-source-url="/api/jobs/${escapeHtml(jobId)}/source">
    <summary><span>手動標記精彩球</span><small>原片會在展開後才載入</small></summary>
    <div class="annotation-body">
      <p class="annotation-help">播放原始影片，把值得收錄的「實際回合」起點與終點記下來；前後留白會由剪輯器另外加入。</p>
      <video class="annotation-video" controls playsinline preload="none" aria-label="原始影片標記播放器"></video>
      <div class="annotation-seek" aria-label="快速移動播放位置">
        <button type="button" data-seek="-10">−10 秒</button>
        <button type="button" data-seek="-1">−1 秒</button>
        <span class="annotation-current">0:00.0</span>
        <button type="button" data-seek="1">+1 秒</button>
        <button type="button" data-seek="10">+10 秒</button>
      </div>
      <div class="annotation-boundaries">
        <label><span>回合起點</span><input class="annotation-start" type="number" min="0" step="0.1" inputmode="decimal" placeholder="秒" /></label>
        <button class="annotation-mark-start" type="button">用目前時間</button>
        <label><span>回合終點</span><input class="annotation-end" type="number" min="0" step="0.1" inputmode="decimal" placeholder="秒" /></label>
        <button class="annotation-mark-end" type="button">用目前時間</button>
      </div>
      <div class="annotation-meta">
        <label><span>這一球</span><select class="annotation-label"><option value="highlight">值得收錄</option><option value="exclude">不該收錄</option></select></label>
        <label><span>備註（可不填）</span><input class="annotation-note" type="text" maxlength="300" placeholder="例如：反拉、長回合、關鍵分" /></label>
      </div>
      <button class="annotation-save" type="button">儲存這一球</button>
      <p class="annotation-message" hidden aria-live="polite"></p>
      <div class="annotation-list" aria-live="polite"><p>展開後讀取標記…</p></div>
    </div>
  </details>`;
}

function renderResultPanel(result, jobId) {
  const reel = result.files.find((file) => file.kind === "reel");
  const pointFiles = result.files.filter((file) => file.kind === "point" || file.kind === "clip");
  const analysis = result.files.find((file) => file.kind === "analysis");
  const annotationPanel = renderAnnotationPanel(jobId);
  if (!reel) {
    return `<div class="downloads">${result.files
      .map((file) => {
        const url = fileAccessUrl(file.url, { download: true });
        return `<a href="${escapeHtml(url)}" download>${escapeHtml(file.name)}</a>`;
      })
      .join("")}</div>${annotationPanel}`;
  }

  const previewUrl = fileAccessUrl(reel.url);
  const downloadUrl = fileAccessUrl(reel.url, { download: true });
  const webShareAvailable = typeof navigator.share === "function";
  const shareAction = webShareAvailable
    ? `<button class="share-button" type="button" data-url="${escapeHtml(reel.url)}" data-filename="${escapeHtml(reel.name)}">分享／存到相簿</button>`
    : "";
  const saveHint = webShareAvailable
    ? "可從手機分享選單選擇「儲存影片」。"
    : "下載後若要放進相簿，請開啟 MP4，再使用手機的分享或儲存影片功能。";
  const pointLinks = pointFiles
    .map((file, index) => {
      const url = fileAccessUrl(file.url, { download: true });
      return `<a href="${escapeHtml(url)}" download>得分 ${index + 1}</a>`;
    })
    .join("");
  const analysisLink = analysis
    ? `<a href="${escapeHtml(fileAccessUrl(analysis.url, { download: true }))}" download>分析報告</a>`
    : "";

  return `<div class="result-panel">
    <div class="reel-heading"><span>BEST POINTS REEL</span><b>${escapeHtml(reel.name)}</b></div>
    <video controls playsinline preload="metadata" aria-label="得分集錦預覽">
      <source src="${escapeHtml(previewUrl)}" type="video/mp4" />
    </video>
    <div class="result-actions">
      <a class="result-primary" href="${escapeHtml(downloadUrl)}" download>下載 MP4</a>
      ${shareAction}
    </div>
    <p class="save-hint">${saveHint}</p>
    <details class="more-files">
      <summary>單分片段與分析檔</summary>
      <div class="downloads">${pointLinks}${analysisLink}</div>
    </details>
  </div>${annotationPanel}`;
}

function showAnnotationMessage(panel, message, isError = false) {
  const element = panel.querySelector(".annotation-message");
  element.textContent = message;
  element.hidden = !message;
  element.classList.toggle("error", isError);
}

function renderAnnotations(panel, annotations) {
  const list = panel.querySelector(".annotation-list");
  if (!annotations.length) {
    list.innerHTML = "<p>還沒有人工標記。找到精彩球後，記下回合起點與終點即可。</p>";
    return;
  }
  list.innerHTML = annotations
    .map((annotation, index) => {
      const label = annotation.label === "highlight" ? "值得收錄" : "不該收錄";
      const note = annotation.note ? `<small>${escapeHtml(annotation.note)}</small>` : "";
      return `<article class="annotation-item ${escapeHtml(annotation.label)}">
        <div><b>${index + 1}. ${label}</b><span>${formatTimestamp(annotation.start)}–${formatTimestamp(annotation.end)} · ${Number(annotation.duration).toFixed(1)} 秒</span>${note}</div>
        <div class="annotation-item-actions">
          <button class="annotation-preview" type="button" data-start="${annotation.start}" data-end="${annotation.end}">播放</button>
          <button class="annotation-delete" type="button" data-annotation-id="${escapeHtml(annotation.id)}">刪除</button>
        </div>
      </article>`;
    })
    .join("");
}

async function loadAnnotations(panel) {
  const list = panel.querySelector(".annotation-list");
  list.innerHTML = "<p>正在讀取標記…</p>";
  try {
    const response = await apiFetch(`/api/jobs/${panel.dataset.jobId}/annotations`);
    const payload = await response.json();
    renderAnnotations(panel, payload.annotations || []);
  } catch (error) {
    list.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

async function activateAnnotationPanel(panel) {
  const video = panel.querySelector(".annotation-video");
  if (!video.dataset.loaded) {
    video.src = fileAccessUrl(panel.dataset.sourceUrl);
    video.preload = "metadata";
    video.dataset.loaded = "true";
    video.load();
  }
  await loadAnnotations(panel);
}

function markAnnotationBoundary(panel, boundary) {
  const video = panel.querySelector(".annotation-video");
  if (!Number.isFinite(video.currentTime)) return;
  const input = panel.querySelector(`.annotation-${boundary}`);
  input.value = video.currentTime.toFixed(1);
  if (boundary === "start") {
    const end = panel.querySelector(".annotation-end");
    if (!end.value || Number(end.value) <= video.currentTime) {
      end.value = Math.min(video.duration || Infinity, video.currentTime + 3).toFixed(1);
    }
  }
  showAnnotationMessage(panel, "");
}

async function saveAnnotation(panel, button) {
  const start = Number(panel.querySelector(".annotation-start").value);
  const end = Number(panel.querySelector(".annotation-end").value);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
    showAnnotationMessage(panel, "請先設定有效的回合起點與終點。", true);
    return;
  }
  button.disabled = true;
  const originalLabel = button.textContent;
  button.textContent = "儲存中…";
  try {
    await apiFetch(`/api/jobs/${panel.dataset.jobId}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start,
        end,
        label: panel.querySelector(".annotation-label").value,
        note: panel.querySelector(".annotation-note").value.trim(),
      }),
    });
    panel.querySelector(".annotation-note").value = "";
    showAnnotationMessage(panel, "已儲存；這個標記之後重跑模型也會保留。", false);
    await loadAnnotations(panel);
  } catch (error) {
    showAnnotationMessage(panel, error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

async function deleteAnnotation(panel, button) {
  if (!window.confirm("確定要刪除這個人工標記嗎？")) return;
  button.disabled = true;
  try {
    await apiFetch(
      `/api/jobs/${panel.dataset.jobId}/annotations/${button.dataset.annotationId}`,
      { method: "DELETE" },
    );
    showAnnotationMessage(panel, "標記已刪除。", false);
    await loadAnnotations(panel);
  } catch (error) {
    showAnnotationMessage(panel, error.message, true);
    button.disabled = false;
  }
}

function uploadIsActive(upload) {
  const updatedAt = Date.parse(upload.updated_at);
  return Number.isFinite(updatedAt) && Date.now() - updatedAt <= uploadActiveWindowMs;
}

function hasLocalResumeSession(upload) {
  const expectedPath = `/api/uploads/${upload.id}`;
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (!key?.startsWith("pingpong-upload:")) continue;
    const saved = localStorage.getItem(key);
    if (saved && new URL(saved, window.location.origin).pathname === expectedPath) return true;
  }
  return false;
}

function forgetLocalResumeSession(uploadId) {
  const expectedPath = `/api/uploads/${uploadId}`;
  const matchingKeys = [];
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (!key?.startsWith("pingpong-upload:")) continue;
    const saved = localStorage.getItem(key);
    if (saved && new URL(saved, window.location.origin).pathname === expectedPath) {
      matchingKeys.push(key);
    }
  }
  for (const key of matchingKeys) localStorage.removeItem(key);
}

function uploadProgress(upload) {
  const raw = upload.size ? Math.min(100, (upload.offset / upload.size) * 100) : 0;
  const value = upload.offset < upload.size ? Math.min(raw, 99.9) : 100;
  const decimals = value > 0 && value < 100 ? 1 : 0;
  return { value, label: `${value.toFixed(decimals)}%` };
}

function uploadUpdatedLabel(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "最近更新";
  return `最後更新 ${date.toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit" })}`;
}

function driveImportProgress(record) {
  if (!record.size) return null;
  const raw = Math.min(100, (record.offset / record.size) * 100);
  const value = record.offset < record.size ? Math.min(raw, 99.9) : 100;
  const decimals = value > 0 && value < 100 ? 1 : 0;
  return { value, label: `${value.toFixed(decimals)}%` };
}

function renderDriveImport(record) {
  const statusLabels = {
    queued: "等待下載",
    resolving: "檢查連結",
    downloading: "Drive 下載中",
    failed: "匯入失敗",
  };
  const details = record.error
    ? escapeHtml(record.error)
    : record.status === "queued"
      ? "已交給電腦，輪到這支影片時會自動開始。"
      : record.status === "resolving"
        ? "正在確認公開權限、檔名與影片格式。"
        : "影片會直接下載到這台電腦，完成後自動排入 GPU 剪輯。";
  const progress = driveImportProgress(record);
  const progressMeta = record.size
    ? `${formatBytes(record.offset)} / ${formatBytes(record.size)}`
    : record.offset
      ? `已下載 ${formatBytes(record.offset)}`
      : "準備連線至 Google Drive";
  const progressBar =
    record.status === "downloading" || record.status === "resolving"
      ? `<div class="job-progress-meta"><span>${escapeHtml(progressMeta)} · ${escapeHtml(uploadUpdatedLabel(record.updated_at))}</span><b>${progress?.label || "下載中"}</b></div><div class="job-progress${progress ? "" : " indeterminate"}"><span${progress ? ` style="width:${progress.value}%"` : ""}></span></div>`
      : "";
  const actions =
    record.status === "failed"
      ? `<div class="import-actions"><button class="delete-import-button" type="button" data-import-id="${escapeHtml(record.id)}">刪除</button><button class="retry-import-button" type="button" data-import-id="${escapeHtml(record.id)}">從目前進度重試</button></div>`
      : record.status === "queued"
        ? `<div class="import-actions"><button class="delete-import-button" type="button" data-import-id="${escapeHtml(record.id)}">取消這筆匯入</button></div>`
        : "";
  const filename = record.filename || "Google Drive 影片";
  const status = statusLabels[record.status] || record.status;

  return `<article class="job ${escapeHtml(record.status)}">
    <div class="job-title"><strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong><span class="status ${escapeHtml(record.status)}">${escapeHtml(status)}</span></div>
    <p class="job-detail">${details}</p>
    ${progressBar}${actions}
  </article>`;
}

async function retryDriveImport(button) {
  const importId = button.dataset.importId;
  if (!importId) return;
  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = "重新排入中…";
  try {
    await apiFetch(`/api/drive-imports/${importId}/retry`, { method: "POST" });
    lastImportsSignature = "";
    await loadActivity();
  } catch (error) {
    window.alert(`無法重試：${error.message}`);
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

async function deleteDriveImport(button) {
  const importId = button.dataset.importId;
  if (!importId || !window.confirm("確定移除這筆 Google Drive 匯入與電腦上的暫存進度？\n\nGoogle Drive 裡的原始影片不受影響。")) return;
  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = "移除中…";
  try {
    await apiFetch(`/api/drive-imports/${importId}`, { method: "DELETE" });
    lastImportsSignature = "";
    await loadActivity();
  } catch (error) {
    window.alert(`無法移除：${error.message}`);
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

function updateDriveButton() {
  elements.driveButton.disabled =
    !authReady || driveSubmitting || !elements.driveUrl.value.trim();
}

function showDriveMessage(message, isError = false) {
  elements.driveMessage.textContent = message;
  elements.driveMessage.classList.toggle("error", isError);
  elements.driveMessage.hidden = !message;
}

async function submitDriveLink(event) {
  event.preventDefault();
  const url = elements.driveUrl.value.trim();
  if (!url || driveSubmitting) return;
  driveSubmitting = true;
  updateDriveButton();
  showDriveMessage("正在把連結交給這台電腦…");
  try {
    await apiFetch("/api/drive-imports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    elements.driveUrl.value = "";
    lastImportsSignature = "";
    showDriveMessage("已開始背景匯入。現在可以關閉此頁，稍後再回來看進度。");
    await loadActivity();
  } catch (error) {
    showDriveMessage(error.message, true);
  } finally {
    driveSubmitting = false;
    updateDriveButton();
  }
}

function renderUpload(upload) {
  const progress = uploadProgress(upload);
  const active = upload.transfer_active;
  const resumableHere = upload.local_resume;
  const statusClass = active ? "uploading" : "waiting";
  const statusText = active ? "上傳中" : "等待續傳";
  const details = resumableHere
    ? "這台裝置保留了續傳位置；若曾重新整理，請在上方重新選擇同一支影片。"
    : active
      ? "來源裝置正在傳送；此頁會自動更新電腦已收到的進度。"
      : "目前沒有收到新分塊；請回來源裝置重新選擇同一支影片續傳。";
  const transferred = `${formatBytes(upload.offset)} / ${formatBytes(upload.size)}`;

  return `<article class="job ${statusClass}">
    <div class="job-title"><strong title="${escapeHtml(upload.filename)}">${escapeHtml(upload.filename)}</strong><span class="status ${statusClass}">${statusText}</span></div>
    <p class="job-detail">${details}</p>
    <div class="job-progress-meta"><span>${escapeHtml(transferred)} · ${escapeHtml(uploadUpdatedLabel(upload.updated_at))}</span><b>${progress.label}</b></div>
    <div class="job-progress"><span style="width:${progress.value}%"></span></div>
    <div class="upload-actions"><button class="delete-upload-button" type="button" data-upload-id="${escapeHtml(upload.id)}" data-filename="${escapeHtml(upload.filename)}" data-transferred="${escapeHtml(transferred)}">刪除這筆上傳</button></div>
  </article>`;
}

async function deleteUploadSession(button) {
  const uploadId = button.dataset.uploadId;
  if (!uploadId) return;
  const filename = button.dataset.filename || "這支影片";
  const transferred = button.dataset.transferred || "已上傳的資料";
  const confirmed = window.confirm(
    `確定刪除「${filename}」這筆未完成上傳？\n\n電腦上的 ${transferred} 會永久刪除，手機裡的原始影片不受影響。`,
  );
  if (!confirmed) return;

  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = "刪除中…";
  try {
    await apiFetch(`/api/uploads/${uploadId}`, {
      method: "DELETE",
      headers: { "Tus-Resumable": "1.0.0" },
    });
    forgetLocalResumeSession(uploadId);
    lastUploadsSignature = "";
    await loadActivity();
  } catch (error) {
    window.alert(`無法刪除這筆上傳：${error.message}`);
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

function renderJob(job) {
  const result = job.result;
  const filename = result?.source_name || `影片 ${job.upload_id.slice(0, 8)}`;
  const progress = Math.round(job.progress * 100);
  const statusText =
    job.status === "completed"
      ? "完成"
      : job.status === "failed"
        ? "失敗"
        : job.status === "processing"
          ? "分析中"
          : "排隊中";
  const summary = result?.summary;
  const pointCount = summary?.point_count ?? summary?.highlight_count ?? 0;
  const details = job.error
    ? escapeHtml(job.error)
    : result
      ? pointCount
        ? `選出 ${pointCount} 個精彩得分，已剪成得分集錦。`
        : "這次沒有足夠可靠的得分回合；可下載分析報告檢查訊號。"
      : escapeHtml(jobStage(job));
  const stats = result
    ? `<div class="job-stats"><span><b>${pointCount}</b> 個得分</span>${summary.reel_duration ? `<span><b>${formatDuration(summary.reel_duration)}</b> 集錦</span>` : ""}<span><b>${formatDuration(result.media.duration)}</b> 原片</span></div>`
    : "";
  const resultPanel = result ? renderResultPanel(result, job.id) : "";
  const progressBar =
    job.status === "processing" || job.status === "queued"
      ? `<div class="job-progress-meta"><span>${escapeHtml(jobStage(job))}</span><b>${progress}%</b></div><div class="job-progress"><span style="width:${progress}%"></span></div>`
      : "";
  return `<article class="job ${escapeHtml(job.status)}">
    <div class="job-title"><strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong><span class="status ${escapeHtml(job.status)}">${statusText}</span></div>
    <p class="job-detail">${details}</p>
    ${progressBar}${stats}${resultPanel}
  </article>`;
}

function renderActivity(imports, uploads, jobs) {
  const uploadViews = uploads.map((upload) => ({
    ...upload,
    transfer_active: uploadIsActive(upload),
    local_resume: hasLocalResumeSession(upload),
  }));
  const total = imports.length + uploadViews.length + jobs.length;
  elements.emptyJobs.hidden = total > 0;
  elements.jobCount.textContent = total
    ? `${total} ${total > 1 ? "個項目" : "支影片"}`
    : "等待第一支影片";

  const importsSignature = JSON.stringify(imports);
  if (importsSignature !== lastImportsSignature) {
    lastImportsSignature = importsSignature;
    elements.importList.innerHTML = imports.map(renderDriveImport).join("");
  }

  const uploadsSignature = JSON.stringify(uploadViews);
  if (uploadsSignature !== lastUploadsSignature) {
    lastUploadsSignature = uploadsSignature;
    elements.uploadList.innerHTML = uploadViews.map(renderUpload).join("");
  }

  const jobsSignature = JSON.stringify(jobs);
  if (jobsSignature !== lastJobsSignature) {
    lastJobsSignature = jobsSignature;
    elements.jobList.innerHTML = jobs.map(renderJob).join("");
  }
}

async function loadActivity() {
  if (!token || activityLoading) return;
  activityLoading = true;
  try {
    const [importsResponse, uploadsResponse, jobsResponse] = await Promise.all([
      apiFetch("/api/drive-imports"),
      apiFetch("/api/uploads"),
      apiFetch("/api/jobs"),
    ]);
    const [{ imports }, { uploads }, { jobs }] = await Promise.all([
      importsResponse.json(),
      uploadsResponse.json(),
      jobsResponse.json(),
    ]);
    renderActivity(imports, uploads, jobs);
  } catch (error) {
    if (String(error.message).includes("401")) elements.tokenWarning.hidden = false;
  } finally {
    activityLoading = false;
  }
}

function selectVideo(file) {
  if (!file) return;
  const extension = file.name.split(".").at(-1)?.toLowerCase();
  if (!file.type.startsWith("video/") && !["mov", "mp4", "m4v", "mkv"].includes(extension)) {
    elements.filePrompt.textContent = "這個檔案看起來不是影片";
    elements.fileMeta.textContent = "請選擇 MOV、MP4、M4V 或 MKV 檔案";
    return;
  }
  selectedFile = file;
  elements.dropZone.classList.add("selected");
  elements.filePrompt.textContent = selectedFile.name;
  elements.fileMeta.textContent = `${formatBytes(selectedFile.size)} · 選好後可直接開始`;
  elements.uploadButton.querySelector("span").textContent = "上傳這支影片";
  elements.uploadButton.disabled = !authReady;
}

elements.videoInput.addEventListener("change", () => {
  selectVideo(elements.videoInput.files?.[0] || null);
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}

elements.dropZone.addEventListener("dragleave", () => {
  elements.dropZone.classList.remove("dragging");
});

elements.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
  selectVideo(event.dataTransfer?.files?.[0] || null);
});

elements.uploadButton.addEventListener("click", startUpload);
elements.driveForm.addEventListener("submit", submitDriveLink);
elements.driveUrl.addEventListener("input", updateDriveButton);
elements.pauseButton.addEventListener("click", () => {
  paused = !paused;
  elements.pauseButton.textContent = paused ? "繼續" : "暫停";
  elements.transferLabel.textContent = paused ? "已暫停（已傳部分會保留）" : "正在傳送影片";
});
elements.jobList.addEventListener("click", (event) => {
  const shareButton = event.target.closest(".share-button");
  if (shareButton) {
    shareOrSave(shareButton);
    return;
  }
  const panel = event.target.closest(".annotation-panel");
  if (!panel) return;
  const video = panel.querySelector(".annotation-video");
  const seekButton = event.target.closest("button[data-seek]");
  if (seekButton) {
    const duration = Number.isFinite(video.duration) ? video.duration : Infinity;
    video.currentTime = Math.max(0, Math.min(duration, video.currentTime + Number(seekButton.dataset.seek)));
    return;
  }
  if (event.target.closest(".annotation-mark-start")) {
    markAnnotationBoundary(panel, "start");
    return;
  }
  if (event.target.closest(".annotation-mark-end")) {
    markAnnotationBoundary(panel, "end");
    return;
  }
  const saveButton = event.target.closest(".annotation-save");
  if (saveButton) {
    saveAnnotation(panel, saveButton);
    return;
  }
  const previewButton = event.target.closest(".annotation-preview");
  if (previewButton) {
    video.currentTime = Number(previewButton.dataset.start);
    video.dataset.stopAt = previewButton.dataset.end;
    video.play().catch(() => showAnnotationMessage(panel, "瀏覽器無法播放這個原片編碼。", true));
    return;
  }
  const deleteButton = event.target.closest(".annotation-delete");
  if (deleteButton) deleteAnnotation(panel, deleteButton);
});
elements.jobList.addEventListener(
  "toggle",
  (event) => {
    const panel = event.target.closest(".annotation-panel");
    if (panel?.open) activateAnnotationPanel(panel);
  },
  true,
);
elements.jobList.addEventListener(
  "timeupdate",
  (event) => {
    const video = event.target.closest(".annotation-video");
    if (!video) return;
    const panel = video.closest(".annotation-panel");
    panel.querySelector(".annotation-current").textContent = formatTimestamp(video.currentTime);
    const stopAt = Number(video.dataset.stopAt);
    if (Number.isFinite(stopAt) && video.currentTime >= stopAt) {
      video.pause();
      delete video.dataset.stopAt;
    }
  },
  true,
);
elements.jobList.addEventListener(
  "durationchange",
  (event) => {
    const video = event.target.closest(".annotation-video");
    if (!video || !Number.isFinite(video.duration)) return;
    const panel = video.closest(".annotation-panel");
    panel.querySelector(".annotation-start").max = String(video.duration);
    panel.querySelector(".annotation-end").max = String(video.duration);
  },
  true,
);
elements.uploadList.addEventListener("click", (event) => {
  const button = event.target.closest(".delete-upload-button");
  if (button) deleteUploadSession(button);
});
elements.importList.addEventListener("click", (event) => {
  const retryButton = event.target.closest(".retry-import-button");
  if (retryButton) {
    retryDriveImport(retryButton);
    return;
  }
  const deleteButton = event.target.closest(".delete-import-button");
  if (deleteButton) deleteDriveImport(deleteButton);
});
elements.refreshButton.addEventListener("click", loadActivity);
elements.accessForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const recoveredToken = accessTokenFrom(elements.accessValue.value);
  if (!recoveredToken) {
    showAccessMessage("找不到有效的存取碼，請貼上包含 #token= 的完整連結。");
    return;
  }
  localStorage.setItem("pingpong-upload-token", recoveredToken);
  showAccessMessage("已儲存，正在重新連線…");
  window.location.replace(`${window.location.pathname}${window.location.search}`);
});

window.addEventListener("beforeunload", (event) => {
  if (!uploadRunning) return;
  event.preventDefault();
  event.returnValue = "";
});

async function initialize() {
  if (!token) {
    elements.tokenWarning.hidden = false;
    showAccessMessage("需要先解鎖，才能顯示這台電腦上的處理 session。");
    return;
  }
  try {
    const [configResponse] = await Promise.all([apiFetch("/api/config"), loadActivity()]);
    const config = await configResponse.json();
    chunkSize = config.chunk_size || chunkSize;
    authReady = true;
    elements.tokenWarning.hidden = true;
    elements.uploadButton.disabled = !selectedFile;
    updateDriveButton();
  } catch (error) {
    elements.tokenWarning.hidden = false;
    showAccessMessage("存取碼無效或已更換，請重新貼上最新的完整連結。");
  }
  setInterval(loadActivity, 2500);
}

initialize();
