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
  highlightLibraryTotal: document.querySelector("#highlightLibraryTotal"),
  highlightLibrarySearch: document.querySelector("#highlightLibrarySearch"),
  highlightLibrarySourceFilter: document.querySelector("#highlightLibrarySourceFilter"),
  highlightLibrarySourceSummary: document.querySelector("#highlightLibrarySourceSummary"),
  highlightLibrarySourceOptions: document.querySelector("#highlightLibrarySourceOptions"),
  highlightLibraryLifecycle: document.querySelector("#highlightLibraryLifecycle"),
  highlightLibraryStorage: document.querySelector("#highlightLibraryStorage"),
  highlightLibraryAfter: document.querySelector("#highlightLibraryAfter"),
  highlightLibraryMinimum: document.querySelector("#highlightLibraryMinimum"),
  highlightLibrarySort: document.querySelector("#highlightLibrarySort"),
  highlightLibraryVisible: document.querySelector("#highlightLibraryVisible"),
  highlightLibraryTopSix: document.querySelector("#highlightLibraryTopSix"),
  highlightLibraryEachTopSix: document.querySelector("#highlightLibraryEachTopSix"),
  highlightLibraryClearFilters: document.querySelector("#highlightLibraryClearFilters"),
  highlightLibraryEmpty: document.querySelector("#highlightLibraryEmpty"),
  highlightLibraryGrid: document.querySelector("#highlightLibraryGrid"),
  compilationSelectionCount: document.querySelector("#compilationSelectionCount"),
  compilationSelectionDuration: document.querySelector("#compilationSelectionDuration"),
  compilationName: document.querySelector("#compilationName"),
  compilationSelectionList: document.querySelector("#compilationSelectionList"),
  compilationClear: document.querySelector("#compilationClear"),
  compilationCreate: document.querySelector("#compilationCreate"),
  compilationMessage: document.querySelector("#compilationMessage"),
  compilationList: document.querySelector("#compilationList"),
  highlightPreview: document.querySelector("#highlightPreview"),
  highlightPreviewClose: document.querySelector("#highlightPreviewClose"),
  highlightPreviewTitle: document.querySelector("#highlightPreviewTitle"),
  highlightPreviewMeta: document.querySelector("#highlightPreviewMeta"),
  highlightPreviewVideo: document.querySelector("#highlightPreviewVideo"),
  annotationDevCount: document.querySelector("#annotationDevCount"),
  annotationDevEmpty: document.querySelector("#annotationDevEmpty"),
  annotationDevList: document.querySelector("#annotationDevList"),
  annotationWorkspace: document.querySelector("#annotationWorkspace"),
  annotationWorkspaceClose: document.querySelector("#annotationWorkspaceClose"),
  annotationWorkspaceFilename: document.querySelector("#annotationWorkspaceFilename"),
  annotationWorkspaceVideo: document.querySelector("#annotationWorkspaceVideo"),
  annotationWorkspaceCurrent: document.querySelector("#annotationWorkspaceCurrent"),
  annotationWorkspaceStart: document.querySelector("#annotationWorkspaceStart"),
  annotationWorkspaceEnd: document.querySelector("#annotationWorkspaceEnd"),
  annotationWorkspaceMarkStart: document.querySelector("#annotationWorkspaceMarkStart"),
  annotationWorkspaceMarkEnd: document.querySelector("#annotationWorkspaceMarkEnd"),
  annotationWorkspaceForm: document.querySelector("#annotationWorkspaceForm"),
  annotationWorkspaceLabel: document.querySelector("#annotationWorkspaceLabel"),
  annotationWorkspaceNoteTags: Array.from(
    document.querySelectorAll('input[name="annotation-note-tag"]'),
  ),
  annotationWorkspaceNoteOtherToggle: document.querySelector(
    "#annotationWorkspaceNoteOtherToggle",
  ),
  annotationWorkspaceNoteOtherField: document.querySelector(
    "#annotationWorkspaceNoteOtherField",
  ),
  annotationWorkspaceNoteOther: document.querySelector("#annotationWorkspaceNoteOther"),
  annotationWorkspaceSave: document.querySelector("#annotationWorkspaceSave"),
  annotationWorkspaceMessage: document.querySelector("#annotationWorkspaceMessage"),
  annotationWorkspaceCount: document.querySelector("#annotationWorkspaceCount"),
  annotationWorkspaceList: document.querySelector("#annotationWorkspaceList"),
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
let libraryActivityLoading = false;
let authReady = false;
let driveSubmitting = false;
let lastImportsSignature = "";
let lastUploadsSignature = "";
let lastJobsSignature = "";
let lastAnnotationDevSignature = "";
let lastLibrarySignature = "";
let lastCompilationsSignature = "";
let libraryHighlights = [];
let librarySources = [];
let libraryCompilations = [];
let selectedHighlightIds = [];
let selectedLibrarySourceIds = new Set();
let compilationSubmitting = false;
let highlightPreviewReturnFocus = null;
const jobRenderSignatures = new Map();
const expandedResultJobIds = new Set();
const compilationRenderSignatures = new Map();
const expandedCompilationIds = new Set();
let annotationWorkspaceJobId = "";
let annotationWorkspaceStart = null;
let annotationWorkspaceEnd = null;
let annotationWorkspaceReturnFocus = null;
let annotationWorkspaceComposing = false;

const annotationNoteMaxLength = 300;

const uploadActiveWindowMs = 60 * 1000;
const desktopLibraryMedia = window.matchMedia("(min-width: 901px)");
const archiveInProgressStates = new Set(["pending", "queued", "uploading", "verifying"]);

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
  "editing-point-reel": "剪接得分集錦",
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
  if (job.stage.startsWith("saving-highlight-")) {
    return `儲存第 ${job.stage.split("-").at(-1)} 個精彩球素材`;
  }
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

function renderResultPanel(result, jobId, sourceName) {
  const reel = result.files.find((file) => file.kind === "reel");
  const pointFiles = result.files.filter(
    (file) => file.kind === "highlight" || file.kind === "point" || file.kind === "clip",
  );
  const analysis = result.files.find((file) => file.kind === "analysis");
  if (!reel) {
    const analysisLink = analysis
      ? `<a href="${escapeHtml(fileAccessUrl(analysis.url, { download: true }))}" download>下載分析報告</a>`
      : "";
    return `<div class="library-result-notice">
      <b>${pointFiles.length} 個精彩球已存入素材庫</b>
      <span>請用電腦開啟下方素材庫，跨影片篩選、排序，再決定要組成哪一支集錦。</span>
      ${analysisLink}
    </div>`;
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

  const open = expandedResultJobIds.has(jobId) ? " open" : "";
  return `<details class="result-panel" data-result-job-id="${escapeHtml(jobId)}"${open}>
    <summary class="reel-heading">
      <span class="sr-only">${escapeHtml(sourceName)} 的剪輯結果：</span>
      <span class="reel-heading-copy"><span>BEST POINTS REEL</span><b>${escapeHtml(reel.name)}</b></span>
      <span class="reel-toggle-label"><span class="reel-toggle-closed">播放與下載</span><span class="reel-toggle-open">收合</span></span>
      <i class="reel-toggle-icon" aria-hidden="true"></i>
    </summary>
    <div class="result-panel-body">
      <video controls playsinline preload="metadata" aria-label="${escapeHtml(sourceName)} 的得分集錦預覽">
        <source data-src="${escapeHtml(previewUrl)}" type="video/mp4" />
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
    </div>
  </details>`;
}

function formatLibraryDate(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "日期未知";
  return date.toLocaleDateString("zh-TW", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

function localDateKey(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return String(value).slice(0, 10);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function highlightArchiveState(highlight) {
  const state = String(highlight.storage?.archive_state || "").trim().toLowerCase();
  return state || "unregistered";
}

function highlightAvailability(highlight) {
  const availability = String(highlight.availability || "").trim().toLowerCase();
  if (["local", "remote_only", "unavailable"].includes(availability)) {
    return availability;
  }
  if (highlight.media_url) return "local";
  if (highlight.storage?.remote_verified) return "remote_only";
  return "unavailable";
}

function highlightIsPlayable(highlight) {
  if (highlightAvailability(highlight) !== "local") return false;
  if (typeof highlight.playable === "boolean") return highlight.playable;
  return Boolean(highlight.media_url);
}

function highlightIsCompilable(highlight) {
  if (highlightAvailability(highlight) !== "local") return false;
  if (typeof highlight.compilable === "boolean") return highlight.compilable;
  return Boolean(highlight.media_url);
}

function highlightMatchesStorageFilter(highlight, filter) {
  if (filter === "all") return true;
  const availability = highlightAvailability(highlight);
  const archiveState = highlightArchiveState(highlight);
  if (["local", "remote_only", "unavailable"].includes(filter)) {
    return availability === filter;
  }
  if (filter === "in_progress") return archiveInProgressStates.has(archiveState);
  return archiveState === filter;
}

function highlightStorageStatus(highlight) {
  const availability = highlightAvailability(highlight);
  const archiveState = highlightArchiveState(highlight);
  const progressLabels = {
    pending: "等待封存",
    queued: "等待上傳",
    uploading: "pCloud 上傳中",
    verifying: "pCloud 驗證中",
  };

  if (availability === "remote_only") {
    return {
      label: "僅 pCloud · 需取回",
      className: "remote",
      title: "pCloud 封存已驗證，但本機檔案不存在；取回後才能預覽或建立集錦。",
    };
  }

  if (availability === "unavailable") {
    if (archiveInProgressStates.has(archiveState)) {
      return {
        label: `${progressLabels[archiveState]} · 無本機檔`,
        className: "progress",
        title: "封存尚未完成，而且本機檔案目前不存在。",
      };
    }
    if (archiveState === "failed") {
      return {
        label: "封存失敗 · 無本機檔",
        className: "failed",
        title: "pCloud 封存失敗，而且本機檔案目前不存在。",
      };
    }
    return {
      label: archiveState === "verified" ? "pCloud 已驗證 · 目前不可用" : "檔案不可用",
      className: "unavailable",
      title: "目前沒有可供預覽或建立集錦的本機檔案。",
    };
  }

  if (archiveState === "verified") {
    return {
      label: "本機 + pCloud",
      className: "verified",
      title: "本機可直接使用，pCloud 封存也已完成驗證。",
    };
  }
  if (archiveInProgressStates.has(archiveState)) {
    return {
      label: `本機 · ${progressLabels[archiveState]}`,
      className: "progress",
      title: "本機檔案可直接使用；pCloud 封存正在處理。",
    };
  }
  if (archiveState === "failed") {
    return {
      label: "本機 · 封存失敗",
      className: "failed",
      title: "本機檔案仍可使用，但上一次 pCloud 封存失敗。",
    };
  }
  return {
    label: "僅本機",
    className: "local",
    title: "本機檔案可直接使用，尚未完成 pCloud 封存。",
  };
}

function filteredLibraryHighlights() {
  const query = elements.highlightLibrarySearch.value.trim().toLocaleLowerCase("zh-TW");
  const lifecycle = elements.highlightLibraryLifecycle.value;
  const storageFilter = elements.highlightLibraryStorage.value;
  const after = elements.highlightLibraryAfter.value;
  const minimum = Number(elements.highlightLibraryMinimum.value) || 0;
  const filtered = libraryHighlights.filter((highlight) => {
    const searchText = [highlight.source_name, highlight.job_id, highlight.id]
      .map((value) => String(value || ""))
      .join(" ")
      .toLocaleLowerCase("zh-TW");
    if (query && !searchText.includes(query)) {
      return false;
    }
    const active = highlight.active !== false;
    if (lifecycle === "active" && !active) return false;
    if (lifecycle === "inactive" && active) return false;
    if (!highlightMatchesStorageFilter(highlight, storageFilter)) return false;
    if (selectedLibrarySourceIds.size && !selectedLibrarySourceIds.has(highlight.job_id)) {
      return false;
    }
    if (after && localDateKey(highlight.source_date) < after) return false;
    return Number(highlight.relative_score) >= minimum;
  });

  const sort = elements.highlightLibrarySort.value;
  return filtered.sort((left, right) => {
    if (sort === "newest") {
      return Date.parse(right.source_date) - Date.parse(left.source_date);
    }
    if (sort === "oldest") {
      return Date.parse(left.source_date) - Date.parse(right.source_date);
    }
    if (sort === "duration") return right.duration - left.duration;
    if (sort === "timeline") {
      return left.job_id.localeCompare(right.job_id) || left.start - right.start;
    }
    return (
      right.relative_score - left.relative_score ||
      right.score - left.score ||
      left.source_rank - right.source_rank
    );
  });
}

function selectedHighlights() {
  const byId = new Map(libraryHighlights.map((highlight) => [highlight.id, highlight]));
  return selectedHighlightIds
    .map((id) => byId.get(id))
    .filter((highlight) => highlight && highlightIsCompilable(highlight));
}

function showCompilationMessage(message, isError = false) {
  elements.compilationMessage.textContent = message;
  elements.compilationMessage.hidden = !message;
  elements.compilationMessage.classList.toggle("error", isError);
}

function renderCompilationSelection() {
  const selected = selectedHighlights();
  const duration = selected.reduce((total, highlight) => total + highlight.duration, 0);
  elements.compilationSelectionCount.textContent = `${selected.length} 球`;
  elements.compilationSelectionDuration.textContent = `約 ${formatDuration(duration)}`;
  const selectionTooLarge = selected.length > 100;
  elements.compilationClear.disabled = selected.length === 0 || compilationSubmitting;
  elements.compilationCreate.disabled =
    selected.length === 0 || selectionTooLarge || compilationSubmitting;
  if (selectionTooLarge) {
    showCompilationMessage("一次最多可建立 100 球；請先移除部分素材。", true);
  } else if (elements.compilationMessage.textContent.startsWith("一次最多可建立 100 球")) {
    showCompilationMessage("");
  }
  if (!selected.length) {
    elements.compilationSelectionList.innerHTML = "<p>從左邊勾選精彩球。</p>";
    return;
  }
  elements.compilationSelectionList.innerHTML = selected
    .map(
      (highlight, index) => `<article class="compilation-selection-item" data-highlight-id="${escapeHtml(highlight.id)}">
        <span>${String(index + 1).padStart(2, "0")}</span>
        <div><b>${escapeHtml(highlight.source_name)}</b><small>#${highlight.source_rank} · ${formatTimestamp(highlight.start)} · ${highlight.duration.toFixed(1)} 秒</small></div>
        <div class="compilation-selection-actions">
          <button type="button" data-selection-move="-1" aria-label="往前移"${index === 0 ? " disabled" : ""}>↑</button>
          <button type="button" data-selection-move="1" aria-label="往後移"${index === selected.length - 1 ? " disabled" : ""}>↓</button>
          <button type="button" data-selection-remove aria-label="移除">×</button>
        </div>
      </article>`,
    )
    .join("");
}

function indexedLibrarySources(serverSources, highlights) {
  const sources = new Map(
    (serverSources || []).map((source) => [String(source.job_id), source]),
  );
  for (const highlight of highlights) {
    const jobId = String(highlight.job_id || "");
    if (!jobId || sources.has(jobId)) continue;
    sources.set(jobId, {
      job_id: jobId,
      name: highlight.source_name || jobId,
      source_date: highlight.source_date,
    });
  }
  return [...sources.values()].sort(
    (left, right) =>
      Date.parse(right.source_date) - Date.parse(left.source_date) ||
      String(left.name).localeCompare(String(right.name), "zh-TW"),
  );
}

function renderHighlightSourceOptions() {
  const availableIds = new Set(librarySources.map((source) => source.job_id));
  selectedLibrarySourceIds = new Set(
    [...selectedLibrarySourceIds].filter((jobId) => availableIds.has(jobId)),
  );
  elements.highlightLibrarySourceSummary.textContent = selectedLibrarySourceIds.size
    ? `${selectedLibrarySourceIds.size} 個來源`
    : "全部來源";
  elements.highlightLibrarySourceOptions.innerHTML = `<p>可複選；沒有勾選時顯示全部來源。</p>${librarySources
    .map(
      (source) => `<label><input type="checkbox" value="${escapeHtml(source.job_id)}"${selectedLibrarySourceIds.has(source.job_id) ? " checked" : ""} /><span><b>${escapeHtml(source.name)}</b><small>${formatLibraryDate(source.source_date)} · ${escapeHtml(source.job_id.slice(0, 6))}</small></span></label>`,
    )
    .join("")}`;
}

function renderHighlightLibrary() {
  const availableIds = new Set(
    libraryHighlights.filter(highlightIsCompilable).map((highlight) => highlight.id),
  );
  selectedHighlightIds = selectedHighlightIds.filter((id) => availableIds.has(id));
  const visible = filteredLibraryHighlights();
  const selectedIds = new Set(selectedHighlightIds);
  const compilableVisible = visible.filter(highlightIsCompilable).length;
  elements.highlightLibraryTotal.textContent = String(libraryHighlights.length);
  elements.highlightLibraryVisible.textContent = `顯示 ${visible.length} / ${libraryHighlights.length} 個 · ${compilableVisible} 個可剪輯`;
  elements.highlightLibraryTopSix.disabled = compilableVisible === 0;
  elements.highlightLibraryEachTopSix.disabled = compilableVisible === 0;
  elements.highlightLibraryEmpty.hidden = visible.length > 0;
  if (!visible.length) {
    const emptyTitle = elements.highlightLibraryEmpty.querySelector("b");
    const emptyDetail = elements.highlightLibraryEmpty.querySelector("span");
    if (libraryHighlights.length) {
      emptyTitle.textContent = "沒有符合目前篩選的素材";
      emptyDetail.textContent = "放寬版本、來源、日期、封存狀態或相對分數條件，就能再次顯示既有片段。";
    } else {
      emptyTitle.textContent = "素材庫目前是空的";
      emptyDetail.textContent = "新影片完成分析後，所有達門檻的精彩球會各自存進來。";
    }
  }
  elements.highlightLibraryGrid.innerHTML = visible
    .map((highlight) => {
      const selected = selectedIds.has(highlight.id);
      const relativePercent = Math.round(highlight.relative_score * 100);
      const playable = highlightIsPlayable(highlight);
      const compilable = highlightIsCompilable(highlight);
      const storageStatus = highlightStorageStatus(highlight);
      const inactive = highlight.active === false;
      const remoteOnly = highlightAvailability(highlight) === "remote_only";
      const unavailableHint = remoteOnly
        ? "需先從 pCloud 取回本機"
        : "目前沒有可用的本機檔案";
      const unavailableAction = remoteOnly ? "取回後可加入" : "目前不可加入";
      return `<article class="highlight-card${selected ? " selected" : ""}${inactive ? " inactive" : ""}${compilable ? "" : " unavailable"}" data-highlight-id="${escapeHtml(highlight.id)}">
        <div class="highlight-card-top">
          <div class="highlight-card-score"><b>${relativePercent}</b><small>REL / 100</small></div>
          <div class="highlight-card-source"><b title="${escapeHtml(highlight.source_name)}">${escapeHtml(highlight.source_name)}</b><span>${formatLibraryDate(highlight.source_date)} · 來源排名 #${highlight.source_rank}${highlight.recommended ? " · 推薦" : ""}</span></div>
        </div>
        <div class="highlight-card-state">
          <span class="highlight-storage-badge ${escapeHtml(storageStatus.className)}" title="${escapeHtml(storageStatus.title)}">${escapeHtml(storageStatus.label)}</span>
          ${inactive ? '<span class="highlight-lifecycle-badge">歷史版本</span>' : ""}
        </div>
        <div class="highlight-card-meta"><span>原片 ${formatTimestamp(highlight.start)}</span><span>${highlight.duration.toFixed(1)} 秒</span></div>
        <div class="highlight-card-actions">
          <label class="highlight-card-select${compilable ? "" : " disabled"}"${compilable ? "" : ` title="${escapeHtml(unavailableHint)}"`}><input type="checkbox" data-library-select${selected ? " checked" : ""}${compilable ? "" : " disabled"} /> ${compilable ? "加入集錦" : unavailableAction}</label>
          <button type="button" data-highlight-preview${playable && highlight.media_url ? "" : ` disabled title="${escapeHtml(unavailableHint)}"`}>${playable && highlight.media_url ? "預覽" : "無法預覽"}</button>
        </div>
      </article>`;
    })
    .join("");
  renderCompilationSelection();
}

function renderCompilation(compilation) {
  const statusLabels = {
    queued: "等待 GPU",
    processing: "GPU 剪輯中",
    completed: "完成",
    failed: "失敗",
  };
  const detail = `${compilation.item_count} 球 · ${compilation.source_count} 個來源 · ${formatDuration(compilation.duration ?? compilation.estimated_duration)}`;
  const error = compilation.error
    ? `<small class="error">${escapeHtml(compilation.error)}</small>`
    : "";
  if (compilation.status !== "completed" || !compilation.file_url) {
    return `<article class="compilation-output" data-compilation-id="${escapeHtml(compilation.id)}"><div class="compilation-output-pending"><div><b>${escapeHtml(compilation.name)}</b><small>${detail}</small>${error}</div><span class="compilation-status">${escapeHtml(statusLabels[compilation.status] || compilation.status)}</span></div></article>`;
  }
  const previewUrl = fileAccessUrl(compilation.file_url);
  const downloadUrl = fileAccessUrl(compilation.file_url, { download: true });
  const open = expandedCompilationIds.has(compilation.id) ? " open" : "";
  return `<details class="compilation-output" data-compilation-id="${escapeHtml(compilation.id)}"${open}>
    <summary><div><b>${escapeHtml(compilation.name)}</b><small>${detail}</small></div><span class="compilation-status">完成</span></summary>
    <div class="compilation-output-body">
      <video controls playsinline preload="none"><source data-src="${escapeHtml(previewUrl)}" type="video/mp4" /></video>
      <div class="compilation-output-actions"><a href="${escapeHtml(downloadUrl)}" download>下載 MP4</a></div>
    </div>
  </details>`;
}

function createCompilationElement(compilation) {
  const template = document.createElement("template");
  template.innerHTML = renderCompilation(compilation).trim();
  return template.content.firstElementChild;
}

function hydrateCompilation(details) {
  const video = details.querySelector("video");
  const source = video?.querySelector("source[data-src]");
  if (!video || !source || source.hasAttribute("src")) return;
  source.src = source.dataset.src;
  video.load();
}

function dehydrateCompilation(details) {
  const video = details.querySelector("video");
  const source = video?.querySelector("source[src]");
  if (!video || !source) return;
  video.pause();
  source.removeAttribute("src");
  video.load();
}

function renderCompilations() {
  if (!libraryCompilations.length) {
    elements.compilationList.innerHTML = "<p>還沒有自訂集錦。</p>";
    compilationRenderSignatures.clear();
    expandedCompilationIds.clear();
    return;
  }

  elements.compilationList.querySelector(":scope > p")?.remove();
  const existingNodes = new Map(
    [...elements.compilationList.children]
      .filter((node) => node.dataset.compilationId)
      .map((node) => [node.dataset.compilationId, node]),
  );
  const liveIds = new Set();

  libraryCompilations.forEach((compilation, index) => {
    const id = String(compilation.id);
    const signature = JSON.stringify(compilation);
    liveIds.add(id);
    let node = existingNodes.get(id);
    if (!node || compilationRenderSignatures.get(id) !== signature) {
      const replacement = createCompilationElement(compilation);
      if (node) node.replaceWith(replacement);
      node = replacement;
      compilationRenderSignatures.set(id, signature);
    }
    const nodeAtIndex = elements.compilationList.children[index];
    if (nodeAtIndex !== node) {
      elements.compilationList.insertBefore(node, nodeAtIndex || null);
    }
    if (node.matches("details[open]")) hydrateCompilation(node);
  });

  existingNodes.forEach((node, id) => {
    if (!liveIds.has(id)) node.remove();
  });
  compilationRenderSignatures.forEach((_, id) => {
    if (!liveIds.has(id)) compilationRenderSignatures.delete(id);
  });
  expandedCompilationIds.forEach((id) => {
    if (!liveIds.has(id)) expandedCompilationIds.delete(id);
  });
}

function addHighlightsToSelection(highlights) {
  const selected = new Set(selectedHighlightIds);
  for (const highlight of highlights) {
    if (highlightIsCompilable(highlight) && !selected.has(highlight.id)) {
      selected.add(highlight.id);
      selectedHighlightIds.push(highlight.id);
    }
  }
  renderHighlightLibrary();
}

function openHighlightPreview(highlightId, returnFocus = null) {
  const highlight = libraryHighlights.find((item) => item.id === highlightId);
  if (!highlight || !highlightIsPlayable(highlight) || !highlight.media_url) return;
  elements.highlightPreviewTitle.textContent = `來源排名 #${highlight.source_rank}`;
  elements.highlightPreviewMeta.textContent = `${highlight.source_name} · ${formatTimestamp(highlight.start)} · 相對分數 ${Math.round(highlight.relative_score * 100)}`;
  elements.highlightPreviewVideo.src = fileAccessUrl(highlight.media_url);
  highlightPreviewReturnFocus = returnFocus;
  elements.highlightPreview.hidden = false;
  document.body.classList.add("highlight-preview-open");
  elements.highlightPreviewVideo.load();
  elements.highlightPreviewVideo.play().catch(() => {});
  elements.highlightPreviewClose.focus({ preventScroll: true });
}

function closeHighlightPreview() {
  if (elements.highlightPreview.hidden) return;
  elements.highlightPreviewVideo.pause();
  elements.highlightPreviewVideo.removeAttribute("src");
  elements.highlightPreviewVideo.load();
  elements.highlightPreview.hidden = true;
  document.body.classList.remove("highlight-preview-open");
  highlightPreviewReturnFocus?.focus({ preventScroll: true });
  highlightPreviewReturnFocus = null;
}

async function createCompilation() {
  if (!selectedHighlightIds.length || selectedHighlightIds.length > 100 || compilationSubmitting) {
    return;
  }
  const original = elements.compilationCreate.textContent;
  compilationSubmitting = true;
  elements.compilationCreate.disabled = true;
  elements.compilationCreate.textContent = "排入 GPU…";
  showCompilationMessage("");
  try {
    const response = await apiFetch("/api/compilations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: elements.compilationName.value.trim(),
        highlight_ids: selectedHighlightIds,
      }),
    });
    const compilation = await response.json();
    showCompilationMessage(`「${compilation.name}」已排入 GPU，沒有一分鐘硬性限制。`);
    elements.compilationName.value = "";
    await loadActivity();
  } catch (error) {
    showCompilationMessage(error.message, true);
  } finally {
    compilationSubmitting = false;
    elements.compilationCreate.textContent = original;
    elements.compilationCreate.disabled = selectedHighlightIds.length === 0;
    renderCompilationSelection();
  }
}

function annotationWorkspaceIsOpen() {
  return !elements.annotationWorkspace.hidden;
}

function showAnnotationWorkspaceMessage(message, isError = false) {
  elements.annotationWorkspaceMessage.textContent = message;
  elements.annotationWorkspaceMessage.hidden = !message;
  elements.annotationWorkspaceMessage.classList.toggle("error", isError);
}

function renderAnnotationWorkspaceBoundaries() {
  elements.annotationWorkspaceStart.textContent =
    annotationWorkspaceStart === null ? "尚未設定" : formatTimestamp(annotationWorkspaceStart);
  elements.annotationWorkspaceEnd.textContent =
    annotationWorkspaceEnd === null ? "尚未設定" : formatTimestamp(annotationWorkspaceEnd);
}

function resetAnnotationWorkspaceNote() {
  for (const checkbox of elements.annotationWorkspaceNoteTags) checkbox.checked = false;
  elements.annotationWorkspaceNoteOtherToggle.checked = false;
  elements.annotationWorkspaceNoteOther.value = "";
  elements.annotationWorkspaceNoteOtherField.hidden = true;
}

function updateAnnotationWorkspaceNoteOther() {
  const isOther = elements.annotationWorkspaceNoteOtherToggle.checked;
  elements.annotationWorkspaceNoteOtherField.hidden = !isOther;
  if (isOther) {
    elements.annotationWorkspaceNoteOther.focus({ preventScroll: true });
  } else {
    elements.annotationWorkspaceNoteOther.value = "";
  }
}

function annotationWorkspaceNoteValue() {
  const selectedTags = elements.annotationWorkspaceNoteTags
    .filter((checkbox) => checkbox.checked)
    .map((checkbox) => checkbox.value);
  const other = elements.annotationWorkspaceNoteOtherToggle.checked
    ? elements.annotationWorkspaceNoteOther.value.trim()
    : "";
  if (other) selectedTags.push(other);
  return selectedTags.join("、");
}

function markAnnotationWorkspaceBoundary(boundary) {
  const current = elements.annotationWorkspaceVideo.currentTime;
  if (!Number.isFinite(current)) return;
  const rounded = Math.round(current * 10) / 10;
  if (boundary === "start") {
    annotationWorkspaceStart = rounded;
    if (annotationWorkspaceEnd !== null && annotationWorkspaceEnd <= rounded) {
      annotationWorkspaceEnd = null;
    }
  } else {
    annotationWorkspaceEnd = rounded;
    elements.annotationWorkspaceVideo.pause();
  }
  renderAnnotationWorkspaceBoundaries();
  showAnnotationWorkspaceMessage("");
}

function seekAnnotationWorkspace(seconds) {
  const video = elements.annotationWorkspaceVideo;
  const duration = Number.isFinite(video.duration) ? video.duration : Infinity;
  delete video.dataset.stopAt;
  video.currentTime = Math.max(0, Math.min(duration, video.currentTime + seconds));
}

function renderAnnotationWorkspaceList(annotations) {
  elements.annotationWorkspaceCount.textContent = `${annotations.length} 個回合`;
  if (!annotations.length) {
    elements.annotationWorkspaceList.innerHTML =
      "<p>還沒有標記。播放原片後按 I、O、Enter 就能存下第一球。</p>";
    return;
  }
  elements.annotationWorkspaceList.innerHTML = annotations
    .map((annotation, index) => {
      const label = annotation.label === "highlight" ? "值得收錄" : "不該收錄";
      const note = annotation.note ? `<small>${escapeHtml(annotation.note)}</small>` : "";
      return `<article class="annotation-workspace-item ${escapeHtml(annotation.label)}">
        <button class="annotation-workspace-preview" type="button" data-start="${annotation.start}" data-end="${annotation.end}" aria-label="播放第 ${index + 1} 個標記">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <div><b>${label}</b><time>${formatTimestamp(annotation.start)}–${formatTimestamp(annotation.end)} · ${Number(annotation.duration).toFixed(1)} 秒</time>${note}</div>
        </button>
        <button class="annotation-workspace-delete" type="button" data-annotation-id="${escapeHtml(annotation.id)}" aria-label="刪除第 ${index + 1} 個標記">×</button>
      </article>`;
    })
    .join("");
}

async function loadAnnotationWorkspaceList() {
  if (!annotationWorkspaceJobId) return;
  elements.annotationWorkspaceList.innerHTML = "<p>正在讀取標記…</p>";
  try {
    const response = await apiFetch(
      `/api/jobs/${annotationWorkspaceJobId}/annotations`,
    );
    const payload = await response.json();
    renderAnnotationWorkspaceList(payload.annotations || []);
  } catch (error) {
    elements.annotationWorkspaceList.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

function openAnnotationWorkspace(button) {
  annotationWorkspaceJobId = button.dataset.jobId || "";
  if (!annotationWorkspaceJobId) return;
  annotationWorkspaceReturnFocus = button;
  annotationWorkspaceStart = null;
  annotationWorkspaceEnd = null;
  annotationWorkspaceComposing = false;
  renderAnnotationWorkspaceBoundaries();
  showAnnotationWorkspaceMessage("");
  elements.annotationWorkspaceFilename.textContent =
    button.dataset.sourceName || "原始影片";
  elements.annotationWorkspaceLabel.value = "highlight";
  resetAnnotationWorkspaceNote();
  elements.annotationWorkspaceCurrent.textContent = "0:00.0";
  elements.annotationWorkspace.hidden = false;
  document.body.classList.add("annotation-workspace-open");
  elements.annotationWorkspaceVideo.src = fileAccessUrl(
    `/api/jobs/${annotationWorkspaceJobId}/source`,
  );
  elements.annotationWorkspaceVideo.preload = "metadata";
  elements.annotationWorkspaceVideo.load();
  loadAnnotationWorkspaceList();
  elements.annotationWorkspace.focus({ preventScroll: true });
}

function closeAnnotationWorkspace() {
  if (!annotationWorkspaceIsOpen()) return;
  const returnJobId = annotationWorkspaceJobId;
  const returnFocus = annotationWorkspaceReturnFocus;
  elements.annotationWorkspaceVideo.pause();
  elements.annotationWorkspaceVideo.removeAttribute("src");
  elements.annotationWorkspaceVideo.load();
  elements.annotationWorkspace.hidden = true;
  document.body.classList.remove("annotation-workspace-open");
  annotationWorkspaceJobId = "";
  annotationWorkspaceReturnFocus = null;
  annotationWorkspaceComposing = false;
  const currentLauncher = Array.from(
    elements.annotationDevList.querySelectorAll(".open-annotation-workspace"),
  ).find((button) => button.dataset.jobId === returnJobId);
  (currentLauncher || (returnFocus?.isConnected ? returnFocus : null))?.focus();
}

function toggleAnnotationWorkspacePlayback() {
  const video = elements.annotationWorkspaceVideo;
  if (video.paused) {
    video
      .play()
      .catch(() => showAnnotationWorkspaceMessage("瀏覽器無法播放這個原片編碼。", true));
  } else {
    video.pause();
  }
}

async function saveAnnotationWorkspace() {
  if (elements.annotationWorkspaceSave.disabled) return;
  if (
    annotationWorkspaceStart === null ||
    annotationWorkspaceEnd === null ||
    annotationWorkspaceEnd <= annotationWorkspaceStart
  ) {
    showAnnotationWorkspaceMessage("請先按 I 設起點，再按 O 設終點。", true);
    return;
  }
  const note = annotationWorkspaceNoteValue();
  if (note.length > annotationNoteMaxLength) {
    showAnnotationWorkspaceMessage("精彩標籤合計不能超過 300 個字。", true);
    if (elements.annotationWorkspaceNoteOtherToggle.checked) {
      elements.annotationWorkspaceNoteOther.focus({ preventScroll: true });
    }
    return;
  }
  if (
    elements.annotationWorkspaceNoteOtherToggle.checked &&
    !elements.annotationWorkspaceNoteOther.value.trim()
  ) {
    showAnnotationWorkspaceMessage("請填寫其他標籤，或取消選取「其他…」。", true);
    elements.annotationWorkspaceNoteOther.focus({ preventScroll: true });
    return;
  }
  const button = elements.annotationWorkspaceSave;
  const originalHtml = button.innerHTML;
  button.disabled = true;
  button.textContent = "儲存中…";
  try {
    await apiFetch(`/api/jobs/${annotationWorkspaceJobId}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: annotationWorkspaceStart,
        end: annotationWorkspaceEnd,
        label: elements.annotationWorkspaceLabel.value,
        note,
      }),
    });
    annotationWorkspaceStart = null;
    annotationWorkspaceEnd = null;
    resetAnnotationWorkspaceNote();
    renderAnnotationWorkspaceBoundaries();
    showAnnotationWorkspaceMessage("已儲存。可以直接繼續播放並標下一球。", false);
    await loadAnnotationWorkspaceList();
  } catch (error) {
    showAnnotationWorkspaceMessage(error.message, true);
  } finally {
    button.disabled = false;
    button.innerHTML = originalHtml;
  }
}

async function deleteAnnotationWorkspaceItem(button) {
  if (!window.confirm("確定要刪除這個人工標記嗎？")) return;
  button.disabled = true;
  try {
    await apiFetch(
      `/api/jobs/${annotationWorkspaceJobId}/annotations/${button.dataset.annotationId}`,
      { method: "DELETE" },
    );
    showAnnotationWorkspaceMessage("標記已刪除。", false);
    await loadAnnotationWorkspaceList();
  } catch (error) {
    showAnnotationWorkspaceMessage(error.message, true);
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

function renderAnnotationDevJob(job, index) {
  const result = job.result;
  const filename = result?.source_name || `影片 ${job.upload_id.slice(0, 8)}`;
  const duration = Number.isFinite(result?.media?.duration)
    ? `${formatDuration(result.media.duration)} 原片`
    : "處理完成";
  return `<article class="annotation-dev-item">
    <span class="annotation-dev-index">${String(index + 1).padStart(2, "0")}</span>
    <div class="annotation-dev-copy">
      <strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong>
      <small>${duration} · 開啟工作區後才會載入原始影片</small>
    </div>
    <button class="annotation-dev-open open-annotation-workspace" type="button" data-job-id="${escapeHtml(job.id)}" data-source-name="${escapeHtml(filename)}" aria-label="開啟 ${escapeHtml(filename)} 的標記工作區">
      <span>開啟標記</span><small>I · O · Enter</small>
    </button>
  </article>`;
}

function renderAnnotationDevelopment(jobs) {
  const completedJobs = jobs.filter(
    (job) => job.status === "completed" && job.result,
  );
  elements.annotationDevCount.textContent = completedJobs.length
    ? `${completedJobs.length} 支可標記影片`
    : "等待可標記影片";
  elements.annotationDevEmpty.hidden = completedJobs.length > 0;
  elements.annotationDevList.innerHTML = completedJobs
    .map(renderAnnotationDevJob)
    .join("");
}

function renderJob(job) {
  const jobId = String(job.id);
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
  const resultPanel = result ? renderResultPanel(result, jobId, filename) : "";
  const progressBar =
    job.status === "processing" || job.status === "queued"
      ? `<div class="job-progress-meta"><span>${escapeHtml(jobStage(job))}</span><b>${progress}%</b></div><div class="job-progress"><span style="width:${progress}%"></span></div>`
      : "";
  return `<article class="job ${escapeHtml(job.status)}" data-job-id="${escapeHtml(jobId)}">
    <div class="job-title"><strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong><span class="status ${escapeHtml(job.status)}">${statusText}</span></div>
    <p class="job-detail">${details}</p>
    ${progressBar}${stats}${resultPanel}
  </article>`;
}

function createJobElement(job) {
  const template = document.createElement("template");
  template.innerHTML = renderJob(job).trim();
  return template.content.firstElementChild;
}

function hydrateResultPanel(panel) {
  const video = panel.querySelector(".result-panel-body > video");
  const source = video?.querySelector("source[data-src]");
  if (!video || !source || source.hasAttribute("src")) return;
  source.src = source.dataset.src;
  video.load();
}

function dehydrateResultPanel(panel) {
  const video = panel.querySelector(".result-panel-body > video");
  const source = video?.querySelector("source[src]");
  if (!video || !source) return;
  video.pause();
  source.removeAttribute("src");
  video.load();
}

function renderJobs(jobs) {
  const existingNodes = new Map(
    [...elements.jobList.children].map((node) => [node.dataset.jobId, node]),
  );
  const liveJobIds = new Set();
  const expandableJobIds = new Set();

  jobs.forEach((job, index) => {
    const jobId = String(job.id);
    const signature = JSON.stringify(job);
    liveJobIds.add(jobId);
    if (job.status === "completed" && job.result) expandableJobIds.add(jobId);

    let node = existingNodes.get(jobId);
    if (!node || jobRenderSignatures.get(jobId) !== signature) {
      const replacement = createJobElement(job);
      if (node) node.replaceWith(replacement);
      node = replacement;
      jobRenderSignatures.set(jobId, signature);
    }

    const nodeAtIndex = elements.jobList.children[index];
    if (nodeAtIndex !== node) {
      elements.jobList.insertBefore(node, nodeAtIndex || null);
    }

    const resultPanel = node.querySelector(".result-panel[data-result-job-id]");
    if (resultPanel?.open) hydrateResultPanel(resultPanel);
  });

  existingNodes.forEach((node, jobId) => {
    if (!liveJobIds.has(jobId)) node.remove();
  });
  jobRenderSignatures.forEach((_, jobId) => {
    if (!liveJobIds.has(jobId)) jobRenderSignatures.delete(jobId);
  });
  expandedResultJobIds.forEach((jobId) => {
    if (!expandableJobIds.has(jobId)) expandedResultJobIds.delete(jobId);
  });
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
    renderJobs(jobs);
  }

  const annotationDevSignature = JSON.stringify(
    jobs
      .filter((job) => job.status === "completed" && job.result)
      .map((job) => ({
        id: job.id,
        sourceName: job.result.source_name,
        duration: job.result.media?.duration ?? null,
      })),
  );
  if (annotationDevSignature !== lastAnnotationDevSignature) {
    lastAnnotationDevSignature = annotationDevSignature;
    renderAnnotationDevelopment(jobs);
  }
}

async function loadLibraryActivity() {
  if (!token || !desktopLibraryMedia.matches || libraryActivityLoading) return;
  libraryActivityLoading = true;
  try {
    const [highlightsResponse, compilationsResponse] = await Promise.all([
      apiFetch("/api/highlights?lifecycle=all"),
      apiFetch("/api/compilations"),
    ]);
    const [highlightsPayload, compilationsPayload] = await Promise.all([
      highlightsResponse.json(),
      compilationsResponse.json(),
    ]);
    const librarySignature = JSON.stringify(highlightsPayload);
    if (librarySignature !== lastLibrarySignature) {
      lastLibrarySignature = librarySignature;
      libraryHighlights = highlightsPayload.highlights || [];
      librarySources = indexedLibrarySources(highlightsPayload.sources, libraryHighlights);
      renderHighlightSourceOptions();
      renderHighlightLibrary();
    }
    const compilationsSignature = JSON.stringify(compilationsPayload);
    if (compilationsSignature !== lastCompilationsSignature) {
      lastCompilationsSignature = compilationsSignature;
      libraryCompilations = compilationsPayload.compilations || [];
      renderCompilations();
    }
    elements.highlightLibraryVisible.removeAttribute("title");
  } catch (error) {
    if (String(error.message).includes("401")) {
      elements.tokenWarning.hidden = false;
    } else {
      elements.highlightLibraryVisible.textContent = "素材庫暫時無法更新";
      elements.highlightLibraryVisible.title = error.message;
    }
  } finally {
    libraryActivityLoading = false;
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
    void loadLibraryActivity();
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
  if (shareButton) shareOrSave(shareButton);
});
elements.jobList.addEventListener(
  "toggle",
  (event) => {
    const panel = event.target;
    if (!panel.matches?.(".result-panel[data-result-job-id]")) return;
    const jobId = panel.dataset.resultJobId;
    if (panel.open) {
      expandedResultJobIds.add(jobId);
      hydrateResultPanel(panel);
    } else {
      expandedResultJobIds.delete(jobId);
      dehydrateResultPanel(panel);
    }
  },
  true,
);
for (const control of [
  elements.highlightLibrarySearch,
  elements.highlightLibraryLifecycle,
  elements.highlightLibraryStorage,
  elements.highlightLibraryAfter,
  elements.highlightLibraryMinimum,
  elements.highlightLibrarySort,
]) {
  control.addEventListener(control.tagName === "INPUT" ? "input" : "change", renderHighlightLibrary);
}
elements.highlightLibraryTopSix.addEventListener("click", () => {
  addHighlightsToSelection(
    filteredLibraryHighlights().filter(highlightIsCompilable).slice(0, 6),
  );
});
elements.highlightLibraryEachTopSix.addEventListener("click", () => {
  const grouped = new Map();
  for (const highlight of filteredLibraryHighlights()) {
    if (!highlightIsCompilable(highlight)) continue;
    if (!grouped.has(highlight.job_id)) grouped.set(highlight.job_id, []);
    grouped.get(highlight.job_id).push(highlight);
  }
  const selected = [];
  for (const highlights of grouped.values()) {
    selected.push(...highlights.sort((left, right) => left.source_rank - right.source_rank).slice(0, 6));
  }
  addHighlightsToSelection(selected);
});
elements.highlightLibraryClearFilters.addEventListener("click", () => {
  elements.highlightLibrarySearch.value = "";
  selectedLibrarySourceIds = new Set();
  renderHighlightSourceOptions();
  elements.highlightLibraryLifecycle.value = "active";
  elements.highlightLibraryStorage.value = "all";
  elements.highlightLibraryAfter.value = "";
  elements.highlightLibraryMinimum.value = "0";
  elements.highlightLibrarySort.value = "quality";
  renderHighlightLibrary();
});
elements.highlightLibrarySourceOptions.addEventListener("change", (event) => {
  const checkbox = event.target.closest('input[type="checkbox"]');
  if (!checkbox) return;
  if (checkbox.checked) {
    selectedLibrarySourceIds.add(checkbox.value);
  } else {
    selectedLibrarySourceIds.delete(checkbox.value);
  }
  elements.highlightLibrarySourceSummary.textContent = selectedLibrarySourceIds.size
    ? `${selectedLibrarySourceIds.size} 個來源`
    : "全部來源";
  renderHighlightLibrary();
});
elements.highlightLibraryGrid.addEventListener("change", (event) => {
  const checkbox = event.target.closest("[data-library-select]");
  if (!checkbox) return;
  const card = checkbox.closest("[data-highlight-id]");
  const highlightId = card?.dataset.highlightId;
  if (!highlightId) return;
  const highlight = libraryHighlights.find((item) => item.id === highlightId);
  if (!highlight || !highlightIsCompilable(highlight)) {
    checkbox.checked = false;
    selectedHighlightIds = selectedHighlightIds.filter((id) => id !== highlightId);
    renderHighlightLibrary();
    return;
  }
  if (checkbox.checked && !selectedHighlightIds.includes(highlightId)) {
    selectedHighlightIds.push(highlightId);
  } else if (!checkbox.checked) {
    selectedHighlightIds = selectedHighlightIds.filter((id) => id !== highlightId);
  }
  renderHighlightLibrary();
});
elements.highlightLibraryGrid.addEventListener("click", (event) => {
  const button = event.target.closest("[data-highlight-preview]");
  if (!button) return;
  openHighlightPreview(
    button.closest("[data-highlight-id]")?.dataset.highlightId,
    button,
  );
});
elements.compilationSelectionList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-highlight-id]");
  if (!item) return;
  const index = selectedHighlightIds.indexOf(item.dataset.highlightId);
  if (index < 0) return;
  if (event.target.closest("[data-selection-remove]")) {
    selectedHighlightIds.splice(index, 1);
  } else {
    const move = Number(event.target.closest("[data-selection-move]")?.dataset.selectionMove);
    const next = index + move;
    if (!Number.isFinite(move) || next < 0 || next >= selectedHighlightIds.length) return;
    [selectedHighlightIds[index], selectedHighlightIds[next]] = [
      selectedHighlightIds[next],
      selectedHighlightIds[index],
    ];
  }
  renderHighlightLibrary();
});
elements.compilationClear.addEventListener("click", () => {
  selectedHighlightIds = [];
  showCompilationMessage("");
  renderHighlightLibrary();
});
elements.compilationCreate.addEventListener("click", createCompilation);
elements.compilationList.addEventListener(
  "toggle",
  (event) => {
    const details = event.target;
    if (!details.matches?.("details.compilation-output[data-compilation-id]")) return;
    const id = details.dataset.compilationId;
    if (details.open) {
      expandedCompilationIds.add(id);
      hydrateCompilation(details);
    } else {
      expandedCompilationIds.delete(id);
      dehydrateCompilation(details);
    }
  },
  true,
);
elements.highlightPreviewClose.addEventListener("click", closeHighlightPreview);
elements.highlightPreview.addEventListener("click", (event) => {
  if (event.target === elements.highlightPreview) closeHighlightPreview();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.highlightPreview.hidden) {
    event.preventDefault();
    closeHighlightPreview();
  }
});
elements.annotationDevList.addEventListener("click", (event) => {
  const workspaceButton = event.target.closest(".open-annotation-workspace");
  if (workspaceButton) openAnnotationWorkspace(workspaceButton);
});
elements.annotationWorkspaceClose.addEventListener("click", closeAnnotationWorkspace);
elements.annotationWorkspaceMarkStart.addEventListener("click", () => {
  markAnnotationWorkspaceBoundary("start");
});
elements.annotationWorkspaceMarkEnd.addEventListener("click", () => {
  markAnnotationWorkspaceBoundary("end");
});
elements.annotationWorkspaceNoteOtherToggle.addEventListener(
  "change",
  updateAnnotationWorkspaceNoteOther,
);
elements.annotationWorkspaceForm.addEventListener("submit", (event) => {
  event.preventDefault();
  saveAnnotationWorkspace();
});
elements.annotationWorkspace.addEventListener("compositionstart", () => {
  annotationWorkspaceComposing = true;
});
elements.annotationWorkspace.addEventListener("compositionend", () => {
  annotationWorkspaceComposing = false;
});
elements.annotationWorkspace.addEventListener("click", (event) => {
  const seekButton = event.target.closest("button[data-workspace-seek]");
  if (seekButton) {
    seekAnnotationWorkspace(Number(seekButton.dataset.workspaceSeek));
    return;
  }
  const previewButton = event.target.closest(".annotation-workspace-preview");
  if (previewButton) {
    const video = elements.annotationWorkspaceVideo;
    video.currentTime = Number(previewButton.dataset.start);
    video.dataset.stopAt = previewButton.dataset.end;
    video
      .play()
      .catch(() => showAnnotationWorkspaceMessage("瀏覽器無法播放這個原片編碼。", true));
    return;
  }
  const deleteButton = event.target.closest(".annotation-workspace-delete");
  if (deleteButton) deleteAnnotationWorkspaceItem(deleteButton);
});
elements.annotationWorkspaceVideo.addEventListener(
  "pointerup",
  () => {
    window.setTimeout(() => {
      if (annotationWorkspaceIsOpen()) {
        elements.annotationWorkspace.focus({ preventScroll: true });
      }
    }, 0);
  },
  { capture: true },
);
elements.annotationWorkspaceVideo.addEventListener("timeupdate", () => {
  const video = elements.annotationWorkspaceVideo;
  elements.annotationWorkspaceCurrent.textContent = formatTimestamp(video.currentTime);
  const stopAt = Number(video.dataset.stopAt);
  if (Number.isFinite(stopAt) && video.currentTime >= stopAt) {
    video.pause();
    delete video.dataset.stopAt;
  }
});
elements.annotationWorkspaceVideo.addEventListener("error", () => {
  showAnnotationWorkspaceMessage("瀏覽器無法播放這個原片編碼。", true);
});
document.addEventListener(
  "keydown",
  (event) => {
    if (!annotationWorkspaceIsOpen()) return;
    if (annotationWorkspaceComposing || event.isComposing || event.keyCode === 229) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeAnnotationWorkspace();
      return;
    }
    if (event.target.matches('input[type="text"], textarea')) {
      if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        elements.annotationWorkspaceForm.requestSubmit();
      }
      return;
    }
    if (event.target.matches('input[type="checkbox"][name^="annotation-note-tag"]')) {
      if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        elements.annotationWorkspaceForm.requestSubmit();
      }
      return;
    }
    if (
      event.target.matches(
        'input, select, textarea, button, a, [contenteditable="true"]',
      )
    ) {
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      event.stopPropagation();
      toggleAnnotationWorkspacePlayback();
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      event.stopPropagation();
      const direction = event.key === "ArrowLeft" ? -1 : 1;
      seekAnnotationWorkspace(direction * (event.shiftKey ? 5 : 1));
      return;
    }
    if (event.code === "KeyI") {
      event.preventDefault();
      event.stopPropagation();
      markAnnotationWorkspaceBoundary("start");
      return;
    }
    if (event.code === "KeyO") {
      event.preventDefault();
      event.stopPropagation();
      markAnnotationWorkspaceBoundary("end");
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      saveAnnotationWorkspace();
    }
  },
  { capture: true },
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
desktopLibraryMedia.addEventListener("change", (event) => {
  if (event.matches) {
    void loadLibraryActivity();
  } else {
    closeHighlightPreview();
    for (const details of elements.compilationList.querySelectorAll(
      "details.compilation-output[open]",
    )) {
      details.open = false;
      dehydrateCompilation(details);
    }
    expandedCompilationIds.clear();
  }
});
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
