const i18n = window.HighlightCraftI18n;
const { clear: clearLocalizedElement, setHtml, setText, t } = i18n;

const elements = {
  languageToggle: document.querySelector("#languageToggle"),
  loginView: document.querySelector("#loginView"),
  loginForm: document.querySelector("#loginForm"),
  loginUsername: document.querySelector("#loginUsername"),
  loginPassword: document.querySelector("#loginPassword"),
  loginButton: document.querySelector("#loginButton"),
  loginMessage: document.querySelector("#loginMessage"),
  appShell: document.querySelector("#appShell"),
  appFooter: document.querySelector("#appFooter"),
  sessionControls: document.querySelector("#sessionControls"),
  accountAvatar: document.querySelector("#accountAvatar"),
  accountName: document.querySelector("#accountName"),
  accountRole: document.querySelector("#accountRole"),
  logoutButton: document.querySelector("#logoutButton"),
  guideButton: document.querySelector("#guideButton"),
  quickGuide: document.querySelector("#quickGuide"),
  accountSecurity: document.querySelector("#accountSecurity"),
  changePasswordForm: document.querySelector("#changePasswordForm"),
  currentPassword: document.querySelector("#currentPassword"),
  changedPassword: document.querySelector("#changedPassword"),
  changedPasswordConfirm: document.querySelector("#changedPasswordConfirm"),
  changePasswordSubmit: document.querySelector("#changePasswordSubmit"),
  changePasswordMessage: document.querySelector("#changePasswordMessage"),
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
  adminPanel: document.querySelector("#adminPanel"),
  adminRefreshButton: document.querySelector("#adminRefreshButton"),
  storageSummary: document.querySelector("#storageSummary"),
  createUserForm: document.querySelector("#createUserForm"),
  newUsername: document.querySelector("#newUsername"),
  newDisplayName: document.querySelector("#newDisplayName"),
  newPassword: document.querySelector("#newPassword"),
  newRole: document.querySelector("#newRole"),
  createUserButton: document.querySelector("#createUserButton"),
  createUserMessage: document.querySelector("#createUserMessage"),
  adminUserCount: document.querySelector("#adminUserCount"),
  adminUserList: document.querySelector("#adminUserList"),
  adminJobCount: document.querySelector("#adminJobCount"),
  adminPendingCount: document.querySelector("#adminPendingCount"),
  adminPendingList: document.querySelector("#adminPendingList"),
  adminJobList: document.querySelector("#adminJobList"),
  adminPagination: document.querySelector("#adminPagination"),
  adminPrevButton: document.querySelector("#adminPrevButton"),
  adminNextButton: document.querySelector("#adminNextButton"),
  adminPageLabel: document.querySelector("#adminPageLabel"),
  adminPasswordDialog: document.querySelector("#adminPasswordDialog"),
  adminPasswordForm: document.querySelector("#adminPasswordForm"),
  adminResetPassword: document.querySelector("#adminResetPassword"),
  adminPasswordMessage: document.querySelector("#adminPasswordMessage"),
  adminPasswordCancel: document.querySelector("#adminPasswordCancel"),
  annotationDevBlock: document.querySelector("#annotationDevBlock"),
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
let lastAnnotationDevSignature = "";
let currentUser = null;
let activityTimer = null;
let adminLoading = false;
let authGeneration = 0;
let requestController = new AbortController();
let identityRefreshPromise = null;
let adminOffset = 0;
let adminTotal = 0;
const adminLimit = 20;
const jobRenderSignatures = new Map();
const expandedResultJobIds = new Set();
let annotationWorkspaceJobId = "";
let annotationWorkspaceStart = null;
let annotationWorkspaceEnd = null;
let annotationWorkspaceReturnFocus = null;
let annotationWorkspaceComposing = false;
let adminPasswordResolver = null;
let latestActivityPayload = null;
let latestAdminPayload = null;
let latestAnnotationPayload = null;
let languageSwitchLocks = 0;
let languageSwitchEpoch = 0;

const annotationNoteMaxLength = 300;

const uploadActiveWindowMs = 60 * 1000;

const stageKeys = {
  queued: "stage.queued",
  "queued-after-restart": "stage.queued-after-restart",
  starting: "stage.starting",
  probing: "stage.probing",
  "audio-analysis": "stage.audio-analysis",
  "motion-analysis": "stage.motion-analysis",
  "detecting-points": "stage.detecting-points",
  "editing-point-reel": "stage.editing-point-reel",
  completed: "stage.completed",
  failed: "stage.failed",
};

function lockLanguageSwitch() {
  const epoch = languageSwitchEpoch;
  let released = false;
  languageSwitchLocks += 1;
  elements.languageToggle.disabled = true;
  return () => {
    if (released || epoch !== languageSwitchEpoch) return;
    released = true;
    languageSwitchLocks = Math.max(0, languageSwitchLocks - 1);
    elements.languageToggle.disabled = languageSwitchLocks > 0;
  };
}

function renderStorageLoading() {
  elements.storageSummary.innerHTML = `
    <article><span>${t("storage.used")}</span><b>${t("common.loading")}</b><small>${t("storage.sourceAndOutput")}</small></article>
    <article><span>${t("storage.sources")}</span><b>—</b><small>${t("storage.waiting")}</small></article>
    <article><span>${t("storage.outputs")}</span><b>—</b><small>${t("storage.waiting")}</small></article>
    <article><span>${t("storage.available")}</span><b>—</b><small>${t("storage.hostDisk")}</small></article>`;
}

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

function fileAccessUrl(path, { download = false } = {}) {
  const url = new URL(path, window.location.origin);
  if (download) url.searchParams.set("download", "true");
  return `${url.pathname}${url.search}`;
}

async function apiFetch(path, options = {}) {
  const {
    headers = {},
    signal = requestController.signal,
    ...requestOptions
  } = options;
  const response = await fetch(path, {
    credentials: "same-origin",
    ...requestOptions,
    headers,
    signal,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      message = (await response.json()).detail || message;
    } catch (_) {
      // Keep the HTTP status as the useful fallback.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response;
}

function userFromPayload(payload) {
  return payload?.user || payload;
}

function isAdmin() {
  return currentUser?.role === "admin";
}

function isUnauthorized(error) {
  return error?.status === 401;
}

function isAborted(error) {
  return error?.name === "AbortError";
}

function sessionIsCurrent(generation, userId = currentUser?.id) {
  return (
    generation === authGeneration &&
    currentUser !== null &&
    String(currentUser.id) === String(userId)
  );
}

function resetUserState(nextUser = null) {
  authGeneration += 1;
  requestController.abort();
  requestController = new AbortController();
  identityRefreshPromise = null;

  authReady = false;
  currentUser = nextUser;
  activityLoading = false;
  adminLoading = false;
  driveSubmitting = false;
  paused = false;
  uploadRunning = false;
  selectedFile = null;
  adminOffset = 0;
  adminTotal = 0;
  latestActivityPayload = null;
  latestAdminPayload = null;
  latestAnnotationPayload = null;
  languageSwitchEpoch += 1;
  languageSwitchLocks = 0;
  elements.languageToggle.disabled = false;

  if (activityTimer) {
    clearInterval(activityTimer);
    activityTimer = null;
  }
  releaseWakeLock();
  finishAdminPassword(null);
  if (annotationWorkspaceIsOpen()) closeAnnotationWorkspace();

  for (const video of [
    ...elements.jobList.querySelectorAll("video"),
    ...elements.adminJobList.querySelectorAll("video"),
  ]) {
    video.pause();
    video.removeAttribute("src");
    for (const source of video.querySelectorAll("source")) source.removeAttribute("src");
    video.load();
  }

  elements.appShell.hidden = true;
  elements.appFooter.hidden = true;
  elements.sessionControls.hidden = true;
  elements.adminPanel.hidden = true;
  elements.annotationDevBlock.hidden = true;
  elements.accountAvatar.textContent = "";
  elements.accountName.textContent = "";
  elements.accountRole.textContent = "";
  elements.videoInput.value = "";
  elements.videoInput.disabled = false;
  elements.dropZone.classList.remove("selected", "dragging");
  setText(elements.filePrompt, "upload.filePrompt");
  setText(elements.fileMeta, "upload.fileMeta");
  elements.uploadButton.disabled = true;
  setText(elements.uploadButton.querySelector("span"), "upload.start");
  elements.transferPanel.hidden = true;
  setText(elements.transferLabel, "transfer.preparing");
  elements.transferPercent.textContent = "0%";
  elements.transferBar.style.width = "0%";
  setText(elements.transferDetail, "transfer.creatingSession");
  elements.pauseButton.hidden = false;
  setText(elements.pauseButton, "transfer.pause");
  elements.driveForm.reset();
  showDriveMessage("");
  updateDriveButton();

  elements.emptyJobs.hidden = false;
  setText(elements.jobCount, "library.waiting");
  elements.importList.replaceChildren();
  elements.uploadList.replaceChildren();
  elements.jobList.replaceChildren();
  setText(elements.annotationDevCount, "annotation.waiting");
  elements.annotationDevEmpty.hidden = false;
  elements.annotationDevList.replaceChildren();
  elements.adminUserCount.textContent = "—";
  elements.adminJobCount.textContent = "—";
  elements.adminPendingCount.textContent = "—";
  elements.adminUserList.innerHTML = `<p class="admin-loading">${t("admin.loadingAccounts")}</p>`;
  elements.adminPendingList.innerHTML = `<p class="admin-loading">${t("admin.loadingPending")}</p>`;
  elements.adminJobList.innerHTML = `<p class="admin-loading">${t("admin.loadingVideos")}</p>`;
  renderStorageLoading();
  elements.adminPagination.hidden = true;
  setText(elements.adminPageLabel, "admin.pageOne");
  elements.createUserForm.reset();
  elements.createUserButton.disabled = false;
  elements.adminRefreshButton.disabled = false;
  elements.createUserMessage.hidden = true;
  elements.quickGuide.open = false;
  elements.accountSecurity.open = false;
  elements.changePasswordForm.reset();
  elements.changePasswordSubmit.disabled = false;
  elements.changePasswordMessage.hidden = true;
  elements.changePasswordMessage.classList.remove("error");
  setText(elements.annotationWorkspaceFilename, "annotation.noVideo");
  elements.annotationWorkspaceCurrent.textContent = "0:00.0";
  setText(elements.annotationWorkspaceStart, "annotation.notSet");
  setText(elements.annotationWorkspaceEnd, "annotation.notSet");
  elements.annotationWorkspaceLabel.value = "highlight";
  resetAnnotationWorkspaceNote();
  elements.annotationWorkspaceSave.disabled = false;
  setHtml(elements.annotationWorkspaceSave, "annotation.saveHtml");
  setText(elements.annotationWorkspaceCount, "annotation.zeroCount");
  elements.annotationWorkspaceList.innerHTML = `<p>${t("annotation.openToLoad")}</p>`;
  showAnnotationWorkspaceMessage("");

  lastImportsSignature = "";
  lastUploadsSignature = "";
  lastJobsSignature = "";
  lastAnnotationDevSignature = "";
  jobRenderSignatures.clear();
  expandedResultJobIds.clear();
  return authGeneration;
}

function showLogin(messageKey = "", parameters = {}) {
  resetUserState();
  elements.loginView.hidden = false;
  elements.loginUsername.value = "";
  elements.loginPassword.value = "";
  if (messageKey) setText(elements.loginMessage, messageKey, parameters);
  else {
    clearLocalizedElement(elements.loginMessage);
    elements.loginMessage.textContent = "";
  }
  elements.loginMessage.hidden = !messageKey;
  elements.loginButton.disabled = false;
  setText(elements.loginButton.querySelector("span"), "login.submit");
  window.setTimeout(() => elements.loginUsername.focus({ preventScroll: true }), 0);
}

function showApplication(user, { reveal = true } = {}) {
  currentUser = user;
  const displayName = user.display_name || user.username;
  clearLocalizedElement(elements.accountName);
  elements.accountName.textContent = displayName;
  elements.accountRole.textContent = `${user.role === "admin" ? "ADMIN" : "USER"} · @${user.username}`;
  elements.accountAvatar.textContent = displayName.trim().charAt(0).toUpperCase() || "U";
  elements.loginView.hidden = reveal;
  elements.appShell.hidden = !reveal;
  elements.appFooter.hidden = !reveal;
  elements.sessionControls.hidden = !reveal;
  elements.adminPanel.hidden = user.role !== "admin";
  elements.annotationDevBlock.hidden = user.role !== "admin";
}

async function refreshIdentityAfterForbidden(expectedGeneration) {
  if (!sessionIsCurrent(expectedGeneration)) return;
  if (identityRefreshPromise) {
    await identityRefreshPromise;
    return;
  }

  const expectedUserId = currentUser.id;
  const refresh = (async () => {
    try {
      const response = await apiFetch("/api/auth/me");
      const user = userFromPayload(await response.json());
      if (!sessionIsCurrent(expectedGeneration, expectedUserId)) return;
      if (!user?.id || String(user.id) !== String(expectedUserId)) {
        await initializeApplication(user);
        return;
      }
      const wasAdmin = isAdmin();
      showApplication(user);
      if (user.role !== "admin") {
        adminLoading = false;
        elements.adminPanel.hidden = true;
        elements.annotationDevBlock.hidden = true;
        if (annotationWorkspaceIsOpen()) closeAnnotationWorkspace();
      } else if (!wasAdmin) {
        await loadAdminDashboard();
      }
    } catch (error) {
      if (isUnauthorized(error) && sessionIsCurrent(expectedGeneration, expectedUserId)) {
        showLogin("error.sessionExpired");
      }
    }
  })();
  identityRefreshPromise = refresh;
  try {
    await refresh;
  } finally {
    if (identityRefreshPromise === refresh) identityRefreshPromise = null;
  }
}

async function handleAuthorizationError(error, generation, messageKey = "error.sessionExpired") {
  if (isAborted(error)) return true;
  if (isUnauthorized(error)) {
    if (sessionIsCurrent(generation)) showLogin(messageKey);
    return true;
  }
  if (error?.status === 403) {
    await refreshIdentityAfterForbidden(generation);
    return true;
  }
  return false;
}

function encodeMetadata(value) {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function resumeKeyHash(value) {
  const text = String(value);
  let first = 0xdeadbeef ^ text.length;
  let second = 0x41c6ce57 ^ text.length;
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    first = Math.imul(first ^ code, 2654435761);
    second = Math.imul(second ^ code, 1597334677);
  }
  first = Math.imul(first ^ (first >>> 16), 2246822507) ^
    Math.imul(second ^ (second >>> 13), 3266489909);
  second = Math.imul(second ^ (second >>> 16), 2246822507) ^
    Math.imul(first ^ (first >>> 13), 3266489909);
  return `${(first >>> 0).toString(16).padStart(8, "0")}${(second >>> 0)
    .toString(16)
    .padStart(8, "0")}`;
}

function resumeStoragePrefix(userId = currentUser?.id) {
  return `pingpong-upload:v2:${resumeKeyHash(userId || "anonymous")}:`;
}

function fingerprint(file, userId = currentUser?.id) {
  const privateFingerprint = resumeKeyHash(
    `${file.name}\u0000${file.size}\u0000${file.lastModified}`,
  );
  return `${resumeStoragePrefix(userId)}${privateFingerprint}`;
}

function localStorageKeys() {
  try {
    return Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index))
      .filter(Boolean);
  } catch (_) {
    return [];
  }
}

function readLocalStorage(key) {
  try {
    return localStorage.getItem(key);
  } catch (_) {
    return null;
  }
}

function writeLocalStorage(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_) {
    // Uploading still works; only browser-side resume discovery is unavailable.
  }
}

function removeLocalStorage(key) {
  try {
    localStorage.removeItem(key);
  } catch (_) {
    // Nothing to remove when browser storage is unavailable.
  }
}

function purgeLegacyResumeKeys() {
  const staleKeys = localStorageKeys().filter(
    (key) => key.startsWith("pingpong-upload:") && !key.startsWith("pingpong-upload:v2:"),
  );
  for (const key of staleKeys) removeLocalStorage(key);
  removeLocalStorage("pingpong-upload-token");
}

async function createSession(
  file,
  { signal = requestController.signal, userId = currentUser?.id } = {},
) {
  const metadata = [
    `filename ${encodeMetadata(file.name)}`,
    `filetype ${encodeMetadata(file.type || "application/octet-stream")}`,
  ].join(",");
  const response = await apiFetch("/api/uploads", {
    method: "POST",
    signal,
    headers: {
      "Tus-Resumable": "1.0.0",
      "Upload-Length": String(file.size),
      "Upload-Metadata": metadata,
    },
  });
  const location = response.headers.get("Location");
  if (!location) throw new Error(t("upload.errorLocation"));
  writeLocalStorage(fingerprint(file, userId), location);
  return { location, offset: 0, jobId: null };
}

async function inspectSession(location, file, { signal = requestController.signal } = {}) {
  const response = await apiFetch(location, {
    method: "HEAD",
    signal,
    headers: { "Tus-Resumable": "1.0.0" },
  });
  const length = Number(response.headers.get("Upload-Length"));
  if (length !== file.size) throw new Error(t("upload.errorSize"));
  return {
    location,
    offset: Number(response.headers.get("Upload-Offset")) || 0,
    jobId: null,
  };
}

async function findOrCreateSession(
  file,
  { signal = requestController.signal, userId = currentUser?.id } = {},
) {
  const fileFingerprint = fingerprint(file, userId);
  const saved = readLocalStorage(fileFingerprint);
  if (saved) {
    try {
      return await inspectSession(saved, file, { signal });
    } catch (error) {
      if (isAborted(error)) throw error;
      if (!String(error.message).includes("404")) console.info("Starting a new upload:", error);
      removeLocalStorage(fileFingerprint);
    }
  }

  const response = await apiFetch("/api/uploads", { signal });
  const { uploads } = await response.json();
  const matches = uploads.filter(
    (upload) => upload.filename === file.name && upload.size === file.size,
  );
  if (matches.length > 1) {
    throw new Error(t("upload.errorDuplicates", { count: matches.length }));
  }
  if (matches.length === 1) {
    const location = `/api/uploads/${matches[0].id}`;
    const session = await inspectSession(location, file, { signal });
    writeLocalStorage(fileFingerprint, location);
    return session;
  }
  return createSession(file, { signal, userId });
}

async function checksumHeader(blob) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  const bytes = new Uint8Array(digest);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return `sha256 ${btoa(binary)}`;
}

async function serverOffset(location, { signal = requestController.signal } = {}) {
  const response = await apiFetch(location, {
    method: "HEAD",
    signal,
    headers: { "Tus-Resumable": "1.0.0" },
  });
  return Number(response.headers.get("Upload-Offset")) || 0;
}

async function sendChunk(location, offset, blob, { signal = requestController.signal } = {}) {
  const checksum = await checksumHeader(blob);
  let lastError = null;
  for (const delay of [0, 700, 1800, 4000]) {
    if (signal.aborted) {
      const error = new Error("Upload request was cancelled");
      error.name = "AbortError";
      throw error;
    }
    if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
    try {
      const headers = {
        "Tus-Resumable": "1.0.0",
        "Upload-Offset": String(offset),
        "Content-Type": "application/offset+octet-stream",
      };
      if (checksum) headers["Upload-Checksum"] = checksum;
      const response = await apiFetch(location, {
        method: "PATCH",
        headers,
        body: blob,
        signal,
      });
      return {
        offset: Number(response.headers.get("Upload-Offset")),
        jobId: response.headers.get("Upload-Job-Id"),
      };
    } catch (error) {
      if (isAborted(error)) throw error;
      lastError = error;
      try {
        const recovered = await serverOffset(location, { signal });
        if (recovered > offset) return { offset: recovered, jobId: null };
      } catch (_) {
        // The retry loop will surface the original transfer error.
      }
    }
  }
  throw lastError || new Error(t("upload.errorChunks"));
}

function setTransferProgress(offset, total, startedAt) {
  const fraction = total ? Math.min(1, offset / total) : 0;
  const percent = Math.round(fraction * 100);
  const elapsed = Math.max(0.25, (performance.now() - startedAt) / 1000);
  const speed = offset / elapsed;
  elements.transferPercent.textContent = `${percent}%`;
  elements.transferBar.style.width = `${percent}%`;
  setText(elements.transferDetail, "upload.progress", {
    offset: formatBytes(offset),
    total: formatBytes(total),
    speed: formatBytes(speed),
  });
}

async function acquireWakeLock() {
  try {
    wakeLock = await navigator.wakeLock?.request("screen");
  } catch (_) {
    wakeLock = null;
  }
  return wakeLock;
}

async function releaseWakeLock(lock = wakeLock) {
  try {
    await lock?.release();
  } catch (_) {
    // The browser may already have released it when the tab lost focus.
  }
  if (wakeLock === lock) wakeLock = null;
}

async function startUpload() {
  if (!selectedFile || uploadRunning || !authReady) return;
  const generation = authGeneration;
  const userId = currentUser.id;
  const signal = requestController.signal;
  const file = selectedFile;
  uploadRunning = true;
  paused = false;
  elements.uploadButton.disabled = true;
  elements.videoInput.disabled = true;
  elements.transferPanel.hidden = false;
  elements.pauseButton.hidden = false;
  setText(elements.pauseButton, "transfer.pause");
  setText(elements.transferLabel, "upload.creatingConnection");
  const startedAt = performance.now();
  const uploadWakeLock = await acquireWakeLock();

  try {
    if (!sessionIsCurrent(generation, userId)) return;
    const session = await findOrCreateSession(file, { signal, userId });
    if (!sessionIsCurrent(generation, userId)) return;
    let offset = session.offset;
    let jobId = session.jobId;
    setTransferProgress(offset, file.size, startedAt);
    setText(elements.transferLabel, offset ? "upload.continuing" : "upload.sending");

    while (offset < file.size) {
      while (paused) await new Promise((resolve) => setTimeout(resolve, 250));
      if (!sessionIsCurrent(generation, userId)) return;
      const end = Math.min(offset + chunkSize, file.size);
      const result = await sendChunk(session.location, offset, file.slice(offset, end), {
        signal,
      });
      if (!sessionIsCurrent(generation, userId)) return;
      if (!Number.isFinite(result.offset) || result.offset <= offset) {
        throw new Error(t("upload.errorOffset"));
      }
      offset = result.offset;
      jobId = result.jobId || jobId;
      setTransferProgress(offset, file.size, startedAt);
    }

    if (!jobId) {
      const response = await apiFetch(session.location, { signal });
      jobId = (await response.json()).job_id;
    }
    if (!sessionIsCurrent(generation, userId)) return;
    setText(elements.transferLabel, "upload.delivered");
    setText(elements.transferDetail, "upload.queuedDetail");
    elements.pauseButton.hidden = true;
    await loadActivity();
  } catch (error) {
    if (!isAborted(error) && sessionIsCurrent(generation, userId)) {
      setText(elements.transferLabel, "upload.paused");
      setText(elements.transferDetail, "upload.pausedDetail", { error: error.message });
      elements.uploadButton.disabled = false;
    }
  } finally {
    if (sessionIsCurrent(generation, userId)) {
      uploadRunning = false;
      elements.videoInput.disabled = false;
    }
    await releaseWakeLock(uploadWakeLock);
  }
}

function jobStage(job) {
  if (job.stage?.startsWith("exporting-point-")) {
    return t("stage.exportingPoint", { number: job.stage.split("-").at(-1) });
  }
  return stageKeys[job.stage] ? t(stageKeys[job.stage]) : job.stage || t("stage.waiting");
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
  const generation = authGeneration;
  const userId = currentUser?.id;
  const signal = requestController.signal;
  const path = button.dataset.url;
  const filename = button.dataset.filename || "best_points_reel.mp4";
  const fallbackUrl = fileAccessUrl(path, { download: true });
  if (!navigator.share) {
    triggerDownload(fallbackUrl, filename);
    return;
  }

  const releaseLanguageSwitch = lockLanguageSwitch();
  button.disabled = true;
  setText(button, "result.preparingShare");
  try {
    const response = await apiFetch(path, { signal });
    const blob = await response.blob();
    if (!sessionIsCurrent(generation, userId)) return;
    const file = new File([blob], filename, { type: blob.type || "video/mp4" });
    if (navigator.canShare && !navigator.canShare({ files: [file] })) {
      triggerDownload(fallbackUrl, filename);
      return;
    }
    await navigator.share({
      files: [file],
      title: t("result.shareTitle"),
    });
  } catch (error) {
    if (!isAborted(error) && sessionIsCurrent(generation, userId)) {
      if (!(await handleAuthorizationError(error, generation))) {
        triggerDownload(fallbackUrl, filename);
      }
    }
  } finally {
    releaseLanguageSwitch();
    if (sessionIsCurrent(generation, userId) && button.isConnected) {
      button.disabled = false;
      setText(button, "result.shareSave");
    }
  }
}

function renderResultPanel(result, jobId, sourceName) {
  const files = Array.isArray(result?.files) ? result.files : [];
  const reel = files.find((file) => file.kind === "reel");
  const pointFiles = files.filter((file) => file.kind === "point" || file.kind === "clip");
  const analysis = files.find((file) => file.kind === "analysis");
  if (!reel) {
    return files.length
      ? `<div class="downloads">${files
      .map((file) => {
        const url = fileAccessUrl(file.url, { download: true });
        return `<a href="${escapeHtml(url)}" download>${escapeHtml(file.name)}</a>`;
      })
      .join("")}</div>`
      : "";
  }

  const previewUrl = fileAccessUrl(reel.url);
  const downloadUrl = fileAccessUrl(reel.url, { download: true });
  const webShareAvailable = typeof navigator.share === "function";
  const shareAction = webShareAvailable
    ? `<button class="share-button" type="button" data-url="${escapeHtml(reel.url)}" data-filename="${escapeHtml(reel.name)}">${t("result.shareSave")}</button>`
    : "";
  const saveHint = webShareAvailable
    ? t("result.mobileSaveHint")
    : t("result.downloadSaveHint");
  const pointLinks = pointFiles
    .map((file, index) => {
      const url = fileAccessUrl(file.url, { download: true });
      return `<a href="${escapeHtml(url)}" download>${t("result.pointDownload", { number: index + 1 })}</a>`;
    })
    .join("");
  const analysisLink = analysis
    ? `<a href="${escapeHtml(fileAccessUrl(analysis.url, { download: true }))}" download>${t("result.analysis")}</a>`
    : "";

  const open = expandedResultJobIds.has(jobId) ? " open" : "";
  return `<details class="result-panel" data-result-job-id="${escapeHtml(jobId)}"${open}>
    <summary class="reel-heading">
      <span class="sr-only">${escapeHtml(t("result.srLabel", { source: sourceName }))}</span>
      <span class="reel-heading-copy"><span>BEST POINTS REEL</span><b>${escapeHtml(reel.name)}</b></span>
      <span class="reel-toggle-label"><span class="reel-toggle-closed">${t("result.playDownload")}</span><span class="reel-toggle-open">${t("result.collapse")}</span></span>
      <i class="reel-toggle-icon" aria-hidden="true"></i>
    </summary>
    <div class="result-panel-body">
      <video controls playsinline preload="metadata" aria-label="${escapeHtml(t("result.previewLabel", { source: sourceName }))}">
        <source data-src="${escapeHtml(previewUrl)}" type="video/mp4" />
      </video>
      <div class="result-actions">
        <a class="result-primary" href="${escapeHtml(downloadUrl)}" download>${t("result.downloadMp4")}</a>
        ${shareAction}
      </div>
      <p class="save-hint">${saveHint}</p>
      <details class="more-files">
        <summary>${t("result.moreFiles")}</summary>
        <div class="downloads">${pointLinks}${analysisLink}</div>
      </details>
    </div>
  </details>`;
}

function annotationWorkspaceIsOpen() {
  return !elements.annotationWorkspace.hidden;
}

function showAnnotationWorkspaceMessage(message, isError = false) {
  clearLocalizedElement(elements.annotationWorkspaceMessage);
  elements.annotationWorkspaceMessage.textContent = message;
  elements.annotationWorkspaceMessage.hidden = !message;
  elements.annotationWorkspaceMessage.classList.toggle("error", isError);
}

function showAnnotationWorkspaceMessageKey(key, parameters = {}, isError = false) {
  setText(elements.annotationWorkspaceMessage, key, parameters);
  elements.annotationWorkspaceMessage.hidden = false;
  elements.annotationWorkspaceMessage.classList.toggle("error", isError);
}

function renderAnnotationWorkspaceBoundaries() {
  if (annotationWorkspaceStart === null) setText(elements.annotationWorkspaceStart, "annotation.notSet");
  else {
    clearLocalizedElement(elements.annotationWorkspaceStart);
    elements.annotationWorkspaceStart.textContent = formatTimestamp(annotationWorkspaceStart);
  }
  if (annotationWorkspaceEnd === null) setText(elements.annotationWorkspaceEnd, "annotation.notSet");
  else {
    clearLocalizedElement(elements.annotationWorkspaceEnd);
    elements.annotationWorkspaceEnd.textContent = formatTimestamp(annotationWorkspaceEnd);
  }
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
  latestAnnotationPayload = annotations;
  setText(
    elements.annotationWorkspaceCount,
    annotations.length === 1 ? "annotation.count.one" : "annotation.count.other",
    { count: annotations.length },
  );
  if (!annotations.length) {
    elements.annotationWorkspaceList.innerHTML = `<p>${t("annotation.empty")}</p>`;
    return;
  }
  const translatedPresetTags = new Map(
    elements.annotationWorkspaceNoteTags.map((checkbox) => [
      checkbox.value,
      t(checkbox.closest("label")?.querySelector("span")?.dataset.i18n || ""),
    ]),
  );
  const localizeNote = (value) => String(value)
    .split("、")
    .map((part) => translatedPresetTags.get(part) || part)
    .join(i18n.language === "en" ? ", " : "、");
  elements.annotationWorkspaceList.innerHTML = annotations
    .map((annotation, index) => {
      const label = t(annotation.label === "highlight" ? "annotation.include" : "annotation.exclude");
      const note = annotation.note ? `<small>${escapeHtml(localizeNote(annotation.note))}</small>` : "";
      return `<article class="annotation-workspace-item ${escapeHtml(annotation.label)}">
        <button class="annotation-workspace-preview" type="button" data-start="${annotation.start}" data-end="${annotation.end}" aria-label="${t("annotation.playLabel", { number: index + 1 })}">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <div><b>${label}</b><time>${formatTimestamp(annotation.start)}–${formatTimestamp(annotation.end)} · ${t("annotation.duration", { seconds: Number(annotation.duration).toFixed(1) })}</time>${note}</div>
        </button>
        <button class="annotation-workspace-delete" type="button" data-annotation-id="${escapeHtml(annotation.id)}" aria-label="${t("annotation.deleteLabel", { number: index + 1 })}">×</button>
      </article>`;
    })
    .join("");
}

async function loadAnnotationWorkspaceList() {
  if (!annotationWorkspaceJobId) return;
  const generation = authGeneration;
  const userId = currentUser?.id;
  const jobId = annotationWorkspaceJobId;
  elements.annotationWorkspaceList.innerHTML = `<p>${t("annotation.loading")}</p>`;
  try {
    const response = await apiFetch(
      `/api/jobs/${jobId}/annotations`,
    );
    const payload = await response.json();
    if (!sessionIsCurrent(generation, userId) || annotationWorkspaceJobId !== jobId) return;
    renderAnnotationWorkspaceList(payload.annotations || []);
  } catch (error) {
    if (
      !(await handleAuthorizationError(error, generation)) &&
      sessionIsCurrent(generation, userId) &&
      annotationWorkspaceJobId === jobId
    ) {
      elements.annotationWorkspaceList.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
    }
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
  if (button.dataset.sourceName) {
    clearLocalizedElement(elements.annotationWorkspaceFilename);
    elements.annotationWorkspaceFilename.textContent = button.dataset.sourceName;
  } else {
    setText(elements.annotationWorkspaceFilename, "annotation.sourceVideo");
  }
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
  latestAnnotationPayload = null;
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
      .catch(() => showAnnotationWorkspaceMessageKey("annotation.playbackError", {}, true));
  } else {
    video.pause();
  }
}

async function saveAnnotationWorkspace() {
  if (elements.annotationWorkspaceSave.disabled) return;
  const generation = authGeneration;
  const userId = currentUser?.id;
  const jobId = annotationWorkspaceJobId;
  if (
    annotationWorkspaceStart === null ||
    annotationWorkspaceEnd === null ||
    annotationWorkspaceEnd <= annotationWorkspaceStart
  ) {
    showAnnotationWorkspaceMessageKey("annotation.rangeError", {}, true);
    return;
  }
  const note = annotationWorkspaceNoteValue();
  if (note.length > annotationNoteMaxLength) {
    showAnnotationWorkspaceMessageKey("annotation.noteLengthError", {}, true);
    if (elements.annotationWorkspaceNoteOtherToggle.checked) {
      elements.annotationWorkspaceNoteOther.focus({ preventScroll: true });
    }
    return;
  }
  if (
    elements.annotationWorkspaceNoteOtherToggle.checked &&
    !elements.annotationWorkspaceNoteOther.value.trim()
  ) {
    showAnnotationWorkspaceMessageKey("annotation.otherError", {}, true);
    elements.annotationWorkspaceNoteOther.focus({ preventScroll: true });
    return;
  }
  const button = elements.annotationWorkspaceSave;
  button.disabled = true;
  setText(button, "annotation.saving");
  try {
    await apiFetch(`/api/jobs/${jobId}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: annotationWorkspaceStart,
        end: annotationWorkspaceEnd,
        label: elements.annotationWorkspaceLabel.value,
        note,
      }),
    });
    if (!sessionIsCurrent(generation, userId) || annotationWorkspaceJobId !== jobId) return;
    annotationWorkspaceStart = null;
    annotationWorkspaceEnd = null;
    resetAnnotationWorkspaceNote();
    renderAnnotationWorkspaceBoundaries();
    showAnnotationWorkspaceMessageKey("annotation.saved");
    await loadAnnotationWorkspaceList();
  } catch (error) {
    if (
      !(await handleAuthorizationError(error, generation)) &&
      sessionIsCurrent(generation, userId) &&
      annotationWorkspaceJobId === jobId
    ) {
      showAnnotationWorkspaceMessage(error.message, true);
    }
  } finally {
    if (sessionIsCurrent(generation, userId) && annotationWorkspaceJobId === jobId) {
      button.disabled = false;
      setHtml(button, "annotation.saveHtml");
    }
  }
}

async function deleteAnnotationWorkspaceItem(button) {
  if (!window.confirm(t("annotation.confirmDelete"))) return;
  const generation = authGeneration;
  const userId = currentUser?.id;
  const jobId = annotationWorkspaceJobId;
  const releaseLanguageSwitch = lockLanguageSwitch();
  button.disabled = true;
  try {
    await apiFetch(
      `/api/jobs/${jobId}/annotations/${button.dataset.annotationId}`,
      { method: "DELETE" },
    );
    if (!sessionIsCurrent(generation, userId) || annotationWorkspaceJobId !== jobId) return;
    showAnnotationWorkspaceMessageKey("annotation.deleted");
    await loadAnnotationWorkspaceList();
  } catch (error) {
    if (
      !(await handleAuthorizationError(error, generation)) &&
      sessionIsCurrent(generation, userId) &&
      annotationWorkspaceJobId === jobId
    ) {
      showAnnotationWorkspaceMessage(error.message, true);
      button.disabled = false;
    }
  } finally {
    releaseLanguageSwitch();
  }
}

function uploadIsActive(upload) {
  const updatedAt = Date.parse(upload.updated_at);
  return Number.isFinite(updatedAt) && Date.now() - updatedAt <= uploadActiveWindowMs;
}

function hasLocalResumeSession(upload) {
  const expectedPath = `/api/uploads/${upload.id}`;
  const expectedPrefix = resumeStoragePrefix();
  for (const key of localStorageKeys()) {
    if (!key.startsWith(expectedPrefix)) continue;
    const saved = readLocalStorage(key);
    try {
      if (saved && new URL(saved, window.location.origin).pathname === expectedPath) return true;
    } catch (_) {
      // Ignore malformed values written by an older build or browser extension.
    }
  }
  return false;
}

function forgetLocalResumeSession(uploadId) {
  const expectedPath = `/api/uploads/${uploadId}`;
  const matchingKeys = [];
  for (const key of localStorageKeys()) {
    if (!key.startsWith("pingpong-upload:")) continue;
    const saved = readLocalStorage(key);
    try {
      if (saved && new URL(saved, window.location.origin).pathname === expectedPath) {
        matchingKeys.push(key);
      }
    } catch (_) {
      // Ignore unrelated malformed local-storage values.
    }
  }
  for (const key of matchingKeys) removeLocalStorage(key);
}

function uploadProgress(upload) {
  const raw = upload.size ? Math.min(100, (upload.offset / upload.size) * 100) : 0;
  const value = upload.offset < upload.size ? Math.min(raw, 99.9) : 100;
  const decimals = value > 0 && value < 100 ? 1 : 0;
  return { value, label: `${value.toFixed(decimals)}%` };
}

function uploadUpdatedLabel(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return t("common.recentlyUpdated");
  return t("common.lastUpdated", {
    time: date.toLocaleTimeString(i18n.locale(), { hour: "2-digit", minute: "2-digit" }),
  });
}

function driveImportProgress(record) {
  if (!record.size) return null;
  const raw = Math.min(100, (record.offset / record.size) * 100);
  const value = record.offset < record.size ? Math.min(raw, 99.9) : 100;
  const decimals = value > 0 && value < 100 ? 1 : 0;
  return { value, label: `${value.toFixed(decimals)}%` };
}

function renderRecordOwner(record, sourceLabel) {
  const createdAt = record.created_at ? formatDateTime(record.created_at) : "";
  return `<div class="admin-job-meta admin-pending-meta">
    <span><b>${escapeHtml(jobOwnerName(record))}</b><small>${escapeHtml(sourceLabel)}${createdAt ? ` · ${escapeHtml(createdAt)}` : ""}</small></span>
  </div>`;
}

function renderDriveImport(record, { admin = false } = {}) {
  const statusKeys = {
    queued: "drive.statusQueued",
    resolving: "drive.statusResolving",
    downloading: "drive.statusDownloading",
    failed: "drive.statusFailed",
  };
  const details = record.error
    ? escapeHtml(record.error)
    : record.status === "queued"
      ? t("drive.detailQueued")
      : record.status === "resolving"
        ? t("drive.detailResolving")
        : t("drive.detailDownloading");
  const progress = driveImportProgress(record);
  const progressMeta = record.size
    ? `${formatBytes(record.offset)} / ${formatBytes(record.size)}`
    : record.offset
      ? t("drive.downloaded", { bytes: formatBytes(record.offset) })
      : t("drive.connecting");
  const progressBar =
    record.status === "downloading" || record.status === "resolving"
      ? `<div class="job-progress-meta"><span>${escapeHtml(progressMeta)} · ${escapeHtml(uploadUpdatedLabel(record.updated_at))}</span><b>${progress?.label || t("drive.downloading")}</b></div><div class="job-progress${progress ? "" : " indeterminate"}"><span${progress ? ` style="width:${progress.value}%"` : ""}></span></div>`
      : "";
  const actions =
    record.status === "failed"
      ? `<div class="import-actions"><button class="delete-import-button" type="button" data-import-id="${escapeHtml(record.id)}" data-label-key="common.delete">${t("common.delete")}</button><button class="retry-import-button" type="button" data-import-id="${escapeHtml(record.id)}">${t("drive.retry")}</button></div>`
      : record.status === "queued"
        ? `<div class="import-actions"><button class="delete-import-button" type="button" data-import-id="${escapeHtml(record.id)}" data-label-key="drive.cancel">${t("drive.cancel")}</button></div>`
        : "";
  const filename = record.filename || t("drive.defaultFilename");
  const status = statusKeys[record.status] ? t(statusKeys[record.status]) : record.status;

  return `<article class="job ${escapeHtml(record.status)}">
    ${admin ? renderRecordOwner(record, "Google Drive") : ""}
    <div class="job-title"><strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong><span class="status ${escapeHtml(record.status)}">${escapeHtml(status)}</span></div>
    <p class="job-detail">${details}</p>
    ${progressBar}${actions}
  </article>`;
}

async function retryDriveImport(button) {
  const importId = button.dataset.importId;
  if (!importId) return;
  const generation = authGeneration;
  const fromAdminDashboard = elements.adminPendingList.contains(button);
  const releaseLanguageSwitch = lockLanguageSwitch();
  button.disabled = true;
  setText(button, "drive.requeuing");
  try {
    await apiFetch(`/api/drive-imports/${encodeURIComponent(importId)}/retry`, {
      method: "POST",
    });
    if (!sessionIsCurrent(generation)) return;
    lastImportsSignature = "";
    await Promise.all([
      loadActivity(),
      fromAdminDashboard ? loadAdminDashboard() : Promise.resolve(),
    ]);
  } catch (error) {
    if (await handleAuthorizationError(error, generation)) return;
    window.alert(t("drive.retryError", { error: error.message }));
  } finally {
    releaseLanguageSwitch();
    if (sessionIsCurrent(generation) && button.isConnected) {
      button.disabled = false;
      setText(button, "drive.retry");
    }
  }
}

async function deleteDriveImport(button) {
  const importId = button.dataset.importId;
  if (!importId || !window.confirm(t("drive.confirmDelete"))) return;
  const generation = authGeneration;
  const fromAdminDashboard = elements.adminPendingList.contains(button);
  const labelKey = button.dataset.labelKey || "common.delete";
  const releaseLanguageSwitch = lockLanguageSwitch();
  button.disabled = true;
  setText(button, "drive.removing");
  try {
    await apiFetch(`/api/drive-imports/${encodeURIComponent(importId)}`, {
      method: "DELETE",
    });
    if (!sessionIsCurrent(generation)) return;
    lastImportsSignature = "";
    await Promise.all([
      loadActivity(),
      fromAdminDashboard ? loadAdminDashboard() : Promise.resolve(),
    ]);
  } catch (error) {
    if (await handleAuthorizationError(error, generation)) return;
    window.alert(t("drive.removeError", { error: error.message }));
  } finally {
    releaseLanguageSwitch();
    if (sessionIsCurrent(generation) && button.isConnected) {
      button.disabled = false;
      setText(button, labelKey);
    }
  }
}

function updateDriveButton() {
  elements.driveButton.disabled =
    !authReady || driveSubmitting || !elements.driveUrl.value.trim();
}

function showDriveMessage(message, isError = false) {
  clearLocalizedElement(elements.driveMessage);
  elements.driveMessage.textContent = message;
  elements.driveMessage.classList.toggle("error", isError);
  elements.driveMessage.hidden = !message;
}

function showDriveMessageKey(key, parameters = {}, isError = false) {
  setText(elements.driveMessage, key, parameters);
  elements.driveMessage.classList.toggle("error", isError);
  elements.driveMessage.hidden = false;
}

async function submitDriveLink(event) {
  event.preventDefault();
  const url = elements.driveUrl.value.trim();
  if (!url || driveSubmitting) return;
  const generation = authGeneration;
  const userId = currentUser?.id;
  driveSubmitting = true;
  updateDriveButton();
  showDriveMessageKey("drive.submitting");
  try {
    await apiFetch("/api/drive-imports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!sessionIsCurrent(generation, userId)) return;
    elements.driveUrl.value = "";
    lastImportsSignature = "";
    showDriveMessageKey("drive.submitted");
    await loadActivity();
  } catch (error) {
    if (!(await handleAuthorizationError(error, generation)) && sessionIsCurrent(generation, userId)) {
      showDriveMessage(error.message, true);
    }
  } finally {
    if (sessionIsCurrent(generation, userId)) {
      driveSubmitting = false;
      updateDriveButton();
    }
  }
}

function renderUpload(upload, { admin = false } = {}) {
  const progress = uploadProgress(upload);
  const active = upload.transfer_active;
  const resumableHere = upload.local_resume;
  const statusClass = active ? "uploading" : "waiting";
  const statusText = t(active ? "upload.statusActive" : "upload.statusWaiting");
  const details = resumableHere
    ? t("upload.resumeLocal")
    : active
      ? t("upload.resumeActive")
      : t("upload.resumeIdle");
  const transferred = `${formatBytes(upload.offset)} / ${formatBytes(upload.size)}`;

  return `<article class="job ${statusClass}">
    ${admin ? renderRecordOwner(upload, t("common.deviceUpload")) : ""}
    <div class="job-title"><strong title="${escapeHtml(upload.filename)}">${escapeHtml(upload.filename)}</strong><span class="status ${statusClass}">${statusText}</span></div>
    <p class="job-detail">${details}</p>
    <div class="job-progress-meta"><span>${escapeHtml(transferred)} · ${escapeHtml(uploadUpdatedLabel(upload.updated_at))}</span><b>${progress.label}</b></div>
    <div class="job-progress"><span style="width:${progress.value}%"></span></div>
    <div class="upload-actions"><button class="delete-upload-button" type="button" data-upload-id="${escapeHtml(upload.id)}" data-filename="${escapeHtml(upload.filename)}" data-transferred="${escapeHtml(transferred)}">${t("upload.deleteRecord")}</button></div>
  </article>`;
}

async function deleteUploadSession(button) {
  const uploadId = button.dataset.uploadId;
  if (!uploadId) return;
  const generation = authGeneration;
  const fromAdminDashboard = elements.adminPendingList.contains(button);
  const filename = button.dataset.filename || t("upload.thisVideo");
  const transferred = button.dataset.transferred || t("upload.transferredData");
  const confirmed = window.confirm(
    t("upload.confirmDelete", { filename, transferred }),
  );
  if (!confirmed) return;

  const releaseLanguageSwitch = lockLanguageSwitch();
  button.disabled = true;
  setText(button, "upload.deleting");
  try {
    await apiFetch(`/api/uploads/${encodeURIComponent(uploadId)}`, {
      method: "DELETE",
      headers: { "Tus-Resumable": "1.0.0" },
    });
    if (!sessionIsCurrent(generation)) return;
    forgetLocalResumeSession(uploadId);
    lastUploadsSignature = "";
    await Promise.all([
      loadActivity(),
      fromAdminDashboard ? loadAdminDashboard() : Promise.resolve(),
    ]);
  } catch (error) {
    if (await handleAuthorizationError(error, generation)) return;
    window.alert(t("upload.deleteError", { error: error.message }));
  } finally {
    releaseLanguageSwitch();
    if (sessionIsCurrent(generation) && button.isConnected) {
      button.disabled = false;
      setText(button, "upload.deleteRecord");
    }
  }
}

function renderAnnotationDevJob(job, index) {
  const result = job.result;
  const filename = result?.source_name || job.source_name || job.filename || t("video.fallbackName", { id: String(job.upload_id || job.id).slice(0, 8) });
  const duration = Number.isFinite(result?.media?.duration)
    ? t("video.sourceDuration", { duration: formatDuration(result.media.duration) })
    : t("video.completed");
  return `<article class="annotation-dev-item">
    <span class="annotation-dev-index">${String(index + 1).padStart(2, "0")}</span>
    <div class="annotation-dev-copy">
      <strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong>
      <small>${t("annotation.loadOnOpen", { duration })}</small>
    </div>
    <button class="annotation-dev-open open-annotation-workspace" type="button" data-job-id="${escapeHtml(job.id)}" data-source-name="${escapeHtml(filename)}" aria-label="${escapeHtml(t("annotation.openLabel", { filename }))}">
      <span>${t("annotation.open")}</span><small>I · O · Enter</small>
    </button>
  </article>`;
}

function renderAnnotationDevelopment(jobs) {
  const completedJobs = jobs.filter(
    (job) => job.status === "completed" && job.result,
  );
  setText(
    elements.annotationDevCount,
    completedJobs.length
      ? (completedJobs.length === 1
        ? "annotation.availableCount.one"
        : "annotation.availableCount.other")
      : "annotation.waiting",
    { count: completedJobs.length },
  );
  elements.annotationDevEmpty.hidden = completedJobs.length > 0;
  elements.annotationDevList.innerHTML = completedJobs
    .map(renderAnnotationDevJob)
    .join("");
}

function jobOwnerName(job) {
  return (
    job.owner?.display_name ||
    job.owner?.username ||
    job.owner_username ||
    job.username ||
    job.user?.display_name ||
    job.user?.username ||
    job.user_id ||
    t("common.unknownUser")
  );
}

function jobSourceName(job) {
  const source = job.source_type || job.source_kind || job.source;
  if (typeof source === "string") {
    if (source.toLowerCase().includes("drive")) return "Google Drive";
    if (source.toLowerCase().includes("upload")) return t("common.deviceUpload");
    return source;
  }
  return job.drive_import_id ? "Google Drive" : t("common.deviceUpload");
}

function formatDateTime(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "";
  return date.toLocaleString(i18n.locale(), {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderJobControls(job, { admin = false } = {}) {
  const jobId = escapeHtml(job.id);
  const canRetry = job.status === "failed";
  const canReprocess = job.status === "completed" || job.status === "failed";
  const canDelete = job.status !== "processing";
  return `<div class="admin-job-meta">
    ${admin ? `<span><b>${escapeHtml(jobOwnerName(job))}</b><small>${escapeHtml(jobSourceName(job))}${job.created_at ? ` · ${escapeHtml(formatDateTime(job.created_at))}` : ""}</small></span>` : ""}
    <span class="admin-job-actions">
      <a href="${escapeHtml(fileAccessUrl(`/api/jobs/${jobId}/source`, { download: true }))}" download>${t("job.downloadSource")}</a>
      ${canRetry ? `<button type="button" data-job-action="retry" data-job-id="${jobId}">${t("job.retry")}</button>` : ""}
      ${canReprocess ? `<button type="button" data-job-action="reprocess" data-job-id="${jobId}">${t("job.reprocess")}</button>` : ""}
      ${canDelete ? `<button class="danger" type="button" data-job-action="delete" data-job-id="${jobId}">${t("job.delete")}</button>` : ""}
    </span>
  </div>`;
}

function renderJob(job, { admin = false } = {}) {
  const jobId = String(job.id);
  const result = job.result;
  const filename = result?.source_name || job.source_name || job.filename || t("video.fallbackName", { id: String(job.upload_id || job.id).slice(0, 8) });
  const progress = Math.round((Number(job.progress) || 0) * 100);
  const statusText =
    job.status === "completed"
      ? t("job.statusComplete")
      : job.status === "failed"
        ? t("job.statusFailed")
        : job.status === "processing"
          ? t("job.statusProcessing")
          : t("job.statusQueued");
  const summary = result?.summary || {};
  const rawPointCount = Number(summary?.point_count ?? summary?.highlight_count ?? 0);
  const pointCount = Number.isFinite(rawPointCount) ? Math.max(0, Math.round(rawPointCount)) : 0;
  const details = job.error
    ? escapeHtml(job.error)
    : result
      ? pointCount
        ? t(pointCount === 1 ? "job.summaryPoints.one" : "job.summaryPoints.other", {
          count: pointCount,
        })
        : t("job.summaryNoPoints")
      : escapeHtml(jobStage(job));
  const stats = result
    ? `<div class="job-stats"><span>${t(pointCount === 1 ? "job.statPoints.one" : "job.statPoints.other", { count: `<b>${pointCount}</b>` })}</span>${summary.reel_duration ? `<span>${t("job.statReel", { duration: `<b>${formatDuration(summary.reel_duration)}</b>` })}</span>` : ""}<span>${t("job.statSource", { duration: `<b>${formatDuration(result.media?.duration)}</b>` })}</span></div>`
    : "";
  const resultPanel = result ? renderResultPanel(result, jobId, filename) : "";
  const progressBar =
    job.status === "processing" || job.status === "queued"
      ? `<div class="job-progress-meta"><span>${escapeHtml(jobStage(job))}</span><b>${progress}%</b></div><div class="job-progress"><span style="width:${progress}%"></span></div>`
      : "";
  return `<article class="job ${escapeHtml(job.status)}" data-job-id="${escapeHtml(jobId)}">
    ${renderJobControls(job, { admin })}
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

function captureResultPanelState(container) {
  return new Map(
    [...container.querySelectorAll(".result-panel[data-result-job-id]")]
      .filter((panel) => panel.open)
      .map((panel) => {
        const video = panel.querySelector(".result-panel-body > video");
        return [panel.dataset.resultJobId, {
          currentTime: Number(video?.currentTime) || 0,
          muted: video?.muted || false,
          paused: video?.paused ?? true,
          playbackRate: Number(video?.playbackRate ?? 1),
          volume: Number(video?.volume ?? 1),
        }];
      }),
  );
}

function restoreResultPanelState(container, states) {
  for (const [jobId, state] of states) {
    const panel = [...container.querySelectorAll(".result-panel[data-result-job-id]")]
      .find((candidate) => candidate.dataset.resultJobId === jobId);
    if (!panel) continue;
    panel.open = true;
    hydrateResultPanel(panel);
    const video = panel.querySelector(".result-panel-body > video");
    if (!video) continue;
    const restorePlayback = () => {
      video.currentTime = state.currentTime;
      video.muted = state.muted;
      video.playbackRate = state.playbackRate;
      video.volume = state.volume;
      if (!state.paused) video.play().catch(() => {});
    };
    if (video.readyState >= 1) restorePlayback();
    else video.addEventListener("loadedmetadata", restorePlayback, { once: true });
  }
}

function renderJobs(jobs) {
  const existingNodes = new Map(
    [...elements.jobList.children].map((node) => [node.dataset.jobId, node]),
  );
  const liveJobIds = new Set();
  const expandableJobIds = new Set();

  jobs.forEach((job, index) => {
    const jobId = String(job.id);
    const signature = JSON.stringify([i18n.language, job]);
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

function firstFinite(object, keys) {
  for (const key of keys) {
    const value = Number(object?.[key]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function renderStorageSummary(payload) {
  const summary = payload?.summary || payload || {};
  const sourceBytes = firstFinite(summary, [
    "source_bytes",
    "sources_bytes",
    "original_bytes",
    "uploads_bytes",
  ]);
  const outputBytes = firstFinite(summary, [
    "output_bytes",
    "outputs_bytes",
    "result_bytes",
  ]);
  const usedBytes =
    firstFinite(summary, ["used_bytes", "total_used_bytes"]) ??
    (sourceBytes !== null && outputBytes !== null ? sourceBytes + outputBytes : null);
  const capacityBytes = firstFinite(summary, [
    "capacity_bytes",
    "total_bytes",
    "disk_total_bytes",
  ]);
  const freeBytes =
    firstFinite(summary, ["free_bytes", "available_bytes", "disk_free_bytes"]) ??
    (capacityBytes !== null && usedBytes !== null ? Math.max(0, capacityBytes - usedBytes) : null);
  const sourceCount = firstFinite(summary, ["source_count", "upload_count", "original_count"]);
  const outputCount = firstFinite(summary, ["output_count", "result_count"]);
  const usedNote = capacityBytes
    ? t("storage.capacityPercent", {
      percent: Math.min(100, ((usedBytes || 0) / capacityBytes) * 100).toFixed(1),
    })
    : t("storage.sourceAndOutput");
  const card = (label, bytes, note) =>
    `<article><span>${label}</span><b>${bytes === null ? "—" : formatBytes(bytes)}</b><small>${note}</small></article>`;
  elements.storageSummary.innerHTML = [
    card(t("storage.used"), usedBytes, usedNote),
    card(
      t("storage.sources"),
      sourceBytes,
      sourceCount === null
        ? t("storage.userSources")
        : t(sourceCount === 1 ? "storage.sourceCount.one" : "storage.sourceCount.other", {
          count: sourceCount,
        }),
    ),
    card(
      t("storage.outputs"),
      outputBytes,
      outputCount === null
        ? t("storage.outputDescription")
        : t(outputCount === 1 ? "storage.outputCount.one" : "storage.outputCount.other", {
          count: outputCount,
        }),
    ),
    card(
      t("storage.available"),
      freeBytes,
      capacityBytes === null
        ? t("storage.hostDisk")
        : t("storage.totalCapacity", { bytes: formatBytes(capacityBytes) }),
    ),
  ].join("");
}

function renderAdminUsers(users) {
  setText(
    elements.adminUserCount,
    users.length === 1 ? "admin.accountCount.one" : "admin.accountCount.other",
    { count: users.length },
  );
  if (!users.length) {
    elements.adminUserList.innerHTML = `<p class="admin-empty">${t("admin.noAccounts")}</p>`;
    return;
  }
  elements.adminUserList.innerHTML = users
    .map((user) => {
      const isSelf = String(user.id) === String(currentUser?.id);
      const active = user.active !== false;
      const name = user.display_name || user.username;
      return `<article class="admin-user ${active ? "" : "inactive"}">
        <span class="admin-user-avatar" aria-hidden="true">${escapeHtml(name.trim().charAt(0).toUpperCase() || "U")}</span>
        <span class="admin-user-identity"><b>${escapeHtml(name)}</b><small>@${escapeHtml(user.username)}${isSelf ? t("admin.currentAccount") : ""}</small></span>
        <span class="admin-user-badges"><i class="role ${escapeHtml(user.role)}">${t(user.role === "admin" ? "common.admin" : "common.user")}</i><i class="active">${t(active ? "admin.active" : "admin.inactive")}</i></span>
        <span class="admin-user-actions">
          <button type="button" data-user-action="name" data-user-id="${escapeHtml(user.id)}" data-current-name="${escapeHtml(user.display_name || "")}">${t("admin.nameAction")}</button>
          <button type="button" data-user-action="password" data-user-id="${escapeHtml(user.id)}"${isSelf ? ` disabled title="${escapeHtml(t("admin.selfPasswordTitle"))}"` : ""}>${t("admin.passwordAction")}</button>
          <button type="button" data-user-action="role" data-user-id="${escapeHtml(user.id)}" data-current-role="${escapeHtml(user.role)}"${isSelf ? ` disabled title="${escapeHtml(t("admin.selfRoleTitle"))}"` : ""}>${t(user.role === "admin" ? "admin.demote" : "admin.promote")}</button>
          <button class="${active ? "danger" : "restore"}" type="button" data-user-action="active" data-user-id="${escapeHtml(user.id)}" data-current-active="${active}"${isSelf ? ` disabled title="${escapeHtml(t("admin.selfActiveTitle"))}"` : ""}>${t(active ? "admin.deactivate" : "admin.activate")}</button>
        </span>
      </article>`;
    })
    .join("");
}

function renderAdminPending(uploads, imports) {
  const records = [
    ...uploads.map((record) => ({ kind: "upload", record })),
    ...imports.map((record) => ({ kind: "drive", record })),
  ].sort(
    (left, right) =>
      (Date.parse(right.record.created_at) || 0) - (Date.parse(left.record.created_at) || 0),
  );
  setText(
    elements.adminPendingCount,
    records.length === 1 ? "admin.pendingCount.one" : "admin.pendingCount.other",
    { count: records.length },
  );
  elements.adminPendingList.innerHTML = records.length
    ? records
      .map(({ kind, record }) =>
        kind === "upload"
          ? renderUpload(
            {
              ...record,
              transfer_active: uploadIsActive(record),
              local_resume: false,
            },
            { admin: true },
          )
          : renderDriveImport(record, { admin: true }),
      )
      .join("")
    : `<p class="admin-empty">${t("admin.noPending")}</p>`;
  return records.length;
}

function renderAdminJobs(payload, pendingCount = 0) {
  const jobs = payload?.jobs || [];
  adminTotal = Number(payload?.total) || jobs.length;
  adminOffset = Number(payload?.offset) || adminOffset;
  const itemCount = adminTotal + pendingCount;
  setText(
    elements.adminJobCount,
    itemCount === 1 ? "admin.itemCount.one" : "admin.itemCount.other",
    { count: itemCount },
  );
  elements.adminJobList.innerHTML = jobs.length
    ? jobs.map((job) => renderJob(job, { admin: true })).join("")
    : `<p class="admin-empty">${t("admin.noVideos")}</p>`;
  const currentPage = Math.floor(adminOffset / adminLimit) + 1;
  const totalPages = Math.max(1, Math.ceil(adminTotal / adminLimit));
  elements.adminPagination.hidden = adminTotal <= adminLimit;
  setText(elements.adminPageLabel, "admin.page", {
    current: currentPage,
    total: totalPages,
  });
  elements.adminPrevButton.disabled = adminOffset <= 0;
  elements.adminNextButton.disabled = adminOffset + adminLimit >= adminTotal;
}

function renderAdminDashboard(payload) {
  renderAdminUsers(payload.users);
  const pendingCount = renderAdminPending(payload.uploads, payload.imports);
  renderAdminJobs(payload.jobs, pendingCount);
  renderStorageSummary(payload.storage);
}

async function changeOwnPassword(event) {
  event.preventDefault();
  const currentPassword = elements.currentPassword.value;
  const newPassword = elements.changedPassword.value;
  const confirmation = elements.changedPasswordConfirm.value;
  elements.changePasswordMessage.classList.remove("error");
  elements.changePasswordMessage.hidden = false;
  if (newPassword !== confirmation) {
    elements.changePasswordMessage.classList.add("error");
    setText(elements.changePasswordMessage, "account.passwordMismatch");
    elements.changedPasswordConfirm.focus();
    return;
  }

  const generation = authGeneration;
  const userId = currentUser?.id;
  elements.changePasswordSubmit.disabled = true;
  setText(elements.changePasswordMessage, "account.updatingPassword");
  try {
    const response = await apiFetch("/api/auth/change-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
      }),
    });
    const updatedUser = userFromPayload(await response.json());
    if (!sessionIsCurrent(generation, userId)) return;
    await initializeApplication(updatedUser);
    if (String(currentUser?.id) === String(userId)) {
      elements.accountSecurity.open = true;
      elements.changePasswordMessage.classList.remove("error");
      setText(elements.changePasswordMessage, "account.passwordUpdated");
      elements.changePasswordMessage.hidden = false;
    }
  } catch (error) {
    if (isAborted(error) || !sessionIsCurrent(generation, userId)) return;
    if (isUnauthorized(error) && String(error.message).includes("Current password")) {
      elements.changePasswordMessage.classList.add("error");
      setText(elements.changePasswordMessage, "account.currentPasswordWrong");
      elements.currentPassword.select();
      return;
    }
    if (await handleAuthorizationError(error, generation)) return;
    elements.changePasswordMessage.classList.add("error");
    setText(elements.changePasswordMessage, "account.passwordUpdateError", {
      error: error.message,
    });
  } finally {
    if (sessionIsCurrent(generation, userId)) elements.changePasswordSubmit.disabled = false;
  }
}

async function loadAdminDashboard() {
  if (!isAdmin() || adminLoading) return;
  const generation = authGeneration;
  const userId = currentUser.id;
  adminLoading = true;
  elements.adminRefreshButton.disabled = true;
  try {
    const responses = await Promise.all([
      apiFetch("/api/admin/users"),
      apiFetch("/api/uploads?scope=all"),
      apiFetch("/api/drive-imports?scope=all"),
      apiFetch(`/api/jobs?scope=all&limit=${adminLimit}&offset=${adminOffset}`),
      apiFetch("/api/storage"),
    ]);
    const [usersPayload, uploadsPayload, importsPayload, jobsPayload, storagePayload] =
      await Promise.all(responses.map((response) => response.json()));
    if (!sessionIsCurrent(generation, userId) || !isAdmin()) return;
    latestAdminPayload = {
      users: usersPayload.users || usersPayload || [],
      uploads: uploadsPayload.uploads || [],
      imports: importsPayload.imports || [],
      jobs: jobsPayload,
      storage: storagePayload,
    };
    renderAdminDashboard(latestAdminPayload);
  } catch (error) {
    if (!(await handleAuthorizationError(error, generation)) && sessionIsCurrent(generation, userId)) {
      elements.adminPendingList.innerHTML = `<p class="admin-empty error">${escapeHtml(t("admin.pendingLoadError", { error: error.message }))}</p>`;
      elements.adminJobList.innerHTML = `<p class="admin-empty error">${escapeHtml(t("admin.dataLoadError", { error: error.message }))}</p>`;
    }
  } finally {
    if (sessionIsCurrent(generation, userId)) {
      adminLoading = false;
      elements.adminRefreshButton.disabled = false;
    }
  }
}

async function createUser(event) {
  event.preventDefault();
  const generation = authGeneration;
  const userId = currentUser?.id;
  const payload = {
    username: elements.newUsername.value.trim(),
    display_name: elements.newDisplayName.value.trim() || null,
    password: elements.newPassword.value,
    role: elements.newRole.value,
  };
  elements.createUserButton.disabled = true;
  elements.createUserMessage.hidden = false;
  elements.createUserMessage.classList.remove("error");
  setText(elements.createUserMessage, "admin.creatingAccount");
  try {
    await apiFetch("/api/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!sessionIsCurrent(generation, userId)) return;
    elements.createUserForm.reset();
    setText(elements.createUserMessage, "admin.accountCreated", {
      username: payload.username,
    });
    await loadAdminDashboard();
  } catch (error) {
    if (!(await handleAuthorizationError(error, generation)) && sessionIsCurrent(generation, userId)) {
      elements.createUserMessage.classList.add("error");
      clearLocalizedElement(elements.createUserMessage);
      elements.createUserMessage.textContent = error.message;
    }
  } finally {
    if (sessionIsCurrent(generation, userId)) elements.createUserButton.disabled = false;
  }
}

async function patchAdminUser(button) {
  const action = button.dataset.userAction;
  const userId = button.dataset.userId;
  if (!action || !userId || button.disabled) return;
  const generation = authGeneration;
  const adminUserId = currentUser?.id;
  const payload = {};
  if (action === "name") {
    const displayName = window.prompt(
      t("admin.displayNamePrompt"),
      button.dataset.currentName || "",
    );
    if (displayName === null) return;
    payload.display_name = displayName.trim() || null;
  } else if (action === "password") {
    const password = await requestAdminPassword();
    if (password === null) return;
    payload.password = password;
  } else if (action === "role") {
    payload.role = button.dataset.currentRole === "admin" ? "user" : "admin";
    if (!window.confirm(t(
      payload.role === "admin" ? "admin.confirmPromote" : "admin.confirmDemote",
    ))) return;
  } else if (action === "active") {
    payload.active = button.dataset.currentActive !== "true";
    if (!window.confirm(t("admin.confirmActive", {
      action: t(payload.active ? "admin.activate" : "admin.deactivate"),
    }))) return;
  } else {
    return;
  }
  const releaseLanguageSwitch = lockLanguageSwitch();
  button.disabled = true;
  try {
    await apiFetch(`/api/admin/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!sessionIsCurrent(generation, adminUserId)) return;
    await loadAdminDashboard();
  } catch (error) {
    if (await handleAuthorizationError(error, generation)) return;
    window.alert(t("admin.updateError", { error: error.message }));
  } finally {
    releaseLanguageSwitch();
    if (sessionIsCurrent(generation, adminUserId) && button.isConnected) {
      button.disabled = false;
    }
  }
}

function requestAdminPassword() {
  if (adminPasswordResolver || elements.adminPasswordDialog.open) {
    return Promise.resolve(null);
  }
  elements.adminPasswordForm.reset();
  elements.adminPasswordMessage.hidden = true;
  elements.adminPasswordDialog.showModal();
  window.setTimeout(() => elements.adminResetPassword.focus({ preventScroll: true }), 0);
  return new Promise((resolve) => {
    adminPasswordResolver = resolve;
  });
}

function finishAdminPassword(value) {
  const resolve = adminPasswordResolver;
  adminPasswordResolver = null;
  if (elements.adminPasswordDialog.open) elements.adminPasswordDialog.close();
  elements.adminPasswordForm.reset();
  elements.adminPasswordMessage.textContent = "";
  elements.adminPasswordMessage.hidden = true;
  if (resolve) resolve(value);
}

async function runJobAction(button) {
  const action = button.dataset.jobAction;
  const jobId = button.dataset.jobId;
  if (!action || !jobId || button.disabled) return;
  const generation = authGeneration;
  const fromAdminDashboard = elements.adminJobList.contains(button);
  const actionKeys = {
    retry: "job.actionRetry",
    reprocess: "job.actionReprocess",
    delete: "job.actionDelete",
  };
  const buttonKeys = {
    retry: "job.retry",
    reprocess: "job.reprocess",
    delete: "job.delete",
  };
  if (!Object.hasOwn(actionKeys, action)) return;
  const actionLabel = t(actionKeys[action]);
  const detail = t(action === "delete" ? "job.deleteDetail" : "job.queueDetail");
  if (!window.confirm(t("job.confirmAction", { action: actionLabel, detail }))) return;
  const releaseLanguageSwitch = lockLanguageSwitch();
  button.disabled = true;
  setText(button, "job.processingAction");
  try {
    await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}${action === "delete" ? "" : `/${action}`}`, {
      method: action === "delete" ? "DELETE" : "POST",
    });
    if (!sessionIsCurrent(generation)) return;
    lastJobsSignature = "";
    if (
      fromAdminDashboard &&
      action === "delete" &&
      adminOffset >= adminTotal - 1
    ) {
      adminOffset = Math.max(0, adminOffset - adminLimit);
    }
    await Promise.all([
      loadActivity(),
      fromAdminDashboard ? loadAdminDashboard() : Promise.resolve(),
    ]);
  } catch (error) {
    if (await handleAuthorizationError(error, generation)) return;
    window.alert(t("job.actionError", { action: t(actionKeys[action]), error: error.message }));
  } finally {
    releaseLanguageSwitch();
    if (sessionIsCurrent(generation) && button.isConnected) {
      button.disabled = false;
      setText(button, buttonKeys[action]);
    }
  }
}

function renderActivity(imports, uploads, jobs) {
  latestActivityPayload = { imports, uploads, jobs };
  const uploadViews = uploads.map((upload) => ({
    ...upload,
    transfer_active: uploadIsActive(upload),
    local_resume: hasLocalResumeSession(upload),
  }));
  const total = imports.length + uploadViews.length + jobs.length;
  elements.emptyJobs.hidden = total > 0;
  setText(
    elements.jobCount,
    total ? (total > 1 ? "library.itemCount" : "library.videoCount") : "library.waiting",
    { count: total },
  );

  const importsSignature = JSON.stringify([i18n.language, imports]);
  if (importsSignature !== lastImportsSignature) {
    lastImportsSignature = importsSignature;
    elements.importList.innerHTML = imports.map(renderDriveImport).join("");
  }

  const uploadsSignature = JSON.stringify([i18n.language, uploadViews]);
  if (uploadsSignature !== lastUploadsSignature) {
    lastUploadsSignature = uploadsSignature;
    elements.uploadList.innerHTML = uploadViews.map(renderUpload).join("");
  }

  const jobsSignature = JSON.stringify([i18n.language, jobs]);
  if (jobsSignature !== lastJobsSignature) {
    lastJobsSignature = jobsSignature;
    renderJobs(jobs);
  }

  const annotationDevSignature = JSON.stringify([
    i18n.language,
    jobs
      .filter((job) => job.status === "completed" && job.result)
      .map((job) => ({
        id: job.id,
        sourceName: job.result.source_name,
        duration: job.result.media?.duration ?? null,
      })),
  ]);
  if (annotationDevSignature !== lastAnnotationDevSignature) {
    lastAnnotationDevSignature = annotationDevSignature;
    renderAnnotationDevelopment(jobs);
  }
}

async function loadActivity() {
  if (!authReady || activityLoading) return;
  const generation = authGeneration;
  const userId = currentUser.id;
  activityLoading = true;
  try {
    const [importsResponse, uploadsResponse, jobs] = await Promise.all([
      apiFetch("/api/drive-imports"),
      apiFetch("/api/uploads"),
      loadAllMyJobs(),
    ]);
    const [{ imports }, { uploads }] = await Promise.all([
      importsResponse.json(),
      uploadsResponse.json(),
    ]);
    if (!sessionIsCurrent(generation, userId)) return;
    renderActivity(imports, uploads, jobs);
  } catch (error) {
    await handleAuthorizationError(error, generation);
  } finally {
    if (sessionIsCurrent(generation, userId)) activityLoading = false;
  }
}

async function loadAllMyJobs() {
  const pageSize = 500;
  const jobs = [];
  const seen = new Set();
  let offset = 0;
  let total = null;
  while (total === null || offset < total) {
    const response = await apiFetch(
      `/api/jobs?scope=mine&limit=${pageSize}&offset=${offset}`,
    );
    const payload = await response.json();
    const page = Array.isArray(payload?.jobs) ? payload.jobs : [];
    total = Number.isFinite(Number(payload?.total)) ? Number(payload.total) : page.length;
    for (const job of page) {
      const jobId = String(job?.id || "");
      if (!jobId || seen.has(jobId)) continue;
      seen.add(jobId);
      jobs.push(job);
    }
    if (!page.length) break;
    offset += page.length;
  }
  return jobs;
}

function selectVideo(file) {
  if (!file) return;
  const extension = file.name.split(".").at(-1)?.toLowerCase();
  if (!file.type.startsWith("video/") && !["mov", "mp4", "m4v", "mkv"].includes(extension)) {
    setText(elements.filePrompt, "file.notVideo");
    setText(elements.fileMeta, "file.chooseVideo");
    return;
  }
  selectedFile = file;
  elements.dropZone.classList.add("selected");
  clearLocalizedElement(elements.filePrompt);
  elements.filePrompt.textContent = selectedFile.name;
  setText(elements.fileMeta, "file.ready", { size: formatBytes(selectedFile.size) });
  setText(elements.uploadButton.querySelector("span"), "file.uploadThis");
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
  setText(elements.pauseButton, paused ? "transfer.resume" : "transfer.pause");
  setText(elements.transferLabel, paused ? "upload.pausedRetained" : "upload.sending");
});
elements.jobList.addEventListener("click", (event) => {
  const actionButton = event.target.closest("button[data-job-action]");
  if (actionButton) {
    runJobAction(actionButton);
    return;
  }
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
      .catch(() => showAnnotationWorkspaceMessageKey("annotation.playbackError", {}, true));
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
  showAnnotationWorkspaceMessageKey("annotation.playbackError", {}, true);
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
elements.guideButton.addEventListener("click", () => {
  elements.quickGuide.open = true;
  elements.quickGuide.scrollIntoView({ behavior: "smooth", block: "center" });
  elements.quickGuide.querySelector("summary")?.focus({ preventScroll: true });
});
elements.changePasswordForm.addEventListener("submit", changeOwnPassword);
elements.adminPasswordForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const password = elements.adminResetPassword.value;
  if (password.length < 8) {
    setText(elements.adminPasswordMessage, "adminPassword.lengthError");
    elements.adminPasswordMessage.hidden = false;
    elements.adminResetPassword.focus();
    return;
  }
  finishAdminPassword(password);
});
elements.adminPasswordCancel.addEventListener("click", () => finishAdminPassword(null));
elements.adminPasswordDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  finishAdminPassword(null);
});
elements.adminPasswordDialog.addEventListener("close", () => {
  if (adminPasswordResolver) finishAdminPassword(null);
});
elements.createUserForm.addEventListener("submit", createUser);
elements.adminRefreshButton.addEventListener("click", loadAdminDashboard);
elements.adminUserList.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-user-action]");
  if (button) patchAdminUser(button);
});
elements.adminPendingList.addEventListener("click", (event) => {
  const retryButton = event.target.closest(".retry-import-button");
  if (retryButton) {
    retryDriveImport(retryButton);
    return;
  }
  const importDeleteButton = event.target.closest(".delete-import-button");
  if (importDeleteButton) {
    deleteDriveImport(importDeleteButton);
    return;
  }
  const uploadDeleteButton = event.target.closest(".delete-upload-button");
  if (uploadDeleteButton) deleteUploadSession(uploadDeleteButton);
});
elements.adminJobList.addEventListener("click", (event) => {
  const actionButton = event.target.closest("button[data-job-action]");
  if (actionButton) {
    runJobAction(actionButton);
    return;
  }
  const shareButton = event.target.closest(".share-button");
  if (shareButton) shareOrSave(shareButton);
});
elements.adminJobList.addEventListener(
  "toggle",
  (event) => {
    const panel = event.target;
    if (!panel.matches?.(".result-panel[data-result-job-id]")) return;
    if (panel.open) hydrateResultPanel(panel);
    else dehydrateResultPanel(panel);
  },
  true,
);
elements.adminPrevButton.addEventListener("click", () => {
  adminOffset = Math.max(0, adminOffset - adminLimit);
  loadAdminDashboard();
});
elements.adminNextButton.addEventListener("click", () => {
  if (adminOffset + adminLimit >= adminTotal) return;
  adminOffset += adminLimit;
  loadAdminDashboard();
});

window.addEventListener("beforeunload", (event) => {
  if (!uploadRunning) return;
  event.preventDefault();
  event.returnValue = "";
});

async function initializeApplication(user) {
  if (!user?.id || !user?.username) {
    showLogin("error.loginPayload");
    return;
  }
  const generation = resetUserState(user);
  const userId = user.id;
  showApplication(user, { reveal: false });
  authReady = true;
  try {
    const configResponse = await apiFetch("/api/config");
    const config = await configResponse.json();
    if (!sessionIsCurrent(generation, userId)) return;
    chunkSize = config.chunk_size || chunkSize;
  } catch (error) {
    if (await handleAuthorizationError(error, generation)) return;
  }
  if (!sessionIsCurrent(generation, userId)) return;
  elements.uploadButton.disabled = !selectedFile;
  updateDriveButton();
  await Promise.all([loadActivity(), loadAdminDashboard()]);
  if (!sessionIsCurrent(generation, userId)) return;
  showApplication(currentUser);
  if (!activityTimer) {
    activityTimer = setInterval(loadActivity, 2500);
  }
}

async function login(event) {
  event.preventDefault();
  const username = elements.loginUsername.value.trim();
  const password = elements.loginPassword.value;
  if (!username || !password) return;
  elements.loginButton.disabled = true;
  setText(elements.loginButton.querySelector("span"), "login.loading");
  elements.loginMessage.hidden = true;
  try {
    const response = await apiFetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    let user = userFromPayload(await response.json());
    if (!user?.username) {
      const meResponse = await apiFetch("/api/auth/me");
      user = userFromPayload(await meResponse.json());
    }
    elements.loginPassword.value = "";
    await initializeApplication(user);
  } catch (error) {
    setText(
      elements.loginMessage,
      isUnauthorized(error) ? "error.loginInvalid" : "error.loginFailed",
      { error: error.message },
    );
    elements.loginMessage.hidden = false;
    elements.loginPassword.select();
  } finally {
    elements.loginButton.disabled = false;
    setText(elements.loginButton.querySelector("span"), "login.submit");
  }
}

async function logout() {
  elements.logoutButton.disabled = true;
  let serverLogoutSucceeded = false;
  try {
    await apiFetch("/api/auth/logout", { method: "POST" });
    serverLogoutSucceeded = true;
  } catch (_) {
    // Always clear this page below, but do not claim the server session was revoked.
  } finally {
    elements.logoutButton.disabled = false;
    showLogin(
      serverLogoutSucceeded
        ? "logout.success"
        : "logout.uncertain",
    );
  }
}

function switchLanguage() {
  const libraryResultState = captureResultPanelState(elements.jobList);
  const adminResultState = captureResultPanelState(elements.adminJobList);
  i18n.toggle();
  lastImportsSignature = "";
  lastUploadsSignature = "";
  lastJobsSignature = "";
  lastAnnotationDevSignature = "";
  jobRenderSignatures.clear();
  if (latestActivityPayload) {
    renderActivity(
      latestActivityPayload.imports,
      latestActivityPayload.uploads,
      latestActivityPayload.jobs,
    );
  }
  if (latestAdminPayload && isAdmin()) renderAdminDashboard(latestAdminPayload);
  if (latestAnnotationPayload) renderAnnotationWorkspaceList(latestAnnotationPayload);
  renderAnnotationWorkspaceBoundaries();
  restoreResultPanelState(elements.jobList, libraryResultState);
  restoreResultPanelState(elements.adminJobList, adminResultState);
}

async function initialize() {
  i18n.apply();
  purgeLegacyResumeKeys();
  try {
    const response = await apiFetch("/api/auth/me");
    await initializeApplication(userFromPayload(await response.json()));
  } catch (error) {
    showLogin(isUnauthorized(error) ? "" : "error.connection");
  }
}

elements.languageToggle.addEventListener("click", switchLanguage);
elements.loginForm.addEventListener("submit", login);
elements.logoutButton.addEventListener("click", logout);
initialize();
