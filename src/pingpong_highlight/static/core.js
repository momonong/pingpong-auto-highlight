// Classic deferred feature module; shared session bindings are declared in core.js.
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

function boundedReadSignal(signal) {
  const controller = new AbortController();
  const cancel = () => controller.abort(signal.reason);
  const timer = setTimeout(() => {
    signal.removeEventListener("abort", cancel);
    controller.abort(new DOMException("Request timed out", "TimeoutError"));
  }, 15000);
  controller.signal.addEventListener("abort", () => {
    clearTimeout(timer);
    signal.removeEventListener("abort", cancel);
  }, { once: true });
  if (signal.aborted) cancel();
  else signal.addEventListener("abort", cancel, { once: true });
  return controller.signal;
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
    signal: ["GET", "HEAD"].includes((requestOptions.method || "GET").toUpperCase())
      ? boundedReadSignal(signal)
      : signal,
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
  resetWorkspace();
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
  updateWorkspaceAccess();
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
