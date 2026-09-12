// Classic deferred feature module; shared session bindings are declared in core.js.
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
    ${!admin && !active ? `<button class="resume-upload-button" type="button">${t("workspace.resume")}</button>` : ""}
    <details class="job-more"><summary>${t("workspace.more")}</summary><div class="upload-actions"><button class="delete-upload-button" type="button" data-upload-id="${escapeHtml(upload.id)}" data-filename="${escapeHtml(upload.filename)}" data-transferred="${escapeHtml(transferred)}">${t("upload.deleteRecord")}</button></div></details>
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
