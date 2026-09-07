// Classic deferred feature module; shared session bindings are declared in core.js.
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
