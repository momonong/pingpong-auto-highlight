function selectVideo(file) {
  if (!file) return;
  openAddFootage();
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
bindPointWorkspace();

elements.uploadList.addEventListener("click", (event) => {
  if (event.target.closest(".resume-upload-button")) {
    openAddFootage();
    elements.videoInput.click();
    return;
  }
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
  await loadActivity();
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
  renderConnectionStatus();
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
