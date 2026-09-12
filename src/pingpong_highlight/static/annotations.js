// Classic deferred feature module; shared session bindings are declared in core.js.
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
    <a class="annotation-dev-open" href="/static/review/index.html?job=${encodeURIComponent(job.id)}">${t("annotation.assistedReview")}</a>
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
