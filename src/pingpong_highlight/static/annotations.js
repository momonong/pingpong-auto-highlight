// Source-bound point review, sharing the existing auth/session and player contracts.
let pointReview = null;
let pointDraft = null;
let pointDirty = false;
let pointBusy = false;
let pointPendingRequest = null;
let pointDraftRevision = 0;
let pointEpoch = 0;
let pointLocalSafe = true;
const pointEl = (id) => document.getElementById(id);
const pointActive = () => (pointReview?.points || []).filter(p => p.active)
  .sort((a, b) => a.start_ms - b.start_ms || a.end_ms - b.end_ms || a.id.localeCompare(b.id));
const pointComplete = p => p.validity === "not_rally" ||
  (p.validity === "valid" && p.boundary_status === "confirmed" && p.rating_status !== "unrated");

function annotationWorkspaceIsOpen() { return !elements.annotationWorkspace.hidden; }
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
function pointDraftKey() {
  return `highlightcraft-point-draft:${currentUser?.id}:${pointReview?.source.id}`;
}
function rememberPointDraft() {
  if (!pointReview) return;
  try {
    if (pointDirty || pointPendingRequest) localStorage.setItem(pointDraftKey(), JSON.stringify({
      draft: pointDraft, revision: pointDraftRevision, pending: pointPendingRequest,
    }));
    else localStorage.removeItem(pointDraftKey());
    pointLocalSafe = true;
  } catch (_) {
    pointLocalSafe = false;
    showAnnotationWorkspaceMessageKey("point.localFailed", {}, true);
  }
}
function pointChanged() {
  if (!pointDraft || pointBusy) return;
  pointDraft.start_ms = pointEl("pointStart").value === "" ? null : Math.round(Number(pointEl("pointStart").value) * 1000);
  pointDraft.end_ms = pointEl("pointEnd").value === "" ? null : Math.round(Number(pointEl("pointEnd").value) * 1000);
  pointDraft.validity = pointEl("pointValidity").value;
  pointDraft.boundary_status = pointEl("pointBoundary").value;
  const value = pointEl("pointRating").value;
  pointDraft.rating_status = ["unrated", "unable"].includes(value) ? value : "rated";
  pointDraft.excitement = pointDraft.rating_status === "rated" ? Number(value) : null;
  pointDraft.reason_tags = [...pointEl("pointReasons").querySelectorAll("input:checked")].map(e => e.value);
  pointDraft.quality_tags = [...pointEl("pointQuality").querySelectorAll("input:checked")].map(e => e.value);
  pointDraft.note = pointEl("pointNote").value;
  pointDirty = true;
  // Keep an uncertain request intact until retried or explicitly reloaded.
  showAnnotationWorkspaceMessageKey("point.draft");
  rememberPointDraft();
  renderAnnotationWorkspaceBoundaries();
}
function renderAnnotationWorkspaceBoundaries() {
  for (const [name, key] of [["Start", "start_ms"], ["End", "end_ms"]]) {
    const output = elements[`annotationWorkspace${name}`];
    if (pointDraft?.[key] == null) setText(output, "annotation.notSet");
    else { clearLocalizedElement(output); output.textContent = formatTimestamp(pointDraft[key] / 1000); }
  }
}
function pointRenderEditor() {
  pointEl("pointFields").disabled = !pointDraft || pointBusy || !!pointPendingRequest;
  setText(pointEl("pointSelection"), "point.selection", {id: pointDraft?.id?.slice(0, 10) || t("point.new")});
  pointEl("pointStart").value = pointDraft?.start_ms == null ? "" : pointDraft.start_ms / 1000;
  pointEl("pointEnd").value = pointDraft?.end_ms == null ? "" : pointDraft.end_ms / 1000;
  pointEl("pointValidity").value = pointDraft?.validity || "pending";
  pointEl("pointBoundary").value = pointDraft?.boundary_status || "pending";
  pointEl("pointRating").value = pointDraft?.rating_status === "rated" ? String(pointDraft.excitement) : pointDraft?.rating_status || "unrated";
  pointEl("pointNote").value = pointDraft?.note || "";
  for (const [id, key, rules] of [["pointReasons", "reason_tags", "reason_tags"], ["pointQuality", "quality_tags", "quality_tags"]]) {
    pointEl(id).innerHTML = Object.entries(pointReview?.rules[rules] || {}).map(([code, labels]) =>
      `<label><input type="checkbox" value="${escapeHtml(code)}" ${pointDraft?.[key].includes(code) ? "checked" : ""}>${escapeHtml(labels[i18n.language === "en" ? 1 : 0])}</label>`).join("");
  }
  renderAnnotationWorkspaceBoundaries();
}
function pointVisible() {
  const filter = pointEl("pointFilter").value;
  return pointActive().filter(p => filter === "all" ||
    (filter === "unrated" && p.rating_status === "unrated" && p.validity !== "not_rally") ||
    (filter === "boundary" && p.boundary_status !== "confirmed" && p.validity !== "not_rally") ||
    (filter === "complete" && pointComplete(p)));
}
function renderAnnotationWorkspaceList(payload) {
  if (!payload?.source) return;
  latestAnnotationPayload = payload;
  const points = pointActive();
  setText(elements.annotationWorkspaceCount, "point.count", {count: points.length, complete: points.filter(pointComplete).length});
  elements.annotationWorkspaceList.innerHTML = pointVisible().map(p => {
    const originKey = {automatic: "point.automatic", manual: "point.manual", legacy: "point.legacyOrigin", split: "point.splitOrigin", merge: "point.mergeOrigin"}[p.origin.kind];
    return `<button type="button" class="point-list-item ${p.id === pointDraft?.id ? "selected" : ""}" data-point-id="${p.id}" aria-pressed="${p.id === pointDraft?.id}">
      <strong>${formatTimestamp(p.start_ms / 1000)}–${formatTimestamp(p.end_ms / 1000)}</strong>
      <span>${escapeHtml(t(`point.${p.validity}`))} · ${escapeHtml(t(`point.${p.boundary_status}`))}</span>
      <span>${escapeHtml(p.rating_status === "rated" ? t(`point.score${p.excitement}`) : t(`point.${p.rating_status}`))}</span>
      <small>${escapeHtml(t(originKey))} · ${p.id.slice(0, 8)}${p.origin.legacy_label ? ` · ${escapeHtml(p.origin.legacy_label)}` : ""}</small></button>`;
  }).join("") || `<p>${t("point.empty")}</p>`;
  const batch = payload.available_proposals;
  setText(pointEl("pointProposalInfo"), "point.importHelp", {count: batch.candidates.length, skipped: batch.skipped});
  if (payload.imports.some(b => b.id === batch.id)) pointEl("pointProposalInfo").append(` ${t("point.imported")}`);
  pointEl("pointCoverageList").innerHTML = payload.coverage.filter(c => c.active).map(c =>
    `<p>${formatTimestamp(c.start_ms / 1000)}–${formatTimestamp(c.end_ms / 1000)} <button type="button" data-coverage-id="${c.id}">${t("point.remove")}</button></p>`).join("") || `<p>${t("point.coverageEmpty")}</p>`;
  if (payload.fully_reviewed) setText(pointEl("pointUnknown"), "point.covered");
  else setText(pointEl("pointUnknown"), "point.unknown", {ranges: (payload.unknown_intervals || []).map(c => `${formatTimestamp(c.start_ms / 1000)}–${formatTimestamp(c.end_ms / 1000)}`).join(", ") || "?"});
  pointEl("pointLegacy").innerHTML = payload.legacy_annotations.map(a =>
    `<p>${formatTimestamp(a.start)}–${formatTimestamp(a.end)} · ${escapeHtml(a.label)}<br>${escapeHtml(a.note)}</p>`).join("");
  pointRenderEditor();
}
function pointSetBusy(busy) {
  pointBusy = busy;
  for (const id of ["pointNew", "pointImport", "pointReload", "pointCoverageAdd", "pointExport", "pointExportLines", "pointFilter"]) pointEl(id).disabled = busy || !pointReview;
  pointEl("pointFields").disabled = busy || !pointDraft || !!pointPendingRequest;
  // Retry sits outside the disabled editor so an uncertain write can be replayed safely.
  pointEl("pointRetry").hidden = !pointPendingRequest;
  pointEl("pointRetry").disabled = busy;
  // Keep scroll position and the player independent of form focus.
}
async function pointWrite(fields) {
  if (pointBusy || !pointReview) return null;
  const epoch = pointEpoch, generation = authGeneration, userId = currentUser?.id;
  if (!pointPendingRequest) pointPendingRequest = {revision: pointDirty ? pointDraftRevision : pointReview.revision,
    request_id: crypto.randomUUID(), ...fields};
  rememberPointDraft();
  pointSetBusy(true);
  showAnnotationWorkspaceMessageKey("point.saving");
  try {
    const response = await apiFetch(`/api/jobs/${annotationWorkspaceJobId}/point-review`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(pointPendingRequest),
      signal: AbortSignal.timeout(20000),
    });
    const payload = await response.json();
    if (epoch !== pointEpoch || !sessionIsCurrent(generation, userId)) return null;
    pointReview = payload; pointPendingRequest = null; pointDirty = false; pointDraftRevision = payload.revision;
    rememberPointDraft();
    showAnnotationWorkspaceMessageKey("point.saved");
    renderAnnotationWorkspaceList(payload);
    return payload;
  } catch (error) {
    if (epoch !== pointEpoch || !sessionIsCurrent(generation, userId)) return null;
    const wasPointSave = pointPendingRequest?.action === "save";
    if (error.status === 409) {
      pointPendingRequest = null;
      showAnnotationWorkspaceMessageKey("point.conflict", {}, true);
    } else {
      // A 4xx other than auth is a definite rejection; allow correction without reusing its ID.
      if (error.status >= 400 && error.status < 500 && error.status !== 401) pointPendingRequest = null;
      showAnnotationWorkspaceMessageKey("point.failed", {detail: error.message}, true);
    }
    pointDirty = pointDirty || wasPointSave;
    rememberPointDraft();
    await handleAuthorizationError(error, generation);
    return null;
  } finally {
    if (epoch === pointEpoch && sessionIsCurrent(generation, userId)) pointSetBusy(false);
  }
}
function pointFieldsPayload() {
  const keys = ["start_ms", "end_ms", "validity", "boundary_status", "rating_status", "excitement", "reason_tags", "quality_tags", "note"];
  return Object.fromEntries(keys.map(k => [k, pointDraft[k]]));
}
async function saveAnnotationWorkspace(next = false) {
  if (pointBusy || !pointDraft) return false;
  if (!pointPendingRequest && !elements.annotationWorkspaceForm.reportValidity()) return false;
  const oldVisible = pointVisible().map(p => p.id), oldId = pointDraft.id;
  const payload = await pointWrite({action: "save", point_id: oldId || null, point: pointFieldsPayload()});
  if (!payload) return false;
  const selected = payload.selected_id || oldId;
  pointDraft = structuredClone(pointActive().find(p => p.id === selected) || null);
  if (next) {
    const after = oldVisible.slice(oldVisible.indexOf(oldId) + 1).find(id => pointVisible().some(p => p.id === id));
    if (after) { await pointSelect(after); return true; }
    showAnnotationWorkspaceMessageKey("point.noNext");
  }
  renderAnnotationWorkspaceList(pointReview);
  elements.annotationWorkspace.focus({preventScroll: true});
  return true;
}
async function pointSelect(id) {
  if (pointBusy || (pointDirty && !(await saveAnnotationWorkspace()))) return;
  pointDraft = structuredClone(pointActive().find(p => p.id === id));
  if (!pointDraft) return;
  pointDraftRevision = pointReview.revision;
  renderAnnotationWorkspaceList(pointReview);
  elements.annotationWorkspaceForm.scrollTop = 0;
  pointPreview();
  elements.annotationWorkspace.focus({preventScroll: true});
}
function pointPlay(video) {
  const epoch = pointEpoch;
  video.play().catch(error => {
    // Seeking, changing source or pausing cancels pending play() normally.
    if (error.name !== "AbortError" && epoch === pointEpoch && annotationWorkspaceIsOpen()) {
      showAnnotationWorkspaceMessageKey("annotation.playbackError", {}, true);
    }
  });
}
function pointPreview() {
  if (!pointDraft) return;
  const video = elements.annotationWorkspaceVideo;
  const pad = Math.max(0, Math.min(10, Number(pointEl("pointPadding").value) || 0));
  video.currentTime = Math.max(0, pointDraft.start_ms / 1000 - pad);
  video.dataset.stopAt = String(Math.min(pointReview.source.duration_ms / 1000, pointDraft.end_ms / 1000 + pad));
  pointPlay(video);
}
async function pointNew() {
  if (!pointReview || pointBusy || (pointDirty && !(await saveAnnotationWorkspace()))) return;
  const start = Math.min(Math.round(elements.annotationWorkspaceVideo.currentTime * 1000), pointReview.source.duration_ms - 1);
  pointDraft = {start_ms: start, end_ms: Math.min(pointReview.source.duration_ms, start + 1000),
    validity: "pending", boundary_status: "pending", rating_status: "unrated", excitement: null,
    reason_tags: [], quality_tags: [], note: ""};
  pointDraftRevision = pointReview.revision; pointDirty = true;
  renderAnnotationWorkspaceList(pointReview); rememberPointDraft();
  showAnnotationWorkspaceMessageKey("point.draft");
  elements.annotationWorkspace.focus({preventScroll: true});
}
function markAnnotationWorkspaceBoundary(boundary) {
  if (!pointDraft || pointBusy || pointPendingRequest) return;
  pointEl(boundary === "start" ? "pointStart" : "pointEnd").value = Math.round(elements.annotationWorkspaceVideo.currentTime * 1000) / 1000;
  pointEl("pointBoundary").value = "pending";
  pointChanged();
}
function seekAnnotationWorkspace(seconds) {
  const video = elements.annotationWorkspaceVideo;
  delete video.dataset.stopAt;
  video.currentTime = Math.max(0, Math.min(video.duration || Infinity, video.currentTime + seconds));
}
function toggleAnnotationWorkspacePlayback() {
  const video = elements.annotationWorkspaceVideo;
  if (!video.paused) video.pause();
  else pointPlay(video);
}
async function loadAnnotationWorkspaceList(restore = false) {
  const epoch = pointEpoch, generation = authGeneration, userId = currentUser?.id;
  pointSetBusy(true);
  try {
    const response = await apiFetch(`/api/jobs/${annotationWorkspaceJobId}/point-review`);
    const payload = await response.json();
    if (epoch !== pointEpoch || !sessionIsCurrent(generation, userId)) return;
    pointReview = payload; pointDraft = null; pointDirty = false; pointPendingRequest = null;
    pointDraftRevision = payload.revision;
    pointEl("coverageEnd").value = payload.source.duration_ms / 1000;
    if (restore) {
      try {
        const draft = JSON.parse(localStorage.getItem(pointDraftKey()) || "null");
        if (draft) {
          pointDraft = draft.draft; pointDraftRevision = draft.revision;
          pointPendingRequest = draft.pending; pointDirty = true;
          showAnnotationWorkspaceMessageKey("point.restored");
        }
      } catch (_) { showAnnotationWorkspaceMessageKey("point.localFailed", {}, true); }
    }
    if (!pointDraft) pointDraft = structuredClone(pointActive().find(p => !pointComplete(p)) || pointActive()[0] || null);
    renderAnnotationWorkspaceList(payload);
    if (!pointDirty) showAnnotationWorkspaceMessageKey("point.loaded");
  } catch (error) {
    if (epoch === pointEpoch && sessionIsCurrent(generation, userId)) {
      showAnnotationWorkspaceMessage(error.message, true);
      await handleAuthorizationError(error, generation);
    }
  } finally { if (epoch === pointEpoch) pointSetBusy(false); }
}
function openAnnotationWorkspace(button) {
  if (!isAdmin()) return;
  annotationWorkspaceJobId = button.dataset.jobId;
  annotationWorkspaceReturnFocus = button; pointEpoch += 1;
  pointReview = null; pointDraft = null; pointDirty = false; pointPendingRequest = null;
  pointEl("pointFilter").value = "all";
  elements.annotationWorkspace.hidden = false;
  document.body.classList.add("annotation-workspace-open");
  clearLocalizedElement(elements.annotationWorkspaceFilename);
  elements.annotationWorkspaceFilename.textContent = button.dataset.sourceName;
  showAnnotationWorkspaceMessageKey("annotation.loading");
  elements.annotationWorkspaceList.replaceChildren();
  elements.annotationWorkspaceVideo.src = fileAccessUrl(`/api/jobs/${annotationWorkspaceJobId}/source`);
  elements.annotationWorkspaceVideo.preload = "metadata";
  elements.annotationWorkspaceVideo.load();
  loadAnnotationWorkspaceList(true);
  elements.annotationWorkspace.focus({preventScroll: true});
}
function closeAnnotationWorkspace(force = false) {
  if (!annotationWorkspaceIsOpen()) return;
  if (force !== true && (pointBusy || (pointDirty && (!pointLocalSafe || !window.confirm(t("point.closeDraft")))))) return;
  const returnFocus = annotationWorkspaceReturnFocus;
  pointEpoch += 1;
  elements.annotationWorkspaceVideo.pause();
  elements.annotationWorkspaceVideo.removeAttribute("src");
  delete elements.annotationWorkspaceVideo.dataset.stopAt;
  elements.annotationWorkspaceVideo.load();
  elements.annotationWorkspace.hidden = true;
  document.body.classList.remove("annotation-workspace-open");
  annotationWorkspaceJobId = ""; annotationWorkspaceReturnFocus = null;
  latestAnnotationPayload = null; pointReview = null; pointDraft = null; pointDirty = false;
  pointPendingRequest = null; pointBusy = false;
  pointRenderEditor();
  elements.annotationWorkspaceList.replaceChildren();
  pointEl("pointCoverageList").replaceChildren();
  pointEl("pointLegacy").replaceChildren();
  if (returnFocus?.isConnected) returnFocus.focus();
}
async function pointStructure(action) {
  if (!pointDraft || pointBusy) return;
  if (pointDirty && !(await saveAnnotationWorkspace())) return;
  if (!pointDraft?.id) { showAnnotationWorkspaceMessageKey("point.noSelection"); return; }
  if (!window.confirm(t(action === "delete" ? "point.confirmDelete" : "point.confirmStructure"))) return;
  const active = pointActive(), index = active.findIndex(p => p.id === pointDraft.id);
  const payload = await pointWrite({action, point_id: pointDraft.id,
    ...(action === "split" ? {split_ms: Math.round(elements.annotationWorkspaceVideo.currentTime * 1000)} : {}),
    ...(action === "merge" ? {other_id: active[index + 1]?.id || null} : {}),
  });
  if (payload) {
    pointDraft = structuredClone(pointActive().find(p => p.id === payload.selected_id) || null);
    renderAnnotationWorkspaceList(payload);
  }
}
async function pointAuxiliary(fields) {
  if (pointBusy || (pointDirty && !(await saveAnnotationWorkspace()))) return;
  const payload = await pointWrite(fields);
  if (payload) renderAnnotationWorkspaceList(payload);
}
async function pointExport(format) {
  if (pointDirty && !(await saveAnnotationWorkspace())) return;
  const generation = authGeneration, userId = currentUser?.id, epoch = pointEpoch;
  try {
    const response = await apiFetch(`/api/jobs/${annotationWorkspaceJobId}/point-review/export?format=${format}`);
    const blob = await response.blob();
    if (epoch !== pointEpoch || !sessionIsCurrent(generation, userId)) return;
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `point-review-${pointReview.source.id}.${format}`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { if (epoch === pointEpoch) showAnnotationWorkspaceMessage(error.message, true); }
}
function bindPointWorkspace() {
  elements.annotationWorkspaceClose.addEventListener("click", () => closeAnnotationWorkspace());
  elements.annotationWorkspaceMarkStart.addEventListener("click", () => markAnnotationWorkspaceBoundary("start"));
  elements.annotationWorkspaceMarkEnd.addEventListener("click", () => markAnnotationWorkspaceBoundary("end"));
  elements.annotationWorkspaceSave.addEventListener("click", () => saveAnnotationWorkspace());
  elements.annotationWorkspaceForm.addEventListener("submit", e => e.preventDefault());
  elements.annotationWorkspaceForm.addEventListener("input", e => {
    if (["pointStart", "pointEnd"].includes(e.target.id)) pointEl("pointBoundary").value = "pending";
    pointChanged();
  });
  pointEl("pointValidity").addEventListener("change", () => {
    if (pointEl("pointValidity").value === "not_rally") pointEl("pointRating").value = "unrated";
    pointChanged();
  });
  pointEl("pointNext").addEventListener("click", () => saveAnnotationWorkspace(true));
  pointEl("pointNew").addEventListener("click", pointNew);
  pointEl("pointPreview").addEventListener("click", pointPreview);
  pointEl("pointFilter").addEventListener("change", () => renderAnnotationWorkspaceList(pointReview));
  for (let i = 0; i < 4; i++) pointEl(`pointRate${i}`).addEventListener("click", () => {
    if (pointDraft?.validity === "not_rally") return;
    pointEl("pointRating").value = String(i); pointChanged(); elements.annotationWorkspace.focus();
  });
  for (const action of ["split", "merge", "delete"]) pointEl(`point${action[0].toUpperCase() + action.slice(1)}`).addEventListener("click", () => pointStructure(action));
  pointEl("pointImport").addEventListener("click", () => pointAuxiliary({action: "import", proposal_batch: pointReview.available_proposals.id}));
  pointEl("pointReload").addEventListener("click", async () => {
    if ((pointDirty || pointPendingRequest) && !window.confirm(t("point.discard"))) return;
    pointDirty = false; pointPendingRequest = null; rememberPointDraft();
    showAnnotationWorkspaceMessage(""); await loadAnnotationWorkspaceList();
  });
  pointEl("pointRetry").addEventListener("click", async () => {
    const payload = await pointWrite({});
    if (payload) { pointDraft = structuredClone(pointActive().find(p => p.id === payload.selected_id) || pointDraft); renderAnnotationWorkspaceList(payload); }
  });
  pointEl("pointExport").addEventListener("click", () => pointExport("json"));
  pointEl("pointExportLines").addEventListener("click", () => pointExport("jsonl"));
  pointEl("pointCoverageAdd").addEventListener("click", () => {
    if (!pointEl("coverageStart").value || !pointEl("coverageEnd").value) return;
    if (window.confirm(t("point.confirmCoverage"))) pointAuxiliary({action: "coverage_add", interval: {
      start_ms: Math.round(Number(pointEl("coverageStart").value) * 1000), end_ms: Math.round(Number(pointEl("coverageEnd").value) * 1000),
    }});
  });
  elements.annotationWorkspace.addEventListener("click", e => {
    const row = e.target.closest("[data-point-id]"); if (row) pointSelect(row.dataset.pointId);
    const seek = e.target.closest("[data-workspace-seek]"); if (seek) seekAnnotationWorkspace(Number(seek.dataset.workspaceSeek));
    const coverage = e.target.closest("[data-coverage-id]");
    if (coverage && window.confirm(t("point.remove"))) pointAuxiliary({action: "coverage_delete", coverage_id: coverage.dataset.coverageId});
  });
  elements.annotationWorkspaceVideo.addEventListener("timeupdate", () => {
    const video = elements.annotationWorkspaceVideo;
    elements.annotationWorkspaceCurrent.textContent = formatTimestamp(video.currentTime);
    if (video.dataset.stopAt && video.currentTime >= Number(video.dataset.stopAt)) { video.pause(); delete video.dataset.stopAt; }
  });
  elements.annotationWorkspaceVideo.addEventListener("error", () => {
    if (annotationWorkspaceIsOpen()) showAnnotationWorkspaceMessageKey("annotation.playbackError", {}, true);
  });
  document.addEventListener("keydown", e => {
    if (!annotationWorkspaceIsOpen() || e.isComposing || e.keyCode === 229 || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest("input, textarea, select, button, a, [contenteditable]")) return;
    if (e.key === "Escape") { e.preventDefault(); closeAnnotationWorkspace(); return; }
    if (pointBusy || pointPendingRequest) return;
    let handled = true;
    if (e.code === "Space") toggleAnnotationWorkspacePlayback();
    else if (["ArrowLeft", "ArrowRight"].includes(e.key)) seekAnnotationWorkspace((e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 5 : 1));
    else if (e.code === "KeyI") markAnnotationWorkspaceBoundary("start");
    else if (e.code === "KeyO") markAnnotationWorkspaceBoundary("end");
    else if (e.code === "KeyN") pointNew();
    else if (e.key === "Enter") saveAnnotationWorkspace(true);
    else if (/^[0-3]$/.test(e.key) && pointDraft && pointDraft.validity !== "not_rally") { pointEl("pointRating").value = e.key; pointChanged(); }
    else handled = false;
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  }, {capture: true});
  window.addEventListener("beforeunload", e => { if (pointDirty || pointBusy) { e.preventDefault(); e.returnValue = ""; } });
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
      <small>${escapeHtml(job.owner?.display_name || job.owner?.username || "")} · ${t("annotation.loadOnOpen", { duration })}</small>
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

async function loadAnnotationDevelopment() {
  if (!isAdmin()) return;
  const generation = authGeneration, userId = currentUser.id;
  elements.annotationDevList.innerHTML = `<p>${t("annotation.loading")}</p>`;
  try {
    const jobs = [];
    for (let offset = 0; ; offset += 100) {
      const response = await apiFetch(`/api/jobs?scope=all&limit=100&offset=${offset}`);
      const page = await response.json();
      if (!sessionIsCurrent(generation, userId) || !isAdmin()) return;
      jobs.push(...page.jobs);
      if (page.jobs.length < 100) break;
    }
    renderAnnotationDevelopment(jobs);
  } catch (error) {
    if (sessionIsCurrent(generation, userId) && !(await handleAuthorizationError(error, generation))) {
      elements.annotationDevList.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
    }
  }
}
