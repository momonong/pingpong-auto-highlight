// Classic deferred feature module; shared session bindings are declared in core.js.
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
  return `<a class="quick-download" href="${escapeHtml(downloadUrl)}" download>${t("result.downloadMp4")}</a><details class="result-panel" data-result-job-id="${escapeHtml(jobId)}"${open}>
    <summary class="reel-heading">
      <span class="sr-only">${escapeHtml(t("result.srLabel", { source: sourceName }))}</span>
      <span class="reel-heading-copy"><span>BEST POINTS REEL</span><b>${escapeHtml(reel.name)}</b></span>
      <span class="reel-toggle-label"><span class="reel-toggle-closed">${t("workspace.play")}</span><span class="reel-toggle-open">${t("result.collapse")}</span></span>
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
    <div class="job-title"><strong title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong><span class="status ${escapeHtml(job.status)}">${statusText}</span></div>
    <p class="job-detail">${details}</p>
    ${progressBar}${stats}${resultPanel}
    ${job.status === "failed" ? `<button type="button" data-job-action="retry" data-job-id="${escapeHtml(jobId)}">${t("job.retry")}</button>` : ""}
    <details class="job-more"><summary>${t("workspace.more")}</summary>${renderJobControls(job, { admin })}</details>
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
