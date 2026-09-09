// Classic deferred feature module; shared session bindings are declared in core.js.
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

  applyLibraryFilter();

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
    if (workspaceView !== "annotations") renderAnnotationDevelopment(jobs);
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
    setActivityNotice("");
  } catch (error) {
    if (sessionIsCurrent(generation, userId)) setActivityNotice("workspace.stale");
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
