// Classic deferred feature module; shared session bindings are declared in core.js.
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
