// Health is independent of login and upload requests. Never infer online from navigator.onLine.
let connectionState = "checking";
let healthController = null;
let healthEpoch = 0;
let healthTimer = null;

function renderConnectionStatus() {
  const pill = document.querySelector("#serviceStatus");
  pill.dataset.state = connectionState;
  setText(pill.querySelector("span"), `service.${connectionState}`);
}

async function checkConnection() {
  const epoch = ++healthEpoch;
  clearTimeout(healthTimer);
  healthController?.abort();
  healthController = new AbortController();
  const controller = healthController;
  const deadline = setTimeout(() => controller.abort(), 6000);
  try {
    if (!navigator.onLine) throw new Error("offline");
    const response = await fetch(HC.url("/api/health"), {
      cache: "no-store", credentials: "same-origin", signal: controller.signal,
    });
    if (!response.ok || (await response.json()).status !== "ok") throw new Error("health");
    if (epoch !== healthEpoch) return;
    connectionState = "online";
  } catch (_) {
    if (epoch !== healthEpoch) return;
    connectionState = navigator.onLine ? "unavailable" : "offline";
    if (authReady) setActivityNotice("workspace.stale");
  } finally {
    clearTimeout(deadline);
    if (epoch === healthEpoch) {
      renderConnectionStatus();
      healthTimer = setTimeout(checkConnection, 5000);
    }
  }
}

window.addEventListener("offline", () => {
  healthEpoch += 1;
  healthController?.abort();
  connectionState = "offline";
  renderConnectionStatus();
  if (authReady) setActivityNotice("workspace.stale");
});
window.addEventListener("online", () => {
  connectionState = "checking";
  renderConnectionStatus();
  checkConnection();
  loadActivity();
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    connectionState = "checking";
    renderConnectionStatus();
    checkConnection();
    loadActivity();
  }
});
// Start after all deferred feature scripts have initialized their bindings.
document.addEventListener("DOMContentLoaded", checkConnection, { once: true });
