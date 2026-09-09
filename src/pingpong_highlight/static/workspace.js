// Workspace navigation never changes account scope or restarts transfers.
let workspaceView = "library";
let libraryFilter = "all";
const workspaceNav = document.querySelector("#workspaceNav");
const addFootage = document.querySelector("#addFootage");
const addVideoButton = document.querySelector("#addVideoButton");
const activityNotice = document.querySelector("#activityNotice");

function setActivityNotice(key) {
  activityNotice.hidden = !key;
  if (key) setText(activityNotice, key);
}

function resetWorkspace() {
  workspaceView = "library";
  libraryFilter = "all";
  addFootage.hidden = true;
  addVideoButton.setAttribute("aria-expanded", "false");
  setActivityNotice("workspace.loading");
  applyLibraryFilter();
}

function openAddFootage() {
  setWorkspaceView("library");
  addFootage.hidden = false;
  addVideoButton.setAttribute("aria-expanded", "true");
  addFootage.scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateWorkspaceAccess() {
  if (!isAdmin()) workspaceView = "library";
  for (const button of workspaceNav.querySelectorAll("[data-view]")) {
    button.hidden = button.dataset.view !== "library" && !isAdmin();
    button.setAttribute("aria-pressed", String(button.dataset.view === workspaceView));
  }
  document.querySelector(".workspace").hidden = workspaceView !== "library";
  elements.adminPanel.hidden = workspaceView !== "admin" || !isAdmin();
  elements.annotationDevBlock.hidden = workspaceView !== "annotations" || !isAdmin();
}

function setWorkspaceView(view) {
  workspaceView = view;
  for (const video of elements.appShell.querySelectorAll("video")) video.pause();
  updateWorkspaceAccess();
  if (workspaceView === "admin") loadAdminDashboard();
  if (workspaceView === "annotations") loadAnnotationDevelopment();
}

function applyLibraryFilter() {
  for (const button of document.querySelectorAll("[data-filter]")) {
    button.setAttribute("aria-pressed", String(button.dataset.filter === libraryFilter));
  }
  const lists = [elements.importList, elements.uploadList, elements.jobList];
  let visible = 0;
  let total = 0;
  for (const list of lists) {
    for (const card of list.children) {
      total += 1;
      const failed = card.classList.contains("failed");
      const ready = list === elements.jobList && card.classList.contains("completed");
      card.hidden = !(libraryFilter === "all" ||
        (libraryFilter === "failed" && failed) ||
        (libraryFilter === "ready" && ready) ||
        (libraryFilter === "active" && !failed && !ready));
      if (!card.hidden) visible += 1;
    }
  }
  document.querySelector("#filterEmpty").hidden = !total || visible > 0;
}

addVideoButton.addEventListener("click", () => {
  if (addFootage.hidden || workspaceView !== "library") openAddFootage();
  else {
    addFootage.hidden = true;
    addVideoButton.setAttribute("aria-expanded", "false");
  }
});
workspaceNav.addEventListener("click", (event) => {
  const button = event.target.closest("[data-view]");
  if (button) setWorkspaceView(button.dataset.view);
});
document.querySelector("#libraryFilters").addEventListener("click", (event) => {
  const button = event.target.closest("[data-filter]");
  if (!button) return;
  libraryFilter = button.dataset.filter;
  applyLibraryFilter();
});
