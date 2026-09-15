"use strict";
// A server-rendered slot supplies the base even on deep links and HTML reloads.
window.HC = Object.freeze({
  root: document.querySelector('meta[name="hc-root-path"]').content,
  url(path) {
    const url = new URL(path, window.location.origin);
    if (url.origin !== window.location.origin) throw new Error("Foreign application URL");
    const root = this.root;
    if (root && url.pathname !== root && !url.pathname.startsWith(root + "/")) {
      url.pathname = root + url.pathname;
    }
    return url.pathname + url.search + url.hash;
  },
});
