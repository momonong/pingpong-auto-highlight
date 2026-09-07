// Run with Playwright resolvable and HIGHLIGHTCRAFT_TEST_PYTHON set to a dev Python.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const net = require('node:net');

(async () => {
  const root = path.resolve(__dirname, '../..');
  fs.mkdirSync(path.join(root, 'data'), {recursive: true});
  const dir = fs.mkdtempSync(path.join(root, 'data/browser-test-'));
  const browser = await chromium.launch({headless: true});
  let server;
  const errors = [];
  try {
    const maker = await browser.newPage();
    const bytes = await maker.evaluate(async () => {
      const canvas = document.createElement('canvas');
      canvas.width = 320; canvas.height = 180;
      const ctx = canvas.getContext('2d');
      const stream = canvas.captureStream(15);
      const recorder = new MediaRecorder(stream, {mimeType: 'video/mp4'});
      const chunks = [];
      recorder.ondataavailable = e => chunks.push(e.data);
      const finished = new Promise(resolve => recorder.onstop = resolve);
      recorder.start();
      for (let i = 0; i < 30; i++) {
        ctx.fillStyle = '#17202b'; ctx.fillRect(0, 0, 320, 180);
        ctx.fillStyle = '#c8ff4d'; ctx.fillRect(i * 10, 80, 20, 20);
        await new Promise(resolve => setTimeout(resolve, 70));
      }
      recorder.stop(); await finished;
      stream.getTracks().forEach(track => track.stop());
      return Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()));
    });
    const media = path.join(dir, 'sample.mp4');
    fs.writeFileSync(media, Buffer.from(bytes));
    await maker.close();
    const port = await new Promise(resolve => {
      const probe = net.createServer();
      probe.listen(0, '127.0.0.1', () => {
        const value = probe.address().port; probe.close(() => resolve(value));
      });
    });
    const url = `http://127.0.0.1:${port}`;
    server = spawn(process.env.HIGHLIGHTCRAFT_TEST_PYTHON || 'python',
      [path.join(__dirname, 'server.py'), path.join(dir, 'state'), String(port), media],
      {cwd: root, windowsHide: true, env: {...process.env, PYTHONPATH: path.join(root, 'src')}});
    let serverLog = '';
    server.stderr.on('data', chunk => serverLog += chunk);
    for (let i = 0; i < 100; i++) {
      try { if ((await fetch(`${url}/api/health`)).ok) break; } catch (_) {}
      if (server.exitCode !== null) throw new Error(serverLog);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.equal((await fetch(`${url}/api/health`)).status, 200);
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
    let page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.locator('#loginUsername').fill('admin');
    await page.locator('#loginPassword').fill('browser-test-password');
    await page.locator('#loginButton').click();
    await page.locator('#appShell').waitFor({state: 'visible'});
    assert.equal(await page.locator('#addFootage').isVisible(), false);
    assert.equal(await page.locator('#adminPanel').isVisible(), false);
    await page.waitForFunction(() => document.querySelector('#serviceStatus').dataset.state === 'online');

    // Real resumable upload: persist one chunk, lose subsequent responses, reload and resume.
    let patches = 0;
    await page.route('**/api/uploads/*', async route => {
      if (route.request().method() === 'PATCH' && ++patches > 1) return route.abort();
      return route.continue();
    });
    await page.locator('#addVideoButton').click();
    await page.locator('#videoInput').setInputFiles(media);
    await page.locator('#uploadButton').click();
    await page.waitForFunction(() => document.querySelector('#transferPercent').textContent !== '0%');
    await page.reload();
    await page.unroute('**/api/uploads/*');
    await page.locator('#appShell').waitFor({state: 'visible'});
    const partial = await context.request.get(`${url}/api/uploads`);
    const uploads = (await partial.json()).uploads;
    assert.equal(uploads.length, 1); assert(uploads[0].offset > 0);
    const originalUploadId = uploads[0].id;
    await page.locator('#addVideoButton').click();
    await page.locator('#videoInput').setInputFiles(media);
    await page.locator('#uploadButton').click();
    await page.locator('#jobList .job.completed').waitFor({timeout: 20000});
    const jobs = (await (await context.request.get(`${url}/api/jobs?scope=mine`)).json()).jobs;
    assert.equal(jobs.length, 1); assert.equal(jobs[0].upload_id, originalUploadId);
    const card = page.locator('#jobList .job').first();
    assert.equal(await card.locator('source').getAttribute('src'), null);
    assert.equal(await card.locator('[data-job-action="delete"]').isVisible(), false);
    const downloadEvent = page.waitForEvent('download');
    await card.locator('.quick-download').click();
    const download = await downloadEvent;
    assert.equal(await download.failure(), null);
    await card.locator('.result-panel > summary').click();
    await page.waitForFunction(() => document.querySelector('#jobList video').readyState >= 2);
    await card.locator('video').evaluate(video => video.play());
    await page.waitForFunction(() => document.querySelector('#jobList video').currentTime > 0.2);
    await card.locator('.result-panel > summary').click();
    await page.waitForFunction(() => !document.querySelector('#jobList source').hasAttribute('src'));

    // Real Drive API and pipeline handoff, with an explicitly synthetic downloader.
    await page.locator('#driveUrl').fill('https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view');
    await page.locator('#driveButton').click();
    await page.waitForFunction(() => document.querySelectorAll('#jobList .job.completed').length === 2, {timeout: 20000});
    await page.locator('[data-filter="active"]').click();
    assert.equal(await page.locator('#filterEmpty').isVisible(), true);
    await page.locator('[data-filter="all"]').click();
    await page.locator('#addVideoButton').click();
    assert(await page.locator('.jobs-card').evaluate(card => card.clientWidth > 1000));
    await page.screenshot({path: path.join(dir, 'desktop.png'), fullPage: true});

    await page.locator('[data-view="admin"]').click();
    await page.locator('#adminPanel').waitFor({state: 'visible'});
    await page.locator('#newUsername').fill('viewer');
    await page.locator('#newDisplayName').fill('Browser Viewer');
    await page.locator('#newPassword').fill('browser-test-password');
    await page.locator('#createUserButton').click();
    await page.locator('#adminUserList').getByText('Browser Viewer').waitFor();
    await page.locator('[data-view="annotations"]').click();
    await page.locator('#annotationDevBlock').waitFor({state: 'visible'});
    await page.locator('.open-annotation-workspace').first().click();
    await page.locator('#annotationWorkspace').waitFor({state: 'visible'});
    await page.locator('#annotationWorkspaceClose').click();
    await page.locator('[data-view="library"]').click();

    // Offline, server failure, malformed health payload and delayed request recovery.
    await context.setOffline(true);
    await page.waitForFunction(() => document.querySelector('#serviceStatus').dataset.state === 'offline');
    assert.equal(await page.locator('#activityNotice').isVisible(), true);
    await context.setOffline(false);
    await page.waitForFunction(() => document.querySelector('#serviceStatus').dataset.state === 'online');
    await page.route('**/api/health', route => route.fulfill({status: 503, body: 'unavailable'}));
    await page.waitForFunction(() => document.querySelector('#serviceStatus').dataset.state === 'unavailable', {timeout: 15000});
    await page.unroute('**/api/health');
    await page.route('**/api/health', route => route.fulfill({status: 200, contentType: 'text/html', body: '<html>proxy error</html>'}));
    await new Promise(resolve => setTimeout(resolve, 5500));
    assert.equal(await page.locator('#serviceStatus').getAttribute('data-state'), 'unavailable');
    await page.unroute('**/api/health');
    await page.waitForFunction(() => document.querySelector('#serviceStatus').dataset.state === 'online', {timeout: 15000});
    await page.route('**/api/health', async route => {
      await new Promise(resolve => setTimeout(resolve, 7500));
      await route.fulfill({status: 200, contentType: 'application/json', body: '{"status":"ok"}'}).catch(() => {});
    });
    await page.waitForFunction(() => document.querySelector('#serviceStatus').dataset.state === 'unavailable', null, {timeout: 15000});
    await page.unroute('**/api/health');
    await page.waitForFunction(() => document.querySelector('#serviceStatus').dataset.state === 'online', null, {timeout: 15000});
    await page.route('**/api/jobs?**', route => route.fulfill({status: 503, body: 'unavailable'}));
    await page.locator('#refreshButton').click();
    await page.waitForFunction(() => !document.querySelector('#activityNotice').hidden);
    assert.equal(await page.locator('#jobList .job').count(), 2);
    await page.unroute('**/api/jobs?**');
    await page.waitForFunction(() => document.querySelector('#activityNotice').hidden);

    const mobileContext = await browser.newContext({
      viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true,
      storageState: await context.storageState(),
    });
    await page.close();
    page = await mobileContext.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.locator('#appShell').waitFor({state: 'visible'});
    assert.equal(await page.locator('#serviceStatus').isVisible(), true);
    assert.equal(await page.locator('[data-view="annotations"]').isVisible(), false);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(dir, 'mobile.png'), fullPage: true});
    await page.locator('#guideButton').click();
    assert.equal(await page.locator('#quickGuide').getAttribute('open'), '');
    await page.locator('#quickGuide > summary').click();
    await page.locator('#languageToggle').click();
    assert.equal(await page.locator('.workspace-heading h1').innerText(), 'My videos');
    await page.locator('#languageToggle').click();
    await page.locator('#logoutButton').click();
    await page.locator('#loginUsername').fill('viewer');
    await page.locator('#loginPassword').fill('browser-test-password');
    await page.locator('#loginButton').click();
    await page.locator('#appShell').waitFor({state: 'visible'});
    assert.equal(await page.locator('#jobList .job').count(), 0);
    assert.equal(await page.locator('[data-view="admin"]').isVisible(), false);
    assert.equal(await page.locator('#addFootage').isVisible(), false);
    await page.setViewportSize({width: 320, height: 740});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({result: 'PASS', artifacts: dir, checks: ['desktop/mobile', 'resume after reload', 'playback/download', 'Drive fixture', 'admin/annotations', 'offline/503/malformed health', 'stale progress/recovery', 'language', 'account isolation']}));
    await context.close();
    await mobileContext.close();
  } finally {
    await browser.close();
    if (server) server.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
