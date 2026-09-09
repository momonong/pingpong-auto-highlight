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
  fs.writeFileSync(path.join(dir, '.point-review-fixture'), 'synthetic-point-review-v1');
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

    page.on('dialog', dialog => dialog.accept());
    await page.locator('#addVideoButton').click();
    await page.locator('#videoInput').setInputFiles(media);
    await page.locator('#uploadButton').click();
    await page.locator('#jobList .job.completed').waitFor({timeout: 20000});
    const job = (await (await context.request.get(`${url}/api/jobs`)).json()).jobs[0];
    const endpoint = `${url}/api/jobs/${job.id}/point-review`;
    const getReview = async () => (await (await context.request.get(endpoint)).json());
    const open = async () => {
      await page.locator('[data-view="annotations"]').click();
      await page.locator('.open-annotation-workspace').first().click();
      await page.waitForFunction(() => !document.querySelector('#pointImport').disabled);
      await page.waitForFunction(() => document.querySelector('#annotationWorkspaceVideo').readyState >= 2);
    };
    const saved = async () => {
      await page.waitForFunction(() => !pointBusy && !pointDirty && !pointPendingRequest);
    };
    await open();
    assert.equal((await getReview()).points.length, 0);
    await page.locator('#pointImport').click(); await saved();
    assert.equal(await page.locator('[data-point-id]').count(), 2);
    await page.locator('[data-point-id]').first().click();
    await page.waitForFunction(() => document.querySelector('#annotationWorkspaceVideo').currentTime > 0.1);
    await page.locator('#annotationWorkspaceVideo').evaluate(v => v.pause());
    await page.evaluate(() => {
      const video = document.querySelector('#annotationWorkspaceVideo');
      pointPlay(video); video.pause();
    });
    await page.waitForTimeout(100);
    assert(!await page.locator('#annotationWorkspaceMessage').evaluate(e => e.classList.contains('error')));
    assert(!(await page.locator('#annotationWorkspace').innerText()).includes('987.654'));
    await page.locator('#pointStart').fill('0.15');
    await page.locator('#pointEnd').fill('0.85');
    await page.locator('#pointValidity').selectOption('valid');
    await page.locator('#pointBoundary').selectOption('confirmed');
    await page.locator('#pointReasons input[value="rally"]').check();
    await page.locator('#pointQuality input[value="occlusion"]').check();
    // Typing numeric keys or Enter inside a text field must not trigger ratings or saves.
    await page.locator('#pointNote').fill('fixture');
    await page.locator('#pointNote').press('3');
    await page.locator('#pointNote').press('Enter');
    assert.equal(await page.locator('#pointRating').inputValue(), 'unrated');
    assert.equal((await getReview()).revision, 1);
    await page.locator('#annotationWorkspace').focus();
    await page.keyboard.press('0');
    await page.keyboard.press('Enter'); await saved();
    let data = await getReview();
    const original = data.points.find(p => p.start_ms === 150);
    assert.equal(original.excitement, 0); assert.equal(original.end_ms, 850);
    assert.deepEqual(original.reason_tags, ['rally']); assert.deepEqual(original.quality_tags, ['occlusion']);
    assert.equal(await page.locator('#pointStart').inputValue(), '1');
    // Unknown and unable remain distinct from ordinary complete points.
    await page.locator('#pointValidity').selectOption('valid');
    await page.locator('#pointBoundary').selectOption('confirmed');
    await page.locator('#pointRating').selectOption('unable');
    await page.locator('#annotationWorkspaceSave').click(); await saved();
    await page.locator('#pointFilter').selectOption('unrated');
    assert.equal(await page.locator('[data-point-id]').count(), 0);
    await page.locator('#pointFilter').selectOption('complete');
    assert.equal(await page.locator('[data-point-id]').count(), 2);
    await page.locator('#pointFilter').selectOption('all');
    await page.locator('[data-point-id]').first().click();
    await page.locator('#annotationWorkspaceVideo').evaluate(v => {v.pause(); v.currentTime = 0.5;});
    await page.locator('#pointSplit').click(); await saved();
    data = await getReview();
    const children = data.points.filter(p => p.active && p.parent_ids.includes(original.id));
    assert.deepEqual(children.map(p => [p.start_ms, p.end_ms]), [[150, 500], [500, 850]]);
    await page.locator('#pointMerge').click(); await saved();
    data = await getReview();
    const merged = data.points.find(p => p.active && p.origin.kind === 'merge');
    assert.deepEqual(merged.parent_ids, children.map(p => p.id));
    assert.equal(merged.excitement, null);
    // Add a missing point and retain unsaved local edits across reload and re-login.
    await page.locator('#annotationWorkspaceVideo').evaluate(v => {v.pause(); v.currentTime = 1.8;});
    await page.locator('#pointNew').click();
    await page.locator('#pointEnd').fill('1.95');
    await page.locator('#pointNote').fill('recover my draft');
    await page.reload(); await page.locator('#appShell').waitFor({state:'visible'}); await open();
    assert.equal(await page.locator('#pointNote').inputValue(), 'recover my draft');
    await page.locator('#pointRating').selectOption('3');
    await page.locator('#annotationWorkspaceSave').click(); await saved();
    // Lose a committed response: retry exactly once and do not duplicate the new point.
    await page.locator('#pointNote').fill('lost response');
    await page.route('**/point-review', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      await route.fetch(); await route.abort();
    });
    await page.locator('#annotationWorkspaceSave').click();
    await page.waitForFunction(() => !pointBusy && !!pointPendingRequest);
    const afterLost = await getReview();
    assert.equal(await page.locator('#pointRetry').isVisible(), true);
    await page.unroute('**/point-review');
    await page.locator('#pointRetry').click(); await saved();
    assert.equal((await getReview()).revision, afterLost.revision);
    // Concurrent windows get a conflict, never silent last-write-wins.
    await page.locator('#pointNote').fill('stale local draft');
    data = await getReview();
    await context.request.post(endpoint, {data: {revision: data.revision, request_id: crypto.randomUUID(),
      action:'coverage_add', interval:{start_ms:0,end_ms:500}}});
    await page.locator('#annotationWorkspaceSave').click();
    await page.waitForFunction(() => !pointBusy && document.querySelector('#annotationWorkspaceMessage').classList.contains('error'));
    assert.equal(await page.locator('#pointNote').inputValue(), 'stale local draft');
    assert(!(await getReview()).points.some(p => p.note === 'stale local draft'));
    await page.locator('#pointReload').click();
    await page.waitForFunction(() => !pointBusy && !pointDirty);
    await page.locator('#coverageStart').fill('0.5'); await page.locator('#coverageEnd').fill('1.5');
    await page.locator('#pointCoverageAdd').click(); await saved();
    assert.deepEqual((await getReview()).unknown_intervals, [{start_ms:1500,end_ms:2000}]);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#pointExportLines').click();
    const download = await downloadEvent;
    const exportPath = path.join(dir, 'review.jsonl'); await download.saveAs(exportPath);
    const exported = fs.readFileSync(exportPath,'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(exported.filter(r=>r.type==='point').length, (await getReview()).points.length);
    await page.locator('#annotationWorkspaceClose').click();
    await page.locator('#languageToggle').click(); await open();
    assert.equal(await page.locator('#annotationWorkspaceTitle').innerText(), 'Point review workspace');
    assert((await page.locator('#pointUnknown').innerText()).includes('Unknown intervals'));
    await page.screenshot({path:path.join(dir,'point-review-en.png'),fullPage:true});
    await page.locator('#annotationWorkspaceClose').click();
    await page.locator('#languageToggle').click();
    await page.locator('#logoutButton').click();
    await page.locator('#loginUsername').fill('admin'); await page.locator('#loginPassword').fill('browser-test-password');
    await page.locator('#loginButton').click(); await page.locator('#appShell').waitFor({state:'visible'}); await open();
    assert((await getReview()).points.some(p=>p.note==='lost response'));
    await page.locator('[data-point-id]').last().click();
    await page.locator('#pointDelete').click(); await saved();
    assert.equal((await getReview()).points.filter(p=>p.active).length,2);
    await page.screenshot({path:path.join(dir,'point-review-zh.png'),fullPage:true});
    await page.locator('[data-point-id]').first().click();
    await page.locator('#annotationWorkspaceVideo').evaluate(v => v.pause());
    await page.screenshot({path:path.join(dir,'point-review-zh.png'),fullPage:true});
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({result:'PASS', artifacts:dir, url, checks:[
      'synthetic original playback', 'candidate import', 'boundaries and stable IDs', 'scores 0/3/unable',
      'split/merge/delete', 'input shortcut guard', 'draft reload', 'lost response retry',
      'concurrent conflict', 'coverage unknown', 'JSONL download', 're-login persistence', 'Chinese/English'
    ]}));
    await context.close();
  } finally {
    await browser.close();
    if (server) server.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

