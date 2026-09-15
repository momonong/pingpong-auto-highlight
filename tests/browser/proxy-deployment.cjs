// Isolated deployment regression. Never run against 18080/18081/18082 or public hosts.
// HC_ACCEPTANCE_URL, HC_ACCEPTANCE_PASSWORD_FILE and HC_ACCEPTANCE_ARTIFACTS are required.
const {chromium, request} = require('playwright');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
(async () => {
  const url = process.env.HC_ACCEPTANCE_URL;
  assert(['https://127.0.0.1:18443/pingpong-highlight', 'http://127.0.0.1:18086'].includes(url), 'Only dedicated integration copies are allowed');
  const prefix = new URL(url).pathname.replace(/\/$/, '');
  const origin = new URL(url).origin;
  const dir = path.resolve(process.env.HC_ACCEPTANCE_ARTIFACTS);
  fs.mkdirSync(dir, {recursive: true});
  const password = fs.readFileSync(process.env.HC_ACCEPTANCE_PASSWORD_FILE, 'utf8').trim();
  const media = path.join(dir, 'AUTOMATION-FIXTURE.mp4');
  assert(!fs.existsSync(media), 'Use a new artifacts directory for every run');
  const made = spawnSync('ffmpeg', ['-v','error','-f','lavfi','-i',
    'testsrc2=size=640x360:rate=30:duration=10','-f','lavfi','-i',
    'sine=frequency=440:duration=10','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',media]);
  assert.equal(made.status, 0, String(made.stderr));
  const browser = await chromium.launch({headless: true});
  const anon = await request.newContext({baseURL:url,ignoreHTTPSErrors:true});
  const contexts = [];
  const checks = [];
  try {
    assert.equal((await anon.get(url+'/api/jobs')).status(), 401);
    const admin = await browser.newContext({ignoreHTTPSErrors:true,locale:'zh-TW'}); contexts.push(admin);
    const page = await admin.newPage();
    const errors=[]; page.on('pageerror', e => errors.push(e.message));
    page.on('dialog', d => d.accept());
    await page.goto(url);
    await page.locator('#loginUsername').fill('admin');
    await page.locator('#loginPassword').fill(password);
    await page.locator('#loginButton').click();
    await page.locator('#appShell').waitFor({state:'visible'});
    checks.push('browser login and anonymous API denial');
    const cookie = (await admin.cookies()).find(c => c.name.startsWith('hc_'));
    assert(cookie); assert.equal(cookie.path, prefix + '/');
    assert.equal(cookie.secure, !!prefix); assert.equal(cookie.httpOnly, true);
    await admin.addCookies([{name:'sibling_service',value:'preserve-me',url:origin},
      {name:'pingpong_session',value:'unrelated-root-cookie',url:origin}]);
    assert.equal((await admin.request.get(url+'/api/auth/me')).status(),200);
    if(prefix){
      const slash=await admin.request.get(origin+prefix+'?x=a%2Fb&job=kept',{maxRedirects:0});
      assert.equal(slash.status(),308);
      assert.equal(slash.headers().location,prefix+'/?x=a%2Fb&job=kept');
      const redirect=await admin.request.get(url+'/static?x=a%2Fb',{maxRedirects:0,
        headers:{'X-Forwarded-Proto':'http','X-Forwarded-Host':'evil.test','Forwarded':'host=evil.test;proto=http'}});
      assert.equal(redirect.status(),307);
      assert.equal(redirect.headers().location,url+'/static/?x=a%2Fb');
      assert.equal((await admin.request.get(origin+'/api/health')).headers()['x-sibling-service'],'preserved');
      const sibling=await admin.newPage(); let sent;
      sibling.on('request',async r=>{if(r.url()===origin+'/')sent=await r.allHeaders();});
      await sibling.goto(origin+'/'); await sibling.waitForTimeout(100);
      assert(!sent.cookie.includes(cookie.name+'=')); assert(sent.cookie.includes('sibling_service=preserve-me'));
      await sibling.close();
    }
    checks.push(prefix ? 'cookie Path/Secure/name isolation and forwarded header sanitization; slash/query preservation' : 'root cookie Path/name and HTTP compatibility');
    const users=[];
    const suffix=Date.now();
    for (const label of ['owner','other']) {
      const username=`acceptance-${label}-${suffix}`;
      const created=await admin.request.post(url+'/api/admin/users', {data:{username,password,display_name:`AUTOMATION FIXTURE ${label}`,role:'user'}});
      assert.equal(created.status(),201,await created.text());
      const ctx=await browser.newContext({ignoreHTTPSErrors:true,locale:'zh-TW'}); contexts.push(ctx);
      assert.equal((await ctx.request.post(url+'/api/auth/login',{data:{username,password}})).status(),200);
      users.push(ctx);
    }
    const [owner,other]=users;
    const bytes=fs.readFileSync(media);
    const upload=await owner.request.post(url+'/api/uploads',{headers:{
      'Tus-Resumable':'1.0.0','Upload-Length':String(bytes.length),'Upload-Metadata':`filename ${Buffer.from('AUTOMATION-FIXTURE.mp4').toString('base64')}`}});
    assert.equal(upload.status(),201,await upload.text());
    const location=upload.headers().location;
    assert(location.startsWith(prefix+'/api/uploads/'));
    const uploadURL = new URL(location,origin).href;
    const half = Math.floor(bytes.length/2);
    const headers = {'Tus-Resumable':'1.0.0','Content-Type':'application/offset+octet-stream'};
    let patch = await owner.request.patch(uploadURL,{headers:{...headers,'Upload-Offset':'0'},data:bytes.subarray(0,half)});
    assert.equal(patch.status(),204,await patch.text());
    // An actual socket disconnect during a PATCH must not advance committed offset.
    const transport=require(url.startsWith('https')?'node:https':'node:http');
    const authCookies=(await owner.cookies()).map(c=>c.name+'='+c.value).join('; ');
    await new Promise(resolve=>{
      const broken=transport.request(uploadURL,{method:'PATCH',rejectUnauthorized:false,
        headers:{...headers,'Upload-Offset':String(half),'Content-Length':String(bytes.length-half),Cookie:authCookies}});
      broken.on('error',()=>resolve()); broken.write(bytes.subarray(half,half+128));
      setTimeout(()=>{broken.destroy();resolve();},150);
    });
    await new Promise(r=>setTimeout(r,150));
    const resumed=await request.newContext({ignoreHTTPSErrors:true,storageState:await owner.storageState()});
    const head=await resumed.head(uploadURL,{headers:{'Tus-Resumable':'1.0.0'}});
    assert.equal(head.status(),200); assert.equal(head.headers()['upload-offset'],String(half));
    const conflict=await resumed.patch(uploadURL,{headers:{...headers,'Upload-Offset':'0'},data:bytes.subarray(half)});
    assert.equal(conflict.status(),409);
    patch=await resumed.patch(uploadURL,{headers:{...headers,'Upload-Offset':String(half)},data:bytes.subarray(half)});
    assert.equal(patch.status(),204,await patch.text()); await resumed.dispose();
    checks.push('TUS POST/PATCH/HEAD; interrupted socket, new client resumes, stale offset rejected');
    let job;
    for(let i=0;i<120;i++){
      const jobs=await (await owner.request.get(url+'/api/jobs?scope=mine')).json(); job=jobs.jobs[0];
      if(job?.status==='completed')break;
      if(job?.status==='failed')throw Error(JSON.stringify(job));
      await new Promise(r=>setTimeout(r,1000));
    }
    assert.equal(job?.status,'completed',JSON.stringify(job));
    const download=await owner.request.get(url+`/api/jobs/${job.id}/source?download=true`);
    assert.equal(download.status(),200);assert(download.headers()['content-disposition'].includes('attachment'));
    assert.deepEqual(await download.body(),bytes); assert(download.headers()['cache-control'].includes('no-store'));
    for(const file of job.result.files){
      assert(file.url.startsWith(prefix+'/api/jobs/'));
      const got=await owner.request.get(origin+file.url+'?download=true');
      assert.equal(got.status(),200); assert(got.headers()['cache-control'].includes('no-store'));
    }
    const base=`/api/jobs/${job.id}/rally-review`;
    assert.deepEqual((await(await other.request.get(url+'/api/jobs?scope=mine')).json()).jobs,[]);
    for(const endpoint of [base,base+'/export',base+'/full-preview',`/api/jobs/${job.id}/source`]){
      assert.equal((await other.request.get(url+endpoint)).status(),404,endpoint);
      assert.equal((await anon.get(url+endpoint)).status(),401,endpoint);
    }
    assert.equal((await other.request.post(url+base+'/full-preview')).status(),404);
    assert.equal((await other.request.post(url+base+'/baseline')).status(),404);
    checks.push('real upload and production processor completed; owner isolation');
    const review=await owner.newPage();review.on('dialog',d=>d.accept());review.on('pageerror',e=>errors.push(e.message));
    await review.goto(url+`/static/review/index.html?job=${job.id}`);
    const before = await (await owner.request.get(url+base)).json();
    assert.deepEqual(before.review.points, []);
    if (!before.full_preview_available) await review.locator('#prepareFull').click();
    await review.waitForFunction(()=>document.querySelector('video').readyState>=2 && document.querySelector('video').src.endsWith('/full-preview'));
    const video=await review.locator('video').evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}));
    assert.equal(video.width,640);assert.equal(video.height,360);assert(Math.abs(video.duration-10)<.1);
    const ranged=await owner.request.get(url+base+'/full-preview',{headers:{Range:'bytes=0-1023'}});
    assert.equal(ranged.status(),206);assert.equal((await ranged.body()).length,1024);
    assert.equal(await review.locator('#modelDetails').getAttribute('open'),null);
    await review.evaluate(()=>{const v=document.querySelector('video');v.pause();v.currentTime=1;});
    await review.locator('#play').focus();await review.keyboard.press('i');
    await review.evaluate(()=>document.querySelector('video').currentTime=2.5);await review.keyboard.press('o');
    await review.locator('#complete').selectOption('uncertain');
    await review.locator('details').filter({has: review.locator('#reason')}).locator('summary').click();
    await review.locator('#reason').fill('BROWSER AUTOMATION FIXTURE - NOT HUMAN GROUND TRUTH');
    await review.locator('#saveContinue').click();
    await review.waitForFunction(()=>!document.querySelector('video').paused && document.querySelector('video').currentTime>2.6);
    let state=await(await owner.request.get(url+base+'/export')).json();
    assert.equal(state.review.points.length,1);assert.deepEqual(state.review.coverage,[]);
    await review.evaluate(()=>{const v=document.querySelector('video');v.pause();v.currentTime=4;});
    await review.waitForFunction(()=>!document.querySelector('#reviewToHere').disabled);
    await review.locator('#reviewToHere').click();await review.waitForFunction(()=>document.querySelector('#reviewProgress').value===40);
    await review.reload();await review.waitForFunction(()=>document.querySelector('video').currentTime>=4);
    await review.evaluate(()=>document.querySelector('video').currentTime=5);await review.locator('#markStart').click();
    await review.reload();await review.waitForFunction(()=>document.querySelector('#start').value==='5');
    assert.equal(await review.locator('#end').inputValue(),'');
    await review.locator('#discard').click();assert.equal(await review.locator('#points button').count(),1);
    state=await(await owner.request.get(url+base+'/export')).json();
    assert.deepEqual(state.review.coverage,[{start_ms:0,end_ms:4000}]);
    assert.equal(state.review.points[0].start_ms,1000);assert.equal(state.review.points[0].end_ms,2500);
    assert.deepEqual(errors,[]);
    checks.push('H264 full-preview with image and Range','I/O and save/resume','explicit coverage only','reload and draft recovery','export');
    const logout=await admin.request.post(url+'/api/auth/logout');
    assert.equal(logout.status(),204); assert(logout.headers()['set-cookie'].includes('Max-Age=0'));
    assert(logout.headers()['set-cookie'].includes('Path='+prefix+'/'));
    assert((await admin.cookies()).some(c=>c.name==='sibling_service'));
    assert.equal((await admin.request.get(url+'/api/auth/me')).status(),401);
    if(prefix)assert.equal(logout.headers()['clear-site-data'],undefined);
    checks.push('source bytes download; result URLs; no shared caching; scoped logout preserves sibling cookie');
    await review.screenshot({path:path.join(dir,'deployment-review.png'),fullPage:true});
    fs.writeFileSync(path.join(dir,'export.json'),JSON.stringify(state,null,2));
    const receipt={result:'PASS',url,job_id:job.id,video,checks,kind:'AUTOMATION FIXTURE ONLY',review_url:url+`/static/review/index.html?job=${job.id}`};
    fs.writeFileSync(path.join(dir,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));
  }finally{await anon.dispose();for(const ctx of contexts)await ctx.close();await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
