// Real-source playback acceptance. Mutation is allowed only on an explicit separate copy.
const {chromium,request}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const url=process.env.HC_ACCEPTANCE_URL;
 assert.match(url||'',/^http:\/\/127\.0\.0\.1:\d+$/);
 const mutate=process.env.HC_MUTATION_COPY==='true';
 if(mutate)assert.equal(new URL(url).port,'18083','Mutations require the dedicated disposable copy on port 18083');
 const receipt=JSON.parse(fs.readFileSync(process.env.HC_REAL_RECEIPT,'utf8'));
 const password=fs.readFileSync(process.env.HC_ACCEPTANCE_PASSWORD_FILE,'utf8').trim();
 const dir=path.resolve(process.env.HC_ACCEPTANCE_ARTIFACTS);fs.mkdirSync(dir,{recursive:true});
 const base=`/api/jobs/${receipt.job_id}/rally-review`;
 const browser=await chromium.launch({headless:true});
 try{
  const ctx=await browser.newContext({locale:'zh-TW',viewport:{width:1440,height:1000}});
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
  await page.goto(url);await page.locator('#loginUsername').fill('admin');await page.locator('#loginPassword').fill(password);await page.locator('#loginButton').click();await page.locator('#appShell').waitFor({state:'visible'});
  const get=async()=>await(await ctx.request.get(url+base+'/export')).json();
  const before=await get();assert.equal(before.source.id,'636c4bb3605b97f3a00e4a7787cc518864bd4a2a9c7b57727d5bdb906bd7adc8');
  assert.deepEqual(before.review.points,[]);assert.deepEqual(before.review.coverage,[]);
  await page.goto(url+`/static/review/index.html?job=${receipt.job_id}`);
  await page.waitForFunction(()=>document.querySelector('video').readyState>=2);
  assert((await page.locator('video').getAttribute('src')).endsWith('/full-preview'));
  assert.equal(await page.locator('#modelDetails').getAttribute('open'),null);
  const samples=[];
  for(const at of [0,110,220]){
   await page.locator('video').evaluate((v,t)=>{v.pause();v.currentTime=t;},at);
   await page.waitForFunction(t=>{const v=document.querySelector('video');return !v.seeking&&v.readyState>=2&&Math.abs(v.currentTime-t)<.1;},at);
   await page.locator('video').evaluate(v=>v.play());await page.waitForFunction(t=>document.querySelector('video').currentTime>t+.15,at);
   await page.locator('video').evaluate(v=>v.pause());
   const sample=await page.locator('video').evaluate(v=>({time:v.currentTime,width:v.videoWidth,height:v.videoHeight,duration:v.duration,decodedFrames:v.getVideoPlaybackQuality().totalVideoFrames}));
   assert.equal(sample.width,640);assert.equal(sample.height,360);assert(Math.abs(sample.duration-223.394)<.1);assert(sample.decodedFrames>0);samples.push(sample);
   await page.screenshot({path:path.join(dir,`real-${at}.png`),fullPage:true});
  }
  const ranged=await ctx.request.get(url+base+'/full-preview',{headers:{Range:'bytes=0-1023'}});assert.equal(ranged.status(),206);assert.equal((await ranged.body()).length,1024);
  const checks=['login','source identity','real frames at 0/110/220 seconds','full-film H264 preview','Range 206','model panel collapsed'];
  if(mutate){
   const outsider=await request.newContext({baseURL:url});
   try{
    assert.equal((await outsider.get(base)).status(),401);
    const username='real-copy-outsider-'+Date.now();
    assert.equal((await ctx.request.post(url+'/api/admin/users',{data:{username,password,role:'user',display_name:'AUTOMATION COPY ONLY'}})).status(),201);
    assert.equal((await outsider.post('/api/auth/login',{data:{username,password}})).status(),200);
    for(const endpoint of [base,base+'/export',base+'/full-preview',`/api/jobs/${receipt.job_id}/source`])assert.equal((await outsider.get(endpoint)).status(),404,endpoint);
    assert.equal((await outsider.post(base+'/full-preview')).status(),404);
    assert.equal((await outsider.post(base,{data:{action:'coverage',revision:0,request_id:'forbidden-copy',interval:{start_ms:0,end_ms:1000}}})).status(),404);
   }finally{await outsider.dispose();}
   await page.locator('video').evaluate(v=>{v.pause();v.currentTime=110;});await page.locator('#play').focus();await page.keyboard.press('i');
   await page.locator('video').evaluate(v=>v.currentTime=113);await page.keyboard.press('o');
   await page.locator('#rally').selectOption('uncertain');await page.locator('#complete').selectOption('uncertain');
   await page.locator('#editDetails > summary').click();await page.locator('#reason').fill('AUTOMATION COPY ONLY - NOT HUMAN GROUND TRUTH');
   await page.locator('#saveContinue').click();await page.waitForFunction(()=>{const v=document.querySelector('video');return !v.paused&&v.currentTime>113.15;});
   let state=await get();assert.equal(state.review.points.length,1);assert.deepEqual(state.review.coverage,[]);
   await page.locator('video').evaluate(v=>{v.pause();v.currentTime=120;});await page.waitForFunction(()=>!document.querySelector('#reviewToHere').disabled);await page.locator('#reviewToHere').click();
   await page.waitForFunction(()=>document.querySelector('#reviewProgress').value>53);
   await page.reload();await page.waitForFunction(()=>document.querySelector('video').currentTime>=120);
   await page.locator('video').evaluate(v=>v.currentTime=150);await page.locator('#markStart').click();await page.reload();await page.waitForFunction(()=>document.querySelector('#start').value==='150');
   assert.equal(await page.locator('#end').inputValue(),'');await page.locator('#discard').click();assert.equal(await page.locator('#points button').count(),1);
   await page.locator('video').evaluate(v=>{v.pause();v.currentTime=v.duration;});await page.waitForFunction(()=>!document.querySelector('#reviewToHere').disabled);await page.locator('#reviewToHere').click();await page.waitForFunction(()=>document.querySelector('#reviewProgress').value===100);
   state=await get();assert.equal(state.review.points[0].start_ms,110000);assert.equal(state.review.points[0].end_ms,113000);assert.equal(state.review.points[0].rally,'uncertain');
   assert.deepEqual(state.review.coverage,[{start_ms:0,end_ms:before.source.duration_ms}]);
   fs.writeFileSync(path.join(dir,'automation-copy-export.json'),JSON.stringify(state,null,2));
   checks.push('owner isolation for real source','I/O and save/resume','save does not imply coverage','explicit whole-film coverage','reload resumes at unreviewed position','incomplete draft restoration/discard','export');
  }else assert.deepEqual(await get(),before);
  assert.deepEqual(errors,[]);
  const result={result:'PASS',url,job_id:receipt.job_id,source_id:before.source.id,kind:mutate?'AUTOMATION ON DISPOSABLE COPY':'REAL PLAYBACK ONLY - HUMAN REVIEW UNCHANGED',samples,checks};
  fs.writeFileSync(path.join(dir,'receipt.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
