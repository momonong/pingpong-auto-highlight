// Real browser + real SQLite API; synthetic proposals are explicitly fixtures.
const {chromium}=require('playwright');
const fs=require('node:fs');const path=require('node:path');const net=require('node:net');
const {spawn,spawnSync}=require('node:child_process');const assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');fs.mkdirSync(path.join(root,'data'),{recursive:true});
 const dir=fs.mkdtempSync(path.join(root,'data/review-browser-'));const media=path.join(dir,'fixture.mp4');
 const made=spawnSync('ffmpeg',['-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=10:duration=10','-c:v','libx264','-pix_fmt','yuv420p',media],{windowsHide:true});assert.equal(made.status,0,made.stderr?.toString());
 const proxy=spawnSync('ffmpeg',['-v','error','-ss','2','-i',media,'-t','8','-c:v','libx264',path.join(dir,'preview.mp4')],{windowsHide:true});assert.equal(proxy.status,0,proxy.stderr?.toString());
 const port=await new Promise(resolve=>{const s=net.createServer();s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>resolve(p));});});
 const url=`http://127.0.0.1:${port}`;
 const server=spawn(process.env.HIGHLIGHTCRAFT_TEST_PYTHON||'python',[path.join(__dirname,'review-server.py'),dir,String(port),media],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'src')}});
 let log='';server.stderr.on('data',c=>log+=c);let browser;
 try{
  for(let i=0;i<100;i++){try{if((await fetch(url)).ok)break;}catch{}if(server.exitCode!==null)throw Error(log);await new Promise(r=>setTimeout(r,100));}
  browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1280,height:900}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.locator('#resume').waitFor();
  await page.locator('#modelDetails summary').click();await page.locator('#proposals button').first().waitFor();
  await page.locator('#editDetails summary').click();await page.locator('#advancedCoverage summary').click();await page.locator('#timing summary').click();
  await page.locator('#original').click();
  await page.locator('#proposals button').first().click();assert.match(await page.locator('#suggestion').textContent(),/盲審/);
  assert(!(await page.locator('body').textContent()).includes('model fixture reason'));
  const playback=page.waitForFunction(()=>document.querySelector('video').currentTime>0.2);
  await page.locator('#preview').click();await playback;await page.locator('#play').click();
  await page.locator('#start').fill('1.2');await page.locator('#end').fill('3.2');
  await page.locator('#rally').selectOption('yes');await page.locator('#complete').selectOption('yes');await page.locator('#highlight').selectOption('include');
  await page.locator('#save').click();await page.locator('#points button').waitFor();
  await page.reload();await page.locator('#points button').click();await page.locator('#editDetails summary').click();await page.locator('#advancedCoverage summary').click();await page.locator('#modelDetails summary').click();await page.locator('#timing summary').click();await page.locator('#original').click();assert.equal(await page.locator('#start').inputValue(),'1.2');assert.equal(await page.locator('#highlight').inputValue(),'include');
  await page.waitForFunction(()=>document.querySelector('video').readyState>=2);
  await page.evaluate(()=>document.querySelector('video').currentTime=2);await page.locator('#split').click();await page.waitForFunction(()=>document.querySelectorAll('#points button').length===2);
  await page.locator('#points button').first().click();assert.equal(await page.locator('#highlight').inputValue(),'unrated');
  const other=await page.locator('#mergeTarget option').last().getAttribute('value');await page.locator('#mergeTarget').selectOption(other);await page.locator('#merge').click();await page.waitForFunction(()=>document.querySelectorAll('#points button').length===1);
  await page.locator('#next').click();await page.locator('#reject').click();await page.waitForFunction(()=>document.querySelectorAll('#points button').length===2);
  await page.locator('#new').click();await page.locator('#start').fill('4');await page.locator('#end').fill('5');await page.locator('#rally').selectOption('yes');await page.locator('#highlight').selectOption('omit');await page.locator('#save').click();await page.waitForFunction(()=>document.querySelectorAll('#points button').length===3);
  await page.locator('#coverStart').fill('0');await page.locator('#coverEnd').fill('5');await page.locator('#coverage').click();await page.waitForFunction(()=>document.querySelector('#stats').textContent.includes('0.00–5.00'));
  await page.locator('#timer').click();await page.waitForTimeout(200);await page.locator('#timer').click();await page.waitForFunction(()=>!document.querySelector('#stats').textContent.includes('計時 0.0 秒'));
  await page.reload();await page.locator('#points button').first().waitFor();assert.equal(await page.locator('#points button').count(),3);
  await page.locator('#points button').last().click();await page.locator('#editDetails summary').click();await page.locator('#advancedCoverage summary').click();await page.locator('#reason').fill('draft survives failure');
  await page.route('**/api/review/*',route=>route.request().method()==='POST'?route.abort():route.continue());
  await page.locator('#save').click();await page.waitForFunction(()=>document.querySelector('#message').classList.contains('error'));
  assert.equal(await page.locator('#reason').inputValue(),'draft survives failure');await page.unroute('**/api/review/*');
  await page.locator('#save').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.startsWith('已保存'));
  const exportURL=await page.locator('#export').getAttribute('href');const exported=await (await page.request.get(url+exportURL)).json();
  assert.equal(exported.review.points.length,3);assert(exported.review.active_ms>0);assert.equal(exported.review.coverage[0].end_ms,5000);
  fs.writeFileSync(path.join(dir,'export.json'),JSON.stringify(exported,null,2));
  await page.locator('#new').click();await page.locator('#compatible').click();
  await page.waitForFunction(()=>document.querySelector('video').readyState>=2);
  await page.evaluate(()=>document.querySelector('video').currentTime=1);await page.locator('#markStart').click();
  assert.equal(await page.locator('#start').inputValue(),'3.000');
  await page.evaluate(()=>document.querySelector('video').currentTime=2);await page.locator('#markEnd').click();
  assert.equal(await page.locator('#end').inputValue(),'4.000');
  await page.locator('#rally').selectOption('yes');await page.locator('#save').click();
  await page.waitForFunction(()=>document.querySelectorAll('#points button').length===4);
  const mapped=await (await page.request.get(url+exportURL)).json();
  assert(mapped.review.points.some(p=>p.start_ms===3000&&p.end_ms===4000));
  await page.screenshot({path:path.join(dir,'review.png'),fullPage:true});assert.deepEqual(errors,[]);
  console.log(JSON.stringify({result:'PASS',artifact:dir,checks:['video playback','blind first review','save reload','split merge','reject','add missed rally','coverage','pause timer','failed request retry','export','compatible preview absolute source timestamps']},null,2));
 }finally{if(browser)await browser.close();server.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
