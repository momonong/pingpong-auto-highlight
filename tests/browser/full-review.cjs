// Whole-film workflow on synthetic video + disposable SQLite. No human efficacy claim.
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),net=require('node:net');
const {spawn,spawnSync}=require('node:child_process');const assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');fs.mkdirSync(path.join(root,'data'),{recursive:true});
 const dir=fs.mkdtempSync(path.join(root,'data/full-review-browser-')),media=path.join(dir,'fixture.mp4');
 const made=spawnSync('ffmpeg',['-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=10:duration=10','-c:v','libx264','-pix_fmt','yuv420p',media],{windowsHide:true});assert.equal(made.status,0);
 const port=await new Promise(r=>{const s=net.createServer();s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>r(p));});});
 const url=`http://127.0.0.1:${port}`;
 const server=spawn(process.env.HIGHLIGHTCRAFT_TEST_PYTHON||'python',[path.join(__dirname,'review-server.py'),dir,String(port),media],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'src')}});
 let browser,log='';server.stderr.on('data',b=>log+=b);
 try{
  for(let i=0;i<150;i++){try{if((await fetch(url)).ok)break;}catch{}if(server.exitCode!==null)throw Error(log);await new Promise(r=>setTimeout(r,100));}
  browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1360,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
  await page.goto(url);await page.waitForFunction(()=>document.querySelector('video').readyState>=2);
  assert((await page.locator('video').getAttribute('src')).endsWith('/full-preview'));
  assert.equal(await page.locator('#modelDetails').getAttribute('open'),null);
  assert.equal(await page.locator('#save').isDisabled(),true);
  assert.equal(await page.locator('#sourceClock').textContent(),'00:00 / 00:10');
  await page.locator('#resume').click();await page.waitForFunction(()=>document.querySelector('video').currentTime>.2);
  await page.evaluate(()=>{const v=document.querySelector('video');v.pause();v.currentTime=1;});
  await page.locator('#play').focus();await page.keyboard.press('i');assert.equal(await page.locator('#start').inputValue(),'1.000');assert.equal(await page.locator('#end').inputValue(),'');
  await page.evaluate(()=>document.querySelector('video').currentTime=2.5);await page.keyboard.press('o');
  await page.locator('#complete').selectOption('yes');assert.equal(await page.locator('#highlight').inputValue(),'unrated');
  await page.locator('#saveContinue').click();await page.waitForFunction(()=>!document.querySelector('video').paused&&document.querySelector('video').currentTime>2.6);
  const exportURL=await page.locator('#export').getAttribute('href');const get=async()=>await(await page.request.get(url+exportURL)).json();
  let state=await get();assert.equal(state.review.points.length,1);assert.deepEqual(state.review.coverage,[]);
  await page.evaluate(()=>{const v=document.querySelector('video');v.pause();v.currentTime=4;});
  await page.waitForFunction(()=>!document.querySelector('#reviewToHere').disabled);await page.locator('#reviewToHere').click();
  await page.waitForFunction(()=>document.querySelector('#reviewProgress').value===40);
  await page.reload();await page.waitForFunction(()=>document.querySelector('video').currentTime>=4);
  assert.match(await page.locator('#resume').textContent(),/00:04/);
  await page.evaluate(()=>document.querySelector('video').currentTime=5);await page.locator('#markStart').click();await page.reload();
  await page.waitForFunction(()=>document.querySelector('#start').value==='5');
  assert.equal(await page.locator('#end').inputValue(),'');assert.equal(await page.locator('#save').isDisabled(),true);
  await page.locator('#discard').click();assert.equal(await page.locator('#start').inputValue(),'');assert.equal(await page.locator('#points button').count(),1);
  await page.evaluate(()=>document.querySelector('video').currentTime=10);await page.waitForFunction(()=>!document.querySelector('#reviewToHere').disabled);await page.locator('#reviewToHere').click();await page.waitForFunction(()=>document.querySelector('#reviewProgress').value===100);
  state=await get();assert.deepEqual(state.review.coverage,[{start_ms:0,end_ms:10000}]);assert.equal(state.review.points[0].highlight,'unrated');
  await page.screenshot({path:path.join(dir,'full-review.png'),fullPage:true});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(dir,'export.json'),JSON.stringify(state,null,2));
  console.log(JSON.stringify({result:'PASS',artifact:dir,checks:['whole film default','one source timeline','model panel optional','keyboard after button focus','save continues playback','save is not coverage','explicit review progress','reload resumes unreviewed','incomplete draft survives reload','explicit discard preserves saved labels','whole-film coverage']},null,2));
 }finally{if(browser)await browser.close();server.kill();fs.writeFileSync(path.join(dir,'server.log'),log);}
})().catch(e=>{console.error(e);process.exitCode=1;});
