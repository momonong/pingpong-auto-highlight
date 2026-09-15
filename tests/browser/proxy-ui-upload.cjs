// Browser resume discovery against the dedicated HTTPS copy, never a human endpoint.
const {chromium,request}=require('playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const url=process.env.HC_ACCEPTANCE_URL;assert.equal(url,'https://127.0.0.1:18443/pingpong-highlight');
 const password=fs.readFileSync(process.env.HC_ACCEPTANCE_PASSWORD_FILE,'utf8').trim();
 const bytes=fs.readFileSync(process.env.HC_FIXTURE_MEDIA);
 const admin=await request.newContext({ignoreHTTPSErrors:true});
 const browser=await chromium.launch({headless:true});
 try{
  for(let i=0;i<30;i++){
   if((await admin.get(url+'/api/health').catch(()=>null))?.status()===200)break;
   await new Promise(r=>setTimeout(r,1000));
  }
  assert.equal((await admin.post(url+'/api/auth/login',{data:{username:'admin',password}})).status(),200);
  const username='proxy-ui-fixture-'+Date.now();
  assert.equal((await admin.post(url+'/api/admin/users',{data:{username,password,role:'user',display_name:'AUTOMATION UI ONLY'}})).status(),201);
  const ctx=await browser.newContext({ignoreHTTPSErrors:true,locale:'zh-TW'}),page=await ctx.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url+'/');await page.locator('#loginUsername').fill(username);await page.locator('#loginPassword').fill(password);
  await page.locator('#loginButton').click();await page.locator('#appShell').waitFor({state:'visible'});
  const filename='AUTOMATION-UI-RESUME.mp4',half=Math.floor(bytes.length/2);
  const upload=await ctx.request.post(url+'/api/uploads',{headers:{'Tus-Resumable':'1.0.0','Upload-Length':String(bytes.length),
    'Upload-Metadata':'filename '+Buffer.from(filename).toString('base64')}});
  assert.equal(upload.status(),201);const location=new URL(upload.headers().location,url).href;
  assert.equal((await ctx.request.patch(location,{headers:{'Tus-Resumable':'1.0.0','Upload-Offset':'0',
    'Content-Type':'application/offset+octet-stream'},data:bytes.subarray(0,half)})).status(),204);
  await page.reload();await page.locator('#appShell').waitFor({state:'visible'});
  await page.locator('#addVideoButton').click();
  await page.locator('#videoInput').setInputFiles({name:filename,mimeType:'video/mp4',buffer:bytes});
  await page.locator('#uploadButton').click();
  await page.locator('#jobList .job.completed').waitFor({timeout:90000});
  const jobs=(await(await ctx.request.get(url+'/api/jobs?scope=mine')).json()).jobs;
  assert.equal(jobs.length,1);assert.equal(jobs[0].upload_id,new URL(location).pathname.split('/').pop());
  const downloaded=await ctx.request.get(url+`/api/jobs/${jobs[0].id}/source?download=true`);
  assert.deepEqual(await downloaded.body(),bytes);assert.deepEqual(errors,[]);
  console.log(JSON.stringify({result:'PASS',job_id:jobs[0].id,checks:['HTTPS subpath browser login','reload and reselect original file',
   'browser discovers existing server upload and resumes same ID','same-origin scoped API fetch','download bytes match'],kind:'AUTOMATION FIXTURE ONLY'}));
 }finally{await browser.close();await admin.dispose();}
})().catch(e=>{console.error(e);process.exitCode=1;});
