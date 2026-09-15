// Destructive operations are limited to the dedicated integration copy and its new fixture.
const {request}=require('playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const url=process.env.HC_ACCEPTANCE_URL;
 assert.equal(url,'https://127.0.0.1:18443/pingpong-highlight');
 const password=fs.readFileSync(process.env.HC_ACCEPTANCE_PASSWORD_FILE,'utf8').trim();
 const ctx=await request.newContext({ignoreHTTPSErrors:true});
 let location;
 try{
  assert.equal((await ctx.post(url+'/api/auth/login',{data:{username:'admin',password}})).status(),200);
  const stale=await(await ctx.get(url+'/api/uploads')).json();
  for(const u of stale.uploads.filter(u=>u.filename==='AUTOMATION-LIMITS-ONLY.mp4'))
    assert.equal((await ctx.delete(url+'/api/uploads/'+u.id,{headers:{'Tus-Resumable':'1.0.0'}})).status(),204);
  const config=await(await ctx.get(url+'/api/config')).json(); assert.equal(config.chunk_size,8*1024**2);
  const upload=await ctx.post(url+'/api/uploads',{headers:{'Tus-Resumable':'1.0.0','Upload-Length':String(80*1024**2),
   'Upload-Metadata':'filename '+Buffer.from('AUTOMATION-LIMITS-ONLY.mp4').toString('base64')}});
  assert.equal(upload.status(),201); location=new URL(upload.headers().location,url).href;
  let offset=0;
  for(const size of [8*1024**2,32*1024**2]){
   const patch=await ctx.patch(location,{headers:{'Tus-Resumable':'1.0.0','Upload-Offset':String(offset),
    'Content-Type':'application/offset+octet-stream'},data:Buffer.alloc(size,1)});
   assert.equal(patch.status(),204);offset+=size;
   assert.equal((await ctx.head(location,{headers:{'Tus-Resumable':'1.0.0'}})).headers()['upload-offset'],String(offset));
  }
  const tooBig=await ctx.patch(location,{headers:{'Tus-Resumable':'1.0.0','Upload-Offset':String(offset),
    'Content-Type':'application/offset+octet-stream'},data:Buffer.alloc(32*1024**2+1,2)});
  assert.equal(tooBig.status(),413);
  assert.equal((await ctx.head(location,{headers:{'Tus-Resumable':'1.0.0'}})).headers()['upload-offset'],String(offset));
  assert.equal((await ctx.delete(location,{headers:{'Tus-Resumable':'1.0.0'}})).status(),204);location=null;
  console.log(JSON.stringify({result:'PASS',checks:['configured browser chunk 8 MiB','8 MiB PATCH','32 MiB PATCH',
   '32 MiB + 1 returns Nginx 413 and does not advance offset','only this unfinished fixture deleted']}));
 }finally{if(location)await ctx.delete(location,{headers:{'Tus-Resumable':'1.0.0'}});await ctx.dispose();}
})().catch(e=>{console.error(e);process.exitCode=1;});
