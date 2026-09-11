'use strict';
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const job = params.get('job');
let sourceId = '', state, pointId = null, proposalId = null, runId = '', busy = false;
let timerStart = null, pendingMs = 0, dirty = false, pendingCommand = null;
let mediaOffset = 0, mediaEnd = Infinity;
const video = $('video');
function sourceTime(){return video.currentTime+mediaOffset;}
function seekSource(seconds){if(seconds<mediaOffset||seconds>mediaEnd){message('此時間不在相容短片內，請切回原片或建立該區段的預覽。',true);return false;}video.currentTime=seconds-mediaOffset;return true;}
function useOriginal(){mediaOffset=0;mediaEnd=state.source.duration_ms/1000;video.src=job?`/api/jobs/${encodeURIComponent(job)}/source`:base()+'/source';$('mediaMode').textContent='原片時間';}
function useCompatible(){const run=state.runs.find(r=>r.id===runId);if(!run?.preview_available)return;mediaOffset=run.preview_range.start_ms/1000;mediaEnd=run.preview_range.end_ms/1000;video.src=base()+'/preview/'+encodeURIComponent(runId);$('mediaMode').textContent=`相容短片 · 原片 ${mediaOffset}–${mediaEnd} 秒（原生播放列顯示短片時間；所有標記使用原片時間）`;}
const labels = {yes:'是',no:'否',uncertain:'不確定',unable:'無法判斷',omit:'不收錄',include:'可收錄',must:'必收錄',unrated:'未判斷'};
const base = () => job ? `/api/jobs/${encodeURIComponent(job)}/rally-review` : `/api/review/${sourceId}`;
const draftKey = () => `hc-review-draft:${base()}`;
function message(text, error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
async function request(url, options={}){const r=await fetch(url,options);if(!r.ok){let p=await r.json();throw new Error(`${r.status}: ${p.detail || 'Request failed'}`);}return r.json();}
function fields(){return {start_ms:Math.round(Number($('start').value)*1000),end_ms:Math.round(Number($('end').value)*1000),rally:$('rally').value,complete:$('complete').value,highlight:$('rally').value==='yes'?$('highlight').value:'unrated',reason:$('reason').value};}
function persistDraft(){dirty=true;localStorage.setItem(draftKey(),JSON.stringify({pointId,proposalId,point:fields()}));}
function edit(p=null, human=false){pointId=human?p.id:null;proposalId=p&&!human?p.id:null;
  const f=human?p:{start_ms:p?.start_ms??Math.round(sourceTime()*1000),end_ms:p?.end_ms??Math.round(sourceTime()*1000)+1000,rally:'uncertain',complete:'uncertain',highlight:'unrated',reason:''};
  $('start').value=f.start_ms/1000;$('end').value=f.end_ms/1000;
  for(const k of ['rally','complete','highlight','reason'])$(k).value=f[k];
  $('editing').textContent=human?'編輯人工紀錄':p?'審核模型候選':'補漏回合';
  $('suggestion').textContent=human?'已保存的人工判斷；不會因重新推論而覆寫。':p?.blind?'盲審樣本：首次人工判斷前隱藏模型建議。':p?`模型建議（未校準）：回合 ${labels[p.rally]}；完整 ${labels[p.complete]}；精彩 ${labels[p.highlight]}\n${p.reason}`:'請先觀看片段，再作判斷。';
  $('split').disabled=!human;$('merge').disabled=!human;
  if(p){seekSource(Math.max(mediaOffset,p.start_ms/1000-1.5));$('coverStart').value=Math.max(mediaOffset,p.start_ms/1000-1.5);$('coverEnd').value=Math.min(mediaEnd,p.end_ms/1000+1.5);}
  dirty=false;localStorage.removeItem(draftKey());
}
function choose(p,human=false){if(busy)return;if(dirty){message('有尚未保存的修改，請先保存；重新載入可恢復伺服器版本。',true);return;}edit(p,human);}
function list(id,items,action,label){$(id).replaceChildren();for(const p of items){const b=document.createElement('button');b.textContent=label(p);b.onclick=()=>action(p);$(id).append(b);}}
const span=p=>`${(p.start_ms/1000).toFixed(2)}–${(p.end_ms/1000).toFixed(2)}s`;
function render(){const r=state.review;const previous=runId; $('runs').replaceChildren();
  $('scope').textContent=`本次分析範圍：${(state.source.scope||[]).map(span).join('、')}。範圍外仍可人工補漏，但尚無本次模型分析。`;
  for(const run of state.runs){const o=new Option(`${run.model_id} · ${run.id.slice(0,8)}`,run.id);$('runs').add(o);}
  runId=state.runs.some(r=>r.id===previous)?previous:state.runs.at(-1)?.id||'';$('runs').value=runId;
  $('compatible').hidden=!state.runs.find(r=>r.id===runId)?.preview_available;
  const reviewed=new Set(r.points.flatMap(p=>p.proposal_ids));
  list('proposals',state.runs.find(r=>r.id===runId)?.proposals||[],p=>{const human=r.points.find(x=>x.proposal_ids.includes(p.id));choose(human||p,!!human);},p=>`${reviewed.has(p.id)?'已審核':'尚未審核'} · ${span(p)}${p.blind?' · 盲審':''}`);
  list('points',r.points,p=>choose(p,true),p=>`${span(p)} · ${labels[p.rally]} · ${labels[p.highlight]}`);
  $('mergeTarget').replaceChildren(new Option('選擇另一筆人工紀錄',''));
  for(const p of r.points)$('mergeTarget').add(new Option(span(p),p.id));
  const seek=p=>{if(seekSource(p.start_ms/1000)){$('coverStart').value=p.start_ms/1000;$('coverEnd').value=Math.min(mediaEnd,p.end_ms/1000);}};
  list('gaps',r.unreviewed,seek,span);list('proposalGaps',r.proposal_gaps,seek,span);
  $('stats').textContent=`已審核 ${r.coverage.map(span).join('、')||'尚無'} · 計時 ${(r.active_ms/1000).toFixed(1)} 秒 · 修改 ${r.modification_count} 次（尚無省時結論）`;
}
function tick(){if(timerStart!==null){pendingMs+=performance.now()-timerStart;timerStart=performance.now();}}
function controls(){for(const e of document.querySelectorAll('input,select,textarea,button'))e.disabled=busy||!!pendingCommand&&!['save','reload'].includes(e.id);$('split').disabled=busy||!!pendingCommand||!pointId;$('merge').disabled=busy||!!pendingCommand||!pointId;}
async function command(action, extra={}){if(busy)return;busy=true;tick();
  const elapsed=Math.min(60000,Math.round(pendingMs));
  const c=pendingCommand||{action,revision:state.review.revision,request_id:crypto.randomUUID(),elapsed_ms:elapsed,...extra};
  pendingCommand=c;controls();
  try{state=await request(base(),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});pendingMs=Math.max(0,pendingMs-c.elapsed_ms);pendingCommand=null;render();
    if(['save','split','merge'].includes(c.action)){dirty=false;localStorage.removeItem(draftKey());edit();}message('已保存；人工紀錄與模型提案分開保存。');
  }catch(e){message(`${e.message}。草稿保留；網路失敗可重按保存，409 請重新載入核對。`,true);if(e.message.startsWith('409'))pendingCommand=null;
  }finally{busy=false;controls();}
}
async function load(){if(busy)return;state=await request(base());render();$('export').href=base()+'/export';
  if(state.runs.find(r=>r.id===runId)?.preview_available)useCompatible();else useOriginal();
  const draft=localStorage.getItem(draftKey());edit();if(draft){const d=JSON.parse(draft);edit(d.point);pointId=d.pointId;proposalId=d.proposalId;for(const k of ['rally','complete','highlight','reason'])$(k).value=d.point[k];dirty=true;localStorage.setItem(draftKey(),draft);message('已恢復尚未保存的草稿；請核對後保存。');}else message('請選候選、人工紀錄，或查看未覆蓋區段。');controls();}
$('save').onclick=()=>command('save',{point:fields(),point_id:pointId,proposal_id:proposalId});
$('import').hidden=!job;$('import').onclick=async()=>{try{state=await request(base()+'/baseline',{method:'POST'});render();message('已匯入既有 baseline；來源版本未知時會明列 UNKNOWN。');}catch(e){message(e.message,true);}};
$('reject').onclick=()=>{ $('rally').value='no';$('highlight').value='unrated';persistDraft();$('save').click();};
$('new').onclick=()=>choose();$('runs').onchange=()=>{runId=$('runs').value;render();if(state.runs.find(r=>r.id===runId)?.preview_available)useCompatible();else useOriginal();};
$('compatible').onclick=useCompatible;$('original').onclick=useOriginal;
$('next').onclick=()=>{const reviewed=new Set(state.review.points.flatMap(p=>p.proposal_ids));const p=state.runs.find(r=>r.id===runId)?.proposals.find(p=>!reviewed.has(p.id));if(p)choose(p);else message('此批次已無未審核候選；請檢查候選未覆蓋區段。');};
$('reload').onclick=async()=>{if(busy)return;timerStart=null;pendingCommand=null;$('timer').textContent='開始計時';await load().catch(e=>message(e.message,true));};
$('markStart').onclick=()=>{$('start').value=sourceTime().toFixed(3);persistDraft();};
$('markEnd').onclick=()=>{$('end').value=sourceTime().toFixed(3);persistDraft();video.pause();};
$('preview').onclick=()=>{if(Number($('end').value)>mediaEnd||Number($('start').value)<mediaOffset){message('候選超出目前相容短片，請切回原片或選擇相應批次。',true);return;}if(!seekSource(Math.max(mediaOffset,Number($('start').value)-1.5)))return;video.dataset.stopAt=Math.min(mediaEnd,Number($('end').value)+1.5);video.play().catch(e=>message(e.message,true));};
video.ontimeupdate=()=>{if(video.dataset.stopAt&&sourceTime()>=Number(video.dataset.stopAt)){video.pause();delete video.dataset.stopAt;}};
$('play').onclick=()=>{delete video.dataset.stopAt;video.paused?video.play().catch(e=>message(e.message,true)):video.pause();};
$('back').onclick=()=>{video.currentTime=Math.max(0,video.currentTime-2);};$('forward').onclick=()=>{video.currentTime=Math.min(video.duration,video.currentTime+2);};
$('split').onclick=()=>command('split',{point_id:pointId,split_ms:Math.round(sourceTime()*1000)});
$('merge').onclick=()=>command('merge',{point_id:pointId,other_id:$('mergeTarget').value});
const coverage=()=>({start_ms:Math.round(Number($('coverStart').value)*1000),end_ms:Math.round(Number($('coverEnd').value)*1000)});
$('coverage').onclick=()=>command('coverage',{interval:coverage()});$('uncover').onclick=()=>command('coverage_remove',{interval:coverage()});
$('timer').onclick=()=>{if(timerStart===null){timerStart=performance.now();$('timer').textContent='暫停計時';}else{tick();timerStart=null;$('timer').textContent='開始計時';command('timer');}};
setInterval(()=>{if(timerStart!==null&&!busy)command('timer');},15000);
document.addEventListener('visibilitychange',()=>{if(document.hidden&&timerStart!==null){tick();timerStart=null;$('timer').textContent='開始計時';if(!busy)command('timer');}});
window.addEventListener('beforeunload',e=>{if(dirty||pendingMs>0||timerStart!==null){e.preventDefault();e.returnValue='';}});
for(const id of ['start','end','rally','complete','highlight','reason'])$(id).oninput=persistDraft;
document.addEventListener('keydown',e=>{if(e.isComposing||/INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName))return;if(e.key.toLowerCase()==='i')$('markStart').click();if(e.key.toLowerCase()==='o')$('markEnd').click();if(e.code==='Space'){e.preventDefault();$('play').click();}});
(async()=>{if(job){$('sources').hidden=true;sourceId=job;}else{const sources=await request('/api/review/sources');for(const s of sources)$('sources').add(new Option(s.name,s.id));sourceId=sources[0]?.id;if(!sourceId){message('尚無實驗來源，請先執行預標註指令。');return;}$('sources').onchange=async()=>{if(dirty||timerStart!==null){$('sources').value=sourceId;message('切換影片前請保存並暫停計時。',true);return;}sourceId=$('sources').value;pointId=null;proposalId=null;await load();};}await load();})().catch(e=>message(e.message,true));
