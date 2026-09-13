'use strict';
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const job = params.get('job');
let sourceId = '', state, pointId = null, proposalId = null, runId = '', busy = false;
let timerStart = null, pendingMs = 0, dirty = false, pendingCommand = null;
let mediaOffset = 0, mediaEnd = Infinity;
let mediaMode = 'original', pendingSeek = null, reviewAnchor = 0;
const video = $('video');
function sourceTime(){return Math.min(mediaEnd,video.currentTime+mediaOffset);}
function timeLabel(seconds){const n=Math.max(0,Math.floor(seconds||0));return String(Math.floor(n/60)).padStart(2,'0')+':'+String(n%60).padStart(2,'0');}
function seekSource(seconds){if(seconds<mediaOffset||seconds>mediaEnd){message('目前正在查看實驗短片，請按「整片播放版本」或「原始影片」查看這個位置。',true);return false;}if(video.readyState<1)pendingSeek=seconds;else video.currentTime=seconds-mediaOffset;return true;}
function setMedia(url,start,end,mode,label){video.pause();delete video.dataset.stopAt;mediaOffset=start;mediaEnd=end;mediaMode=mode;pendingSeek=start;video.src=HC.url(url);$('mediaMode').textContent=label;$('mediaProblem').hidden=true;updateClock();}
function useOriginal(){setMedia(job?`/api/jobs/${encodeURIComponent(job)}/source`:base()+'/source',0,state.source.duration_ms/1000,'original','整支原片');}
function useFull(){setMedia(base()+'/full-preview',0,state.source.duration_ms/1000,'full','整片播放 · 時間與原片一致');}
function useCompatible(){const run=state.runs.find(r=>r.id===runId);if(!run?.preview_available)return;setMedia(base()+'/preview/'+encodeURIComponent(runId),run.preview_range.start_ms/1000,run.preview_range.end_ms/1000,'segment','實驗短片 · 下方大字顯示原片時間');}
function updateClock(){if(!state)return;$('sourceClock').textContent=`${timeLabel(sourceTime())} / ${timeLabel(state.source.duration_ms/1000)}`;$('reviewToHere').textContent=`確認 ${timeLabel(reviewAnchor)}–${timeLabel(sourceTime())} 已檢查`;$('reviewToHere').disabled=busy||dirty||!!pendingCommand||sourceTime()<=reviewAnchor;}
function validBounds(){const p=fields();return $('start').value!==''&&$('end').value!==''&&p.start_ms>=0&&p.end_ms>p.start_ms&&p.end_ms<=state.source.duration_ms;}
const labels = {yes:'是',no:'否',uncertain:'不確定',unable:'無法判斷',omit:'不收錄',include:'可收錄',must:'必收錄',unrated:'未判斷'};
const base = () => HC.url(job ? `/api/jobs/${encodeURIComponent(job)}/rally-review` : `/api/review/${sourceId}`);
const draftKey = () => `hc-review-draft:${base()}`;
function message(text, error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
async function request(url, options={}){const r=await fetch(HC.url(url),options);if(!r.ok){let p=await r.json();throw new Error(`${r.status}: ${p.detail || 'Request failed'}`);}return r.json();}
function fields(){return {start_ms:Math.round(Number($('start').value)*1000),end_ms:Math.round(Number($('end').value)*1000),rally:$('rally').value,complete:$('complete').value,highlight:$('rally').value==='yes'?$('highlight').value:'unrated',reason:$('reason').value};}
function persistDraft(){dirty=true;localStorage.setItem(draftKey(),JSON.stringify({pointId,proposalId,point:fields(),emptyStart:$('start').value==='',emptyEnd:$('end').value===''}));controls();}
function edit(p=null, human=false){pointId=human?p.id:null;proposalId=p&&!human?p.id:null;
  const f=human?p:{start_ms:p?.start_ms??Math.round(sourceTime()*1000),end_ms:p?.end_ms??Math.round(sourceTime()*1000)+1000,rally:'uncertain',complete:'uncertain',highlight:'unrated',reason:''};
  $('start').value=p?f.start_ms/1000:'';$('end').value=p?f.end_ms/1000:'';
  for(const k of ['rally','complete','highlight','reason'])$(k).value=f[k];
  $('editing').textContent=human?'修改已保存的回合':p?'檢查模型提案':'記錄一個回合';
  $('editHint').textContent=p?'核對開始與結束，確認後保存。':'播放影片：發球前按 I，該分結束後按 O，再保存。';$('reject').hidden=!p||human;
  $('suggestion').textContent=human?'已保存的人工判斷；不會因重新推論而覆寫。':p?.blind?'盲審樣本：首次人工判斷前隱藏模型建議。':p?`模型建議（未校準）：回合 ${labels[p.rally]}；完整 ${labels[p.complete]}；精彩 ${labels[p.highlight]}\n${p.reason}`:'請先觀看片段，再作判斷。';
  $('split').disabled=!human;$('merge').disabled=!human;
  if(p){seekSource(Math.max(mediaOffset,p.start_ms/1000-1.5));$('coverStart').value=Math.max(mediaOffset,p.start_ms/1000-1.5);$('coverEnd').value=Math.min(mediaEnd,p.end_ms/1000+1.5);}
  dirty=false;localStorage.removeItem(draftKey());controls();
}
function choose(p,human=false){if(busy)return;if(dirty){message('這個回合還沒保存。請先保存，或按「放棄未保存修改」。',true);return;}edit(p,human);}
function list(id,items,action,label){$(id).replaceChildren();for(const p of items){const b=document.createElement('button');b.textContent=label(p);b.onclick=()=>action(p);$(id).append(b);}}
const span=p=>`${(p.start_ms/1000).toFixed(2)}–${(p.end_ms/1000).toFixed(2)}s`;
function render(){const r=state.review;const previous=runId; $('runs').replaceChildren();
  $('scope').textContent=`本次分析範圍：${(state.source.scope||[]).map(span).join('、')}。範圍外仍可人工補漏，但尚無本次模型分析。`;
  for(const run of state.runs){const o=new Option(`${run.model_id} · ${run.id.slice(0,8)}`,run.id);$('runs').add(o);}
  runId=state.runs.some(r=>r.id===previous)?previous:state.runs.at(-1)?.id||'';$('runs').value=runId;
  const currentRun=state.runs.find(r=>r.id===runId);if(currentRun)$('scope').textContent+=` 此批次 ${currentRun.proposals.length} 個候選、${currentRun.output_error_count||0} 個視窗輸出失敗；空清單不代表沒有回合。`;
  $('compatible').hidden=!state.runs.find(r=>r.id===runId)?.preview_available;
  $('full').hidden=!state.full_preview_available;$('prepareFull').hidden=!!state.full_preview_available;
  const reviewed=new Set(r.points.flatMap(p=>p.proposal_ids));
  list('proposals',state.runs.find(r=>r.id===runId)?.proposals||[],p=>{const human=r.points.find(x=>x.proposal_ids.includes(p.id));choose(human||p,!!human);},p=>`${reviewed.has(p.id)?'已審核':'尚未審核'} · ${span(p)}${p.blind?' · 盲審':''}`);
  list('points',r.points,p=>choose(p,true),p=>`${span(p)} · ${labels[p.rally]} · ${labels[p.highlight]}`);
  $('pointCount').textContent=`（${r.points.length} 筆）`;$('emptyPoints').hidden=r.points.length>0;
  const checked=r.coverage.reduce((sum,p)=>sum+p.end_ms-p.start_ms,0),duration=state.source.duration_ms;
  const unresolved=r.points.filter(p=>['uncertain','unable'].includes(p.rally)||p.rally==='yes'&&['uncertain','unable'].includes(p.complete)).length;
  $('reviewProgress').value=checked/duration*100;$('progressText').textContent=`已檢查 ${timeLabel(checked/1000)} / ${timeLabel(duration/1000)}（${Math.round(checked/duration*100)}%）。${unresolved?`仍有 ${unresolved} 筆回合／完整性待確認。`:''}`;
  const resumeAt=r.unreviewed[0]?.start_ms/1000||0;$('resume').textContent=r.unreviewed.length?(resumeAt?`從 ${timeLabel(resumeAt)} 繼續審核`:'從頭開始審核'):'全片已檢查 · 再看一次';
  $('mergeTarget').replaceChildren(new Option('選擇另一筆人工紀錄',''));
  for(const p of r.points)$('mergeTarget').add(new Option(span(p),p.id));
  const seek=p=>{if(seekSource(p.start_ms/1000)){$('coverStart').value=p.start_ms/1000;$('coverEnd').value=Math.min(mediaEnd,p.end_ms/1000);}};
  list('gaps',r.unreviewed,seek,span);list('proposalGaps',r.proposal_gaps,seek,span);
  $('stats').textContent=`已審核 ${r.coverage.map(span).join('、')||'尚無'} · 計時 ${(r.active_ms/1000).toFixed(1)} 秒 · 修改 ${r.modification_count} 次（尚無省時結論）`;
}
function tick(){if(timerStart!==null){pendingMs+=performance.now()-timerStart;timerStart=performance.now();}}
function controls(){for(const e of document.querySelectorAll('input,select,textarea,button'))e.disabled=busy||!!pendingCommand&&!['save','reload'].includes(e.id);$('split').disabled=busy||!!pendingCommand||!pointId;$('merge').disabled=busy||!!pendingCommand||!pointId;if(state){$('save').disabled=busy||(!pendingCommand&&!validBounds());$('saveContinue').disabled=busy||!!pendingCommand||!validBounds();$('preview').disabled=busy||!!pendingCommand||!validBounds();updateClock();}}
async function command(action, extra={}){if(busy)return;busy=true;tick();
  const elapsed=Math.min(60000,Math.round(pendingMs));
  const c=pendingCommand||{action,revision:state.review.revision,request_id:crypto.randomUUID(),elapsed_ms:elapsed,...extra};
  pendingCommand=c;controls();
  try{state=await request(base(),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});pendingMs=Math.max(0,pendingMs-c.elapsed_ms);pendingCommand=null;render();
    if(['save','split','merge'].includes(c.action)){dirty=false;localStorage.removeItem(draftKey());edit();}message('已保存。可以繼續標下一個回合；檢查完一段後，再確認檢查進度。');return true;
  }catch(e){message(`${e.message}。草稿保留；網路失敗可重按保存，409 請重新載入核對。`,true);if(e.message.startsWith('409'))pendingCommand=null;
  }finally{busy=false;controls();}
}
async function load(){if(busy)return;state=await request(base());render();$('export').href=base()+'/export';
  reviewAnchor=state.review.unreviewed[0]?.start_ms/1000||0;
  if(state.full_preview_available)useFull();else useOriginal();seekSource(reviewAnchor);
  const draft=localStorage.getItem(draftKey());edit();if(draft){const d=JSON.parse(draft);edit(d.point);pointId=d.pointId;proposalId=d.proposalId;for(const k of ['rally','complete','highlight','reason'])$(k).value=d.point[k];if(d.emptyStart)$('start').value='';if(d.emptyEnd)$('end').value='';dirty=true;localStorage.setItem(draftKey(),draft);message('已恢復未保存的回合。請保存，或按「放棄未保存修改」。');}else message('先按「從頭開始審核」或「繼續審核」，在回合開始／結束時按 I／O。');controls();}
$('save').onclick=()=>command('save',{point:fields(),point_id:pointId,proposal_id:proposalId});
$('saveContinue').onclick=async()=>{const end=fields().end_ms/1000;if(await command('save',{point:fields(),point_id:pointId,proposal_id:proposalId})){delete video.dataset.stopAt;if(seekSource(end))video.play().catch(e=>message(e.message,true));}};
$('discard').onclick=()=>{edit();message('已放棄未保存修改；已保存的標註保持不變。');};
$('resume').onclick=()=>{if(dirty){message('請先保存或放棄目前回合的修改，再繼續審核。',true);return;}reviewAnchor=state.review.unreviewed[0]?.start_ms/1000||0;if(mediaMode==='segment'){state.full_preview_available?useFull():useOriginal();}delete video.dataset.stopAt;if(seekSource(reviewAnchor))video.play().catch(e=>message(e.message,true));};
$('reviewToHere').onclick=async()=>{const end=Math.min(state.source.duration_ms,Math.round(sourceTime()*1000));if(dirty||end<=Math.round(reviewAnchor*1000))return;if(await command('coverage',{interval:{start_ms:Math.round(reviewAnchor*1000),end_ms:end}})){reviewAnchor=end/1000;updateClock();message('檢查進度已保存；下次會從尚未檢查的位置繼續。');}};
$('prepareFull').onclick=async()=>{if(busy)return;const position=sourceTime();busy=true;controls();message('正在準備整片播放版本，原片與標註會保留。請稍候…');try{state=await request(base()+'/full-preview',{method:'POST'});render();useFull();seekSource(Math.min(position,mediaEnd));message('整片已可播放，請按「從頭開始審核」或「繼續審核」。');}catch(e){message(`整片準備失敗：${e.message}`,true);}finally{busy=false;controls();}};
$('import').hidden=!job;$('import').onclick=async()=>{try{state=await request(base()+'/baseline',{method:'POST'});render();message('已匯入既有 baseline；來源版本未知時會明列 UNKNOWN。');}catch(e){message(e.message,true);}};
$('reject').onclick=()=>{ $('rally').value='no';$('highlight').value='unrated';persistDraft();$('save').click();};
$('new').onclick=()=>choose();$('runs').onchange=()=>{runId=$('runs').value;render();};
$('compatible').onclick=useCompatible;$('original').onclick=useOriginal;$('full').onclick=useFull;
$('next').onclick=()=>{const reviewed=new Set(state.review.points.flatMap(p=>p.proposal_ids));const p=state.runs.find(r=>r.id===runId)?.proposals.find(p=>!reviewed.has(p.id));if(p)choose(p);else message('此批次已無未審核候選；請檢查候選未覆蓋區段。');};
$('reload').onclick=async()=>{if(busy)return;timerStart=null;pendingCommand=null;$('timer').textContent='開始計時';await load().catch(e=>message(e.message,true));};
$('markStart').onclick=()=>{$('start').value=sourceTime().toFixed(3);if(Number($('end').value)<=sourceTime())$('end').value='';if(!pointId&&!proposalId)$('rally').value='yes';persistDraft();message('起點已設好。繼續播放，該分結束時按 O。');};
$('markEnd').onclick=()=>{if($('start').value===''){message('請先在回合開始前按 I 設起點。',true);return;}$('end').value=sourceTime().toFixed(3);persistDraft();video.pause();message('終點已設好。確認是否完整，再按「保存回合，繼續播放」。');};
$('preview').onclick=()=>{if(Number($('end').value)>mediaEnd||Number($('start').value)<mediaOffset){message('候選超出目前相容短片，請切回原片或選擇相應批次。',true);return;}if(!seekSource(Math.max(mediaOffset,Number($('start').value)-1.5)))return;video.dataset.stopAt=Math.min(mediaEnd,Number($('end').value)+1.5);video.play().catch(e=>message(e.message,true));};
video.ontimeupdate=()=>{updateClock();if(video.dataset.stopAt&&sourceTime()>=Number(video.dataset.stopAt)){video.pause();delete video.dataset.stopAt;}};
video.onloadedmetadata=()=>{if(pendingSeek!==null){video.currentTime=Math.max(0,pendingSeek-mediaOffset);pendingSeek=null;}updateClock();};
function mediaFailure(){if(mediaMode==='original'){$('mediaProblem').textContent='此瀏覽器可能無法播放原片影像。請按「準備整片播放」，建立可從頭看到尾的播放版本。';}else $('mediaProblem').textContent='影片載入失敗，請重新載入頁面後重試。';$('mediaProblem').hidden=false;}
video.onerror=mediaFailure;video.onloadeddata=()=>{if(video.videoWidth===0||video.videoHeight===0)mediaFailure();else $('mediaProblem').hidden=true;};
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
document.addEventListener('keydown',e=>{if(e.isComposing||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;if(e.key.toLowerCase()==='i')$('markStart').click();if(e.key.toLowerCase()==='o')$('markEnd').click();if(e.code==='Space'&&e.target.tagName!=='BUTTON'){e.preventDefault();$('play').click();}});
(async()=>{if(job){$('sources').hidden=true;sourceId=job;}else{const sources=await request('/api/review/sources');sources.forEach((s,i)=>{const o=new Option(`影片 ${i+1} · ${timeLabel(s.duration_ms/1000)}`,s.id);o.title=s.name;$('sources').add(o);});sourceId=sources[0]?.id;if(!sourceId){message('尚無影片可審核，請先載入影片。');return;}$('sources').onchange=async()=>{if(dirty||timerStart!==null||pendingMs>0){$('sources').value=sourceId;message('切換影片前請保存並暫停計時。',true);return;}sourceId=$('sources').value;pointId=null;proposalId=null;await load();};}await load();})().catch(e=>message(e.message,true));
