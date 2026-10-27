'use strict';
const $ = s => document.querySelector(s);
const player = $('#player');
let pack, saved, queue = [], saving = false, pendingKey;
function el(tag, text, attrs={}) {const e=document.createElement(tag);e.textContent=text;Object.assign(e,attrs);return e;}
async function api(url, options={}) {
  let r=await fetch(url,options);
  if(r.status===403){await fetch('/');r=await fetch(url,options);}
  if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();
}
function latest(token,field){return [...saved.records,...queue].filter(r=>r.token===token&&r.field===field).at(-1)?.value || '';}
function persistPending(){try{localStorage.setItem(pendingKey,JSON.stringify(queue));}catch(e){/* server save remains authoritative */}}
function record(token,field,value){
  queue.push({id:crypto.randomUUID(),token,field,value:value||null,presentation_sequence:$('#sequence').value});
  persistPending();flush();
}
async function flush(){
  if(saving)return;saving=true;$('#retry').hidden=true;$('#export').hidden=true;
  try{
    while(queue.length){$('#save-status').textContent='保存中…';
      saved=await api('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(queue[0])});queue.shift();persistPending();}
    $('#save-status').textContent=`已保存 · ${saved.revision} 筆紀錄`;
  }catch(e){$('#save-status').textContent='保存失敗，請保留此頁並按重試。';$('#retry').hidden=false;}
  finally{saving=false;$('#export').hidden=queue.length>0;updateGate();}
}
$('#retry').onclick=flush;
function updateGate(){
  const all=pack.selected_clips.every((_,i)=>latest(`point-${i+1}`,'complete')==='完整');
  $('#gate').textContent=all?'六段已評為完整，可進入合輯比較。':'合輯比較尚未開啟：請先完成六段採用切點的完整性核對。';
  for(const o of $('#section-nav').options)if(o.value==='reels')o.disabled=!all || queue.length>0;
  if(!all && $('#section-nav').value==='reels'){$('#section-nav').value='point:0';render();}
}
async function play(token,label){
  player.pause();$('#playing').textContent=label;$('#play-status').textContent='載入中…';
  player.src=`/media/${encodeURIComponent(token)}`;player.load();
  for(const id of ['replay','pause','back','forward'])$('#'+id).disabled=false;
  try{await player.play();}catch(e){$('#play-status').textContent='播放未啟動，請按影片控制列播放或從頭重播。';}
}
$('#replay').onclick=()=>{player.currentTime=0;player.play().catch(()=>{});};
$('#pause').onclick=()=>player.paused?player.play().catch(()=>{}):player.pause();
$('#back').onclick=()=>{player.currentTime=Math.max(0,player.currentTime-1);};
$('#forward').onclick=()=>{if(Number.isFinite(player.duration))player.currentTime=Math.min(player.duration,player.currentTime+1);};
player.ontimeupdate=()=>{$('#play-status').textContent=`${player.currentTime.toFixed(1)} / ${Number.isFinite(player.duration)?player.duration.toFixed(1):'…'} 秒${player.ended?' · 播放結束':''}`;};
player.onerror=()=>{$('#play-status').textContent='影片播放失敗，請保留頁面回報；評分尚未因此確認。';};
function rating(card,token,field,label,values){
  const l=el('label',label);const s=el('select','',{'ariaLabel':`${token} ${label}`});s.append(el('option','待確認',{value:''}));
  for(const v of values)s.append(el('option',v,{value:v}));s.value=latest(token,field);
  s.onchange=()=>{record(token,field,s.value);updateGate();};l.append(s);card.append(l);
}
function card(token,title,ratings=true){
  const c=el('section','',{className:'card'});c.append(el('h3',title));const b=el('button',`播放 ${title}`);b.onclick=()=>play(token,title);c.append(b);
  if(ratings){rating(c,token,'complete','完整性',['完整','不完整','無法判定']);rating(c,token,'rhythm','觀看節奏',['太急','合適','太拖']);
    const label=el('label','備註');const n=el('textarea','',{value:latest(token,'note'),maxLength:2000,ariaLabel:`${token} 備註`});
    n.oninput=()=>record(token,'note',n.value);label.append(n);c.append(label);}
  $('#cards').append(c);return c;
}
function context(i){const row=pack.contexts[i];const b=el('button','播放原片前後文');b.onclick=()=>play(`context-${i+1}`,'原片前後文');$('#context').append(b);}
function render(){
  player.pause();player.removeAttribute('src');player.load();$('#playing').textContent='請選擇下方版本播放';$('#play-status').textContent='';
  for(const id of ['replay','pause','back','forward'])$('#'+id).disabled=true;
  $('#cards').replaceChildren();$('#context').replaceChildren();$('#source-info').replaceChildren();$('#disclosure').open=false;
  const [type,index]=$('#section-nav').value.split(':');const i=Number(index);const reverse=$('#sequence').value==='1';
  if(type==='pair'){
    const p=pack.pairs[i];$('#section-title').textContent=`比較 ${i+1} · ${p.stage_label}`;
    const rows=reverse?[...p.media].reverse():p.media;rows.forEach(m=>card(m.token,`版本 ${m.token}`));
    for(const m of rows)$('#source-info').append(el('p',`${m.token}：原片 ${m.requested_source_range.join('–')} 秒`));
    $('#source-info').append(el('p',p.difference));
  }else if(type==='point'){
    $('#section-title').textContent=`第 ${i+1} 段採用切點核對`;card(`point-${i+1}`,`第 ${i+1} 段`);context(i);
    $('#source-info').append(el('p',`原片 ${pack.selected_clips[i].requested_source_range.join('–')} 秒；事件範圍仍屬助手觀察，待真人確認。`));
  }else{
    $('#section-title').textContent='合輯比較：分開評轉場與順序';
    const rows=reverse?[...pack.reels].reverse():pack.reels;rows.forEach(m=>card(m.token,`合輯 ${m.token}`,false));
    const c=el('section','',{className:'card'});rating(c,'reel','transition','轉場（K／M）',['K','M','無差異','無法判定']);rating(c,'reel','order','順序（K／N）',['K','N','無差異','無法判定']);$('#cards').append(c);
    $('#source-info').append(el('p','固定六段與切點；K／M 只比較轉場，K／N 只比較順序。'));
  }
}
$('#sequence').onchange=()=>{record('review','presentation_order',$('#sequence').value);render();};
$('#section-nav').onchange=render;
window.addEventListener('beforeunload',e=>{if(queue.length){e.preventDefault();e.returnValue='';}});
(async()=>{
  try{
    [pack,saved]=await Promise.all([api('/api/pack'),api('/api/feedback')]);$('#fixture').hidden=!pack.fixture;
    pendingKey=`rally-edit-v2:${saved.plan_sha256}:${saved.actor}`;
    try{const q=JSON.parse(localStorage.getItem(pendingKey)||'[]');if(Array.isArray(q))queue=q;}catch(e){}
    $('#sequence').value=latest('review','presentation_order')||'0';
    pack.pairs.forEach((p,i)=>$('#section-nav').append(el('option',`比較 ${i+1} · ${p.stage_label}`,{value:`pair:${i}`})));
    pack.selected_clips.forEach((p,i)=>$('#section-nav').append(el('option',`第 ${i+1} 段採用切點核對`,{value:`point:${i}`})));
    $('#section-nav').append(el('option','合輯比較',{value:'reels'}));$('#section-nav').disabled=false;
    $('#save-status').textContent=`已讀取 · ${saved.revision} 筆紀錄`;render();updateGate();if(queue.length)flush();
  }catch(e){$('#save-status').textContent='讀取失敗，請重新整理。';$('#export').hidden=true;}
})();
