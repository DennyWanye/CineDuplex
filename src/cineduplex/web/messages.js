'use strict';
// Voice messages reuse the resident call protocol and its reviewed output.
// A draft stays local until pointer-up; it never enters the network queue.
if(document.getElementById('message-room')) (()=>{
 const ui=id=>document.getElementById(id), button=ui('hold-talk');
 let mode='messages',recording=null,press=null,busy=false,awaiting=false,connecting=false;
 let silenceTimer=null,idleTimer=null,replyTimer=null,viewTimer=null,playback=null;
 let latestUser=null,captionOn=false,records=[],answers=new Map(),bytesHeld=0;
 let generation=0,flushing=false;
 const limitSeconds=60, maxBytes=24*1024*1024;
 function notice(message=''){ui('chat-notice').textContent=message;ui('chat-notice').hidden=!message;}
 function hint(text){ui('record-hint').textContent=text;}
 function refresh(){
  ui('reset-message-session').hidden=mode!=='messages'||!call;
  ui('reset-message-session').disabled=busy||!!recording||!!press||connecting||stopping;
  if(mode!=='messages')return;
  button.disabled=busy||awaiting||stopping||(!call&&!ready);
  ui('enter-call').disabled=busy||!!recording||!!press||connecting||stopping;
  ui('cancel-reply').hidden=!awaiting;
  const locked=!!call||!!press||!!recording||busy||connecting||stopping;
  for(const id of ['voice','language','mic-device'])ui(id).disabled=locked;
  ui('settings-note').textContent=locked?'当前录音或对话结束后可更换声音和麦克风。':'设置用于下一次对话。通话中也可以直接说“请改用中文”。';
  if(!press&&!recording&&!busy){
   ui('hold-label').textContent=awaiting?'正在等待小橙回答…':'按住说话';
   hint(awaiting?'小橙正在准备回答':call||ready?'松开发送，上滑取消':ui('service-message').textContent);
  }
 }
 function wav(chunks,rate){
  const n=chunks.reduce((sum,b)=>sum+b.byteLength,0),buffer=new ArrayBuffer(44+n),v=new DataView(buffer);
  const text=(offset,s)=>{for(let i=0;i<s.length;i++)v.setUint8(offset+i,s.charCodeAt(i));};
  text(0,'RIFF');v.setUint32(4,36+n,true);text(8,'WAVE');text(12,'fmt ');v.setUint32(16,16,true);
  v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);
  v.setUint16(32,2,true);v.setUint16(34,16,true);text(36,'data');v.setUint32(40,n,true);
  let offset=44;for(const chunk of chunks){new Uint8Array(buffer,offset,chunk.byteLength).set(new Uint8Array(chunk));offset+=chunk.byteLength;}
  return new Blob([buffer],{type:'audio/wav'});
 }
 function scroll(){ui('message-list').scrollTop=ui('message-list').scrollHeight;}
 function row(who){
  ui('chat-empty').hidden=true;
  const el=document.createElement('article');el.className='message-row '+who;
  const avatar=document.createElement(who==='user'?'span':'img');avatar.className='message-avatar';
  if(who==='user')avatar.textContent='我';else{avatar.src=entryUrl('xiaocheng.png');avatar.alt='小橙';}
  const content=document.createElement('div');content.className='message-content';
  const player=document.createElement('button');player.className='voice-bubble';player.disabled=true;player.setAttribute('aria-pressed','false');
  const symbol=document.createElement('span');symbol.className='play-symbol';symbol.textContent='▶';
  const wave=document.createElement('span');wave.className='bubble-wave';wave.setAttribute('aria-hidden','true');
  const duration=document.createElement('span');duration.className='bubble-duration';duration.textContent='…';
  player.append(symbol,wave,duration);
  const status=document.createElement('p');status.className='message-status';
  const caption=document.createElement('p');caption.className='message-caption';caption.hidden=true;
  content.append(player,status,caption);el.append(avatar,content);ui('message-list').append(el);
  const item={el,player,symbol,wave,duration,status,caption,who,chunks:[],rate:who==='user'?16000:24000,bytes:0,peaks:[],url:null,complete:false};
  player.onclick=()=>replayMessage(item);records.push(item);scroll();return item;
 }
 function appendAudio(item,pcm){
  item.chunks.push(pcm);item.bytes+=pcm.byteLength;bytesHeld+=pcm.byteLength;
  item.duration.textContent=(item.bytes/item.rate/2).toFixed(1)+'″';
  // Keep a bounded envelope of every received packet, including earlier speech.
  const data=new DataView(pcm);let packetPeak=0;
  for(let i=0;i<data.byteLength;i+=2)packetPeak=Math.max(packetPeak,Math.abs(data.getInt16(i,true))/32768);
  item.peaks.push(packetPeak);
  const bars=Math.min(32,item.peaks.length);
  item.wave.replaceChildren();for(let k=0;k<bars;k++){
   let peak=0;const from=Math.floor(k*item.peaks.length/bars),to=Math.floor((k+1)*item.peaks.length/bars);
   for(let i=from;i<to;i++)peak=Math.max(peak,item.peaks[i]);
   const bar=document.createElement('i');bar.style.height=(4+Math.min(1,peak*3)*22)+'px';item.wave.append(bar);
  }
  item.player.setAttribute('aria-label',(item.who==='user'?'我的':'小橙的')+'语音，'+(item.bytes/item.rate/2).toFixed(1)+'秒');
 }
 function playable(item){
  if(!item.bytes||item.failed)return;
  if(item.url)URL.revokeObjectURL(item.url);
  item.url=URL.createObjectURL(wav(item.chunks,item.rate));item.player.disabled=false;item.complete=true;
  while(records.length>60||bytesHeld>maxBytes){
   const old=records[0];if(old===item||!old.complete)break;
   if(playback?.item===old)stopReplay();records.shift();bytesHeld-=old.bytes;
   if(old.url)URL.revokeObjectURL(old.url);old.el.remove();
   for(const [key,value] of answers)if(value===old)answers.delete(key);
  }
 }
 function stopReplay(){
  if(!playback)return;const {audio,item}=playback;playback=null;audio.pause();audio.removeAttribute('src');audio.load();
  item.player.setAttribute('aria-pressed','false');item.symbol.textContent='▶';item.status.textContent=item.who==='user'?(item.deliveryStatus||'保存在本页'):'点击重听';
 }
 async function replayMessage(item){
  if(recording||press||awaiting||busy){notice('请先完成当前录音或回答，再重听语音。');return;}
  if(playback?.item===item){stopReplay();return;}stopReplay();clearPlayback();
  const audio=new Audio(item.url);audio.preload='auto';audio.setAttribute('playsinline','');
  playback={audio,item};audio.onended=()=>{if(playback?.audio===audio)stopReplay();};
  try{await selectPlaybackDevice(audio);await audio.play();if(playback?.audio!==audio)return;
   item.symbol.textContent='Ⅱ';item.player.setAttribute('aria-pressed','true');item.status.textContent='正在播放';
  }catch{stopReplay();notice('播放未开启，请再次点击语音。');}
 }
 function endWaiting(message){
  awaiting=false;clearTimeout(replyTimer);if(latestUser&&message)latestUser.status.textContent=message;
  armIdle();refresh();
 }
 function armIdle(){clearTimeout(idleTimer);if(call)idleTimer=setTimeout(async()=>{
  if(mode==='messages'&&!busy&&!recording&&!press&&!awaiting){await stop(true,'消息会话已休息');notice('已暂停连接。下一条语音将开始新的对话，当前消息仍可重听。');}
 },60000);}
 function keepSession(){
  if(silenceTimer)return;
  silenceTimer=setInterval(()=>{if(mode==='messages'&&call&&!busy&&!stopping)enqueue(new ArrayBuffer(12800));},400);
 }
 function onEvent(e){
  if(mode!=='messages')return;
  if(e.type==='safety_outcome'&&e.outcome==='application_stop'&&
     ['output_audio_text_mismatch','output_audio_semantics_unrecognized'].includes(e.reason)){
   const item=answers.get(e.event_id);
   if(item){
    if(playback?.item===item)stopReplay();clearPlayback();
    item.failed=true;item.complete=true;bytesHeld-=item.bytes;
    item.bytes=0;item.chunks=[];item.player.disabled=true;
    if(item.url)URL.revokeObjectURL(item.url);item.url=null;
    item.duration.textContent='—';item.status.textContent='声音异常，回答已停止';
    item.player.setAttribute('aria-label','声音异常，无法重听这段回答');
   }
  }
  if((e.type==='audio'||e.type==='text_delta'||e.type==='event_end')&&e.epoch>=epoch){
   let item=answers.get(e.event_id);
   if(!item){item=row('assistant');item.id=e.event_id;item.epoch=e.epoch;answers.set(e.event_id,item);}
   if(e.type==='audio'){
    const raw=atob(e.pcm),data=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)data[i]=raw.charCodeAt(i);
    if(e.sample_rate!==24000||data.length%2)return;
    appendAudio(item,data.buffer);item.status.textContent='正在接收回答';
   }else{item.caption.textContent=e.snapshot||e.text||'';item.caption.hidden=!captionOn||!item.caption.textContent;}
  }
  if(e.type==='audio_end'&&e.epoch>=epoch){const item=answers.get(e.event_id);if(item){playable(item);item.status.textContent='回答已收到';}endWaiting();}
  if(e.type==='stop_audio'||e.type==='guardrail_cancelled'){
   for(const item of answers.values())if(!item.complete&&item.bytes){playable(item);item.status.textContent='回答已中止';}
  }
  if(e.type==='guardrail_pending')hint('小橙正在准备回答');
  if(['guardrail_block','error','response_warning','session_warning'].includes(e.type))notice(e.message);
 }
 function onStop(reason){
  clearInterval(silenceTimer);silenceTimer=null;clearTimeout(idleTimer);clearTimeout(replyTimer);
  if(mode==='messages'){
   if(awaiting){if(latestUser){latestUser.deliveryStatus='已发送 · 回答未完成';latestUser.status.textContent=latestUser.deliveryStatus;}notice(reason);}
   for(const item of answers.values())if(!item.complete){if(item.bytes)playable(item);item.status.textContent=item.bytes?'回答未完成 · 已收到部分':'回答未完成';}
   awaiting=false;refresh();
  }
 }
 async function releaseRecorder(rec){
  if(!rec)return;clearTimeout(rec.limit);rec.stream?.getTracks().forEach(t=>t.stop());
  rec.source?.disconnect();rec.node?.disconnect();if(rec.ctx)await rec.ctx.close().catch(()=>{});
 }
 async function beginPress(y,pointerId){
  if(mode!=='messages'||press||recording||busy||awaiting||stopping||(!ready&&!call))return;
  notice();stopReplay();clearPlayback();clearTimeout(idleTimer);
  const token=++generation;press={token,y,pointerId,cancel:false};
  ui('hold-label').textContent='等待麦克风权限…';hint('首次使用，请允许麦克风');refresh();
  let rec={chunks:[],samples:0,energy:0,ctx:null,stream:null,token};
  try{
   if(!isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('请使用受信任的 HTTPS 浏览器页面。');
   const selected=ui('mic-device').value;
   rec.stream=await navigator.mediaDevices.getUserMedia({audio:{...(selected?{deviceId:{exact:selected}}:{}),channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
   if(press?.token!==token){await releaseRecorder(rec);return;}
   rec.ctx=new AudioContext();await rec.ctx.resume();await rec.ctx.audioWorklet.addModule(entryUrl('capture.worklet.js'));
   if(press?.token!==token){await releaseRecorder(rec);return;}
   rec.node=new AudioWorkletNode(rec.ctx,'continuous-capture');
   rec.node.port.onmessage=e=>{
    if(e.data.flushed){rec.flushed?.();return;}
    if(recording!==rec)return;
    const valid=e.data.validSamples||6400;rec.chunks.push(e.data.pcm);rec.samples+=valid;rec.energy+=e.data.rms**2*valid;
    if(press&&!press.cancel){ui('hold-label').textContent='正在录音 '+(rec.samples/16000).toFixed(0)+'″';hint('松开发送，上滑取消');}
   };
   rec.source=rec.ctx.createMediaStreamSource(rec.stream);rec.source.connect(rec.node);rec.node.connect(rec.ctx.destination);
   recording=rec;button.dataset.recording='true';ui('hold-label').textContent='正在录音…';hint('松开发送，上滑取消');
   rec.stream.getAudioTracks()[0].onended=()=>finishPress(true);
   rec.limit=setTimeout(()=>finishPress(false),limitSeconds*1000);
  }catch(e){if(press?.token===token){press=null;notice(microphoneError(e));}await releaseRecorder(rec);resetButton();}
 }
 function resetButton(){button.dataset.recording='false';button.dataset.cancel='false';refresh();}
 async function finishPress(cancel=false){
  if(flushing)return;
  const held=press;press=null;const rec=recording;
  if(!rec){++generation;resetButton();return;}
  flushing=true;busy=true;resetButton();
  try{
   // Drain the final fractional packet before disconnecting the worklet.
   await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>reject(Error('录音收尾失败，请重新录制。')),1500);
    rec.flushed=()=>{clearTimeout(timeout);resolve();};rec.node.port.postMessage({type:'flush'});
   });
   recording=null;rec.stream.getAudioTracks()[0].onended=null;await releaseRecorder(rec);
   if(cancel||held?.cancel||mode!=='messages'){notice('录音已取消，未发送。');return;}
   if(rec.samples<6400){notice('录音太短，请按住说完一句话再松开。');return;}
   if(Math.sqrt(rec.energy/rec.samples)<.002){notice('没有收到清晰声音，请检查麦克风后重试。');return;}
   await sendRecording(rec);
  }catch(e){recording=null;await releaseRecorder(rec);notice(e.message);if(call)await stop();}
  finally{flushing=false;busy=false;resetButton();armIdle();}
 }
 async function sendRecording(rec){
  latestUser=row('user');const item=latestUser;for(const pcm of rec.chunks)appendAudio(item,pcm);
  playable(item);item.deliveryStatus='正在发送…';item.status.textContent=item.deliveryStatus;ui('hold-label').textContent='正在发送…';hint('正在发送语音');
  try{
   if(!call){connecting=true;await connect();connecting=false;answers.clear();}
   keepSession();awaiting=true;const id=call;
   const begin=performance.now();
   for(let i=0;i<rec.chunks.length;i++){
    if(call!==id)throw Error('连接已结束，语音未完整发送。');
    enqueue(rec.chunks[i]);await new Promise(r=>setTimeout(r,Math.max(0,begin+(i+1)*400-performance.now())));
   }
   while(sendQueue.length&&call===id)await new Promise(r=>setTimeout(r,50));
   if(call!==id)throw Error('连接已结束，请重新发送语音。');
   item.deliveryStatus='已发送';item.status.textContent=item.deliveryStatus;
   if(awaiting)replyTimer=setTimeout(async()=>{notice('本次回答等待超时，请再试一次。');await stop(true,'回答超时');},120000);
  }catch(e){connecting=false;item.deliveryStatus='发送未完成，请重新录制';item.status.textContent=item.deliveryStatus;awaiting=false;if(call||context)await stop();throw e;}
 }
 async function switchMode(next){
  if(busy||recording||press||connecting||stopping)return;
  if(next===mode)return;stopReplay();
  if(call||permissionPending)await stop(true,next==='call'?'已切换到实时通话':'已返回语音消息');
  mode=next;document.body.dataset.mode=mode;ui('back-messages').hidden=mode!=='call';notice();
  if(mode==='call')ui('start').focus();else{await status();refresh();button.focus();}
 }
 button.addEventListener('pointerdown',e=>{if(e.button!==0)return;e.preventDefault();button.setPointerCapture(e.pointerId);beginPress(e.clientY,e.pointerId);});
 button.addEventListener('pointermove',e=>{if(!press||press.pointerId!==e.pointerId)return;press.cancel=press.y-e.clientY>65;button.dataset.cancel=String(press.cancel);
  if(press.cancel){ui('hold-label').textContent='松开取消';hint('这条录音不会发送');}});
 button.addEventListener('pointerup',e=>{if(press?.pointerId===e.pointerId)finishPress();});
 button.addEventListener('pointercancel',()=>finishPress(true));
 button.addEventListener('lostpointercapture',()=>{if(press)finishPress(true);});
 button.addEventListener('contextmenu',e=>e.preventDefault());
 button.addEventListener('keydown',e=>{if([' ','Enter'].includes(e.key)){e.preventDefault();if(!e.repeat)beginPress(0,'keyboard');}else if(e.key==='Escape')finishPress(true);});
 button.addEventListener('keyup',e=>{if([' ','Enter'].includes(e.key)){e.preventDefault();finishPress();}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden&&(press||recording))finishPress(true);});
 window.addEventListener('blur',()=>{if(press||recording)finishPress(true);});
 ui('enter-call').onclick=()=>switchMode('call');ui('back-messages').onclick=()=>switchMode('messages');
 ui('cancel-reply').onclick=async()=>{stopReplay();await stop(true,'已停止回答');notice('已停止回答。下一条语音将开始新的对话。');};
 ui('reset-message-session').onclick=async()=>{await stop(true,'已结束当前对话');notice('下一条语音将开始新的对话，已有消息仍可重听。');refresh();};
 ui('chat-settings').onclick=()=>ui('settings-dialog').showModal();ui('message-privacy').onclick=()=>ui('privacy-dialog').showModal();
 ui('message-captions').onclick=()=>{captionOn=!captionOn;ui('message-captions').textContent=captionOn?'字幕：开':'字幕：关';ui('message-captions').setAttribute('aria-pressed',String(captionOn));for(const item of records)item.caption.hidden=!captionOn||!item.caption.textContent;};
 viewTimer=setInterval(()=>{
  refresh();if(mode!=='messages')return;
  for(const item of answers.values())if(item.complete&&item.bytes&&playback?.item!==item){
   const active=[...sources].some(s=>!s.cineduplexCancelled&&s.cineduplexMessageId===item.id);
   if(active)item.status.textContent='正在播放';else if(item.status.textContent==='正在播放')item.status.textContent='点击重听';
  }
 },300);
 window.addEventListener('pagehide',()=>{press=null;generation++;releaseRecorder(recording);recording=null;clearInterval(viewTimer);clearInterval(silenceTimer);clearTimeout(idleTimer);clearTimeout(replyTimer);stopReplay();for(const item of records)if(item.url)URL.revokeObjectURL(item.url);});
 window.voiceMessages={onEvent,onStop};refresh();
})();
