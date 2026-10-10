'use strict';
const $=id=>document.getElementById(id);
const entryBase=new URL('.',document.currentScript.src).pathname;
const entryUrl=path=>entryBase+path.replace(/^\//,'');
let ready=false,call=null,context=null,stream=null,capture=null,micSource=null;
let sendQueue=[],sending=false,audioSeq=0,eventSeq=0,epoch=0,playAt=0,sources=new Set();
let uploadAbort=null;
let started=0,timer=null,pollAbort=null,stopping=false,replaying=false,utterances=new Map();
let sentPackets=0,receivedPackets=0,playedPackets=0;
let permissionPending=false,micRequest=0,micChecking=false,silentSince=0;
let micLabel="",lastLevelUpdate=0;
let pendingSpeech=new Map(),completedReplies=0,lastBufferWait=0;
let producerRatio=1,playbackUnderruns=0;
let nativeState='L';
let speakingCompute=[];
let playbackBus=null,playbackBridge=null,playbackClock=null;
let diagnosticQueue=[],diagnosticSeq=0,diagnosticCall=null,diagnosticBusy=false,diagnosticTimer=null,diagnosticDropped=0,lastProcessedSeconds=0;
function diagnostic(type,details={}){
 if(!diagnosticCall)return;
 diagnosticQueue.push({type,client_seq:diagnosticSeq++,at:Date.now()/1000,
  elapsed_ms:Date.now()-started,...details});
 if(diagnosticQueue.length>1024){diagnosticQueue.shift();diagnosticDropped++;}
}
async function flushDiagnostics(){
 if(diagnosticBusy||!diagnosticCall||!diagnosticQueue.length)return;
 diagnosticBusy=true;const id=diagnosticCall,batch=diagnosticQueue.slice(0,32);
 try{await api('/api/calls/'+id+'/diagnostics',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({events:batch}),keepalive:true,signal:AbortSignal.timeout(2000)});diagnosticQueue.splice(0,batch.length);$('log-status').textContent='诊断日志已保存 · '+id;}
 catch(e){$('log-status').textContent='诊断日志发送失败：'+e.message;}
 finally{diagnosticBusy=false;}
}
async function sampleTransport(){
 if(!context||!call)return;
 diagnostic('transport_sample',{context_time:context.currentTime,state:context.state,diagnostics_dropped:diagnosticDropped,
  base_latency:context.baseLatency,output_latency:context.outputLatency,source_nodes:sources.size,
  pending_replies:pendingSpeech.size,send_queue:sendQueue.length,ahead_seconds:Math.max(0,playAt-context.currentTime)});
 if(playbackBridge){
  const stats=await playbackBridge.rx.getStats();
  stats.forEach(s=>{if(s.type==='inbound-rtp'&&s.kind==='audio')diagnostic('rtc_state',{
   jitter:s.jitter,jitter_buffer_delay:s.jitterBufferDelay,packets_lost:s.packetsLost,
   concealed_samples:s.concealedSamples,total_samples_received:s.totalSamplesReceived});});
 }
 await flushDiagnostics();
}
function playbackConnected(peer){
 return new Promise((resolve,reject)=>{
  const timeout=setTimeout(()=>finish(Error('通话播放音轨连接超时')),5000);
  function finish(error){clearTimeout(timeout);peer.removeEventListener('connectionstatechange',changed);error?reject(error):resolve();}
  function changed(){if(peer.connectionState==='connected')finish();else if(['failed','closed'].includes(peer.connectionState))finish(Error('通话播放音轨连接失败'));}
  peer.addEventListener('connectionstatechange',changed);changed();
 });
}
async function setupPlayback(){
 playbackBus=context.createGain();
 // Keep the MediaStream destination clock running through silent gaps. A
 // stopped source must deliver zeros rather than stall the local RTP track.
 playbackClock=context.createConstantSource();playbackClock.offset.value=0;
 playbackClock.connect(playbackBus);playbackClock.start();
 if(typeof RTCPeerConnection==='undefined'){
  playbackBus.connect(context.destination);$('echo-status').textContent='浏览器不支持通话播放参考，请使用耳机';return;
 }
 // Local peer pair only: no STUN/TURN server and no microphone track. The
 // received playback track gives browser AEC its documented remote reference.
 const tx=new RTCPeerConnection({iceServers:[]}),rx=new RTCPeerConnection({iceServers:[]});
 const destination=context.createMediaStreamDestination();
 const player=$('speaker');playbackBridge={tx,rx,destination,player};
 const pendingTx=[],pendingRx=[];
 tx.onicecandidate=e=>{if(e.candidate){if(rx.remoteDescription)rx.addIceCandidate(e.candidate).catch(()=>{});else pendingRx.push(e.candidate);}};
 rx.onicecandidate=e=>{if(e.candidate){if(tx.remoteDescription)tx.addIceCandidate(e.candidate).catch(()=>{});else pendingTx.push(e.candidate);}};
 let timeout;
 const received=new Promise((resolve,reject)=>{timeout=setTimeout(()=>reject(Error('通话播放参考连接超时')),5000);
  rx.ontrack=e=>{player.srcObject=new MediaStream([e.track]);resolve();};});
 // Attach the rejection handler before negotiation to avoid an unhandled
 // rejection when an unsupported browser fails setRemoteDescription.
 received.catch(()=>{});
 try{
  tx.addTrack(destination.stream.getAudioTracks()[0],destination.stream);
  await tx.setLocalDescription(await tx.createOffer());await rx.setRemoteDescription(tx.localDescription);
  for(const candidate of pendingRx)await rx.addIceCandidate(candidate);
  await rx.setLocalDescription(await rx.createAnswer());await tx.setRemoteDescription(rx.localDescription);
  for(const candidate of pendingTx)await tx.addIceCandidate(candidate);
  await received;clearTimeout(timeout);await Promise.all([playbackConnected(tx),playbackConnected(rx)]);playbackBus.connect(destination);
  await Promise.race([player.play(),new Promise((_,reject)=>setTimeout(()=>reject(Error('浏览器未开启通话声音播放')),5000))]);
  $('echo-status').textContent='通话播放参考已连接 · 麦克风持续收音';
 }catch(e){
  clearTimeout(timeout);closePlaybackBridge();playbackBus.connect(context.destination);
  $('echo-status').textContent='通话播放参考不可用，请使用耳机 · '+e.message;
 }
}
function closePlaybackBridge(){
 if(!playbackBridge)return;
 const {tx,rx,destination,player}=playbackBridge;playbackBridge=null;
 player.pause();player.srcObject?.getTracks().forEach(t=>t.stop());player.srcObject=null;player.muted=false;
 destination.stream.getTracks().forEach(t=>t.stop());tx.close();rx.close();
}
function describeEcho(track){
 const setting=track.getSettings().echoCancellation;
 const state=setting===true||typeof setting==='string'?'已开启':setting===false?'未开启':'浏览器未报告';
 $('echo-status').textContent='麦克风回声消除'+state+' · '+(playbackBridge?'通话播放参考已连接':'请使用耳机');
}
function showState(){
 const playing=sources.size>0,preparing=pendingSpeech.size>0;
 $('state').textContent=playing?'正在播放回答，也在收音':preparing?'正在准备回答，也在收音':states[nativeState]||nativeState;
 $('orb').textContent=playing?'播放':preparing?'准备':{L:'聆听',S:'回答',B:'回应'}[nativeState]||nativeState;
}
function transport(){ $('transport').textContent=(replaying?'录音流':'麦克风')+' · 已送 '+sentPackets+' 包 · 已收 '+receivedPackets+' 包 · 播放完成 '+playedPackets+' 包 · 缓冲等待 '+playbackUnderruns+' 次'; }
const states={L:'正在听你说',S:'正在回答，也在听',B:'简短回应，也在听'};
const endReasons={hangup:'通话已结束',context_limit:'本次通话已达到模型上下文上限，请开始新通话',
 input_timeout:'收音连接中断，通话已结束',input_backlog:'模型处理速度落后，通话已结束',
 model_error:'模型发生错误，通话已结束',codec_error:'语音解码发生错误，通话已结束',
 service_stop:'服务已停止',guardrail_blocked:'内容未通过安全审核，通话已停止',
 guardrail_unavailable:'安全审核未完成，通话已停止'};
async function api(url,opts){const r=await fetch(entryUrl(url),opts),j=await r.json();if(!r.ok){const e=Error(j.error||'连接失败');e.status=r.status;throw e;}return j;}
function error(message){if(message)diagnostic('client_error',{message});$('error').hidden=!message;$('error').className='error';$('error').textContent=message;}
function event(message){const d=document.createElement('div');d.className='event';d.textContent=new Date().toLocaleTimeString()+' · '+message;$('events').prepend(d);while($('events').children.length>20)$('events').lastChild.remove();}
function clearPlayback(){if(sources.size)diagnostic('audio_cancel',{reason:'interrupt_or_stop',source_nodes:sources.size});pendingSpeech.clear();for(const s of sources){s.cineduplexCancelled=true;try{s.stop()}catch{}}sources.clear();if(playbackBridge)playbackBridge.player.muted=true;if(context)playAt=context.currentTime;}
function bufferTarget(ratio){return Math.min(3.2,1.0+Math.max(0,ratio-1)*12);}
function initialReserve(reply){
 const recent=speakingCompute.slice(-12).sort((a,b)=>a-b);
 const modelRatio=recent.length?recent[Math.floor((recent.length-1)*.9)]/400:1;
 const rate=Math.max(1.25,producerRatio,modelRatio);
 // Reserve the measured next-block production deficit relative to the
 // first block already available. Do not cap a known deficit at 1.1s.
 const firstSeconds=reply.totalLength/reply.rate;
 reply.initialLead=Math.min(2.2,Math.max(.35,1.6*rate-firstSeconds+.25));
 const text=utterances.get(reply.id)?.textContent||'';
 const chinese=(text.match(/[\u3400-\u9fff]/g)||[]).length;
 const words=(text.match(/[A-Za-z]+(?:'[A-Za-z]+)?/g)||[]).length;
 const estimatedSeconds=chinese*.24+words*.4;
 // Reserve for a long answer only. Length is a conservative hint, while
 // measured production rate controls its size. Input remains continuous;
 // native interrupts always cancel scheduled audio immediately.
 if(estimatedSeconds<=6)return reply.initialLead;
 return Math.min(6,Math.max(3.5,reply.initialLead,
  .6+Math.max(.25,rate-1)*(estimatedSeconds-1)*1.5));
}
function newReply(e){return {id:e.event_id,chunks:[],length:0,totalLength:0,packets:0,rate:e.sample_rate||24000,
 epoch:e.epoch,firstAt:performance.now(),streaming:!!e.streaming,started:false,finished:false,nodes:0,completed:false,
 target:1.0,initialLead:.85,lastAudioAt:null,ratios:[],rebuffering:false};}
function receiveAudio(e){
 if(!context||e.epoch<epoch)return;
 const {pcm,sample_rate:rate,event_id:id}=e;
 const raw=atob(pcm),buf=new ArrayBuffer(raw.length),view=new DataView(buf);for(let i=0;i<raw.length;i++)view.setUint8(i,raw.charCodeAt(i));
 let reply=pendingSpeech.get(id);
 if(!reply){reply=newReply(e);pendingSpeech.set(id,reply);}
 if(rate!==reply.rate||raw.length%2)throw Error('输出音频格式发生变化，请重新连接');
 const out=new Float32Array(raw.length/2);for(let i=0;i<out.length;i++)out[i]=view.getInt16(i*2,true)/32768;
 const at=Number.isFinite(e.at)?e.at:performance.now()/1000;
 if(reply.lastAudioAt!==null&&!e.final_chunk&&out.length/rate>=.25){
  const ratio=Math.max(0,at-reply.lastAudioAt)/(e.codec_advance_seconds||out.length/rate);reply.ratios.push(ratio);
  // Initial reserve is a timed runway, not another full codec block.
  // Learning a slightly slow producer must not double first-audio waiting.
 }
 reply.lastAudioAt=at;
 // A terminal codec block proves no future audio needs a runway.
 if(e.final_chunk===true)reply.finished=true;
 reply.chunks.push(out);reply.length+=out.length;reply.totalLength+=out.length;reply.packets++;
 // If the whole reply completes before its first scheduled sound, replace
 // only that unplayed node. Complete PCM needs no producer reserve. Never
 // move an already playing reply or another reply's playback timeline.
 const first=reply.firstNode;
 if(reply.finished&&first&&sources.size===1&&sources.has(first)&&first.cineduplexStart>context.currentTime+.08){
  diagnostic('audio_cancel',{event_id:id,reason:'complete_before_playback',scheduled_start:first.cineduplexStart,source_nodes:1});
  first.cineduplexCancelled=true;first.stop();sources.delete(first);
  reply.chunks.unshift(first.buffer.getChannelData(0).slice());reply.length+=first.buffer.length;
  reply.carriedPackets=(reply.carriedPackets||0)+first.cineduplexPackets-1;
  reply.started=false;reply.firstNode=null;playAt=context.currentTime;
 }
 if(reply.totalLength/rate>60)throw Error('回答音频超过缓冲上限，请结束后重试');
 diagnostic('audio_receive',{event_id:id,chunk_seq:e.chunk_seq,server_seq:e.seq,server_at:e.at,samples:out.length,sample_rate:rate,context_time:context.currentTime});
 receivedPackets++;transport();
 if(reply.streaming){
  if(reply.started&&!reply.rebuffering&&playAt<context.currentTime-.08){
   playbackUnderruns++;reply.rebuffering=true;diagnostic('rebuffer',{event_id:id,count:playbackUnderruns,ahead_seconds:playAt-context.currentTime});
   reply.target=Math.max(1.6,reply.target,bufferTarget(Math.max(producerRatio,...reply.ratios)));
   event('生成速度不足，正在补充连续播放缓冲 · 第 '+playbackUnderruns+' 次');
   transport();
  }
  if((reply.started&&!reply.rebuffering)||reply.length/rate>=reply.target)scheduleReply(reply);
 }
}
function completeReply(reply){
 if(reply.finished&&!reply.nodes&&!reply.completed&&reply.totalLength){reply.completed=true;completedReplies++;}
}
function scheduleReply(reply){
 if(!context||!reply.length||reply.epoch<epoch)return;
 const audio=context.createBuffer(1,reply.length,reply.rate),out=audio.getChannelData(0);
 let offset=0;for(const chunk of reply.chunks){out.set(chunk,offset);offset+=chunk.length;}
 const packets=reply.chunks.length+(reply.carriedPackets||0);reply.carriedPackets=0;reply.chunks=[];reply.length=0;
 const node=context.createBufferSource();node.buffer=audio;node.connect(playbackBus||context.destination);
 if(playbackBridge)playbackBridge.player.muted=false;
 if(!reply.started){
  reply.initialLead=initialReserve(reply);
  diagnostic('initial_buffer',{event_id:reply.id,lead_seconds:reply.initialLead,producer_ratio:producerRatio});
  lastBufferWait=(performance.now()-reply.firstAt)/1000+(reply.finished?.04:reply.initialLead);
  event('开始播放回答 · 模型开口至播放 '+lastBufferWait.toFixed(1)+' 秒');reply.started=true;
 }
 // First block is smaller; later blocks amortize decoding overhead.
 // A completed reply needs no reserve for an ungenerated next block.
 const lead=reply.finished?.04:(reply.rebuffering?Math.max(1,reply.initialLead):(reply.totalLength===audio.length?reply.initialLead:.04));
 reply.rebuffering=false;
 playAt=Math.max(playAt,context.currentTime+lead);node.cineduplexStart=playAt;node.cineduplexPackets=packets;if(!reply.firstNode)reply.firstNode=node;node.start(playAt);diagnostic('audio_schedule',{event_id:reply.id,context_time:context.currentTime,scheduled_start:playAt,duration:audio.duration,ahead_seconds:playAt-context.currentTime,source_nodes:sources.size});playAt+=audio.duration;sources.add(node);reply.nodes++;
 node.onended=()=>{sources.delete(node);reply.nodes--;if(!node.cineduplexCancelled){playedPackets+=packets;completeReply(reply);diagnostic('audio_played',{event_id:reply.id,context_time:context?.currentTime,duration:audio.duration,played_packets:playedPackets});}transport();updateAudioStatus();};
 showState();
}
function finishAudio(e){
 const reply=pendingSpeech.get(e.event_id);pendingSpeech.delete(e.event_id);
 if(!context||!reply||e.epoch<epoch||reply.epoch<epoch)return;
 reply.finished=true;scheduleReply(reply);completeReply(reply);
 if(reply.ratios.length){
  const sorted=reply.ratios.slice().sort((a,b)=>a-b);
  producerRatio=sorted[Math.min(sorted.length-1,Math.floor(sorted.length*.9))];
 }
 const text=utterances.get(e.event_id)?.textContent||'',duration=reply.totalLength/reply.rate;
 if(text.length>=8&&duration<Math.max(.6,text.length*.06)){
  $('response-warning').hidden=false;
  $('response-warning').textContent='模型这轮声音不完整（'+duration.toFixed(2)+'秒）。连接仍在收音，可以继续说话。';
  event('回答声音不完整 · '+duration.toFixed(2)+' 秒');
 }
 updateAudioStatus();
}
function updateAudioStatus(){
 let buffered=0;for(const reply of pendingSpeech.values())buffered+=reply.length/reply.rate;
 if(pendingSpeech.size)$('audio-status').textContent='正在生成下一段声音 · 已缓存 '+buffered.toFixed(1)+' 秒';
 else if(sources.size)$('audio-status').textContent='正在播放连续语音';
 else $('audio-status').textContent='等待回答 · 已播放 '+completedReplies+' 次';
 showState();
}
function transcript(e){const id=e.event_id||'current';let p=utterances.get(id);if(!p){p=document.createElement('div');p.className='utterance';$('transcript').append(p);utterances.set(id,p);}p.textContent=e.snapshot||e.text||'';$('transcript').scrollTop=$('transcript').scrollHeight;}
async function poll(id){
 while(call===id&&!stopping){
  try{
   const pollBegin=performance.now();const j=await api('/api/calls/'+id+'/events?after='+eventSeq,{signal:pollAbort.signal});
   if(call!==id)return;
   if(j.events.some(e=>e.type==='audio'))diagnostic('transport_sample',{poll_wait_ms:performance.now()-pollBegin});
   for(const e of j.events){
    eventSeq=e.seq;
    if(e.type==='audio_start'&&e.epoch>=epoch){pendingSpeech.set(e.event_id,newReply(e));updateAudioStatus();}
    if(e.type==='audio'){receiveAudio(e);updateAudioStatus();}
    if(e.type==='audio_end')finishAudio(e);
    if(e.type==='response_warning'){$('response-warning').hidden=false;$('response-warning').textContent=e.message;event(e.message);}
    if(e.type==='session_warning'){$('response-warning').hidden=false;$('response-warning').textContent=e.message;event(e.message);}
    if(e.type==='guardrail_pending'){$('guardrail-status').textContent=e.message;}
    if(e.type==='guardrail_pass'){$('guardrail-status').textContent=e.message;}
    if(e.type==='guardrail_block'){clearPlayback();$('guardrail-status').textContent=e.message;error(e.message);event(e.message);}
    if(e.type==='stop_audio'){epoch=e.epoch;clearPlayback();updateAudioStatus();event('模型回到聆听，已清空待播放声音');}
    if(e.type==='state_change'){nativeState=e.to;showState();event(states[e.to]||e.to);}
    if(e.type==='generation_end'){nativeState='L';showState();}
    if(e.type==='text_delta'||e.type==='event_end')transcript(e);
    if(e.type==='metrics'){
     lastProcessedSeconds=e.processed_seconds;
     $('backlog').textContent=e.pending_seconds.toFixed(1)+' 秒';
     if(e.state==='S'||e.state==='B'){
      speakingCompute.push(e.compute_ms);if(speakingCompute.length>30)speakingCompute.shift();
      const average=speakingCompute.reduce((a,b)=>a+b,0)/speakingCompute.length;
      $('compute').textContent=Math.round(average)+' ms';
     }
     nativeState=e.state;showState();
     if(e.pending_seconds>3)$('state').textContent+=' · 处理速度落后于输入';
    }
    if(e.type==='error'){error(e.message);event('通话发生错误');await stop(false);return;}
    if(e.type==='closed'||e.type==='session_stopping'){const reason=endReasons[e.reason]||('通话结束：'+e.reason);event(reason);await stop(false,reason);return;}
   }
   if(j.closed){await stop(false);return;}
  }catch(e){if(e.name==='AbortError')return;error(e.message);await stop();return;}
 }
}
function monitorMic(rms){
 if(replaying)return;
 const db=rms>0?20*Math.log10(rms):-Infinity;
 if(Date.now()-lastLevelUpdate>600){$('mic-info').textContent=micLabel+' · 输入音量 '+(Number.isFinite(db)?Math.round(db)+' dB':'静音');lastLevelUpdate=Date.now();}
 if(rms<.0005){if(!silentSince)silentSince=Date.now();}else silentSince=0;
 const track=stream?.getAudioTracks()[0];
 const weak=!!silentSince&&Date.now()-silentSince>8000;
 $('mic-warning').hidden=!(weak||track?.muted);
 $('mic-warning').textContent=track?.muted?'浏览器的麦克风轨道已静音。请检查系统或耳机的静音开关。':'当前输入几乎是静音。如果你正在说话，请结束后选择正确的麦克风，并点击检查麦克风。';
}
async function listMics(){
 if(!navigator.mediaDevices?.enumerateDevices)return;
 const current=$('mic-device').value;
 const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audioinput');
 const options=[new Option('系统默认麦克风','')];
 devices.forEach((d,i)=>options.push(new Option(d.label||'麦克风 '+(i+1),d.deviceId)));
 $('mic-device').replaceChildren(...options);if(options.some(o=>o.value===current))$('mic-device').value=current;
}
function enqueue(pcm,rms=0){monitorMic(rms);if(micChecking){$('level').value=rms;return;}if(!call||stopping)return;$('level').value=rms;sendQueue.push(pcm);if(sendQueue.length>12){error('网络发送积压，通话已结束。');stop();return;}pump();}
async function uploadAudio(id,seq,pcm){
 // A response can be lost after acceptance. The server deduplicates the
 // unchanged sequence number under its input lock; never retry new bytes.
 for(let attempt=0;attempt<3;attempt++){
  if(call!==id||stopping)return false;
  const abort=new AbortController();uploadAbort=abort;
  const timeout=setTimeout(()=>abort.abort(),1200);
  try{
   await api('/api/calls/'+id+'/audio',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Audio-Seq':String(seq)},body:pcm,signal:abort.signal});
   return call===id&&!stopping;
  }catch(e){
   if(call!==id||stopping)return false;
   const transient=['AbortError','TimeoutError','TypeError'].includes(e.name)||[502,503,504].includes(e.status);
   if(!transient||attempt===2)throw e;
   diagnostic('transport_sample',{reason:'audio_retry_same_sequence',send_queue:sendQueue.length,upload_ms:1200});
   await new Promise(r=>setTimeout(r,50*(attempt+1)));
  }finally{clearTimeout(timeout);if(uploadAbort===abort)uploadAbort=null;}
 }
 return false;
}
async function pump(){
 const id=call;if(!id||sending===id)return;sending=id;
 try{while(sendQueue.length&&call===id&&!stopping){
   const pcm=sendQueue[0],uploadBegin=performance.now();if(!await uploadAudio(id,audioSeq,pcm))break;
   audioSeq++;sentPackets++;if(audioSeq%10===0)diagnostic('transport_sample',{upload_ms:performance.now()-uploadBegin,send_queue:sendQueue.length,sent_packets:sentPackets});transport();sendQueue.shift();
 }}catch(e){if(call===id){error(e.message);await stop();}}finally{if(sending===id)sending=false;}
}
async function connect(requestId=null){
 error('');context=new AudioContext();await context.resume();
 await setupPlayback();
 if(requestId!==null&&requestId!==micRequest)throw Error('通话启动已取消');
 const j=await api('/api/calls',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({variant:$('variant').value,voice:$('voice').value,language:$('language').value})});
 if(requestId!==null&&requestId!==micRequest){await api('/api/calls/'+j.id+'/stop',{method:'POST',body:''});throw Error('通话启动已取消');}
 call=j.id;audioSeq=0;eventSeq=0;epoch=0;sendQueue=[];playAt=context.currentTime;started=Date.now();stopping=false;
 sentPackets=0;receivedPackets=0;playedPackets=0;completedReplies=0;lastBufferWait=0;producerRatio=1;playbackUnderruns=0;speakingCompute=[];$('compute').textContent='尚未回答';pendingSpeech.clear();transport();updateAudioStatus();
 diagnosticCall=call;diagnosticQueue=[];diagnosticSeq=0;diagnosticDropped=0;lastProcessedSeconds=0;
 $('call-id').textContent=call;$('report').href=entryUrl('/api/calls/'+call+'/diagnostics');$('report').hidden=false;$('mark-problem').disabled=false;
 const settings=stream?.getAudioTracks()[0]?.getSettings()||{};
 diagnostic('client_config',{precision:j.decode_precision,runtime_sha256:j.tp_runtime_sha256||j.runtime_sha256||j.source_sha256,tensor_parallel_size:j.tensor_parallel_size||1,voice:$('voice').value,language:$('language').value,
  echo_cancellation:settings.echoCancellation,noise_suppression:settings.noiseSuppression,auto_gain_control:settings.autoGainControl,
  input_sample_rate:settings.sampleRate,context_sample_rate:context.sampleRate,reference_connected:!!playbackBridge,browser:navigator.userAgent});
 diagnosticTimer=setInterval(()=>sampleTransport().catch(()=>{}),2000);
 nativeState='L';$('response-warning').hidden=true;
 pollAbort=new AbortController();utterances=new Map();$('transcript').replaceChildren();$('events').replaceChildren();
 $('start').disabled=true;$('replay').disabled=true;$('variant').disabled=true;$('voice').disabled=true;$('language').disabled=true;$('stop').disabled=false;
 $('orb').classList.add('active');$('orb').textContent='聆听';$('state').textContent=states.L;
 event('通话已连接 · '+$('language').selectedOptions[0].textContent+' · '+($('voice').value==='male'?'男声':'女声')+' · '+($('variant').value==='trained'?'本次训练模型 · 稳定模式':'原始模型'));
 timer=setInterval(()=>{let s=Math.floor((Date.now()-started)/1000);$('duration').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');$('playback').textContent=Math.max(0,playAt-context.currentTime).toFixed(1)+' 秒';},250);
 poll(call);return call;
}
async function start(checkOnly=false){
 if(permissionPending||call||micChecking)return;
 const requestId=++micRequest;permissionPending=true;$('start').disabled=true;$('stop').disabled=false;
 $('mic-check').disabled=true;$('mic-device').disabled=true;
 $('state').textContent='等待麦克风授权，请在浏览器或系统提示中允许';
 try{
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('麦克风需要受信任的 HTTPS 或 localhost 页面。请使用局域网证书入口。');
  // Permission precedes session creation, so denying it does not occupy GPU.
  const selected=$('mic-device').value;
  const acquired=await navigator.mediaDevices.getUserMedia({audio:{...(selected?{deviceId:{exact:selected}}:{}),channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
  if(requestId!==micRequest){acquired.getTracks().forEach(t=>t.stop());return;}
  stream=acquired;permissionPending=false;await listMics();
  const track=stream.getAudioTracks()[0];micLabel=track.label||'浏览器麦克风';silentSince=0;
  $('mic-device').disabled=true;$('mic-check').disabled=true;
  if(checkOnly){micChecking=true;context=new AudioContext();await context.resume();await setupPlayback();$('state').textContent='正在检查收音。请说一句话，观察音量条。';$('orb').textContent='收音';$('start').disabled=true;$('stop').disabled=false;}else await connect(requestId);
  describeEcho(track);
  await context.audioWorklet.addModule(entryUrl('/capture.worklet.js'));
  capture=new AudioWorkletNode(context,'continuous-capture');capture.port.onmessage=e=>enqueue(e.data.pcm,e.data.rms);
  micSource=context.createMediaStreamSource(stream);micSource.connect(capture);capture.connect(context.destination);
  stream.getAudioTracks()[0].onended=()=>stop();
 }catch(e){if(requestId!==micRequest)return;permissionPending=false;error(e.message);await stop();}
}
async function stop(remote=true,reason='通话已结束'){
 if(stopping)return;stopping=true;const id=call;diagnostic('call_stop',{reason,sent_packets:sentPackets,received_packets:receivedPackets,played_packets:playedPackets});clearInterval(diagnosticTimer);call=null;$('mark-problem').disabled=true;
 micRequest++;permissionPending=false;micChecking=false;$('mic-device').disabled=false;$('mic-check').disabled=false;
 pollAbort?.abort();uploadAbort?.abort();uploadAbort=null;sending=false;sendQueue=[];clearInterval(timer);clearPlayback();
 if(capture){capture.port.onmessage=null;capture.disconnect();capture=null;}if(micSource){micSource.disconnect();micSource=null;}
 if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
 closePlaybackBridge();if(playbackClock){playbackClock.stop();playbackClock.disconnect();playbackClock=null;}if(playbackBus){playbackBus.disconnect();playbackBus=null;}
 if(context){await context.close().catch(()=>{});context=null;}
 if(id&&remote){try{await api('/api/calls/'+id+'/stop',{method:'POST',body:'',signal:AbortSignal.timeout(3000)});}catch{}}
 $('orb').classList.remove('active');$('orb').textContent='待机';$('state').textContent=reason;$('level').value=0;
 $('audio-status').textContent='播放已停止 · 已播放 '+completedReplies+' 次回答';
 if(context===null)$('echo-status').textContent='收音与通话播放参考已停止';
 $('stop').disabled=true;$('variant').disabled=false;$('voice').disabled=false;$('language').disabled=false;replaying=false;stopping=false;
 const flushDeadline=Date.now()+2500;while(diagnosticQueue.length&&Date.now()<flushDeadline){if(diagnosticBusy)await new Promise(r=>setTimeout(r,50));else await flushDiagnostics();}
 await status();
}
async function replay(){
 let file=$('file').files[0];if(!file)return;$('replay').disabled=true;
 try{
  if(file.size>10*1024*1024)throw Error('测试录音最多10MB');
  const decode=new OfflineAudioContext(1,1,16000);
  const audio=await decode.decodeAudioData(await file.arrayBuffer());
  if(audio.duration>310)throw Error('流式测试录音最多310秒');
  const offline=new OfflineAudioContext(1,Math.round((audio.duration+12)*16000),16000);
  const src=offline.createBufferSource();src.buffer=audio;src.connect(offline.destination);src.start();
  const data=(await offline.startRendering()).getChannelData(0);replaying=true;const id=await connect();
  event('录音流式测试，持续输入包含结尾12秒静音');
  const begin=performance.now();
  for(let p=0; p<data.length && call===id; p+=6400){
   const bytes=new ArrayBuffer(12800),view=new DataView(bytes);let rms=0;
   for(let i=0;i<6400;i++){let v=data[p+i]||0;rms+=v*v;view.setInt16(i*2,Math.max(-32768,Math.min(32767,Math.round(v*32768))),true);}
   enqueue(bytes,Math.sqrt(rms/6400));await new Promise(r=>setTimeout(r,Math.max(0,begin+(p/16000+.4)*1000-performance.now())));
  }
  if(call===id){event('测试音频已发送，等待模型处理');while(call===id&&sendQueue.length)await new Promise(r=>setTimeout(r,200));
   const deadline=Date.now()+120000;await new Promise(r=>setTimeout(r,2000));
   while(call===id&&(lastProcessedSeconds<data.length/16000-.1||pendingSpeech.size||sources.size)&&Date.now()<deadline)await new Promise(r=>setTimeout(r,250));
   if(call===id)await stop();}
 }catch(e){error(e.message);await stop();}
}
async function status(){
 if(call||micChecking)return;
 try{const s=await api('/api/status');ready=!!s.ready;
  $('connection').textContent=ready?(s.active?'已有通话进行中':'实时模型已就绪'):(s.phase==='storage_disconnected'?'存储连接已断开':s.phase==='updating'?'正在更新实时接口':'实时模型正在加载');
  if(s.active)ready=false;
  $('model').textContent=(s.model||'模型未加载')+' · '+(s.control_mode==='native-full-duplex'?'原生听说控制':'等待实时接口')+(s.decode_precision==='bf16'?' · BF16 原精度 · 无权重量化':s.decode_precision?.startsWith('int4')?' · INT4 权重量化':'');
  if(s.trained_adapter_scale!==undefined)$('model').textContent+=' · 训练适配强度 '+Math.round(s.trained_adapter_scale*100)+'%';
  if(Number.isInteger(s.tensor_parallel_size)&&s.tensor_parallel_size>1)$('model').textContent+=' · '+s.tensor_parallel_size+' GPU';
  const g=s.guardrail;
  $('guardrail-status').textContent=!g?'安全护栏：当前服务未报告接入状态':!g.enabled?'安全护栏未启用':
   '阿里云护栏：回答文字审核已启用 · 输入音频'+(g.input_audio?'审核并在发现风险时停止':'未审核')+
   ' · 回答音频'+(g.output_audio?'审核已启用':'未单独审核')+' · 审核通过后发送回答';
  $('guardrail-privacy').hidden=!(g?.enabled&&(g.input_audio||g.output_audio));
  $('guardrail-playback-hint').textContent=g?.enabled?'回答生成后先审核，通过后发送文字和声音；等待期间麦克风继续收音。':
   '文字和声音逐步到达。生成较慢时会先多缓存一点声音，再连续播放；麦克风保持收音。';
 }catch(e){ready=false;$('connection').textContent='实时接口未连接';$('model').textContent=e.message;}
 $('start').disabled=!ready||permissionPending;$('replay').disabled=!ready||permissionPending||!$('file').files.length;
}
$('start').onclick=()=>start(false);$('mic-check').onclick=()=>start(true);$('stop').onclick=()=>stop();$('replay').onclick=replay;$('file').onchange=status;
$('mark-problem').onclick=()=>{diagnostic('mark_problem',{context_time:context?.currentTime,source_nodes:sources.size,pending_replies:pendingSpeech.size,ahead_seconds:Math.max(0,playAt-(context?.currentTime||0))});flushDiagnostics();event('已标记声音问题的位置');};
window.addEventListener('pagehide',()=>{if(diagnosticCall&&diagnosticQueue.length)navigator.sendBeacon(entryUrl('/api/calls/'+diagnosticCall+'/diagnostics'),JSON.stringify({events:diagnosticQueue.slice(-32)}));if(call)navigator.sendBeacon(entryUrl('/api/calls/'+call+'/stop'),'');stream?.getTracks().forEach(t=>t.stop());});
listMics().catch(()=>{});status();setInterval(status,5000);
