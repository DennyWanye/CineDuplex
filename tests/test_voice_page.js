// Focused checks of the actual shared runtime's user-view state and mute path.
// No hardware microphone, browser permissions, GPU or cloud requests.
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.join(__dirname,'../src/cineduplex/web');
const html=fs.readFileSync(path.join(root,'realtime.html'),'utf8');
assert.equal(/<input\b|<textarea\b|contenteditable=/i.test(html),false,'voice page has no text or file input');
assert.equal(html.includes('id="replay"'),false,'no hidden file replay control on the user page');
assert.ok(/id="caption-panel"[^>]*hidden/.test(html),'captions default off');
assert.ok(html.indexOf('id="settings-dialog"')<html.indexOf('id="voice"'),'voice choice is inside settings');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size,ids.length,'no duplicate DOM identities');
const elements=new Map();
function element(id){
 if(id==='file'||id==='replay')return null;
 if(!elements.has(id))elements.set(id,{hidden:false,disabled:false,textContent:'',value:'',children:[],style:{},attrs:{},
   classList:{toggle(){},remove(){}},setAttribute(k,v){this.attrs[k]=v;},getAttribute(k){return this.attrs[k];},
   prepend(){},append(x){this.children.push(x);},replaceChildren(){this.children=[];}});
 return elements.get(id);
}
element('voice-wave').children=Array.from({length:9},()=>({style:{}}));
const source=fs.readFileSync(path.join(root,'realtime.js'),'utf8').split("$('start').onclick=")[0];
const sandbox={assert,document:{body:{dataset:{view:'voice'}},currentScript:{src:'https://example.test/cineduplex/realtime.js'},getElementById:element,createElement:()=>({})},
 URL,window:{isSecureContext:true},navigator:{mediaDevices:{}},performance:{now:()=>0},clearInterval(){},setInterval(){},setTimeout,AbortSignal,Uint8Array,Date};
const checks=`
 prefixDeliveryDeficits=[0,0,0];assert.equal(initialReserve({reviewedPrefix:true}),1.6,'short completed replies must not under-buffer the next stream');
 prefixDeliveryDeficits=[1.5,1.8,2];assert.ok(initialReserve({reviewedPrefix:true})>=1.6);
 ready=true;refreshUserView();assert.equal($('voice-title').textContent,'你好，我是小橙');
 assert.equal($('call-controls').hidden,true);assert.equal($('duration').hidden,true);
 toggleCaptions();assert.equal($('caption-panel').hidden,false);assert.equal($('transcript').attrs['aria-live'],'polite');
 toggleCaptions();assert.equal($('caption-panel').hidden,true);assert.equal($('transcript').attrs['aria-live'],'off');
 permissionPending=true;userStarting=true;refreshUserView();assert.equal(document.body.dataset.callState,'permission');
 assert.equal($('call-controls').hidden,false,'permission can be cancelled');
 permissionPending=false;refreshUserView();assert.equal(document.body.dataset.callState,'connecting');
 userStarting=false;call='test';capture={port:{},disconnect(){}};nativeState='L';refreshUserView();
 assert.equal(document.body.dataset.callState,'listening');
 pendingSpeech.set('reply',{});refreshUserView();assert.equal(document.body.dataset.callState,'preparing');
 pendingSpeech.clear();context={currentTime:10};
 const future={cineduplexStart:11,buffer:{duration:2,sampleRate:24000,getChannelData(){return new Float32Array(48000);}}};
 sources.add(future);refreshUserView();assert.equal(document.body.dataset.callState,'listening','scheduled but unplayed audio is not speaking');
 context.currentTime=11.1;refreshUserView();assert.equal(document.body.dataset.callState,'speaking');sources.clear();
 const track={enabled:true,stop(){this.stopped=true;},getSettings(){return {};}};
 stream={getAudioTracks:()=>[track],getTracks:()=>[track]};sendQueue=[new Uint8Array([5,6]).buffer];
 toggleMute();assert.equal(track.enabled,false);assert.equal(userMuted,true);
 assert.deepEqual([...new Uint8Array(sendQueue[0])],[0,0],'queued microphone PCM becomes silence');
 pump=()=>{};enqueue(new Uint8Array([7,8]).buffer,.1);
 assert.deepEqual([...new Uint8Array(sendQueue[1])],[0,0],'new PCM remains silent while muted');
 refreshUserView();assert.equal(document.body.dataset.callState,'muted');
 toggleMute();assert.equal(track.enabled,true);assert.equal($('mute').attrs['aria-pressed'],'false');
 // Stopping closes microphone and context and sends the original session stop.
 const requests=[];api=async(route)=>{requests.push(route);return {ready:true,guardrail:{enabled:true,input_audio:true,output_audio:true}};};
 context={currentTime:12,async close(){this.closed=true;}};const savedContext=context;
 const pendingRequest=micRequest;
 stop(true).then(async()=>{
  assert.equal(track.stopped,true);assert.equal(savedContext.closed,true);assert.equal(call,null);
  assert.equal(capture,null);assert.ok(micRequest>pendingRequest,'late permission result is invalidated');
  assert.ok(requests.includes('/api/calls/test/stop'));assert.equal($('call-controls').hidden,true);
  assert.equal($('voice-title').textContent,'通话已结束');assert.equal($('start').disabled,false);
  $('transcript').children.push({textContent:'previous call'});
  let grant;const lateTrack={stop(){this.stopped=true;}};
  navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>{grant=resolve;});
  const attempt=start();assert.equal(permissionPending,true);assert.equal($('transcript').children.length,0,'restart clears previous captions before permission resolves');await stop();
  grant({getTracks:()=>[lateTrack]});await attempt;
  assert.equal(lateTrack.stopped,true,'permission granted after cancellation immediately closes the microphone');
  assert.equal(requests.includes('/api/calls'),false,'cancelled permission does not create a model call');
  navigator.mediaDevices.getUserMedia=async()=>{throw Object.assign(Error('denied'),{name:'NotAllowedError'});};
  await start();assert.equal($('error').hidden,false);assert.ok($('error').textContent.includes('麦克风权限未开启'));
  assert.equal(requests.includes('/api/calls'),false,'denied microphone never occupies a model call');
  $('output-device').value='speaker';
  let selected='';await selectPlaybackDevice({async setSinkId(value){selected=value;}});assert.equal(selected,'speaker');
  await assert.rejects(()=>selectPlaybackDevice({}),/浏览器不支持/,'unsupported sink must not silently select another output');
  await assert.rejects(()=>selectPlaybackDevice({async setSinkId(){throw Error('output device disconnected');}}),/disconnected/);
  $('output-device').value='';await selectPlaybackDevice({});
  endingCall='just-ended';api=async()=>({ready:true,active:'just-ended'});await status();
  assert.equal($('service-message').textContent,'上一通通话正在结束，请稍等');
  api=async()=>({ready:true,active:'other-call'});await status();assert.ok($('service-message').textContent.includes('另一通'));
  api=async()=>({ready:true,active:null});await status();assert.equal(endingCall,null);
  console.log('PASS: voice-only controls, mute/stop, restart cleanup, output routing and ending identity');
 }).catch(e=>{console.error(e);process.exitCode=1;});
`;
sandbox.console=console;sandbox.process=process;
vm.runInNewContext(source+checks,sandbox,{filename:'voice-page-runtime'});
