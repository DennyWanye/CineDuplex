// Exercise the actual page functions with an AudioContext clock, without
// a microphone, network request, model, or browser permissions.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const file=path.join(__dirname,'../src/cineduplex/web/realtime.js');
const source=fs.readFileSync(file,'utf8').split("$('start').onclick=")[0];
const elements=new Map();
const document={currentScript:{src:'https://example.test/cineduplex/realtime.js'},
 getElementById(id){if(!elements.has(id))elements.set(id,{textContent:'',children:[],prepend(){},append(){}});return elements.get(id);},
 createElement(){return {textContent:'',className:''};}};
const sandbox={assert,document,URL,performance:{now:()=>0},atob:s=>Buffer.from(s,'base64').toString('binary'),
 pcm:Buffer.alloc(24000*2).toString('base64'),nodes:[]};
vm.runInNewContext(source+`
 context={currentTime:10,createBuffer(channels,length,rate){const data=new Float32Array(length);return {length,duration:length/rate,getChannelData:()=>data};},
 createBufferSource(){const node={connect(){},start(at){this.at=at;},stop(){this.onended?.();}};nodes.push(node);return node;}};
 playbackBus={};producerRatio=10;speakingCompute=[10000];playAt=10;
 const full={event_id:'approved',epoch:0,sample_rate:24000,streaming:true,response_complete:true};
 pendingSpeech.set(full.event_id,newReply(full));
 for(let i=0;i<4;i++)receiveAudio({...full,type:'audio',pcm,chunk_seq:i,final_chunk:i===3});
 assert.equal(nodes.length,0,'whole approved answer waits for audio_end, not a guessed runway');
 finishAudio({...full,type:'audio_end'});
 assert.equal(nodes.length,1);assert.equal(nodes[0].buffer.length,96000);
 assert.equal(nodes[0].at,10.04);assert.equal(nodes[0].cineduplexPackets,4);
 nodes[0].onended();assert.equal(playedPackets,4);assert.equal(completedReplies,1);
 const native={event_id:'native',epoch:0,sample_rate:24000,streaming:true};
 pendingSpeech.set(native.event_id,newReply(native));receiveAudio({...native,type:'audio',pcm,chunk_seq:0});
 assert.equal(nodes.length,2,'unreviewed native streaming retains incremental playback');
 assert.ok(nodes[1].at>context.currentTime+.04,'live producer still gets its reserve');
 clearPlayback();assert.equal(sources.size,0);
 epoch=1;receiveAudio({...native,type:'audio',pcm});assert.equal(nodes.length,2,'stale epoch is not played');
`,sandbox,{filename:file});
console.log('PASS: complete approved PCM starts at 40ms; all 4 packets preserved; native reserve and cancellation retained');
