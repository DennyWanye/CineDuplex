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
const sandbox={assert,document,URL,Buffer,performance:{now:()=>0},atob:s=>Buffer.from(s,'base64').toString('binary'),
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
 const reviewed={event_id:'prefix',epoch:0,sample_rate:24000,streaming:true,reviewed_prefix:true};
 pendingSpeech.set(reviewed.event_id,newReply(reviewed));pendingSpeech.get(reviewed.event_id).target=1;playAt=context.currentTime;
 receiveAudio({...reviewed,type:'audio',pcm,chunk_seq:1});
 assert.equal(nodes.length,3);assert.ok(Math.abs(nodes[2].at-11.6)<1e-6,'reviewed prefix reserves the missing runway');
 assert.equal(nodes[2].buffer.length,24000);
 context.currentTime=13;
 receiveAudio({...reviewed,type:'audio',pcm,chunk_seq:2});
 assert.equal(nodes.length,3,'after an actual underrun, one short block does not restart playback');
 assert.equal(pendingSpeech.get('prefix').target,6.4);
 receiveAudio({...reviewed,type:'audio',pcm,chunk_seq:3});
 assert.equal(nodes.length,3);
 receiveAudio({...reviewed,type:'audio',pcm,chunk_seq:4});
 assert.equal(nodes.length,3,'a real cloud stall requires multiple future blocks, not the first-sound cap');
 for(let i=5;i<=8;i++)receiveAudio({...reviewed,type:'audio',pcm,chunk_seq:i});
 assert.equal(nodes.length,4);assert.equal(nodes[3].buffer.length,24000*7);
 clearPlayback();assert.equal(sources.size,0);
 epoch=1;receiveAudio({...native,type:'audio',pcm});assert.equal(nodes.length,4,'stale epoch is not played');
 // Reproduce the real five-second receipt stall. The first approved block
 // must stay queued; the next two blocks share one uninterrupted clock.
 context.currentTime=10;playAt=10;epoch=0;prefixDeliveryDeficits=[0,0,0];playbackUnderruns=0;
 const paced={...reviewed,event_id:'paced'};pendingSpeech.set(paced.event_id,newReply(paced));
 performance.now=()=>10000;
 receiveAudio({...paced,type:'audio',pcm:Buffer.alloc(30720*2).toString('base64')});
 assert.equal(nodes.length,4,'one short reviewed prefix does not start an incomplete stream');
 context.currentTime=15.04;performance.now=()=>15040;
 receiveAudio({...paced,type:'audio',pcm:Buffer.alloc(38400*2).toString('base64')});
 assert.equal(nodes.length,5);assert.ok(Math.abs(nodes[4].at-16.64)<1e-6,'coalesced blocks retain the measured next-receipt reserve');
 assert.equal(nodes[4].buffer.length,69120);
 context.currentTime=17.7;performance.now=()=>17700;
 receiveAudio({...paced,type:'audio',pcm:Buffer.alloc(38400*2).toString('base64')});
 assert.equal(playbackUnderruns,0);assert.ok(Math.abs(nodes[5].at-19.52)<1e-6);
 clearPlayback();context.currentTime=20;playAt=20;
 const brief={...reviewed,event_id:'brief'};pendingSpeech.set(brief.event_id,newReply(brief));
 receiveAudio({...brief,type:'audio',pcm:Buffer.alloc(12000*2).toString('base64'),final_chunk:true});
 finishAudio({...brief,type:'audio_end'});assert.equal(nodes[6].at,20.04,'complete short answer has no multi-block wait');
`,sandbox,{filename:file});
console.log('PASS: complete approved PCM starts at 40ms; all 4 packets preserved; native reserve and cancellation retained');
