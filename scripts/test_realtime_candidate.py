#!/usr/bin/env python3
"""Bounded real HTTP/GPU test of an already admitted shadow service.

Uses an existing diagnostic recording, never starts a GPU worker or training.
Logs and output audio remain on the actual NAS. This is not a listening review.
"""
import argparse
import base64
import json
from pathlib import Path
import threading
import time
import urllib.request
import wave


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--upstream',default='http://127.0.0.1:18092')
    p.add_argument('--variant',choices=('trained-quarter:female:zh','trained-quarter:male:zh','base:female:zh','base:male:zh'),default='trained-quarter:female:zh')
    p.add_argument('--skip-warmup',action='store_true',help='Use only after the same resident variant has qualified in a prior call')
    args=p.parse_args()
    if not Path('/srv/nas/.nas-online').is_file():raise RuntimeError('actual NAS required')
    if args.root.exists():raise RuntimeError('preserve the original test attempt')
    args.root.mkdir()
    def api(route,data=None,headers=None):
        req=urllib.request.Request(args.upstream+route,data=data,
            headers=headers or {'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
    state=api('/status')
    if not state.get('ready') or state.get('active'):raise RuntimeError('candidate not idle/ready')
    call=api('/calls',json.dumps({'variant':args.variant}).encode())
    call_id=call['id'];route='/calls/'+call_id
    (args.root/'IDENTITY.json').write_text(json.dumps(call,ensure_ascii=False,indent=2))
    events=[];audio=bytearray();done=threading.Event();poll_errors=[]
    def poll():
        after=0
        try:
            with (args.root/'events.jsonl').open('a',buffering=1) as log:
                while not done.is_set():
                    result=api(route+'/events?after='+str(after))
                    for event in result['events']:
                        events.append(event)
                        if event['type']=='audio':audio.extend(base64.b64decode(event['pcm']))
                        log.write(json.dumps({k:v for k,v in event.items() if k!='pcm'},ensure_ascii=False)+'\n')
                    after=result['after']
                    if result['closed']:return
        except Exception as e:poll_errors.append(type(e).__name__+': '+str(e))
    thread=threading.Thread(target=poll,daemon=True);thread.start()
    seq=0
    def send(pcm):
        nonlocal seq
        result=api(route+'/audio',pcm,{'Content-Type':'application/octet-stream','X-Audio-Seq':str(seq)})
        seq+=1
        return result
    failure=None
    try:
        warmup_packets=0
        if not args.skip_warmup:
            # Compilation/qualification is measured separately. Sparse silence
            # keeps the input alive without overflowing the native input queue.
            deadline=time.monotonic()+120;next_input=0
            while not any(e['type']=='metrics' for e in events):
                if any(e['type'] in ('error','closed') for e in events) or poll_errors:
                    raise RuntimeError('candidate warmup failed; see saved events')
                if time.monotonic()>deadline:raise TimeoutError('candidate warmup exceeded 120 seconds')
                if time.monotonic()>=next_input:
                    send(bytes(12800));next_input=time.monotonic()+5
                time.sleep(.1)
            # Drain the warmup before sending the same known two-turn fixture.
            warmup_packets=seq
            deadline=time.monotonic()+30
            while events[-1].get('type')!='closed':
                metrics=[e for e in events if e['type']=='metrics']
                if metrics and metrics[-1]['processed_seconds']>=seq*.4:break
                if time.monotonic()>deadline:raise TimeoutError('warmup queue did not drain')
                time.sleep(.1)
        with wave.open(str(args.input)) as wav:
            if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(16000,1,2):raise ValueError('fixture format')
            pcm=wav.readframes(wav.getnframes())
        pcm+=bytes(12*16000*2)
        pcm+=bytes((-len(pcm))%12800)
        start=time.monotonic()
        for offset in range(0,len(pcm),12800):
            if any(e['type'] in ('error','closed') for e in events):raise RuntimeError('candidate ended during fixture')
            target=start+offset/32000
            time.sleep(max(0,target-time.monotonic()))
            send(pcm[offset:offset+12800])
        deadline=time.monotonic()+30
        while True:
            metrics=[e for e in events if e['type']=='metrics']
            if metrics and metrics[-1]['processed_seconds']>=seq*.4:break
            if any(e['type'] in ('error','closed') for e in events):raise RuntimeError('candidate failed to drain')
            if time.monotonic()>deadline:raise TimeoutError('fixture backlog exceeded 30 seconds')
            time.sleep(.1)
    except Exception as e:failure=type(e).__name__+': '+str(e)
    finally:
        try:api(route+'/stop',b'')
        except Exception as e:poll_errors.append('stop: '+str(e))
        thread.join(timeout=25);done.set()
        with wave.open(str(args.root/'output.wav'),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(audio)
    # The event stream closes before asynchronous model/codec cleanup ends.
    # Confirm idle and inspect the final server log instead of treating a
    # successful stop POST as completed resource cleanup.
    final=api('/status');deadline=time.monotonic()+20
    while final.get('active')==call_id and time.monotonic()<deadline:
        time.sleep(.2);final=api('/status')
    server_folder=args.root.parent/'calls'/call_id
    saved_events=[json.loads(line) for line in (server_folder/'events.jsonl').read_text().splitlines() if line]
    summary_path=server_folder/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
    if final.get('active')==call_id:failure=failure or 'model cleanup exceeded 20 seconds'
    metrics=[e for e in events if e['type']=='metrics']
    speaking=[e for e in metrics if e.get('state') in ('S','B')]
    responses={}
    for e in events:
        if e['type']=='text_delta':
            key=str(e.get('event_id','unknown'));responses[key]=responses.get(key,'')+e.get('text','')
    result={'call_id':call_id,'fixture':str(args.input),'test_kind':'real_http_gpu_no_browser_no_listening',
        'sent_packets':seq,'processed_seconds':metrics[-1]['processed_seconds'] if metrics else 0,
        'responses':responses,'errors':[e for e in saved_events if e['type']=='error'],
        'failure':failure,'poll_errors':poll_errors,'output_seconds':len(audio)/48000,
        'speaking_windows':len(speaking),'speaking_mean_ms':sum(e['compute_ms'] for e in speaking)/len(speaking) if speaking else None,
        'speaking_max_ms':max((e['compute_ms'] for e in speaking),default=None),
        'peak_pending_seconds':max((e['pending_seconds'] for e in metrics),default=0),
        'final_status':final,'server_summary':summary,'warmup_packets':locals().get('warmup_packets')}
    (args.root/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if failure or poll_errors or result['errors']:raise SystemExit(1)


if __name__=='__main__':main()
