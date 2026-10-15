#!/usr/bin/env python3
"""Prepare a pinned, private decode of actual official-reference speech codes.

Preparation never starts GPU work. Apply with the existing idle-only updater,
then invoke the normal admitted call probe. No browser audio is emitted.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import uuid
import re

from cineduplex.train.nas_transport import NAS

BODY = '''    def codec_probe(self):
        import hashlib,importlib.util
        codec=self.app.codec
        try:
            source=self.app.root/REFERENCE_FILE
            payload=source.read_bytes()
            if len(payload)>6000000 or hashlib.sha256(payload).hexdigest()!=REFERENCE_SHA:
                raise RuntimeError('official generated-code evidence identity')
            reference=json.loads(payload)
            if not NORMAL_CALL and (not reference.get('ok') or not any(p.get('phase')=='load_same_weights' for p in reference['phases'])):
                raise RuntimeError('no completed same-weight reference')
            codec.select_voice(self.app.root,self.voice)
            official=None
            if OFFICIAL_DECODER:
                source=self.app.root.parents[1]/'output/official-backend-20261013-v1/source/third_party/Step-Audio2/token2wav.py'
                if hashlib.sha256(source.read_bytes()).hexdigest()!=DECODER_SHA:
                    raise RuntimeError('official decoder source identity')
                spec=importlib.util.spec_from_file_location('private_official_token2wav',str(source))
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                # Reuse the exact resident FP32 flow/vocoder and independently
                # verified prompt codes. No new weights or 496MB tokenizer read.
                official=module.Token2wav.__new__(module.Token2wav)
                official.float16=False;official.flow=codec.flow;official.hift=codec.hift
                codes,lens,mel,mel_lens,embedding=codec.prompt
                official.cache={'same-prompt':(codes,lens,embedding,mel,mel_lens)}
                voice_wave=self.app.root.parents[1]/codec.voice_specs[self.voice]['audio']
                # TorchAudio 2.10 delegates load() to an optional TorchCodec.
                # These pinned prompts are already verified mono PCM16 WAVs;
                # read those exact samples without another codec dependency.
                with wave.open(str(voice_wave)) as wav:
                    if (wav.getnchannels(),wav.getsampwidth())!=(1,2):
                        raise RuntimeError('voice prompt PCM identity')
                    rate=wav.getframerate()
                    audio=torch.from_numpy(np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768.).unsqueeze(0)
                if rate!=24000:audio=module.torchaudio.transforms.Resample(rate,24000)(audio)
                original_len=int(module.mel_spectrogram(audio).shape[-1])
                original_lens=torch.tensor([original_len],device=mel_lens.device,dtype=mel_lens.dtype)
                official.cache['original-length']=(codes,lens,embedding,mel,original_lens)
            segments=[];current=None
            def finish(reason):
                nonlocal current
                if current and current['raw_codes']:
                    current['end_reason']=reason;segments.append(current)
                current=None
            for event in reference.get('events',[]):
                case=event['case']
                if current and current['case']!=case:finish('next_recorded_question')
                if event['type']=='speaking_token':
                    if current is None:current={'case':case,'text_ids':[],'raw_codes':[]}
                    text=event.get('text_token');speech=event.get('stoken')
                    if type(text) is not int or type(speech) is not int:
                        raise RuntimeError('non-scalar native generated token')
                    current['text_ids'].append(text)
                    if 151696<=speech<158352:
                        if speech>=151696+6561:
                            raise RuntimeError('official generated token exceeds actual codec vocabulary')
                        current['raw_codes'].append(speech-151696)
                elif event['type']=='speaking_done':finish(event.get('reason','done'))
                elif event['type']=='state_change' and event.get('to')=='L':finish('native_listen_transition')
            finish('bounded_reference_end')
            if NORMAL_CALL:
                for item in reference['segments']:
                    original=self.app.root/item['file']
                    payload=original.read_bytes()
                    if hashlib.sha256(payload).hexdigest()!=item['sha256']:
                        raise RuntimeError('normal generated codes changed')
                    data=json.loads(payload)
                    codes=data['speech_codes'] if RECORDED_AB else data['raw_codes']
                    if (not RECORDED_AB and not data.get('native_model_codes')) or not codes or any(type(c) is not int or not 0<=c<6561 for c in codes):
                        raise RuntimeError('normal native codec range')
                    segments.append({'case':item['file'],'text':item['text'] if RECORDED_AB else data['text'],'raw_codes':codes,'end_reason':'archived_native_event'})
            if not segments:raise RuntimeError('reference produced no speech codes')
            if sum(len(s['raw_codes']) for s in segments)>2400:
                raise RuntimeError('total recorded-answer decode budget')
            record={'reference_call':REFERENCE_CALL,'reference_sha256':REFERENCE_SHA,
                'source_kind':'recorded-native-shadow-codes' if RECORDED_AB else 'normal-call-native-codes' if NORMAL_CALL else 'official-reference',
                'voice':self.voice,'microphone':False,'speaker':False,'released_audio_events':0,
                'quality_accepted':False,'segments':[]}
            timeline=None
            if TIMELINE_SOURCE:
                path=self.app.root.parents[1]/TIMELINE_SOURCE
                if hashlib.sha256(path.read_bytes()).hexdigest()!=TIMELINE_SHA:
                    raise RuntimeError('codec timeline source identity')
                spec=importlib.util.spec_from_file_location('private_codec_timeline',str(path))
                timeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(timeline)
                record['timeline_source_sha256']=TIMELINE_SHA
            lookahead=int(codec.flow.pre_lookahead_len)
            for index,segment in enumerate(segments):
                if self.closed.is_set():raise RuntimeError('private decode canceled')
                codes=segment['raw_codes']
                if not 1<=len(codes)<=1200 or len(segments)>16:
                    raise RuntimeError('recorded-answer decode budget')
                if not NORMAL_CALL:
                    segment['text']=self.app.fw.tokenizer.decode(segment['text_ids'],skip_special_tokens=True)
                codec.reset()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(20261009)
                    full_float=codec.complete(codes);full=codec.encode_pcm(full_float)
                    extra=[]
                    if official is not None:
                        for label in ['same-prompt','original-length']:
                            torch.manual_seed(20261009)
                            blob=official(codes,label)
                            with wave.open(io.BytesIO(blob)) as wav:
                                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,24000):
                                    raise RuntimeError('official decoder PCM format')
                                pcm=wav.readframes(wav.getnframes())
                            target=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.
                            if len(target)!=len(full_float):raise RuntimeError('official decoder sample count differs')
                            relative=float(np.linalg.norm(target-np.clip(full_float,-1,1))/max(np.linalg.norm(full_float),1e-8))
                            segment.setdefault('official_comparison',[]).append({'mode':label,
                                'relative_l2':relative,'source_sha256':DECODER_SHA,
                                'native_prompt_frames':int(codec.prompt[3].item()),
                                'official_prompt_frames':int(official.cache[label][4].item())})
                            extra.append(('official-'+label,pcm))
                def stream_parts(emit):
                    offset=0;block=36;pieces=[]
                    while len(codes)-offset>=block+lookahead:
                        pieces.append(emit(codes[offset:offset+block+lookahead],False))
                        offset+=block;block=40
                    if offset<len(codes):pieces.append(emit(codes[offset:],True))
                    return b''.join(pieces)
                codec.reset()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(20261009)
                    streamed=stream_parts(lambda c,last:codec.pcm(c,last=last))
                if official is not None:
                    official.mel_cache_len=8;official.source_cache_len=8*480
                    official.speech_window=codec.window
                    official.stream_lookahead_len=lookahead
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(20261009)
                        official.set_stream_cache('same-prompt')
                        original_stream=stream_parts(lambda c,last:official.stream(c,'same-prompt',last_chunk=last))
                    a=np.frombuffer(streamed,dtype='<i2').astype(np.float32)
                    b=np.frombuffer(original_stream,dtype='<i2').astype(np.float32)
                    if len(a)!=len(b):raise RuntimeError('official stream sample count differs')
                    segment['official_stream_relative_l2']=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-8))
                    extra.append(('official-stream',original_stream))
                if timeline is not None:
                    timeline.install(codec)
                    try:
                        codec.reset()
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            torch.manual_seed(20261009)
                            corrected=stream_parts(lambda c,last:codec.pcm(c,last=last))
                        segment['absolute_noise_trace']=codec._absolute_noise_trace
                        extra.append(('absolute-noise-stream',corrected))
                    finally:timeline.uninstall(codec)
                segment['audio']=[]
                for mode,pcm in [('full',full),('stream',streamed)]+extra:
                    name=f'official-{index:02d}-{mode}.wav'
                    with wave.open(str(self.folder/name),'wb') as wav:
                        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm)
                    samples=np.frombuffer(pcm,dtype='<i2')
                    segment['audio'].append({'mode':mode,'file':name,'samples':len(samples),
                        'sha256':hashlib.sha256((self.folder/name).read_bytes()).hexdigest(),
                        'clipped_samples':int(np.sum(np.abs(samples.astype(np.int32))>=32767))})
                record['segments'].append(segment)
            atomic(self.folder/'official-audio.json',record)
            atomic(self.folder/'probe.json',{'kind':'official-code-decode','quality_accepted':False,
                'segments':len(segments),'released_audio_events':0})
        except Exception as exc:self.emit({'type':'error','message':str(exc)[:400]})
        finally:
            try:codec.reset()
            finally:
                self.closed.set()
                try:
                    self.emit({'type':'closed','reason':'private_official_decode_complete'})
                    self.archive.close()
                finally:
                    with self.app.lock:
                        if self.app.active is self:self.app.active=None
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--prefix',required=True)
    p.add_argument('--reference-call',required=True,type=uuid.UUID)
    p.add_argument('--tp-version',required=True,type=int)
    p.add_argument('--call-version',required=True,type=int)
    p.add_argument('--official-decoder',action='store_true')
    p.add_argument('--noise-timeline',action='store_true')
    p.add_argument('--normal-call',action='store_true',help='Decode two immutable native code archives from a completed normal call')
    p.add_argument('--normal-event',help='One finished native event, including a closed call rejected by output consistency review')
    p.add_argument('--recorded-ab',action='store_true',help='Decode the two frozen native-shadow answers with the corrected live codec framing')
    a=p.parse_args()
    if a.normal_event and not a.normal_call:p.error('--normal-event requires --normal-call')
    if '/' in a.prefix.removeprefix('output/') or not a.prefix.startswith('output/realtime-tp4-'):
        p.error('existing admitted root required')
    n=NAS(json.loads(Path(a.config).read_text())['nas']);pre=a.prefix+'/'
    reference=str(a.reference_call)
    reference_file='calls/'+reference+'/official-reference.json'
    if a.recorded_ab:
        if a.normal_call:raise ValueError('select one native recording kind')
        result=json.loads(n.read(pre+'calls/'+reference+'/recorded-ab/RESULT.json',timeout=15))
        if not result.get('completed') or result['errors'] or len(result['cases'])!=2:
            raise ValueError('completed two-answer native comparison required')
        segments=[]
        for row in result['cases']:
            item=row['speech_code_file'];file=item['path'].removeprefix(pre)
            if not file.startswith('calls/'+reference+'/recorded-ab/'):
                raise ValueError('original native code ownership')
            blob=n.read(item['path'],timeout=15)
            if hashlib.sha256(blob).hexdigest()!=item['sha256']:raise ValueError('original native code pin')
            segments.append({'file':file,'sha256':item['sha256'],'text':row['generated_text']})
        raw=json.dumps({'segments':segments}).encode()
        reference_file=f'NORMAL-CODEC-{a.tp_version}-INPUT.json';n.put(pre+reference_file,raw)
    elif a.normal_call:
        events=n.read(pre+'calls/'+reference+'/events.jsonl',timeout=15)
        records=[json.loads(line) for line in events.splitlines()]
        if not any(e['type']=='closed' for e in records) or any(e['type']=='error' for e in records):
            raise RuntimeError('completed normal call without runtime errors required')
        if a.normal_event:
            if not re.fullmatch(r'evt-[1-9][0-9]*',a.normal_event):raise ValueError('native event identity')
            finished=[e for e in records if e['type']=='generation_end']
            index=int(a.normal_event.split('-')[1])-1
            if index>=len(finished) or finished[index].get('reason')!='eos':
                raise RuntimeError('selected native generation must finish naturally before private decode')
            ids=[a.normal_event]
        else:
            ids=[e['event_id'] for e in records if e['type']=='event_end']
            if ids!=['evt-1','evt-2']:raise RuntimeError('two completed native answers required')
        segments=[]
        for event in ids:
            file='calls/'+reference+'/'+event+'-codes.json'
            blob=n.read(pre+file,timeout=15)
            if len(blob)>100000:raise RuntimeError('native code archive budget')
            segments.append({'file':file,'sha256':hashlib.sha256(blob).hexdigest()})
        raw=json.dumps({'segments':segments}).encode()
        reference_file=f'NORMAL-CODEC-{a.tp_version}-INPUT.json'
        if n.stat(pre+reference_file) is None:
            n.put(pre+reference_file,raw)
        elif n.read(pre+reference_file,timeout=15)!=raw:
            raise RuntimeError('existing immutable native code manifest differs')
    else:
        raw=n.read(pre+reference_file,timeout=15)
        if len(raw)>6000000 or not json.loads(raw).get('ok'):
            raise RuntimeError('completed bounded official reference required')
    original=n.read(pre+'call-runtime-v53.py',timeout=15).decode()
    if hashlib.sha256(original.encode()).hexdigest()!='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214':
        raise RuntimeError('original call source changed')
    tree=ast.parse(original)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Call')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='codec_probe')
    lines=original.splitlines(True)
    source=f'REFERENCE_CALL={reference!r}\nREFERENCE_SHA={hashlib.sha256(raw).hexdigest()!r}\n'
    source+=f'REFERENCE_FILE={reference_file!r}\nNORMAL_CALL={bool(a.normal_call or a.recorded_ab)!r}\nRECORDED_AB={a.recorded_ab!r}\n'
    decoder=n.read('output/official-backend-20261013-v1/source/third_party/Step-Audio2/token2wav.py',timeout=15) if a.official_decoder else b''
    source+=f'OFFICIAL_DECODER={a.official_decoder!r}\nDECODER_SHA={hashlib.sha256(decoder).hexdigest()!r}\n'
    timeline_source='output/official-backend-20261013-v1/codec_stream_timeline_v1.py' if a.noise_timeline else ''
    timeline_blob=n.read(timeline_source,timeout=15) if timeline_source else b''
    source+=f'TIMELINE_SOURCE={timeline_source!r}\nTIMELINE_SHA={hashlib.sha256(timeline_blob).hexdigest()!r}\n'
    source+=''.join(lines[:method.lineno-1])+BODY+''.join(lines[method.end_lineno:])
    ast.parse(source);call_name=f'call-runtime-v{a.call_version}.py'
    call_pin=n.put(pre+call_name,source.encode())
    tp=n.read(pre+'tp-runtime-v73.py',timeout=15).decode()
    if hashlib.sha256(tp.encode()).hexdigest()!='2d9405253fa14781c83fd5db6868e403f8a9b7b8d10f3d9d167f3c2e5c7ecb6a':
        raise RuntimeError('original TP source changed')
    tp=tp.replace("CALL_RUNTIME_NAME='call-runtime-v53.py'",'CALL_RUNTIME_NAME='+repr(call_name)).replace(
        "CALL_RUNTIME_SHA='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214'",
        'CALL_RUNTIME_SHA='+repr(call_pin['sha256']))
    ast.parse(tp)
    pins={'call':call_pin,'tp':n.put(pre+f'tp-runtime-v{a.tp_version}.py',tp.encode()),
          'reference_call':reference,'quality_accepted':False}
    print(n.put('output/official-backend-20261013-v1/AUDIO-'+str(a.tp_version)+'-PINS.json',json.dumps(pins).encode()))
    print(json.dumps(pins))


if __name__=='__main__':main()
