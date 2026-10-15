#!/usr/bin/env python3
"""Prepare private recorded-file checks using the existing live guardrail client.

No credentials are copied, no audio is published, and preparation starts no
GPU work. The existing admitted call owns cancellation and all review IDs.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import uuid

from cineduplex.train.nas_transport import NAS

BODY='''    def codec_probe(self):
        import hashlib
        try:
            import re,unicodedata
            from difflib import SequenceMatcher
            def diagnostic_similarity(expected,actual):
                def normalize(text):
                    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())
                return {'ratio':SequenceMatcher(None,normalize(expected),normalize(actual),autojunk=False).ratio(),
                    'scope':'diagnostic only; does not replace the deployed safety gate'}
            client=self.app.guardrail_client
            original_rpc=client.rpc
            slice_trace=[]
            if SLICE_ORDER_TRACE:
                def observed_rpc(action,*args,**kwargs):
                    payload=original_rpc(action,*args,**kwargs)
                    if action=='VoiceModerationResult' and payload.get('Code')==200:
                        parts=payload.get('Data',{}).get('SliceDetails',[])
                        slice_trace.append([{key:part[key] for key in
                            ('StartTime','EndTime','StartTimestamp','EndTimestamp','Text','RiskLevel','Labels')
                            if key in part} for part in parts])
                    return payload
                client.rpc=observed_rpc
            status=client.config.status()
            if not all(status.get(k) for k in ['enabled','input_audio','input_semantic','output_text','output_audio']):
                raise RuntimeError('existing complete protection required')
            record={'microphone':False,'speaker':False,'released_audio_events':0,
                'quality_accepted':False,'kind':'independent actual cloud transcript and safety','cases':[]}
            for index,case in enumerate(TEST_CASES):
                blob=(self.app.root.parents[1]/case['path']).read_bytes()
                if hashlib.sha256(blob).hexdigest()!=case['sha256']:
                    raise RuntimeError('recorded cloud-check waveform identity')
                with wave.open(io.BytesIO(blob)) as wav:
                    if (wav.getnchannels(),wav.getsampwidth())!=(1,2):raise RuntimeError('recorded PCM format')
                    rate=wav.getframerate();pcm=wav.readframes(wav.getnframes())
                if len(pcm)>rate*2*TEST_AUDIO_MAX_SECONDS:raise RuntimeError('recorded cloud-check budget')
                try:
                    audio=client.check_audio(pcm,sample_rate=rate,data_id=self.id+'-reference-'+str(index),cancelled=self.closed)
                finally:
                    if SLICE_ORDER_TRACE:
                        atomic(self.folder/'slice-order-trace.json',{'case':case,'responses':slice_trace,
                            'microphone':False,'speaker':False,'generation_changed':False})
                semantic=client.check_text(audio.transcript,direction=case['direction'],session_id=self.id,
                    chat_id='reference-'+str(index)+'-spoken') if audio.transcript.strip() else None
                expected=client.check_text(case['expected'],direction=case['direction'],session_id=self.id,
                    chat_id='reference-'+str(index)+'-expected')
                record['cases'].append({**case,'audio_review':audio.record(),
                    'asr':audio.transcript,'semantic_review':semantic.record() if semantic else None,
                    'expected_text_review':expected.record(),
                    'alignment':diagnostic_similarity(case['expected'],audio.transcript)})
                atomic(self.folder/'official-cloud-check.json',record)
            atomic(self.folder/'probe.json',{'kind':'official-cloud-check','cases':len(TEST_CASES),
                'quality_accepted':False,'released_audio_events':0})
        except Exception as exc:self.emit({'type':'error','message':str(exc)[:400]})
        finally:
            if 'original_rpc' in locals():client.rpc=original_rpc
            self.closed.set()
            try:
                self.emit({'type':'closed','reason':'private_reference_review_complete'})
                self.archive.close()
            finally:
                with self.app.lock:
                    if self.app.active is self:self.app.active=None
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--prefix',required=True)
    p.add_argument('--decode-call',type=uuid.UUID)
    p.add_argument('--compare-voice-call',type=uuid.UUID,help='Second private decode of exactly the same native codes using the other existing voice')
    p.add_argument('--chronological-review',action='store_true')
    p.add_argument('--acoustic-ab-call',type=uuid.UUID)
    p.add_argument('--context-ab-call',type=uuid.UUID)
    p.add_argument('--context-base-only',action='store_true')
    p.add_argument('--travel-comparison',action='store_true')
    p.add_argument('--output-contract-comparison',action='store_true')
    p.add_argument('--resume-cloud-call',type=uuid.UUID)
    p.add_argument('--answer-only',action='store_true',help='Cross-check complete and streaming generated answers; reuse prior input evidence')
    p.add_argument('--max-audio-seconds',type=int,choices=(16,48),default=16)
    p.add_argument('--slice-order-trace',action='store_true',help='Private original response slice timing; never store signed URLs or credentials')
    p.add_argument('--decode-mode',choices=('official-same-prompt','official-stream'))
    p.add_argument('--tp-version',required=True,type=int);p.add_argument('--call-version',required=True,type=int)
    a=p.parse_args()
    if sum(bool(x) for x in (a.decode_call,a.acoustic_ab_call,a.context_ab_call))!=1:p.error('select one existing recorded comparison call')
    if a.decode_mode and not a.decode_call:p.error('decode-mode requires decode-call')
    if a.compare_voice_call and not (a.decode_call and a.answer_only):p.error('voice comparison requires answer-only decode-call')
    if a.output_contract_comparison and not a.context_ab_call:p.error('output contract requires existing comparison call')
    if a.travel_comparison and not a.context_ab_call:p.error('travel comparison requires context call')
    if a.context_base_only and not a.context_ab_call:p.error('base-only requires context comparison')
    context_call=a.context_ab_call
    if context_call:a.acoustic_ab_call=context_call
    if '/' in a.prefix.removeprefix('output/') or not a.prefix.startswith('output/realtime-tp4-'):
        p.error('existing admitted root required')
    n=NAS(json.loads(Path(a.config).read_text())['nas']);pre=a.prefix+'/'
    record=json.loads(n.read(pre+'calls/'+str(a.acoustic_ab_call or a.decode_call)+
        ('/recorded-ab/RESULT.json' if a.acoustic_ab_call else '/official-audio.json'),timeout=15))
    if record.get('microphone') or record.get('speaker') or record.get('released_audio_events') or record.get('public_audio_released'):
        raise RuntimeError('private recorded-file comparison required')
    repair_only=bool(context_call and record.get('context_repair_only'))
    if a.acoustic_ab_call and (not record.get('completed') or record['errors'] or len(record['cases'])!=(2 if repair_only else 8 if a.travel_comparison else 4 if context_call else 6)):
        raise RuntimeError('completed bounded recorded A/B required')
    if context_call:
        expected={(arm,q) for arm in ('original','context_refresh') for q in ('repeat-q1','repeat-q2')}
        if repair_only:expected={('context_refresh',q) for q in ('repeat-q1','repeat-q2')}
        base_expected={(arm,q) for arm in ('context_refresh','context_refresh_base') for q in ('repeat-q1','repeat-q2')}
        if a.travel_comparison:
            expected={(arm,f'travel-q{i}') for arm in ('original','context_refresh') for i in range(1,5)}
            base_expected=expected
        if a.output_contract_comparison:
            expected={('original_output',f'travel-q{i}') for i in range(1,5)};base_expected=expected
            if not record.get('output_contract_validation',{}).get('speech_id_masked'):raise RuntimeError('actual output contract validation missing')
        if {(r['arm'],r['case']) for r in record['cases']} not in (expected,base_expected) or not (record.get('context_policy') or a.output_contract_comparison):
            raise RuntimeError('context comparison identities differ')
    if not a.acoustic_ab_call and not 1<=len(record['segments'])<=(2 if a.answer_only else 1):
        raise RuntimeError('bounded recorded answers required')
    cases=[] if a.answer_only or a.acoustic_ab_call else [{'id':'original-question','direction':'input',
        'path':'output/realtime-tp4-20261013-after-dsv41-v1/morning-asr/repeat-q1.wav',
        'sha256':'e1fddda8ce7a43d04df525e9d43f2c4926669d18f6d64330f1d2954c5c3c50ef',
        'expected':'介绍一下中国天安门'}]
    for index,segment in enumerate(record.get('segments',[])):
        for mode in (('official-same-prompt','official-stream') if a.answer_only else ('official-same-prompt',)):
            if a.decode_mode and mode!=a.decode_mode:continue
            waveform=next(x for x in segment['audio'] if x['mode']==mode)
            if Path(waveform['file']).name!=waveform['file']:
                raise RuntimeError('unsafe recorded waveform name')
            cases.append({'id':str(index)+'-'+mode,'direction':'output',
                'path':pre+'calls/'+str(a.decode_call)+'/'+waveform['file'],
                'sha256':waveform['sha256'],'expected':segment['text']})
    if a.compare_voice_call:
        other=json.loads(n.read(pre+'calls/'+str(a.compare_voice_call)+'/official-audio.json',timeout=15))
        if (other.get('microphone') or other.get('speaker') or other.get('released_audio_events')
                or {record.get('voice'),other.get('voice')}!={'female','male'}
                or len(other['segments'])!=len(record['segments'])):
            raise ValueError('private existing voice pair required')
        for index,(left,right) in enumerate(zip(record['segments'],other['segments'])):
            if any(left[key]!=right[key] for key in ('case','text','raw_codes')):
                raise ValueError('voice comparison must hold exact native codes and text fixed')
            for mode in ('official-same-prompt','official-stream'):
                if a.decode_mode and mode!=a.decode_mode:continue
                waveform=next(x for x in right['audio'] if x['mode']==mode)
                if Path(waveform['file']).name!=waveform['file']:raise ValueError('unsafe voice waveform')
                cases.append({'id':other['voice']+'-'+str(index)+'-'+mode,'direction':'output',
                    'path':pre+'calls/'+str(a.compare_voice_call)+'/'+waveform['file'],
                    'sha256':waveform['sha256'],'expected':right['text']})
    if a.acoustic_ab_call:
        for row in record['cases']:
            if a.context_base_only and row['arm']!='context_refresh_base':continue
            if a.output_contract_comparison and row['case'] not in ('travel-q3','travel-q4'):continue
            if 'wav' not in row:continue
            wav=row['wav']
            if not wav['path'].startswith(pre+'calls/'+str(a.acoustic_ab_call)+'/recorded-ab/'):
                raise RuntimeError('acoustic waveform outside original call')
            cases.append({'id':row['arm']+'-'+row['case'],'direction':'output',
                'path':wav['path'],'sha256':wav['sha256'],'expected':row['generated_text']})
        if not cases:raise RuntimeError('no generated acoustic waveforms')
    if a.resume_cloud_call:
        previous=json.loads(n.read(pre+'calls/'+str(a.resume_cloud_call)+'/official-cloud-check.json',timeout=15))
        checked={(r['id'],r['path'],r['sha256'],r['expected']) for r in previous['cases']}
        valid={(r['id'],r['path'],r['sha256'],r['expected']) for r in cases}
        if not checked<=valid:raise RuntimeError('resume evidence does not belong to this exact comparison')
        cases=[r for r in cases if (r['id'],r['path'],r['sha256'],r['expected']) not in checked]
        if not cases:raise RuntimeError('all cases already checked; no duplicate cloud call')
    original=n.read(pre+'call-runtime-v53.py',timeout=15).decode()
    if hashlib.sha256(original.encode()).hexdigest()!='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214':
        raise RuntimeError('original call source differs')
    cls=next(x for x in ast.parse(original).body if isinstance(x,ast.ClassDef) and x.name=='Call')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='codec_probe')
    lines=original.splitlines(True)
    source='SLICE_ORDER_TRACE='+repr(a.slice_order_trace)+'\nTEST_AUDIO_MAX_SECONDS='+repr(48 if a.travel_comparison or a.output_contract_comparison else a.max_audio_seconds)+'\nTEST_CASES='+repr(cases)+'\n'+''.join(lines[:method.lineno-1])+BODY+''.join(lines[method.end_lineno:])
    if a.chronological_review:source=f'# Chronological private review binding {a.tp_version}\n'+source
    ast.parse(source);name=f'call-runtime-v{a.call_version}.py';call=n.put(pre+name,source.encode())
    tp=n.read(pre+'tp-runtime-v73.py',timeout=15).decode()
    if hashlib.sha256(tp.encode()).hexdigest()!='2d9405253fa14781c83fd5db6868e403f8a9b7b8d10f3d9d167f3c2e5c7ecb6a':
        raise RuntimeError('original TP source differs')
    tp=tp.replace("CALL_RUNTIME_NAME='call-runtime-v53.py'",'CALL_RUNTIME_NAME='+repr(name)).replace(
        "CALL_RUNTIME_SHA='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214'",
        'CALL_RUNTIME_SHA='+repr(call['sha256']))
    guard_pin=None
    if a.chronological_review:
        from prepare_chronological_guardrail import prepare
        tp,guard_pin=prepare(n,pre,a.tp_version,tp)
    ast.parse(tp)
    pins={'call':call,'tp':n.put(pre+f'tp-runtime-v{a.tp_version}.py',tp.encode()),
        'quality_accepted':False,'new_credentials':False,'cases':cases,'guardrail_bundle':guard_pin}
    print(n.put('output/official-backend-20261013-v1/CLOUD-'+str(a.tp_version)+'-PINS.json',json.dumps(pins).encode()))
    print(json.dumps({k:v for k,v in pins.items() if k!='cases'}))


if __name__=='__main__':main()
