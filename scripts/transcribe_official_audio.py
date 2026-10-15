#!/usr/bin/env python3
"""Independently transcribe private generated WAVs with the existing CPU ASR.

Uses recorded files only. Does not start a model service, access live devices,
or change the voice model's BF16 precision. All artifacts stay on the NAS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
import uuid

from cineduplex.train.nas_transport import NAS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--prefix',required=True)
    p.add_argument('--decode-call',required=True,type=uuid.UUID)
    a=p.parse_args()
    if '/' in a.prefix.removeprefix('output/') or not a.prefix.startswith('output/realtime-tp4-'):
        p.error('existing admitted root required')
    n=NAS(json.loads(Path(a.config).read_text())['nas']);pre=a.prefix+'/calls/'+str(a.decode_call)+'/'
    payload=n.read(pre+'official-audio.json',timeout=15)
    if len(payload)>1000000:raise RuntimeError('audio metadata budget')
    source=json.loads(payload)
    if source['microphone'] or source['speaker'] or source['released_audio_events']:
        raise RuntimeError('private recorded-file evidence required')
    destination='output/official-backend-20261013-v1/ASR-'+str(a.decode_call)+'.json'
    evidence_sha=hashlib.sha256(payload).hexdigest()
    if n.stat(destination) is not None:
        saved=json.loads(n.read(destination,timeout=15))
        if saved['audio_evidence_sha256']!=evidence_sha:raise RuntimeError('existing ASR evidence changed')
        print(json.dumps(saved,ensure_ascii=False));return
    results=[]
    for index,segment in enumerate(source['segments']):
        for audio in segment['audio']:
            name=audio['file']
            if Path(name).name!=name or not name.endswith('.wav'):raise RuntimeError('unsafe generated audio name')
            raw=n.read(pre+name,timeout=15)
            if len(raw)>4000000 or hashlib.sha256(raw).hexdigest()!=audio['sha256']:
                raise RuntimeError('actual generated waveform differs')
            rel=pre+name.removesuffix('.wav')+'-16k.wav'
            data=n.run(shlex.join(['docker','exec','cineduplex-web','ffmpeg','-v','error',
                '-i',n.path(pre+name),'-ac','1','-ar','16000','-f','wav','pipe:1']),timeout=25)
            if len(data)>3000000:raise RuntimeError('converted PCM budget')
            sha=hashlib.sha256(data).hexdigest()
            if n.stat(rel) is None:n.put(rel,data)
            elif hashlib.sha256(n.read(rel,timeout=15)).hexdigest()!=sha:
                raise RuntimeError('existing converted PCM differs')
            log_path=rel.removesuffix('.wav')+'-asr.log'
            began=time.monotonic()
            if n.stat(log_path) is None:
                text=n.run(shlex.join(['timeout','180','docker','exec','cineduplex-web',
                    'whisper-bounded','-m','/opt/drama-asr/ggml-base.bin','-f',n.path(rel),
                    '-l','zh','-t','4','-ng','-np']),timeout=190)
                n.put(log_path,text)
            else:text=n.read(log_path,timeout=15)
            results.append({'segment':index,'case':segment['case'],'mode':audio['mode'],
                'voice':source['voice'],'model_text':segment['text'],'asr':text.decode(errors='replace'),
                'wave_sha256':audio['sha256'],'input16k_sha256':sha,
                'samples':audio['samples'],'codes':len(segment['raw_codes']),
                'clipped_samples':audio['clipped_samples'],'seconds':time.monotonic()-began})
    record={'audio_evidence_sha256':evidence_sha,'reference_call':source['reference_call'],
        'decode_call':str(a.decode_call),'microphone':False,'speaker':False,
        'asr':'existing CPU whisper base, independent of generator','quality_accepted':False,
        'subjective_listening':False,'results':results}
    print(n.put(destination,json.dumps(record,ensure_ascii=False,indent=2).encode()))
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
