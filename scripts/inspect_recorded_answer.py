#!/usr/bin/env python3
"""Trace an existing answer's archived PCM to the bytes actually released.

No generation, service changes, input transcript conditioning or audio devices.
The independent CPU recognizer is diagnostic, not a listening-quality verdict.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import time
import uuid
import wave

from cineduplex.train.nas_transport import NAS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wav(pcm):
    out = io.BytesIO()
    with wave.open(out, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(pcm)
    return out.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=Path)
    p.add_argument('--prefix', required=True)
    p.add_argument('--receipt', required=True)
    p.add_argument('--event', choices=('evt-1', 'evt-2'), required=True)
    p.add_argument('--destination', required=True)
    args = p.parse_args()
    n = NAS(json.loads(args.config.read_text())['nas'])
    for name in (args.prefix, args.destination):
        if not name.startswith('output/'):
            raise ValueError('private NAS output paths required')
        n.path(name)
    if Path(args.receipt).name != args.receipt:
        raise ValueError('receipt filename required')
    dest = args.destination.rstrip('/')
    if n.stat(dest+'/RESULT.json') is not None:
        raise ValueError('already completed; reuse saved evidence')
    payload = n.read(args.prefix+'/'+args.receipt, timeout=15)
    if len(payload) > 2_000_000:
        raise ValueError('receipt budget')
    record = json.loads(payload)
    call = str(uuid.UUID(record['call']))
    folder = args.prefix+'/calls/'+call
    events = record['events']
    if record.get('error') or record['after'].get('active'):
        raise ValueError('completed replay required')
    summary = json.loads(n.read(folder+'/summary.json', timeout=15))
    if summary['reason'] != 'hangup':
        raise ValueError('complete uninterrupted archive required')
    size = n.stat(folder+'/output.pcm')
    if size is None or size > 6_000_000:
        raise ValueError('PCM budget')
    output = n.read(folder+'/output.pcm', timeout=15)
    order = [e['event_id'] for e in events if e['type'] == 'event_end']
    offset = 0
    checks = []
    selected = None
    # Code duration is independently tied to each archived native answer.
    for event_id in order:
        if event_id not in ('evt-1', 'evt-2'):
            raise ValueError('two-answer diagnostic only')
        codes = json.loads(n.read(folder+'/'+event_id+'-codes.json', timeout=15))
        length = len(codes['raw_codes'])*960*2
        pcm = output[offset:offset+length]
        if len(pcm) != length:
            raise ValueError('native code duration and archive differ')
        end = 0
        for e in events:
            if e['type'] != 'audio' or e.get('event_id') != event_id:
                continue
            begin = e['sample_offset']*2
            stop = begin+e['samples']*2
            if begin != end or not begin < stop <= length:
                raise ValueError('released audio has gaps, overlap or range error')
            encoded = base64.b64encode(pcm[begin:stop])
            if digest(encoded) != e['pcm_sha256'] or len(encoded) != e['encoded_pcm_bytes']:
                raise ValueError('released bytes differ from archived PCM')
            checks.append(dict(event_id=event_id, seq=e['seq'], start=begin, end=stop,
                               pcm_sha256=digest(pcm[begin:stop])))
            end = stop
        if end != length:
            raise ValueError('incomplete audio release')
        if event_id == args.event:
            selected = (pcm, codes)
        offset += length
    if offset != len(output) or selected is None:
        raise ValueError('archive accounting mismatch')
    pcm, codes = selected
    report = dict(at=time.time(), call=call, event_id=args.event,
                  input_receipt_sha256=digest(payload), archive_sha256=digest(output),
                  released_byte_checks=checks, model_text=codes['text'],
                  captions=[e['snapshot'] for e in events
                            if e['type']=='text_delta' and e.get('event_id')==args.event],
                  microphone=False, speaker=False, generation=False,
                  subjective_listening=False, quality_accepted=False, clips=[])

    def preserve(path, data):
        if n.stat(path) is None:
            return n.put(path, data)
        old=n.read(path, timeout=15)
        if old != data:
            raise ValueError('existing immutable artifact differs')
        return dict(path=path, sha256=digest(data), bytes=len(data))

    # Same samples, full context and a first-5-second excerpt. Never add the
    # expected text as a recognizer prompt, and never send ASR to generation.
    for label, data in (('full', pcm), ('opening', pcm[:5*48000])):
        path=dest+'/'+label+'.wav'
        pin=preserve(path, wav(data))
        sixteen=dest+'/'+label+'-16k.wav'
        if n.stat(sixteen) is None:
            converted=n.run(shlex.join(['docker','exec','cineduplex-web','ffmpeg',
                '-v','error','-i',n.path(path),'-ac','1','-ar','16000','-f','wav','pipe:1']),timeout=25)
            preserve(sixteen, converted)
        log=dest+'/'+label+'-asr.log'
        began=time.monotonic()
        reused=n.stat(log) is not None
        if reused:
            result=n.read(log,timeout=15)
        else:
            result=n.run(shlex.join(['timeout','90','docker','exec','cineduplex-web',
                'whisper-bounded','-m','/opt/drama-asr/ggml-base.bin','-f',n.path(sixteen),
                '-l','zh','-t','4','-ng','-np']),timeout=100)
            preserve(log,result)
        row=dict(label=label, seconds=len(data)/48000, wave=pin,
                 cpu_asr=result.decode(errors='replace'), asr_reused=reused,
                 operation_seconds=round(time.monotonic()-began,3))
        report['clips'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    print(n.put(dest+'/RESULT.json',json.dumps(report,ensure_ascii=False,indent=2).encode()),flush=True)


if __name__ == '__main__':
    main()
