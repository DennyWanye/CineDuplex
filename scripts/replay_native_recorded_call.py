#!/usr/bin/env python3
"""Replay pinned reviewer PCM through the admitted native call API.

This diagnostic has no browser, microphone or playback device. It keeps the
normal input/output safety path and never supplies a transcript to generation.
It submits once, records the returned identity, and stops only its own call.
"""
import argparse
import hashlib
import io
import json
import shlex
import time
import uuid
import wave
from pathlib import Path

from cineduplex.cluster.guard import guard
from cineduplex.train.nas_transport import NAS

TP73 = '2d9405253fa14781c83fd5db6868e403f8a9b7b8d10f3d9d167f3c2e5c7ecb6a'
CALL53 = '472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214'


def response_outcomes(events):
    """Account for real user cancellations without hiding missing replies/leaks."""
    starts = {e['event_id']: e for e in events if e['type'] == 'event_start'
              and e.get('event_kind') == 'response'}
    completed = {e['event_id'] for e in events if e['type'] == 'event_end'
                 and not e.get('interrupt') and e['event_id'] in starts}
    onsets = {(e.get('input_seq'), e.get('revision')) for e in events
              if e['type'] == 'ingress_speech'}
    cancelled, leaks = {}, []
    for eid, start in starts.items():
        if eid in completed:
            continue
        stops = [e for e in events if e['type'] == 'stop_audio'
                 and e.get('reason') == 'ingress_confirmed_speech'
                 and e.get('epoch', 0) > start['epoch'] and e['seq'] > start['seq']
                 and (e.get('input_seq'), e.get('ingress_revision')) in onsets]
        if not stops:
            continue
        stop = min(stops, key=lambda e: e['seq'])
        cancelled[eid] = stop['seq']
        leaks += [e['seq'] for e in events if e.get('event_id') == eid
                  and e['seq'] > stop['seq'] and e['type'] in
                  ('audio', 'text_delta', 'event_end', 'audio_end')]
    latest = max(starts, key=lambda eid: starts[eid]['epoch']) if starts else None
    return {'started': len(starts), 'completed': sorted(completed),
            'confirmed_user_cancelled': cancelled, 'stale_release_sequences': leaks,
            'latest_reply_complete': latest in completed,
            'accounted': len(completed) + len(cancelled)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=Path)
    p.add_argument('--prefix', required=True)
    p.add_argument('--request', required=True)
    p.add_argument('--record', required=True)
    p.add_argument('--runtime-pins', help='Explicit immutable normal-call candidate pins')
    p.add_argument('--settling-seconds',type=int,choices=(24,48),default=24)
    p.add_argument('--variant',choices=('trained-quarter','base'),default='trained-quarter',
                   help='Explicit same-resident-weight causal control; no new model')
    a = p.parse_args()
    if Path(a.record).name != a.record or not a.record.endswith('.json'):
        raise ValueError('immutable receipt filename required')
    cfg = json.loads(a.config.read_text())
    nas = NAS(cfg['nas'])
    spec_raw = nas.read(a.request, timeout=15)
    spec = json.loads(spec_raw)
    continuous = spec.get('archived_continuous', False)
    allow_cancellation = spec.get('allow_confirmed_input_cancellation', False)
    if not isinstance(continuous, bool):
        raise ValueError('continuous archive flag must be boolean')
    if not isinstance(allow_cancellation, bool) or (allow_cancellation and not continuous):
        raise ValueError('cancellation accounting is only for original continuous calls')
    if continuous:
        if not a.runtime_pins or len(spec['cases']) != 1:
            raise ValueError('one pinned original continuous call required')
        case = spec['cases'][0]
        parts = Path(case['path']).parts
        if (len(parts) != 5 or parts[0] != 'output'
                or parts[1] != 'realtime-tp4-20261013-v13' or parts[2] != 'calls'
                or parts[4] != 'input.pcm' or case.get('format') != 'pcm16-16000-mono'):
            raise ValueError('original reviewer PCM archive required')
        source_call = str(uuid.UUID(parts[3]))
        summary_pin = spec['source_summary']
        if summary_pin['path'] != str(Path(case['path']).with_name('summary.json')):
            raise ValueError('original archive summary path differs')
        summary_raw = nas.read(summary_pin['path'], timeout=15)
        if hashlib.sha256(summary_raw).hexdigest() != summary_pin['sha256']:
            raise ValueError('original archive summary changed')
        source_summary = json.loads(summary_raw)
        if source_summary['id'] != source_call:
            raise ValueError('original archive call identity differs')
        expected_answers = spec['expected_min_answers']
        if isinstance(expected_answers, bool) or not isinstance(expected_answers, int) or not 1 <= expected_answers <= 4:
            raise ValueError('bounded expected original answers required')
    else:
        expected_answers = len(spec['cases'])
    if spec.get('recognized_text_input') or any('recognized_text' in c for c in spec['cases']):
        raise ValueError('recorded audio only; no transcript conditioning')
    if not 1 <= len(spec['cases']) <= (4 if a.runtime_pins else 2):
        raise ValueError('bounded causal replay case budget')
    packets = []
    inputs = []
    for case in spec['cases']:
        raw = nas.read(case['path'], timeout=15)
        if len(raw) > (6_000_000 if continuous else 512000) or hashlib.sha256(raw).hexdigest() != case['sha256']:
            raise ValueError('original recording changed')
        if continuous:
            pcm = raw
            if len(pcm) % 2 or abs(len(pcm)/32000-source_summary['received_seconds']) > 1e-6:
                raise ValueError('original PCM duration and receipt differ')
        else:
            with wave.open(io.BytesIO(raw)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                    raise ValueError('original mono PCM16/16k required')
                pcm = wav.readframes(wav.getnframes())
        inputs.append({'id': case['id'], 'sha256': case['sha256'],
                       'start_sample': len(packets) * 6400,
                       'samples': len(pcm) // 2})
        packets.extend(pcm[i:i + 12800].ljust(12800, b'\0') for i in range(0, len(pcm), 12800))
        packets.extend([bytes(12800)] * (a.settling_seconds*5//2))  # fixed diagnostic settling tail
    if len(packets) > (600 if a.runtime_pins else 170):
        raise ValueError('recorded replay duration budget')
    destination = a.prefix.rstrip('/') + '/' + a.record
    if nas.stat(destination) is not None:
        raise ValueError('existing receipt; do not repeat an unknown submission')
    guard(cfg['cluster'], heartbeat=True)

    def api(route, data=None, headers=()):
        cmd = ['curl', '-fsS', '--max-time', '20']
        if data is not None:
            cmd += ['-X', 'POST', '--data-binary', '@-']
        for h in headers:
            cmd += ['-H', h]
        return json.loads(nas.run(shlex.join(cmd + ['http://127.0.0.1:18092' + route]),
                                  data=data, timeout=25))

    expected_tp=TP73
    expected_call=CALL53
    candidate=None
    if a.runtime_pins:
        candidate=json.loads(nas.read(a.runtime_pins,timeout=15))
        if candidate.get('baseline')!=TP73 or candidate.get('safety_changed') or candidate.get('weights_changed'):
            raise ValueError('normal native candidate contract differs')
        if candidate.get('call_runtime')!='original Call53':
            call_kind=candidate.get('call_runtime')
            allowed_kind=(call_kind=='Call53 with bounded idle progress fix' or
                (call_kind=='Call53 with idle fix and direct-answer instruction' and candidate.get('direct_answer_policy') is True))
            if (not allowed_kind
                or candidate.get('call_base_sha256')!=CALL53 or not candidate.get('idle_grace_after_processing')):
                raise ValueError('normal call changes are not the bounded idle fix')
            call_pin=candidate['call'];call_source=nas.read(call_pin['path'],timeout=15)
            if not call_pin['path'].startswith(a.prefix.rstrip('/')+'/') or hashlib.sha256(call_source).hexdigest()!=call_pin['sha256']:
                raise ValueError('normal call source identity differs')
            expected_call=call_pin['sha256']
        item=candidate['tp']
        if not item['path'].startswith(a.prefix.rstrip('/')+'/') or hashlib.sha256(nas.read(item['path'],timeout=15)).hexdigest()!=item['sha256']:
            raise ValueError('candidate source identity')
        expected_tp=item['sha256']
    before = api('/status')
    safety = before.get('guardrail', {})
    if not (before.get('ready') and before.get('active') is None
            and before.get('tp_runtime_sha256') == expected_tp
            and before.get('decode_precision') == 'bf16'
            and before.get('quantization') == 'none'
            and all(safety.get(k) for k in ('enabled', 'input_audio', 'input_semantic', 'output_text', 'output_audio'))):
        raise RuntimeError('original native ready/idle identity and full safety required')
    intent = {'at': time.time(), 'before': before, 'inputs': inputs,
              'request_sha256': hashlib.sha256(spec_raw).hexdigest(),
              'microphone': False, 'speaker': False, 'subjective_listening': False,
              'scope': 'recorded fixture isolation with digital settling; not original call timing',
              'settling_seconds':a.settling_seconds,'runtime_pins':candidate}
    if continuous:
        intent.update(scope='original continuous reviewer PCM; no inserted pauses between utterances; digital settling only after archive end',
                      source_original_call=source_call, source_summary=summary_pin,
                      expected_min_answers=expected_answers, archived_continuous=True,
                      allow_confirmed_input_cancellation=allow_cancellation)
    intent['variant']=a.variant
    nas.put(destination.removesuffix('.json') + '-INTENT.json', json.dumps(intent).encode())
    call = None
    events = []
    cursor = 0
    error = None
    try:
        # Never retry this POST after a transport failure. Reconcile /status.
        started = api('/calls', json.dumps({'variant':a.variant+':female:zh'}).encode(),
                      ('Content-Type: application/json',))
        call = str(uuid.UUID(started['id']))
        nas.put(destination.removesuffix('.json') + '-CALL.json', json.dumps(started).encode())
        if started.get('guardrail_runtime_sha256') != expected_call:
            raise RuntimeError('normal native call identity differs')
        if candidate and candidate.get('guardrail_bundle'):
            live=api('/status')
            if live.get('guardrail_bundle_sha256')!=candidate['guardrail_bundle']['sha256']:
                raise RuntimeError('normal candidate safety bundle identity differs')

        def collect():
            nonlocal cursor
            batch = api('/calls/' + call + '/events?after=' + str(cursor))
            cursor = batch['after']
            for e in batch['events']:
                if 'pcm' in e:
                    e = {**e, 'pcm_sha256': hashlib.sha256(e['pcm'].encode()).hexdigest(),
                         'encoded_pcm_bytes': len(e['pcm'])}
                    del e['pcm']
                events.append(e)
            if len(events) > 6000:
                raise RuntimeError('event budget')
            if any(e.get('type') == 'error' for e in batch['events']):
                raise RuntimeError('native/safety error; preserved in own call archive')
            return batch.get('closed', False)

        begun = time.monotonic()
        for seq, pcm in enumerate(packets):
            delay = begun + seq * .4 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            api('/calls/' + call + '/audio', pcm,
                ('Content-Type: application/octet-stream', 'X-Audio-Seq: ' + str(seq)))
            if seq % 5 == 4 and collect():
                raise RuntimeError('native call ended during recorded input')
        deadline = time.monotonic() + 35
        target = len(packets) * .4
        while time.monotonic() < deadline:
            processed = max((e.get('processed_seconds', 0) for e in events if e.get('type') == 'metrics'), default=0)
            answers=sum(e.get('type')=='event_end' and not e.get('interrupt') for e in events)
            # API counters are rounded decimals; e.g. 538*.4 is
            # 215.20000000000002, while processed_seconds is 215.2.
            # That representational difference must not keep a completed
            # replay waiting until the normal no-input timeout.
            accounted = response_outcomes(events) if allow_cancellation else None
            terminal = (accounted['started'] == expected_answers
                        and accounted['accounted'] == expected_answers
                        and accounted['latest_reply_complete'] and not accounted['stale_release_sequences']) if accounted else answers >= expected_answers
            if processed + 1e-6 >= target and terminal:
                break
            if collect():
                break
        answers=sum(e.get('type')=='event_end' and not e.get('interrupt') for e in events)
        accounted = response_outcomes(events) if allow_cancellation else None
        if accounted:
            if (accounted['started'] != expected_answers or accounted['accounted'] != expected_answers
                    or not accounted['latest_reply_complete'] or accounted['stale_release_sequences']):
                raise RuntimeError('incomplete/unsafe continuous response outcomes: '+json.dumps(accounted))
        elif answers<expected_answers:
            raise RuntimeError('incomplete reviewed answers: '+str(answers)+'/'+str(expected_answers))
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:400]
    finally:
        if call is not None:
            try:
                api('/calls/' + call + '/stop', b'{}', ('Content-Type: application/json',))
                end = time.monotonic() + 30
                while time.monotonic() < end:
                    batch = api('/calls/' + call + '/events?after=' + str(cursor))
                    cursor = batch['after']; events.extend(batch['events'])
                    after = api('/status')
                    if after.get('active') is None:
                        break
                else:
                    raise RuntimeError('own call cleanup did not reach idle')
            except Exception as exc:
                error = (error or '') + '; cleanup: ' + str(exc)[:200]
        after = api('/status')
        receipt = {**intent, 'call': call, 'events': events, 'error': error,
                   'after': after, 'quality_accepted': False,
                   'native_transcript_conditioning': False}
        if allow_cancellation:
            receipt['response_outcomes'] = response_outcomes(events)
        pin = nas.put(destination, json.dumps(receipt, ensure_ascii=False).encode())
        print(json.dumps({'pin': pin, 'call': call, 'error': error,
                          'ready': after.get('ready'), 'active': after.get('active'),
                          'event_count': len(events)}, ensure_ascii=False), flush=True)
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
