"""Private, read-only native-input diagnosis on the admitted resident model.

Invoked by an owned codec-probe call, never as a standalone GPU process.
No transcript, expected answer, altered sampler, output audio or device I/O.
The original upstream preprocessor and scatter layout are independent oracles;
equality here does not prove semantic understanding or subjective quality.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import time
import types
import wave

import numpy as np
import torch
import torch.nn.functional as F


def difference(left, right):
    if left.shape != right.shape:
        return {'equal': False, 'left_shape': list(left.shape), 'right_shape': list(right.shape)}
    a, b = left.float(), right.float()
    delta = a - b
    return {'equal': bool(torch.equal(left, right)), 'shape': list(left.shape),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            'max_abs': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / b.norm().clamp_min(1e-9))}


def pinned(root, spec):
    path = root / spec['path']
    raw = path.read_bytes()
    if len(raw) > 2000000 or hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise RuntimeError('diagnostic source/recording identity changed: ' + spec['path'])
    return raw


def reference_preprocessor(source):
    # Execute only the named pure functions from the pinned upstream snapshot.
    # Do not import its alternative backend or construct another model.
    import librosa
    tree = ast.parse(source)
    names = {'_mel_filters', 'log_mel_spectrogram', 'compute_token_num'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SingleTurnGenerationFramework')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == 'stepaudio_audio_prepreocess')
    nodes.append(fn)
    for n in nodes:
        n.decorator_list = []
    scope = {'torch': torch, 'np': np, 'F': F, 'librosa': librosa,
             'IS_CUDA': True, 'BatchFeature': dict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-upstream-preprocessor>', 'exec'), scope)
    return scope['stepaudio_audio_prepreocess']


def run(call, spec):
    from lychee_fd.runtime import vllm_generation
    from cineduplex.train.realtime_service import conversation_instruction, prepare_adapter, optimize_bf16
    root = Path(call.app.root).parents[1]
    model, fw, tp = call.app.model, call.app.fw, call.app.tensor_parallel
    record = {'schema': 1, 'at': time.time(), 'call': call.id,
              'model_identity': dict(call.app.identity), 'microphone': False,
              'speaker': False, 'released_audio_events': 0, 'recognized_text_input': False,
              'parameters_changed': False, 'quality_accepted': False,
              'cache_trace_requested': bool(spec.get('cache_trace')),
              'frontend': [], 'embeddings': [], 'semantic': [], 'events': [], 'errors': []}
    target = call.folder / 'native-input-contract.json'

    def save():
        temporary = target.with_suffix('.pending')
        temporary.write_text(json.dumps(record, ensure_ascii=False))
        temporary.replace(target)

    def check_budget():
        if call.closed.is_set() or time.monotonic() - begun > 180:
            raise RuntimeError('owned native trace canceled or time budget exceeded')

    begun = time.monotonic()
    save()
    session = None
    started = False
    restore = []
    handles = []
    try:
        if model.get_input_embeddings().weight.dtype != torch.bfloat16:
            raise RuntimeError('original BF16 model required')
        original_source = pinned(root, spec['upstream_runtime']).decode()
        reference_pre = reference_preprocessor(original_source)
        inputs = []
        for case in spec['cases']:
            raw = pinned(root, case)
            with wave.open(io.BytesIO(raw)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                    raise RuntimeError('original PCM16 mono 16k recording required')
                pcm = wav.readframes(wav.getnframes())
            inputs.append((case['id'], pcm))
        if len(inputs) != 2 or sum(len(p) for _, p in inputs) > 512000:
            raise RuntimeError('bounded original two-recording scope required')
        record['inputs'] = spec['cases']
        # No distributed language forward in this phase. Reuse resident audio
        # weights; compare every actual recording window against upstream math.
        prior = spec.get('frontend_receipt')
        if prior:
            receipt = json.loads(pinned(root, prior))
            rows = receipt['frontend']
            if receipt['inputs'] != spec['cases'] or len(rows) != 35 or not all(
                    row['ids_equal'] and row['lens_equal'] and all(row[key]['equal']
                    for key in ('mel', 'features', 'packing')) for row in rows):
                raise RuntimeError('prior frontend evidence does not match the original inputs')
            record['frontend_reused'] = prior
        with torch.inference_mode():
            for case_id, pcm in ([] if prior else inputs):
                session = fw.create_incremental_stream_session(audio_incremental_mode=True)
                for offset in range(0, len(pcm), 12800):
                    check_budget()
                    audio = np.frombuffer(pcm[offset:offset+12800], dtype='<i2').astype(np.float32) / 32768.
                    expected = reference_pre(fw, audio)
                    actual = fw.stepaudio_audio_prepreocess(audio)
                    session._append_incremental_audio(audio, flush_audio_tail=False)
                    mel = expected['feats'][0].T.unsqueeze(0).to(model.device).bfloat16()
                    lens = torch.tensor(expected['feats_lengths'], device=model.device, dtype=torch.int32)
                    encoded, encoded_lens = model.encoder(mel, lens)
                    projected = model.adapter(encoded)
                    length = int(((encoded_lens - 1) // 2 + 1)[0])
                    packed, packed_lens = session._packed_audio_embeds()
                    cached = session._audio_embed_seq_cache[-1]
                    row = {'case': case_id, 'start_sample': offset // 2,
                           'ids_equal': actual['input_ids'] == expected['input_ids'],
                           'lens_equal': actual['feats_lengths'] == expected['feats_lengths'],
                           'mel': difference(actual['feats'][0], expected['feats'][0]),
                           'features': difference(cached, projected[0, :length]),
                           'packing': difference(packed[len(session._audio_embed_seq_cache)-1, :length], cached),
                           'patch_count': actual['input_ids'].count(fw.audio_token_id),
                           'feature_count': length, 'packed_length': int(packed_lens[-1])}
                    record['frontend'].append(row)
                session.close(); session = None
        record['phase'] = 'frontend_complete'
        save()
        # Actual native generation with unchanged four-rank transport/samplers.
        # Only q1 is needed to localize the first wrong answer; q2's entire
        # frontend has already been checked. No hard-coded answer is supplied.
        prepare_adapter(model, .25); optimize_bf16(model, call.app.root)
        profile_id = call.id
        record['profile_id'] = profile_id
        tp.start(call.voice, call.language, .25, profile_id=profile_id); started = True
        tp.install(model)
        fw.last_ss_pos = None
        fw.model._merge_cache_by_past_id.clear()
        vllm_generation.SYSTEM_MESSAGE_PREFIX = '<|BOT|>system\n' + conversation_instruction(call.language) + '<|EOT|>'
        session = fw.create_incremental_stream_session(start_speak_token_factor=1.,
            start_listen_token_factor=1., audio_incremental_mode=True)
        state = {'window': -1, 'pending': None, 'sample_phase': 0}
        original_forward = model.forward
        original_sample = model._sample_from_logits
        restore.extend([(model, 'forward', original_forward), (model, '_sample_from_logits', original_sample)])

        def inspect_forward(current, *args, **kw):
            # The reference uses upstream full masked_scatter, independent of
            # the installed index-select/cache optimization. Never alter kwargs.
            cached = kw.get('pre_computed_audio')
            if cached is None:
                cached = current._realtime_audio
            features, lens = cached
            audio_ids = kw['audio_input_ids']
            mask = torch.arange(features.shape[1], device=features.device)[None, :] < lens[:, None]
            ref_audio = current.get_input_embeddings()(audio_ids)
            ref_audio = ref_audio.masked_scatter(audio_ids.eq(fw.audio_token_id).unsqueeze(-1).expand_as(ref_audio),
                                               features[mask].to(ref_audio.dtype))
            prefix = kw['prefix_input_ids'].shape[1]
            zeros = ref_audio.new_zeros((1, prefix, ref_audio.shape[-1]))
            ref_audio = torch.cat((zeros, ref_audio), 1)[:, kw['cache_position']]
            text = current.get_input_embeddings()(kw['input_ids'])
            if current.no_text_label:
                text[:, prefix:] = 0
            control = current.get_input_embeddings()(kw['control_input_ids'])
            speech = current.get_input_embeddings()(kw['stoken_ids'])
            past = kw.get('past_key_values')
            if past is None or past.get_seq_length() == 0:
                control = torch.cat((zeros, control), 1)
                speech = torch.cat((zeros, speech), 1)
            state['pending'] = (text + control + speech + ref_audio,
                {'window': state['window'], 'state': session.listening_state,
                 'position': kw['cache_position'].tolist(),
                 'audio_tokens': int(audio_ids.numel()), 'features': int(lens.sum()),
                 'patch_count': int(audio_ids.eq(fw.audio_token_id).sum()),
                 'cache_before': int(past.get_seq_length()) if past is not None else 0})
            state['sample_phase'] = 0

        def forward(current, *args, **kw):
            state['sample_phase'] = 0
            try:
                inspect_forward(current, *args, **kw)
            except Exception as exc:
                state['pending'] = None
                record['errors'].append('input observation: ' + str(exc)[:200])
            return original_forward(*args, **kw)

        def inspect_main_input(module, args, kw):
            expected, row = state['pending']
            row['comparison'] = difference(kw['inputs_embeds'], expected)
            record['embeddings'].append(row)
            state['pending'] = None

        def main_input(module, args, kw):
            if state['pending'] is None:
                return
            try:
                inspect_main_input(module, args, kw)
            except Exception as exc:
                record['errors'].append('embedding observation: ' + str(exc)[:200])

        def sample(current, *args, **kw):
            phase = state['sample_phase']; state['sample_phase'] += 1
            # Copy raw logits BEFORE calling the original processor exactly once.
            if phase == 0:
                values, ids = args[0][0, :151643].float().topk(8)
                raw = (values.tolist(), ids.tolist())
            chosen = original_sample(*args, **kw)
            if phase == 0:
                token = int(chosen[0, 0])
                if session.listening_state in ('s', 'b') and token < 151643 and len(record['semantic']) < 40:
                    record['semantic'].append({'window': state['window'],
                        'state': session.listening_state,
                        'input_end_sample': (state['window']+1)*6400,
                        'token': token, 'text': fw.tokenizer.decode([token]),
                        'candidates': [{'id': i, 'text': fw.tokenizer.decode([i]), 'logit': v}
                                       for v, i in zip(*raw)]})
            return chosen

        model.forward = types.MethodType(forward, model)
        model._sample_from_logits = types.MethodType(sample, model)
        handles.append(model.model.register_forward_pre_hook(main_input, with_kwargs=True))
        case_id, pcm = inputs[0]
        pcm += bytes(16000*2*12)
        with torch.inference_mode():
            for seq, offset in enumerate(range(0, len(pcm), 12800)):
                check_budget()
                state['window'] = seq
                packet = pcm[offset:offset+12800]
                audio = np.frombuffer(packet, dtype='<i2').astype(np.float32) / 32768.
                # Match the original Call.feed/loop ingress contract. This
                # synchronous file replay has no pending ingress queue; it is
                # not a replay of the original wall-clock network timing.
                if call.ingress_speech is None:
                    raise RuntimeError('original ingress VAD is required')
                if call.ingress_speech.feed(audio):
                    call.ingress_revision += 1
                evidence = call.ingress_speech.evidence
                turn_control = {'revision': call.ingress_revision, 'caught_up': True,
                    'quiet': bool(evidence and not evidence.active and evidence.quiet_frames >= 20)}
                record.setdefault('ingress', []).append({'window': seq, **turn_control})
                tp.audio(packet, turn_control=turn_control)
                for event in session.advance_stream(audio, audio_is_incremental=True, emit_generation_complete=False):
                    if event['type'] == 'speaking_token':
                        fw.runtime_speech_steps += 1
                    if event['type'] in ('speaking_token', 'state_change', 'speaking_done', 'control_decision'):
                        record['events'].append({'window': seq, **event})
                tp.finish_window()
                if record['errors']:
                    raise RuntimeError('native observation failed after completed window')
                if seq % 5 == 4:
                    save()
                if len(record['semantic']) >= 16:
                    break
        record['ok'] = True
    except Exception as exc:
        record['errors'].append(type(exc).__name__ + ': ' + str(exc)[:500])
        raise
    finally:
        for handle in handles:
            handle.remove()
        for obj, key, value in reversed(restore):
            setattr(obj, key, value)
        if session is not None:
            session.close()
        if started:
            tp.stop()
        fw.model._merge_cache_by_past_id.clear()
        record['seconds'] = time.monotonic() - begun
        save()
