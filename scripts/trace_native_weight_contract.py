"""Bounded read-only audit of resident native weights and training alignment.

Invoked only by an owned, Gateway-admitted diagnostic call. No generation,
microphone, speaker, transcript injection, weight mutation or second model.
Sparse value comparisons are explicitly not a full resident-weight hash.
"""
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import time
import wave

import numpy as np
import torch
import torch.nn.functional as F


def run(call, spec):
    root = Path(call.app.root).parents[1]
    model, fw = call.app.model, call.app.fw
    begun = time.monotonic()
    out = {'schema': 1, 'call': call.id, 'at': time.time(),
           'identity': dict(call.app.identity), 'microphone': False,
           'speaker': False, 'generation': False, 'weight_mutation': False,
           'full_weight_hash': False, 'quality_accepted': False,
           'weights': [], 'config_differences': [], 'training_alignment': [],
           'errors': []}

    def budget():
        if call.closed.is_set() or time.monotonic() - begun > 90:
            raise RuntimeError('weight-contract time/cancel boundary')

    def pinned(item):
        data = (root / item['path']).read_bytes()
        if len(data) > 2000000 or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('small evidence identity: ' + item['path'])
        return data

    try:
        published = json.loads(pinned(spec['config']))
        index = json.loads(pinned(spec['index']))['weight_map']
        fields = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                  'num_attention_heads', 'num_key_value_heads', 'hidden_act',
                  'max_position_embeddings', 'rms_norm_eps', 'rope_theta',
                  'rope_scaling', 'use_sliding_window', 'vocab_size')
        out['configs'] = {}
        for name in ('text_config', 'stoken_layer_config', 'control_layer_config',
                     'merge_layer_config', 'audio_encoder_config'):
            actual = getattr(model.config, name).to_dict()
            keys = fields if name != 'audio_encoder_config' else tuple(published[name])
            row = {k: {'published': published[name].get(k), 'resident': actual.get(k)}
                   for k in keys if k != 'model_type'}
            out['configs'][name] = row
            for key, values in row.items():
                if values['published'] != values['resident']:
                    out['config_differences'].append({'field': name+'.'+key, **values})
        scalars = ('control_branch_layer', 'control_token_chunk_size', 'stoken_delay_num',
                   'adding_text_hiddenstates', 'no_text_label', 'no_stoken_label')
        scalars += tuple(k for k in published if k.endswith('_token_id'))
        out['scalar_config'] = {k: {'published': published[k],
                                  'resident': getattr(model.config, k, None)} for k in scalars}
        for k, values in out['scalar_config'].items():
            if values['published'] != values['resident']:
                out['config_differences'].append({'field': k, **values})
        out['branches'] = {'stoken_index': model.stoken_model_index,
                           'control_index': model.control_model_index,
                           'expected_stoken_index': published['text_config']['num_hidden_layers'] - published['stoken_layer_config']['num_hidden_layers'],
                           'expected_control_index': published['text_config']['num_hidden_layers'] - published['control_branch_layer']}
        out['runtime'] = {k: getattr(fw, k, None) for k in ('window_second',
                          'align_audio_input', 'stoken_delay_num', 'control_token_chunk_size')}
        out['tokenizer'] = {}
        for token in ('<audio_patch>', '<tts_start>', '<tts_end>', '<tts_pad>', '<|EOT|>'):
            out['tokenizer'][token] = fw.tokenizer.encode(token, add_special_tokens=False)
        # Read headers and three deterministic 32-element slices per tensor.
        # Never mmap/load all shards or initialize another model.
        dtype_map = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
                     'I64': torch.int64, 'BOOL': torch.bool}
        read_bytes = 0
        for shard in spec['shards']:
            budget()
            path = root / 'raw/lychee-full-duplex' / shard['path']
            if path.stat().st_size != shard['size']:
                raise ValueError('shard size differs')
            with path.open('rb', buffering=0) as source:
                header_len = struct.unpack('<Q', source.read(8))[0]
                if header_len > 1000000:
                    raise ValueError('header exceeds bounded budget')
                header = json.loads(source.read(header_len)); read_bytes += 8+header_len
                names = sorted(k for k in header if k != '__metadata__')
                if set(names) != {k for k, v in index.items() if v == shard['path']}:
                    raise ValueError('shard/index tensor names differ')
                for name in names:
                    budget()
                    meta = header[name]
                    parent, leaf = name.rsplit('.', 1)
                    module = model.get_submodule(parent)
                    if module.__class__.__name__ == 'LoRALinear':
                        module = module.base
                    value = getattr(module, leaf)
                    dtype = dtype_map[meta['dtype']]
                    row = {'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                           'shape_matches': list(value.shape) == meta['shape'],
                           'dtype_matches': value.dtype == dtype, 'samples': []}
                    if not row['shape_matches'] or not row['dtype_matches'] or value.is_meta:
                        raise ValueError('resident tensor metadata mismatch: '+name)
                    flat = value.detach().reshape(-1)
                    count = min(32, flat.numel()); width = value.element_size()
                    random_start = int(hashlib.sha256(name.encode()).hexdigest()[:12],16) % (flat.numel()-count+1)
                    for start in sorted({0, random_start, flat.numel()-count}):
                        source.seek(8+header_len+meta['data_offsets'][0]+start*width)
                        data = source.read(count*width); read_bytes += len(data)
                        if len(data) != count*width:
                            raise ValueError('short tensor slice')
                        expected = torch.frombuffer(bytearray(data), dtype=dtype)
                        actual = flat[start:start+count].cpu()
                        row['samples'].append({'start': start, 'count': count,
                            'equal': bool(torch.equal(actual, expected))})
                    out['weights'].append(row)
        out['source_bytes_read'] = read_bytes
        out['tensor_count'] = len(out['weights'])
        out['sampled_value_count'] = sum(s['count'] for r in out['weights'] for s in r['samples'])
        out['value_mismatches'] = [r['name'] for r in out['weights'] if not all(s['equal'] for s in r['samples'])]
        # Independently execute the training Dataset's pure frontend on the
        # same two original recordings. Its shell defaults are not checkpoint
        # config; align against the published checkpoint and live framework.
        import librosa
        tree = ast.parse(pinned(spec['dataset']).decode())
        names = {'_mel_filters', 'log_mel_spectrogram', 'compute_token_num'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='LazySupervisedDataset')
        nodes.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='stepaudio_audio_prepreocess'))
        scope = {'torch': torch, 'np': np, 'F': F, 'librosa': librosa, 'BatchFeature': dict}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-training-frontend>','exec'),scope)
        for case in spec['cases']:
            with wave.open(io.BytesIO(pinned(case))) as wav:
                if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth()) != (16000,1,2):
                    raise ValueError('original recording format differs')
                pcm = wav.readframes(wav.getnframes())
            for offset in range(0,len(pcm),12800):
                budget()
                samples = np.frombuffer(pcm[offset:offset+12800],dtype='<i2').astype(np.float32)/32768.
                training = scope['stepaudio_audio_prepreocess'](fw,samples)
                runtime = fw.stepaudio_audio_prepreocess(samples)
                out['training_alignment'].append({'case':case['id'],'start_sample':offset//2,
                    'ids_equal':training['input_ids']==runtime['input_ids'],
                    'lens_equal':training['feats_lengths']==runtime['feats_lengths'],
                    'mel_equal': all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(training['feats'],runtime['feats']))})
    except Exception as exc:
        out['errors'].append(type(exc).__name__+': '+str(exc)[:500])
        raise
    finally:
        out['elapsed_seconds'] = time.monotonic()-begun
        target = call.folder/'native-weight-contract.json'
        temp = target.with_suffix('.pending')
        temp.write_text(json.dumps(out,ensure_ascii=False));temp.replace(target)
