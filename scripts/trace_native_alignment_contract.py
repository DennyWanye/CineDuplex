"""Isolate training-window lookahead using recorded audio and resident encoder."""
import ast
import hashlib
import io
import json
from pathlib import Path
import time
import types
import wave

import librosa
import numpy as np
import torch
import torch.nn.functional as F


def run(call, spec):
    root = Path(call.app.root).parents[1]
    model, fw = call.app.model, call.app.fw
    begun = time.monotonic()
    out = {'schema':1,'call':call.id,'at':time.time(),'microphone':False,
           'speaker':False,'generated_audio':False,'weight_mutation':False,
           'recognized_text_input':False,'quality_accepted':False,
           'mel':[],'lookahead':[],'errors':[]}

    def read(item):
        data=(root/item['path']).read_bytes()
        if len(data)>2000000 or hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('pinned small source changed')
        return data

    def diff(a,b):
        a,b=a.detach().float().cpu(),b.detach().float().cpu()
        if a.shape!=b.shape:raise ValueError('comparison shapes differ')
        delta=a-b
        return {'relative_l2':float(delta.norm()/b.norm().clamp_min(1e-9)),
                'max_abs':float(delta.abs().max()),'equal':bool(torch.equal(a,b)),
                'bf16_equal_fraction':float((a.bfloat16()==b.bfloat16()).float().mean()),
                'shape':list(a.shape)}

    try:
        training=json.loads(read(spec['training_receipt']))
        if not training['run_identity_bound'] or not training['feature_descriptor_rows_equal']:
            raise ValueError('historical training binding failed')
        out['training_receipt']=spec['training_receipt']
        tree=ast.parse(read(spec['dataset']).decode())
        names={'_mel_filters','log_mel_spectrogram','compute_token_num'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LazySupervisedDataset')
        nodes.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='stepaudio_audio_prepreocess'))
        scope={'torch':torch,'np':np,'F':F,'librosa':librosa,'BatchFeature':dict}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-training-frontend>','exec'),scope)
        pre=scope['stepaudio_audio_prepreocess']
        short=types.SimpleNamespace(window_second=.4,align_audio_input=True,
                 audio_token_id=fw.audio_token_id,audio_pad_token_id=fw.audio_pad_token_id)
        long=types.SimpleNamespace(**vars(short));long.window_second=training['training_configuration']['window_second']
        out['windows']={'frozen_training':long.window_second,'runtime':fw.window_second}

        def features(data,options):
            batch=pre(options,data)
            if len(batch['feats'])!=1:raise ValueError('bounded one-window comparison required')
            mel=batch['feats'][0].T.unsqueeze(0).to(model.device).bfloat16()
            lens=torch.tensor(batch['feats_lengths'],device=model.device,dtype=torch.int32)
            encoded,n=model.encoder(mel,lens);adapted=model.adapter(encoded)
            length=int(((n-1)//2+1)[0]);return adapted[:,:length]

        with torch.inference_mode():
            for case in spec['cases']:
                with wave.open(io.BytesIO(read(case))) as wav:
                    if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(16000,1,2):
                        raise ValueError('original recording format')
                    audio=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768.
                for offset in range(0,len(audio),6400):
                    if call.closed.is_set() or time.monotonic()-begun>60:raise RuntimeError('probe time/cancel boundary')
                    clip=audio[offset:offset+6400]
                    a,b=pre(short,clip),fw.stepaudio_audio_prepreocess(clip)
                    out['mel'].append({'case':case['id'],'start_sample':offset,
                        'ids_equal':a['input_ids']==b['input_ids'],
                        'lens_equal':a['feats_lengths']==b['feats_lengths'],
                        **diff(a['feats'][0],b['feats'][0])})
                # Exact same first 400ms. Only future waveform is changed.
                altered=audio.copy();altered[6400:]=0
                causal=features(audio[:6400],short)
                whole=features(audio,long)[:,:causal.shape[1]]
                future_zero=features(altered,long)[:,:causal.shape[1]]
                out['lookahead'].append({'case':case['id'],'source':case,
                    'unchanged_prefix_samples':6400,
                    'long_chunk_vs_causal':diff(whole,causal),
                    'same_prefix_changed_future':diff(whole,future_zero),
                    'interpretation':'future-dependent features, not proof of the semantic/noise root cause'})
    except Exception as exc:
        out['errors'].append(type(exc).__name__+': '+str(exc)[:400]);raise
    finally:
        out['elapsed_seconds']=time.monotonic()-begun
        target=call.folder/'native-alignment-contract.json';temp=target.with_suffix('.pending')
        temp.write_text(json.dumps(out,ensure_ascii=False));temp.replace(target)
